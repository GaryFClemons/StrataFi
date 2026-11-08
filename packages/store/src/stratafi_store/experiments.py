"""Experiments: recording a plan, and running its trials within a budget.

An experiment is created ``planned``, with its trials in plan order (``create_experiment``).
Nothing runs until someone confirms the plan: ``start_experiment`` plans the trials again from
the pinned base documents, refuses if the plan changed (``plan_stale``), saves each distinct
pipeline spec and retrieval config as a variant of its base (``variant_name``; a trial equal
to its base uses the base's version), admits the first ``budget.max_trials`` trials and skips
the rest, and marks the experiment ``running``.

``ExperimentOrchestrator.advance`` (the worker's ``experiment.advance`` job) moves a running
experiment on, always holding its row lock:

1. Trials whose eval run finished record their objective, measures and constraints; trials
   whose pipeline run or eval run failed or was cancelled end so.
2. ``spent`` adds up the finished runs' estimated cost and the hours since the start. Over a
   budget, the experiment is ``stopping``: no new run starts, pending trials are skipped, and
   the runs in flight are cancelled. Finished trials stay valid.
3. Otherwise trials start in plan order: a trial attaches to its pipeline's run, which trials
   that differ only in retrieval share, or starts it while fewer than
   ``execution.max_parallel_runs`` run (with one, every shared prefix runs once and later runs
   find it cached); a trial whose pipeline run succeeded gets its eval run on the dev split, up
   to ``execution.max_parallel_evals`` at once, so the leaderboard fills in while later trials
   embed.
4. With successive halving, a trial's eval run scores its rung's seeded sample of the
   split (``eval_options``), and a trial whose rung isn't the last waits for the rest of its
   rung. When every trial left has its rung's result, ``_decide`` prunes the rung
   (``stratafi_core.adaptive.prune``): the trials clearly behind its leader end ``pruned``
   with the rung's estimate, and the rest move to the next rung for a bigger sample. The last
   rung scores every query, and its trials finish like a grid's.
5. When no trial is pending or running, it ends: ``succeeded``, ``stopped`` (a budget, or
   trials over ``max_trials``) or ``cancelled``.

Runs tell it when they finish: ``finish_hook`` is the run and eval-run orchestrators'
``on_finish``, and queues an ``experiment.advance`` job for each experiment with a trial on the
run. ``reconcile`` (the worker's heartbeat) queues one for any experiment that missed it, or
whose time budget ran out. Each change notifies an ``experiment`` event with the experiment's ID
as ``run_id``, so ``GET /events?run_id=`` follows experiments too.
"""

import json
import statistics
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from stratafi_core.adaptive import Rung, halving_schedule, prune
from stratafi_core.components import ComponentRegistry
from stratafi_core.experiments import (
    Budget,
    ExperimentSpec,
    History,
    check_constraints,
    measures,
    metric_estimate,
    plan_trials,
    rerank_key,
    variant_description,
    variant_name,
)
from stratafi_core.specs import (
    PipelineSpec,
    RetrievalConfig,
    document_hash,
    dump_document,
    spec_hash,
)
from stratafi_store import evalruns, evalsets, runs, specs
from stratafi_store.evalruns import EvalOrchestrator, EvalRun
from stratafi_store.jobs import EVENTS_CHANNEL, JobQueue
from stratafi_store.runs import FinishHook, Orchestrator, Run
from stratafi_store.tables import (
    ACTIVE_EXPERIMENT_STATUSES,
    annotations,
    confirmations,
    eval_query_results,
    eval_runs,
    experiments,
    jobs,
    stage_tasks,
    trials,
)
from stratafi_store.tables import runs as runs_table

ADVANCE_JOB = "experiment.advance"
FINISHED = ("succeeded", "stopped", "failed", "cancelled")
TRIAL_DONE = ("succeeded", "failed", "skipped", "cancelled", "pruned")


class PlanChanged(Exception):
    """The trials planned again from the pinned documents aren't the ones recorded."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Experiment:
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    status: str
    document: dict[str, Any]
    document_hash: str
    corpus_version_id: uuid.UUID
    eval_set_version_id: uuid.UUID
    pipeline_spec_id: uuid.UUID
    retrieval_config_id: uuid.UUID
    plan: dict[str, Any]
    spent: dict[str, Any]
    counts: dict[str, Any]
    stop_reason: str | None
    error_code: str | None
    created_by: str
    started_by: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def finished(self) -> bool:
        return self.status in FINISHED

    def spec(self) -> ExperimentSpec:
        return ExperimentSpec.model_validate(self.document)


@dataclass(frozen=True)
class Trial:
    id: uuid.UUID
    workspace_id: uuid.UUID
    experiment_id: uuid.UUID
    ordinal: int
    grid_index: int
    levels: list[int]
    factor_values: dict[str, str]
    pipeline_hash: str
    retrieval_hash: str
    pipeline_spec_id: uuid.UUID | None
    retrieval_config_id: uuid.UUID | None
    pipeline_run_id: uuid.UUID | None
    eval_run_id: uuid.UUID | None
    status: str
    objective: dict[str, Any] | None
    measures: dict[str, Any]
    constraints: list[dict[str, Any]]
    feasible: bool | None
    error_code: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    rung: int
    rungs: list[dict[str, Any]]


def _experiment(row: Any) -> Experiment:
    return Experiment(**{column.name: getattr(row, column.name) for column in experiments.columns})


def _trial(row: Any) -> Trial:
    return Trial(**{column.name: getattr(row, column.name) for column in trials.columns})


# --- recording and reading --------------------------------------------------------------------


def create_experiment(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    spec: ExperimentSpec,
    document: Mapping[str, Any],
    document_hash: str,
    corpus_version_id: uuid.UUID,
    eval_set_version_id: uuid.UUID,
    pipeline_spec_id: uuid.UUID,
    retrieval_config_id: uuid.UUID,
    plan: Mapping[str, Any],
    trial_rows: Sequence[Mapping[str, Any]],
    created_by: str,
) -> Experiment:
    """A planned experiment with its trials. ``trial_rows`` give each trial's ordinal, grid
    index, levels, factor values and hashes."""
    experiment_id = uuid.uuid4()
    row = conn.execute(
        sa.insert(experiments)
        .values(
            id=experiment_id,
            workspace_id=workspace_id,
            name=spec.name,
            document=dict(document),
            document_hash=document_hash,
            corpus_version_id=corpus_version_id,
            eval_set_version_id=eval_set_version_id,
            pipeline_spec_id=pipeline_spec_id,
            retrieval_config_id=retrieval_config_id,
            plan=dict(plan),
            counts={"trials": {"pending": len(trial_rows)}},
            created_by=created_by,
        )
        .returning(*experiments.columns)
    ).one()
    if trial_rows:
        conn.execute(
            sa.insert(trials),
            [
                {
                    "id": uuid.uuid4(),
                    "workspace_id": workspace_id,
                    "experiment_id": experiment_id,
                    **dict(trial),
                }
                for trial in trial_rows
            ],
        )
    experiment = _experiment(row)
    _notify(conn, experiment)
    return experiment


def get_experiment(
    conn: Connection, workspace_id: uuid.UUID | None, experiment_id: uuid.UUID
) -> Experiment | None:
    query = sa.select(experiments).where(experiments.c.id == experiment_id)
    if workspace_id is not None:
        query = query.where(experiments.c.workspace_id == workspace_id)
    row = conn.execute(query).first()
    return _experiment(row) if row is not None else None


def list_experiments(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    after: tuple[datetime, uuid.UUID] | None = None,
    status: str | None = None,
    include_archived: bool = True,
) -> list[Experiment]:
    """Newest first."""
    query = sa.select(experiments).where(experiments.c.workspace_id == workspace_id)
    if status is not None:
        query = query.where(experiments.c.status == status)
    if not include_archived:
        query = query.where(
            ~sa.exists().where(
                annotations.c.experiment_id == experiments.c.id,
                annotations.c.archived_at.is_not(None),
            )
        )
    if after is not None:
        query = query.where(sa.tuple_(experiments.c.created_at, experiments.c.id) < after)
    rows = conn.execute(
        query.order_by(experiments.c.created_at.desc(), experiments.c.id.desc()).limit(limit)
    )
    return [_experiment(row) for row in rows]


def trials_of(conn: Connection, experiment_id: uuid.UUID) -> list[Trial]:
    rows = conn.execute(
        sa.select(trials).where(trials.c.experiment_id == experiment_id).order_by(trials.c.ordinal)
    )
    return [_trial(row) for row in rows]


def get_trial(
    conn: Connection, workspace_id: uuid.UUID, experiment_id: uuid.UUID, trial_id: uuid.UUID
) -> Trial | None:
    row = conn.execute(
        sa.select(trials).where(
            trials.c.workspace_id == workspace_id,
            trials.c.experiment_id == experiment_id,
            trials.c.id == trial_id,
        )
    ).first()
    return _trial(row) if row is not None else None


def trial_counts(conn: Connection, experiment_id: uuid.UUID) -> dict[str, Any]:
    statuses = dict(
        conn.execute(
            sa.select(trials.c.status, sa.func.count())
            .where(trials.c.experiment_id == experiment_id)
            .group_by(trials.c.status)
        ).all()
    )
    return {"trials": {str(key): int(value) for key, value in statuses.items()}}


def _notify(conn: Connection, experiment: Experiment) -> None:
    event: dict[str, Any] = {
        "type": "experiment",
        "workspace_id": str(experiment.workspace_id),
        "run_id": str(experiment.id),
        "status": experiment.status,
        "counts": dict(experiment.counts or {}),
    }
    conn.execute(
        sa.text("SELECT pg_notify(:channel, :payload)"),
        {"channel": EVENTS_CHANNEL, "payload": json.dumps(event, separators=(",", ":"))},
    )


def _update(conn: Connection, experiment: Experiment, **values: Any) -> Experiment:
    row = conn.execute(
        sa.update(experiments)
        .where(experiments.c.id == experiment.id)
        .values(updated_at=sa.func.now(), **values)
        .returning(*experiments.columns)
    ).one()
    return _experiment(row)


def _set_trial(conn: Connection, trial: Trial, **values: Any) -> None:
    conn.execute(sa.update(trials).where(trials.c.id == trial.id).values(**values))


# --- starting -----------------------------------------------------------------------------------


def _save_variant(
    conn: Connection,
    kind: specs.DocumentKind,
    base: specs.StoredDocument,
    resolved: PipelineSpec | RetrievalConfig,
    created_by: str,
) -> uuid.UUID:
    """The saved version a trial runs: the base's own when the trial changes nothing, else the
    variant saved under its base's name and a digest of its configuration."""
    configuration = spec_hash(resolved)
    if configuration == base.spec_hash:
        return base.id
    document = dump_document(resolved)
    document["name"] = variant_name(base.name, configuration)
    document["description"] = variant_description(base.name, base.version, base.document, document)
    model = type(resolved).model_validate(document)
    stored, _ = specs.save_version(
        conn,
        kind,
        workspace_id=base.workspace_id,
        document=dump_document(model),
        spec_hash=configuration,
        document_hash=document_hash(model),
        created_by=created_by,
    )
    return stored.id


def start_experiment(
    conn: Connection,
    experiment: Experiment,
    registry: ComponentRegistry,
    *,
    started_by: str,
) -> Experiment:
    """Confirm a planned experiment (the module docstring says what that does). The caller
    holds no lock; this takes the experiment's. Raises ``PlanChanged``."""
    locked = conn.execute(
        sa.select(experiments).where(experiments.c.id == experiment.id).with_for_update()
    ).one()
    experiment = _experiment(locked)
    if experiment.status != "planned":
        raise PlanChanged("not_planned")
    spec = experiment.spec()
    base_pipeline = specs.get_by_id(
        conn, "pipeline", experiment.workspace_id, experiment.pipeline_spec_id
    )
    base_retrieval = specs.get_by_id(
        conn, "retrieval", experiment.workspace_id, experiment.retrieval_config_id
    )
    if base_pipeline is None or base_retrieval is None:
        raise PlanChanged("base_missing")
    planned = plan_trials(
        spec,
        PipelineSpec.model_validate(base_pipeline.document),
        RetrievalConfig.model_validate(base_retrieval.document),
        registry,
    )
    recorded = trials_of(conn, experiment.id)
    if planned.issues or [(t.pipeline_hash, t.retrieval_hash) for t in planned.trials] != [
        (t.pipeline_hash, t.retrieval_hash) for t in recorded
    ]:
        raise PlanChanged("plan_stale")
    pipelines: dict[str, uuid.UUID] = {}
    retrievals: dict[str, uuid.UUID] = {}
    admitted = spec.budget.max_trials
    for planned_trial, trial in zip(planned.trials, recorded, strict=True):
        if trial.ordinal >= admitted:
            _set_trial(
                conn, trial, status="skipped", error_code="max_trials", finished_at=sa.func.now()
            )
            continue
        if planned_trial.pipeline_hash not in pipelines:
            pipelines[planned_trial.pipeline_hash] = _save_variant(
                conn, "pipeline", base_pipeline, planned_trial.pipeline, started_by
            )
        if planned_trial.retrieval_hash not in retrievals:
            retrievals[planned_trial.retrieval_hash] = _save_variant(
                conn, "retrieval", base_retrieval, planned_trial.retrieval, started_by
            )
        _set_trial(
            conn,
            trial,
            pipeline_spec_id=pipelines[planned_trial.pipeline_hash],
            retrieval_config_id=retrievals[planned_trial.retrieval_hash],
        )
    experiment = _update(
        conn,
        experiment,
        status="running",
        started_by=started_by,
        started_at=sa.func.now(),
        counts=trial_counts(conn, experiment.id),
        spent=_spent([], [], datetime.now(UTC), datetime.now(UTC)),
    )
    _notify(conn, experiment)
    return experiment


def enqueue_advance(
    conn: Connection, queue: JobQueue, experiment: Experiment, *, max_attempts: int = 3
) -> None:
    queue.enqueue_in(
        conn,
        workspace_id=experiment.workspace_id,
        kind=ADVANCE_JOB,
        payload={"experiment_id": str(experiment.id)},
        resource_class="io",
        priority=5,
        max_attempts=max_attempts,
    )


def finish_hook(queue: JobQueue, *, max_attempts: int = 3) -> FinishHook:
    """The run and eval-run orchestrators' ``on_finish``: queue an ``experiment.advance`` for
    each experiment with a trial on the run."""

    def on_finish(conn: Connection, kind: str, workspace_id: uuid.UUID, run_id: uuid.UUID) -> None:
        column = trials.c.pipeline_run_id if kind == "run" else trials.c.eval_run_id
        waiting = conn.execute(
            sa.select(trials.c.experiment_id).where(column == run_id).distinct()
        ).scalars()
        for experiment_id in waiting:
            queue.enqueue_in(
                conn,
                workspace_id=workspace_id,
                kind=ADVANCE_JOB,
                payload={"experiment_id": str(experiment_id)},
                resource_class="io",
                priority=5,
                max_attempts=max_attempts,
            )

    return on_finish


# --- spending -----------------------------------------------------------------------------------


def run_cost(run: Run | None) -> float:
    """A finished pipeline run's estimated cost, from its manifest."""
    if run is None or not run.manifest:
        return 0.0
    return float((run.manifest.get("cost") or {}).get("estimated") or 0.0)


def eval_cost(run: EvalRun) -> float:
    """What a finished eval run spent: its record counts the shards that ran, not the ones
    reused, while its metrics keep the cost of the configuration (``per_1k_queries``)."""
    return float(((run.manifest or {}).get("cost") or {}).get("estimated") or 0.0)


def _spent(
    pipeline_runs: Iterable[Run], eval_list: Iterable[EvalRun], since: datetime, now: datetime
) -> dict[str, Any]:
    ingest = sum(run_cost(run) for run in pipeline_runs if run.finished)
    evals = sum(eval_cost(run) for run in eval_list if run.finished)
    return {
        "ingest_usd": round(ingest, 6),
        "eval_usd": round(evals, 6),
        "total_usd": round(ingest + evals, 6),
        "hours": round(max(0.0, (now - since).total_seconds()) / 3600, 6),
    }


def over_budget(budget: Budget, spent: Mapping[str, Any]) -> str | None:
    """Which budget the spending is over, if any: ``max_cost_usd`` or ``max_hours``."""
    if budget.max_cost_usd is not None and float(spent["total_usd"]) > budget.max_cost_usd:
        return "max_cost_usd"
    if budget.max_hours is not None and float(spent["hours"]) >= budget.max_hours:
        return "max_hours"
    return None


# --- the orchestrator ---------------------------------------------------------------------------

Cancel = tuple[str, uuid.UUID, uuid.UUID]  # (run or eval_run, workspace, ID)


# --- successive halving -----------------------------------------------------------------


def halving_of(
    experiment: Experiment, spec: ExperimentSpec, trial_list: Sequence[Trial]
) -> list[Rung] | None:
    """Successive halving's rungs for the trials the budget admitted over the split's queries
    (``halving_plan`` showed them); None for another strategy."""
    search = spec.search
    if search.method != "successive-halving":
        return None
    admitted = sum(1 for trial in trial_list if trial.error_code != "max_trials")
    queries = int((experiment.plan.get("eval_set") or {}).get("items") or 0)
    schedule = halving_schedule(admitted, queries, eta=search.eta, min_queries=search.min_queries)
    return schedule or None


def eval_options(
    spec: ExperimentSpec, schedule: Sequence[Rung] | None, rung: int
) -> dict[str, Any]:
    """A trial's eval run options: the experiment's, and with successive halving its rung's
    seeded sample, cut into shards of a size every rung's sample is a multiple of, so a rung
    reuses the shards of the rungs before it (the last rung scores every query)."""
    options = spec.eval_options.dump()
    if not schedule:
        return options
    search = spec.search
    if search.min_queries % spec.eval_options.shard_size:
        options["shard_size"] = search.min_queries
    sample: dict[str, Any] = {"seed": search.seed}
    if rung < len(schedule) - 1:
        sample["size"] = schedule[rung].queries
    options["sample"] = sample
    return options


def rung_record(
    spec: ExperimentSpec, schedule: Sequence[Rung], rung: int, run: EvalRun, *, kept: bool | None
) -> dict[str, Any]:
    """What a trial's ``rungs`` keeps of one rung: its sample, eval run, estimate and verdict
    (None: the experiment stopped before the rung was decided)."""
    objective = spec.objective
    return {
        "rung": rung,
        "queries": schedule[rung].queries if rung < len(schedule) else None,
        "eval_run_id": str(run.id),
        "objective": metric_estimate(
            run.metrics or {}, objective.rule, objective.checkpoint, objective.metric
        ),
        "kept": kept,
        "against_leader": None,
    }


class ExperimentOrchestrator:
    def __init__(
        self,
        engine: Engine,
        queue: JobQueue,
        runs_orchestrator: Orchestrator,
        evals_orchestrator: EvalOrchestrator,
        *,
        max_attempts: int = 3,
    ) -> None:
        self.engine = engine
        self.queue = queue
        self.runs = runs_orchestrator
        self.evals = evals_orchestrator
        self.max_attempts = max_attempts

    def _lock(self, conn: Connection, experiment_id: uuid.UUID) -> Experiment | None:
        row = conn.execute(
            sa.select(experiments).where(experiments.c.id == experiment_id).with_for_update()
        ).first()
        return _experiment(row) if row is not None else None

    def advance(self, experiment_id: uuid.UUID) -> Experiment | None:
        """One step of a running experiment (the module docstring says what it does)."""
        with self.engine.begin() as conn:
            experiment = self._lock(conn, experiment_id)
            if experiment is None or experiment.status not in ACTIVE_EXPERIMENT_STATUSES:
                return experiment
            experiment, cancels = self._advance(conn, experiment)
        self._cancel(cancels)
        return experiment

    def cancel(self, experiment_id: uuid.UUID, workspace_id: uuid.UUID) -> Experiment | None:
        """Cancel an experiment: a planned one at once; a running one stops starting work and
        cancels what's in flight, then ends ``cancelled``."""
        with self.engine.begin() as conn:
            experiment = self._lock(conn, experiment_id)
            if experiment is None or experiment.workspace_id != workspace_id:
                return None
            if experiment.finished:
                return experiment
            if experiment.status == "planned":
                conn.execute(
                    sa.update(trials)
                    .where(trials.c.experiment_id == experiment.id)
                    .values(status="skipped", error_code="cancelled", finished_at=sa.func.now())
                )
                experiment = _update(
                    conn,
                    experiment,
                    status="cancelled",
                    finished_at=sa.func.now(),
                    counts=trial_counts(conn, experiment.id),
                )
                _notify(conn, experiment)
                return experiment
            experiment = _update(conn, experiment, status="cancelling")
            experiment, cancels = self._advance(conn, experiment)
        self._cancel(cancels)
        return experiment

    def _cancel(self, cancels: Sequence[Cancel]) -> None:
        for kind, workspace_id, run_id in cancels:
            orchestrator = self.runs if kind == "run" else self.evals
            cancelled = orchestrator.cancel(run_id, workspace_id)
            if cancelled is None:
                continue
            for job_id in cancelled[1]:
                self.queue.cancel(job_id, workspace_id)

    def _advance(self, conn: Connection, experiment: Experiment) -> tuple[Experiment, list[Cancel]]:
        spec = experiment.spec()
        now = datetime.now(UTC)
        trial_list = trials_of(conn, experiment.id)
        pipeline_runs = runs.runs_by_id(
            conn, {t.pipeline_run_id for t in trial_list if t.pipeline_run_id}
        )
        eval_list = evalruns.eval_runs_by_id(
            conn, {t.eval_run_id for t in trial_list if t.eval_run_id}
        )
        stopping = experiment.status in ("stopping", "cancelling")
        schedule = halving_of(experiment, spec, trial_list)
        code = (experiment.stop_reason or "cancelled") if stopping else None
        for trial in trial_list:
            if trial.status == "running":
                self._settle(conn, spec, trial, pipeline_runs, eval_list, schedule, stop=code)
        spent = _spent(
            pipeline_runs.values(), eval_list.values(), experiment.started_at or now, now
        )
        if experiment.status == "running" and (reason := over_budget(spec.budget, spent)):
            experiment = _update(conn, experiment, status="stopping", stop_reason=reason)
            stopping = True
            for trial in trials_of(conn, experiment.id):
                if trial.status == "running":
                    self._settle(conn, spec, trial, pipeline_runs, eval_list, schedule, stop=reason)
        cancels: list[Cancel] = []
        if stopping:
            code = experiment.stop_reason if experiment.status == "stopping" else "cancelled"
            conn.execute(
                sa.update(trials)
                .where(trials.c.experiment_id == experiment.id, trials.c.status == "pending")
                .values(status="skipped", error_code=code, finished_at=sa.func.now())
            )
            cancels += [
                ("run", run.workspace_id, run.id)
                for run in pipeline_runs.values()
                if not run.finished
            ]
            cancels += [
                ("eval_run", run.workspace_id, run.id)
                for run in eval_list.values()
                if not run.finished
            ]
        else:
            if schedule is not None:
                self._decide(conn, experiment, spec, schedule, eval_list)
            self._schedule(conn, experiment, spec, schedule, pipeline_runs, eval_list)
        trial_list = trials_of(conn, experiment.id)
        values: dict[str, Any] = {"spent": spent, "counts": trial_counts(conn, experiment.id)}
        if all(trial.status in TRIAL_DONE for trial in trial_list):
            if experiment.status == "cancelling":
                values["status"] = "cancelled"
            elif experiment.status == "stopping":
                values["status"] = "stopped"
            elif any(trial.error_code == "max_trials" for trial in trial_list):
                values.update(status="stopped", stop_reason="max_trials")
            else:
                values["status"] = "succeeded"
            values["finished_at"] = sa.func.now()
        experiment = _update(conn, experiment, **values)
        _notify(conn, experiment)
        return experiment, cancels

    def _settle(
        self,
        conn: Connection,
        spec: ExperimentSpec,
        trial: Trial,
        pipeline_runs: Mapping[uuid.UUID, Run],
        eval_list: Mapping[uuid.UUID, EvalRun],
        schedule: Sequence[Rung] | None,
        *,
        stop: str | None,
    ) -> None:
        """Record a running trial whose runs ended. When stopping (``stop`` is why), a trial
        still waiting for its eval run to start ends cancelled, and one waiting for the rest
        of its rung is skipped."""
        stopping = stop is not None
        if trial.eval_run_id is not None:
            run = eval_list[trial.eval_run_id]
            waiting = schedule is not None and trial.rung < len(schedule) - 1
            if run.status == "succeeded" and waiting:
                if stopping:
                    record = rung_record(spec, schedule or [], trial.rung, run, kept=None)
                    _set_trial(
                        conn,
                        trial,
                        status="skipped",
                        error_code=stop,
                        rungs=[*trial.rungs, record],
                        finished_at=sa.func.now(),
                    )
                return
            if run.status == "succeeded":
                metrics = run.metrics or {}
                objective = spec.objective
                pipeline = (
                    pipeline_runs.get(trial.pipeline_run_id) if trial.pipeline_run_id else None
                )
                found = measures(
                    metrics,
                    objective=objective,
                    ingest_cost=run_cost(pipeline),
                    names=[constraint.metric for constraint in spec.constraints],
                )
                checked = check_constraints(spec.constraints, found)
                rungs = trial.rungs
                if schedule is not None:
                    rungs = [*rungs, rung_record(spec, schedule, trial.rung, run, kept=True)]
                _set_trial(
                    conn,
                    trial,
                    status="succeeded",
                    objective=metric_estimate(
                        metrics, objective.rule, objective.checkpoint, objective.metric
                    ),
                    measures=found,
                    constraints=checked,
                    feasible=all(result["passed"] for result in checked),
                    rungs=rungs,
                    finished_at=sa.func.now(),
                )
            elif run.finished:
                status = "cancelled" if run.status == "cancelled" else "failed"
                _set_trial(
                    conn,
                    trial,
                    status=status,
                    error_code=run.error_code or f"eval_{run.status}",
                    finished_at=sa.func.now(),
                )
            return
        pipeline = pipeline_runs.get(trial.pipeline_run_id) if trial.pipeline_run_id else None
        if pipeline is None or not pipeline.finished:
            return
        if pipeline.status != "succeeded" or stopping:
            status = "failed" if pipeline.status == "failed" else "cancelled"
            code = pipeline.error_code or ("run_failed" if status == "failed" else "cancelled")
            _set_trial(conn, trial, status=status, error_code=code, finished_at=sa.func.now())
        elif not _indexed(conn, pipeline):
            _set_trial(
                conn,
                trial,
                status="failed",
                error_code="run_not_indexed",
                finished_at=sa.func.now(),
            )

    def _decide(
        self,
        conn: Connection,
        experiment: Experiment,
        spec: ExperimentSpec,
        schedule: Sequence[Rung],
        eval_list: Mapping[uuid.UUID, EvalRun],
    ) -> None:
        """Decide successive halving's rung once every trial left in it has its result: the
        trials clearly behind the leader end pruned, the rest move to the next rung."""
        alive = [t for t in trials_of(conn, experiment.id) if t.status in ("pending", "running")]
        if not alive:
            return
        rung = min(trial.rung for trial in alive)
        if rung >= len(schedule) - 1:
            return
        results: dict[uuid.UUID, EvalRun] = {}
        for trial in alive:
            run = eval_list.get(trial.eval_run_id) if trial.eval_run_id else None
            if trial.rung != rung or run is None or run.status != "succeeded":
                return
            results[trial.id] = run
        objective = spec.objective
        found = query_values(
            conn,
            [run.id for run in results.values()],
            rule=objective.rule,
            checkpoint=objective.checkpoint,
            metric=objective.metric,
        )
        per_trial = {
            trial.ordinal: {
                row.item_id: row.value
                for row in found.get(results[trial.id].id, [])
                if row.value is not None
            }
            for trial in alive
        }
        shared = sorted(set.intersection(*(set(values) for values in per_trial.values())))
        decision = prune(
            {ordinal: [values[item] for item in shared] for ordinal, values in per_trial.items()},
            keep=schedule[rung].keep,
            goal=objective.goal,
            resamples=spec.eval_options.resamples,
            seed=spec.eval_options.seed,
        )
        for trial in alive:
            verdict = decision.verdicts[trial.ordinal]
            run = results[trial.id]
            record = rung_record(spec, schedule, rung, run, kept=verdict.kept)
            record["against_leader"] = (
                verdict.against_leader.as_dict() if verdict.against_leader else None
            )
            rungs = [*trial.rungs, record]
            if verdict.kept:
                _set_trial(conn, trial, rung=rung + 1, rungs=rungs, eval_run_id=None)
            else:
                _set_trial(
                    conn,
                    trial,
                    status="pruned",
                    objective=record["objective"],
                    rungs=rungs,
                    finished_at=sa.func.now(),
                )

    def _schedule(
        self,
        conn: Connection,
        experiment: Experiment,
        spec: ExperimentSpec,
        schedule: Sequence[Rung] | None,
        pipeline_runs: dict[uuid.UUID, Run],
        eval_list: dict[uuid.UUID, EvalRun],
    ) -> None:
        """Start trials in plan order within the parallel limits: pipeline runs first, then
        the eval runs of trials whose pipeline run succeeded."""
        created_by = experiment.started_by or experiment.created_by
        by_spec = {run.pipeline_spec_id: run for run in pipeline_runs.values()}
        active = sum(1 for run in pipeline_runs.values() if not run.finished)
        for trial in trials_of(conn, experiment.id):
            if trial.status != "pending" or trial.pipeline_spec_id is None:
                continue
            run = by_spec.get(trial.pipeline_spec_id)
            if run is None:
                if active >= spec.execution.max_parallel_runs:
                    continue
                run = runs.create_run(
                    conn,
                    workspace_id=experiment.workspace_id,
                    pipeline_spec_id=trial.pipeline_spec_id,
                    corpus_version_id=experiment.corpus_version_id,
                    spec_hash=trial.pipeline_hash,
                    created_by=created_by,
                )
                self.queue.enqueue_in(
                    conn,
                    workspace_id=experiment.workspace_id,
                    kind=runs.START_JOB,
                    payload={"run_id": str(run.id)},
                    resource_class="io",
                    max_attempts=self.max_attempts,
                )
                by_spec[trial.pipeline_spec_id] = pipeline_runs[run.id] = run
                active += 1
            _set_trial(
                conn, trial, status="running", pipeline_run_id=run.id, started_at=sa.func.now()
            )
        evaluating = sum(1 for run in eval_list.values() if not run.finished)
        for trial in trials_of(conn, experiment.id):
            if evaluating >= spec.execution.max_parallel_evals:
                break
            if trial.status != "running" or trial.eval_run_id is not None:
                continue
            pipeline = pipeline_runs.get(trial.pipeline_run_id) if trial.pipeline_run_id else None
            if (
                pipeline is None
                or pipeline.status != "succeeded"
                or trial.retrieval_config_id is None
            ):
                continue
            evaluation = evalruns.create_eval_run(
                conn,
                workspace_id=experiment.workspace_id,
                pipeline_run_id=pipeline.id,
                retrieval_config_id=trial.retrieval_config_id,
                eval_set_version_id=experiment.eval_set_version_id,
                split=spec.eval_set.split,
                options=eval_options(spec, schedule, trial.rung),
                retrieval_hash=trial.retrieval_hash,
                test_read=False,
                created_by=created_by,
            )
            self.queue.enqueue_in(
                conn,
                workspace_id=experiment.workspace_id,
                kind=evalruns.START_JOB,
                payload={"eval_run_id": str(evaluation.id)},
                resource_class="io",
                priority=5,
                max_attempts=self.max_attempts,
            )
            eval_list[evaluation.id] = evaluation
            _set_trial(conn, trial, eval_run_id=evaluation.id)
            evaluating += 1

    # --- the sweep ---------------------------------------------------------------------------

    def reconcile(self) -> int:
        """Queue an advance for each active experiment that has none queued and either a
        running trial whose run ended without telling it, or a time budget that ran out.
        Returns how many it queued."""
        finished = ("succeeded", "failed", "cancelled")
        with self.engine.connect() as conn:
            waiting = (
                sa.select(experiments.c.id)
                .where(experiments.c.status.in_(ACTIVE_EXPERIMENT_STATUSES))
                .where(
                    ~sa.exists().where(
                        jobs.c.kind == ADVANCE_JOB,
                        jobs.c.status.in_(("queued", "running")),
                        jobs.c.payload["experiment_id"].astext
                        == sa.cast(experiments.c.id, sa.Text),
                    )
                )
            )
            missed = sa.exists().where(
                trials.c.experiment_id == experiments.c.id,
                trials.c.status == "running",
                sa.or_(
                    sa.exists().where(
                        eval_runs.c.id == trials.c.eval_run_id, eval_runs.c.status.in_(finished)
                    ),
                    sa.and_(
                        trials.c.eval_run_id.is_(None),
                        sa.exists().where(
                            runs_table.c.id == trials.c.pipeline_run_id,
                            runs_table.c.status.in_(finished),
                        ),
                    ),
                ),
            )
            hours = experiments.c.document["budget"]["max_hours"].astext.cast(sa.Float)
            late = sa.and_(
                experiments.c.status == "running",
                hours.is_not(None),
                experiments.c.started_at + sa.func.make_interval(0, 0, 0, 0, 0, 0, hours * 3600)
                <= sa.func.now(),
            )
            rows = conn.execute(
                waiting.where(sa.or_(missed, late)).add_columns(experiments.c.workspace_id)
            ).all()
        for row in rows:
            self.queue.enqueue(
                workspace_id=row.workspace_id,
                kind=ADVANCE_JOB,
                payload={"experiment_id": str(row.id)},
                resource_class="io",
                priority=5,
                dedupe_key=f"{ADVANCE_JOB}:{row.id}",
                max_attempts=self.max_attempts,
            )
        return len(rows)


def _indexed(conn: Connection, run: Run) -> bool:
    return any(
        task.status in runs.DONE_TASK_STATUSES
        for task in runs.tasks_of(conn, run.id, stage="index")
    )


def stage_counts(conn: Connection, experiment_id: uuid.UUID) -> dict[str, dict[str, int]]:
    """The experiment's pipeline runs' tasks by stage: how many ran and how many were cached,
    to hold against the plan's unique work."""
    run_ids = (
        sa.select(trials.c.pipeline_run_id)
        .where(trials.c.experiment_id == experiment_id, trials.c.pipeline_run_id.is_not(None))
        .distinct()
    )
    found: dict[str, Counter[str]] = {}
    for row in conn.execute(
        sa.select(stage_tasks.c.stage, stage_tasks.c.status, sa.func.count())
        .where(stage_tasks.c.run_id.in_(run_ids))
        .group_by(stage_tasks.c.stage, stage_tasks.c.status)
    ):
        found.setdefault(row[0], Counter())[row[1]] += row[2]
    return {stage: dict(counter) for stage, counter in found.items()}


# --- what planning reads ------------------------------------------------------------------------

RECENT_EVAL_RUNS = 200


def history(conn: Connection, workspace_id: uuid.UUID) -> History:
    """What the workspace's past work says for estimates: each component version's median task
    time (tasks that ran, not cache hits), and from recent eval runs that scored their queries
    the median per-query time and each reranker's median cost per 1,000 queries."""
    task_ms = {
        (row.component_id, row.component_version): float(row.median)
        for row in conn.execute(
            sa.select(
                stage_tasks.c.component_id,
                stage_tasks.c.component_version,
                sa.func.percentile_cont(0.5)
                .within_group(stage_tasks.c.duration_ms.asc())
                .label("median"),
            )
            .where(
                stage_tasks.c.workspace_id == workspace_id,
                stage_tasks.c.status == "succeeded",
                stage_tasks.c.cache_hit.is_(False),
                stage_tasks.c.duration_ms.is_not(None),
            )
            .group_by(stage_tasks.c.component_id, stage_tasks.c.component_version)
        )
    }
    recent = conn.execute(
        sa.select(eval_runs.c.metrics, eval_runs.c.context)
        .where(
            eval_runs.c.workspace_id == workspace_id,
            eval_runs.c.status == "succeeded",
            eval_runs.c.results_cached.is_(False),
        )
        .order_by(eval_runs.c.created_at.desc())
        .limit(RECENT_EVAL_RUNS)
    ).all()
    per_query: list[float] = []
    costs: dict[str, list[float]] = defaultdict(list)
    for row in recent:
        metrics, context = row.metrics or {}, row.context or {}
        mean = ((metrics.get("latency_ms") or {}).get("total_ms") or {}).get("mean")
        if isinstance(mean, int | float):
            per_query.append(float(mean))
        rerank = (context.get("retrieval") or {}).get("rerank") or {}
        per_1k = (metrics.get("cost") or {}).get("per_1k_queries")
        if rerank.get("component") and isinstance(per_1k, int | float):
            costs[rerank_key(rerank["component"], rerank.get("params") or {})].append(per_1k)
    return History(
        task_ms=task_ms,
        query_ms=statistics.median(per_query) if per_query else None,
        rerank_cost_per_1k={key: statistics.median(values) for key, values in costs.items()},
    )


def evaluated_pairs(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    corpus_version_id: uuid.UUID,
    eval_set_version_id: uuid.UUID,
    split: str,
    options: Mapping[str, Any],
) -> set[tuple[str, str]]:
    """The (pipeline hash, retrieval hash) pairs with a succeeded eval run like a trial's: the
    same eval set version, split and options, over a whole run of the corpus version. A trial
    with one reuses its results."""
    rows = conn.execute(
        sa.select(runs_table.c.spec_hash, eval_runs.c.retrieval_hash)
        .join(runs_table, runs_table.c.id == eval_runs.c.pipeline_run_id)
        .where(
            eval_runs.c.workspace_id == workspace_id,
            eval_runs.c.status == "succeeded",
            eval_runs.c.eval_set_version_id == eval_set_version_id,
            eval_runs.c.split == split,
            # Equal options: each contains the other.
            eval_runs.c.options.contains(dict(options)),
            eval_runs.c.options.contained_by(dict(options)),
            runs_table.c.corpus_version_id == corpus_version_id,
            runs_table.c.until_stage.is_(None),
            # A run without a sample holds JSON null there, not SQL NULL.
            sa.or_(
                runs_table.c.sample.is_(None),
                sa.func.jsonb_typeof(runs_table.c.sample) == "null",
            ),
            runs_table.c.rerun_from.is_(None),
        )
        .distinct()
    ).all()
    return {(row.spec_hash, row.retrieval_hash) for row in rows}


# --- what the results views read ----------------------------------------------------------------


@dataclass(frozen=True)
class QueryValue:
    item_id: str
    value: float | None  # the metric for a query with evidence
    lost_at: str | None


def query_values(
    conn: Connection,
    eval_run_ids: Iterable[uuid.UUID],
    *,
    rule: str,
    checkpoint: str,
    metric: str,
) -> dict[uuid.UUID, list[QueryValue]]:
    """One metric per query, for each eval run's queries with evidence, and where each miss
    was lost: read from the stored scores, without loading them whole."""
    ids = list(eval_run_ids)
    if not ids:
        return {}
    value = eval_query_results.c.score["metrics"][rule][checkpoint][metric].astext.cast(sa.Float)
    rows = conn.execute(
        sa.select(
            eval_query_results.c.eval_run_id,
            eval_query_results.c.item_id,
            value.label("value"),
            eval_query_results.c.lost_at,
        )
        .where(
            eval_query_results.c.eval_run_id.in_(ids),
            eval_query_results.c.has_evidence.is_(True),
        )
        .order_by(eval_query_results.c.eval_run_id, eval_query_results.c.ordinal)
    )
    found: dict[uuid.UUID, list[QueryValue]] = defaultdict(list)
    for row in rows:
        found[row.eval_run_id].append(QueryValue(row.item_id, row.value, row.lost_at))
    return dict(found)


# --- confirming a trial on the test split ------------------------------------------------


class ConfirmationRefused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Confirmation:
    id: uuid.UUID
    workspace_id: uuid.UUID
    experiment_id: uuid.UUID
    trial_id: uuid.UUID
    eval_run_id: uuid.UUID
    eval_set_version_id: uuid.UUID
    ordinal: int
    reason: str | None
    note: str | None
    created_by: str
    created_at: datetime


def confirmations_of(conn: Connection, experiment_id: uuid.UUID) -> list[Confirmation]:
    rows = conn.execute(
        sa.select(confirmations)
        .where(confirmations.c.experiment_id == experiment_id)
        .order_by(confirmations.c.ordinal)
    )
    return [
        Confirmation(**{column.name: getattr(row, column.name) for column in confirmations.columns})
        for row in rows
    ]


def confirm_trial(
    conn: Connection,
    experiment: Experiment,
    trial: Trial,
    queue: JobQueue,
    *,
    reason: str | None,
    note: str | None,
    created_by: str,
    max_attempts: int = 3,
) -> tuple[Confirmation, EvalRun, int]:
    """Evaluate a finished trial once on the locked test split: an eval run of its
    pipeline run and retrieval config, counted on the eval set version. The experiment's first
    read needs no reason; a later one without a reason is refused (``test_split_already_read``),
    so no second read goes unlogged. Returns the confirmation, its eval run and the version's
    count of test reads."""
    conn.execute(
        sa.select(experiments.c.id).where(experiments.c.id == experiment.id).with_for_update()
    ).one()
    if (
        trial.status != "succeeded"
        or trial.pipeline_run_id is None
        or not trial.retrieval_config_id
    ):
        raise ConfirmationRefused(
            "trial_pruned" if trial.status == "pruned" else "trial_not_finished"
        )
    earlier = confirmations_of(conn, experiment.id)
    if earlier and reason is None:
        raise ConfirmationRefused("test_split_already_read")
    version = evalsets.get_version(
        conn, experiment.workspace_id, version_id=experiment.eval_set_version_id
    )
    if version is None or not version.split_counts.get("test", 0):
        raise ConfirmationRefused("no_test_split")
    spec = experiment.spec()
    evaluation = evalruns.create_eval_run(
        conn,
        workspace_id=experiment.workspace_id,
        pipeline_run_id=trial.pipeline_run_id,
        retrieval_config_id=trial.retrieval_config_id,
        eval_set_version_id=version.id,
        split="test",
        options=spec.eval_options.dump(),
        retrieval_hash=trial.retrieval_hash,
        test_read=True,
        created_by=created_by,
    )
    queue.enqueue_in(
        conn,
        workspace_id=experiment.workspace_id,
        kind=evalruns.START_JOB,
        payload={"eval_run_id": str(evaluation.id)},
        resource_class="io",
        priority=5,
        max_attempts=max_attempts,
    )
    counted = evalsets.record_test_read(conn, version.id)
    row = conn.execute(
        sa.insert(confirmations)
        .values(
            id=uuid.uuid4(),
            workspace_id=experiment.workspace_id,
            experiment_id=experiment.id,
            trial_id=trial.id,
            eval_run_id=evaluation.id,
            eval_set_version_id=version.id,
            ordinal=len(earlier) + 1,
            reason=reason,
            note=note,
            created_by=created_by,
        )
        .returning(*confirmations.columns)
    ).one()
    confirmation = Confirmation(
        **{column.name: getattr(row, column.name) for column in confirmations.columns}
    )
    return confirmation, evaluation, counted.test_reads
