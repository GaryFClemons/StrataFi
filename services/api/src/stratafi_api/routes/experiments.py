"""Experiments: plan, create, start, follow and cancel.

``POST /experiments/plan`` checks an experiment and answers its plan without saving anything;
``POST /experiments`` saves it ``planned`` with its trials. The plan pins the corpus version, the
eval set version (frozen, dev split) and the base documents, expands and resolves the trials,
walks the corpus through every distinct pipeline for the unique work (with the cache's answers),
and estimates time and cost from past work (``stratafi_core.experiments``). Embedding revisions
come from the model gateway, with the caller's identity, as a run's would; when the gateway
can't say, the plan warns ``revisions_unresolved`` and counts embedding work as uncached.

Nothing runs until ``POST /experiments/{id}/start`` with ``confirm: true``; a worker then runs
the trials (``stratafi_store.experiments``), and ``experiment`` events on
``GET /events?run_id=`` follow it. ``GET /experiments/{id}/results`` holds the results views
(``stratafi_core.analysis``), computed from the finished trials' eval runs, with the
guardrails: significance groups against the leader, practical equivalence, the recommended
trial and sample-size hints.

A trial chosen on dev is confirmed once on the locked test split: ``POST
/experiments/{id}/trials/{trial_id}/confirm`` starts an eval run of the test split, counted on
the eval set version and audited; a second read for the same experiment needs a reason, which is
logged. Audit events carry IDs, codes and counts only.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, ProblemIssue, problem_responses
from stratafi_api.routes.search import gateway_services
from stratafi_components.embed import EmbedComponent
from stratafi_core.analysis import (
    FactorInfo,
    TrialResult,
    factor_effects,
    file_types,
    guardrails,
    interactions,
    leaderboard,
    pareto,
    slice_metric,
    where_lost,
)
from stratafi_core.components import ComponentError, ComponentRegistry, StageContext
from stratafi_core.evalsets import load_yaml
from stratafi_core.experiments import (
    ExperimentSpec,
    PlanEntry,
    budget_check,
    dump_experiment,
    estimate,
    halving_plan,
    metric_estimate,
    parse_experiment,
    parse_ref,
    plan_trials,
    unique_work,
)
from stratafi_core.hashing import content_hash
from stratafi_core.logs import current_log_context
from stratafi_core.specs import (
    PipelineSpec,
    RetrievalConfig,
    SpecInvalid,
    SpecIssue,
    resolve_pipeline,
    resolve_retrieval,
)
from stratafi_store import corpora, evalruns, evalsets, runs, specs
from stratafi_store import experiments as store
from stratafi_store.audit import record_audit_event
from stratafi_store.jobs import JobQueue

router = APIRouter(tags=["experiments"])

ExperimentStatus = Literal[
    "planned",
    "running",
    "stopping",
    "succeeded",
    "stopped",
    "failed",
    "cancelling",
    "cancelled",
]
TrialStatus = Literal["pending", "running", "succeeded", "failed", "skipped", "cancelled", "pruned"]
MAX_CONTENT_CHARS = 200_000
UNRESOLVED = "unresolved"


class ExperimentIn(BaseModel):
    """An experiment as a document, or as YAML text: one of the two."""

    model_config = ConfigDict(extra="forbid")

    document: dict[str, Any] | None = Field(
        default=None, description="The stratafi.experiment.v1 document"
    )
    content: str | None = Field(
        default=None, min_length=1, max_length=MAX_CONTENT_CHARS, description="The same, as YAML"
    )

    @model_validator(mode="after")
    def _one(self) -> "ExperimentIn":
        if (self.document is None) == (self.content is None):
            raise ValueError("give either document or content")
        return self


class IssueOut(BaseModel):
    path: str
    code: str


class PlannedTrialOut(BaseModel):
    ordinal: int
    grid_index: int
    levels: list[int]
    values: dict[str, str] = Field(description="Factor path → the level's label")
    pipeline_hash: str
    retrieval_hash: str
    evaluated: bool = Field(description="An identical eval run has finished: its results reuse")


class PlanOut(BaseModel):
    valid: bool
    issues: list[IssueOut]
    document: dict[str, Any] | None = Field(description="The experiment, every default filled in")
    plan: dict[str, Any] | None = Field(
        description="Factors, pinned inputs, unique work per stage, estimate and budget check"
    )
    trials: list[PlannedTrialOut]


class ExperimentOut(BaseModel):
    id: uuid.UUID
    name: str
    status: ExperimentStatus
    document: dict[str, Any] = Field(description="The stratafi.experiment.v1 it runs")
    document_hash: str
    corpus_version_id: uuid.UUID
    eval_set_version_id: uuid.UUID
    pipeline_spec_id: uuid.UUID = Field(description="The base pipeline spec version")
    retrieval_config_id: uuid.UUID = Field(description="The base retrieval config version")
    plan: dict[str, Any]
    spent: dict[str, Any] = Field(description="Estimated cost so far, and hours since the start")
    counts: dict[str, Any] = Field(description="Trials by status")
    stop_reason: str | None = Field(description="The budget that stopped it")
    error_code: str | None
    created_by: str
    started_by: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def of(cls, experiment: store.Experiment) -> "ExperimentOut":
        return cls(
            id=experiment.id,
            name=experiment.name,
            status=experiment.status,  # type: ignore[arg-type]
            document=experiment.document,
            document_hash=experiment.document_hash,
            corpus_version_id=experiment.corpus_version_id,
            eval_set_version_id=experiment.eval_set_version_id,
            pipeline_spec_id=experiment.pipeline_spec_id,
            retrieval_config_id=experiment.retrieval_config_id,
            plan=experiment.plan,
            spent=experiment.spent or {},
            counts=experiment.counts or {},
            stop_reason=experiment.stop_reason,
            error_code=experiment.error_code,
            created_by=experiment.created_by,
            started_by=experiment.started_by,
            created_at=experiment.created_at,
            started_at=experiment.started_at,
            finished_at=experiment.finished_at,
        )


class TrialOut(BaseModel):
    id: uuid.UUID
    ordinal: int
    grid_index: int
    levels: list[int]
    values: dict[str, str] = Field(description="Factor path → the level's label")
    status: TrialStatus
    pipeline_hash: str
    retrieval_hash: str
    pipeline_spec_id: uuid.UUID | None
    retrieval_config_id: uuid.UUID | None
    run_id: uuid.UUID | None = Field(
        description="Its pipeline run, shared by its pipeline's trials"
    )
    eval_run_id: uuid.UUID | None
    objective: dict[str, Any] | None = Field(description="The objective's mean with its 95% CI")
    measures: dict[str, Any]
    constraints: list[dict[str, Any]]
    feasible: bool | None = Field(description="Every constraint passed")
    error_code: str | None
    rung: int = Field(description="Successive halving: the rung it's at, or reached")
    rungs: list[dict[str, Any]] = Field(
        description="Successive halving: each finished rung's sample, eval run, estimate and "
        "verdict"
    )
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def of(cls, trial: store.Trial) -> "TrialOut":
        return cls(
            id=trial.id,
            ordinal=trial.ordinal,
            grid_index=trial.grid_index,
            levels=trial.levels,
            values=trial.factor_values,
            status=trial.status,  # type: ignore[arg-type]
            pipeline_hash=trial.pipeline_hash,
            retrieval_hash=trial.retrieval_hash,
            pipeline_spec_id=trial.pipeline_spec_id,
            retrieval_config_id=trial.retrieval_config_id,
            run_id=trial.pipeline_run_id,
            eval_run_id=trial.eval_run_id,
            objective=trial.objective,
            measures=trial.measures,
            constraints=trial.constraints,
            feasible=trial.feasible,
            error_code=trial.error_code,
            rung=trial.rung,
            rungs=trial.rungs,
            started_at=trial.started_at,
            finished_at=trial.finished_at,
        )


class TrialsOut(BaseModel):
    items: list[TrialOut]


class WorkOut(BaseModel):
    planned: dict[str, Any] = Field(description="The plan's unique work per stage")
    done: dict[str, dict[str, int]] = Field(
        description="The experiment's pipeline runs' tasks by stage and status"
    )


class ResultsOut(BaseModel):
    objective: dict[str, Any] = Field(description="The metric, rule, checkpoint and goal shown")
    k_eval: int = Field(description="The eval set's cutoff")
    trials: int = Field(description="Trials with results")
    queries: int = Field(description="Queries with evidence every trial scored")
    leaderboard: list[dict[str, Any]] = Field(description="Best first, ranked")
    effects: list[dict[str, Any]] = Field(
        description="Per factor, each level's paired effect against the first, with a 95% CI"
    )
    interactions: list[dict[str, Any]] = Field(description="Per pair of factors, cell means")
    pareto: dict[str, list[dict[str, Any]]] = Field(
        description="The objective against p95 latency and cost per 1,000 queries; frontiers"
    )
    file_types: dict[str, Any] = Field(
        description="Each gold file type's slice per trial, and per parse factor a routing table"
    )
    where_lost: dict[str, Any] = Field(
        description="Per trial, queries found and lost by stage; the queries most trials missed"
    )
    guardrails: dict[str, Any] = Field(
        description="Against the leader: Holm-adjusted groups, practical equivalence within the "
        "margin, the recommended trial, and sample-size hints"
    )


class ConfirmIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_test_read: Literal[True] = Field(
        description="This reads the locked test split: counted and audited"
    )
    reason: Literal["run_failed", "candidate_changed", "other"] | None = Field(
        default=None,
        description="Needed for every read after the experiment's first; logged",
    )
    note: str | None = Field(
        default=None, max_length=500, description="Kept with the confirmation, never logged"
    )


class ConfirmationOut(BaseModel):
    id: uuid.UUID
    ordinal: int = Field(description="Its place among the experiment's reads of the test split")
    trial_id: uuid.UUID
    trial_ordinal: int
    values: dict[str, str]
    eval_run_id: uuid.UUID
    status: Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]
    reason: str | None
    note: str | None
    dev: dict[str, Any] | None = Field(description="The objective on dev, with its 95% CI")
    test: dict[str, Any] | None = Field(description="The objective on test, once it's scored")
    created_by: str
    created_at: datetime


class ConfirmationsOut(BaseModel):
    items: list[ConfirmationOut]
    test_reads: int = Field(description="How often the eval set version's test split was read")


class StartIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal[True] = Field(description="The plan was reviewed: start it")


# --- planning ---------------------------------------------------------------------------------


def _queue(request: Request) -> JobQueue:
    queue = request.app.state.jobs
    if queue is None:
        raise Problem("unavailable", code="database_not_configured")
    return queue


def _orchestrator(request: Request) -> store.ExperimentOrchestrator:
    orchestrator = request.app.state.experiments
    if orchestrator is None:
        raise Problem("unavailable", code="database_not_configured")
    return orchestrator


def _audit(
    conn: Connection, principal: Principal, action: str, experiment_id: uuid.UUID, **details: Any
) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type="experiment",
        resource_id=str(experiment_id),
        request_id=current_log_context().get("request_id"),
        details=details,
    )


def _document(body: ExperimentIn) -> dict[str, Any]:
    if body.document is not None:
        return body.document
    loaded = load_yaml(body.content or "")
    if not isinstance(loaded, dict):
        raise SpecInvalid([SpecIssue("(root)", "not_an_object")])
    return loaded


class _Planned:
    """A plan and what it pinned, ready to save."""

    def __init__(self) -> None:
        self.issues: list[SpecIssue] = []
        self.spec: ExperimentSpec | None = None
        self.plan: dict[str, Any] | None = None
        self.trials: list[PlannedTrialOut] = []
        self.corpus_version_id: uuid.UUID | None = None
        self.eval_set_version_id: uuid.UUID | None = None
        self.pipeline_spec_id: uuid.UUID | None = None
        self.retrieval_config_id: uuid.UUID | None = None


def _pin_document(
    conn: Connection, kind: specs.DocumentKind, workspace_id: uuid.UUID, ref: str, path: str
) -> tuple[specs.StoredDocument | None, list[SpecIssue]]:
    name, version = parse_ref(ref)
    stored = specs.get_version(conn, kind, workspace_id, name, version)
    return stored, [] if stored is not None else [SpecIssue(path, "not_found")]


def _revisions(
    request: Request,
    planned_pipelines: dict[str, PipelineSpec],
    workspace_id: uuid.UUID,
    tenant: str | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], bool]:
    """Each pipeline's embedding revision, by pipeline hash, and whether all resolved."""
    registry = request.app.state.registry
    services, _ = gateway_services(request, tenant)
    context = StageContext(str(workspace_id), services=services)
    found: dict[str, str] = {}
    resolved = True
    envs: dict[str, dict[str, dict[str, Any]]] = {}
    for pipeline_hash, pipeline in planned_pipelines.items():
        ref = pipeline.stages.embed
        key = content_hash({"component": ref.component, "params": ref.params})
        if key not in found:
            component = cast(EmbedComponent, registry.get(ref.component)())
            try:
                found[key] = component.revision_for(component.resolve_params(ref.params), context)
            except ComponentError:
                found[key] = UNRESOLVED
                resolved = False
        envs[pipeline_hash] = {"embed": {"revision": found[key]}}
    return envs, resolved


def _plan(
    request: Request, conn: Connection, principal: Principal, document: dict[str, Any]
) -> _Planned:
    out = _Planned()
    try:
        spec = parse_experiment(document)
    except SpecInvalid as error:
        out.issues = error.issues
        return out
    out.spec = spec
    workspace_id = principal.workspace.id
    registry = request.app.state.registry
    corpus_name, corpus_number = parse_ref(spec.corpus)
    corpus = corpora.get_corpus_by_name(conn, workspace_id, corpus_name)
    corpus_version = None
    if corpus is not None:
        corpus_version = (
            corpora.get_version(conn, workspace_id, corpus_id=corpus.id, version=corpus_number)
            if corpus_number is not None
            else corpora.latest_version(conn, workspace_id, corpus.id)
        )
    if corpus_version is None:
        out.issues.append(SpecIssue("corpus", "not_found"))
    eval_set = evalsets.get_set_by_name(conn, workspace_id, spec.eval_set.name)
    eval_version = None
    if eval_set is not None:
        eval_version = (
            evalsets.get_version(
                conn, workspace_id, eval_set_id=eval_set.id, version=spec.eval_set.version
            )
            if spec.eval_set.version is not None
            else evalsets.latest_version(conn, workspace_id, eval_set.id)
        )
    if eval_version is None:
        out.issues.append(SpecIssue("eval_set", "not_found"))
    elif not eval_version.frozen:
        out.issues.append(SpecIssue("eval_set", "not_frozen"))
    elif not eval_version.split_counts.get(spec.eval_set.split, 0):
        out.issues.append(SpecIssue("eval_set.split", "no_items"))
    base_pipeline, issues = _pin_document(
        conn, "pipeline", workspace_id, spec.base.pipeline, "base.pipeline"
    )
    out.issues += issues
    base_retrieval, issues = _pin_document(
        conn, "retrieval", workspace_id, spec.base.retrieval, "base.retrieval"
    )
    out.issues += issues
    if out.issues or not (corpus_version and eval_version and base_pipeline and base_retrieval):
        return out
    try:
        pipeline = resolve_pipeline(base_pipeline.document, registry)
        retrieval = resolve_retrieval(base_retrieval.document, registry)
    except SpecInvalid:
        out.issues.append(SpecIssue("base", "invalid_base"))
        return out
    planned = plan_trials(spec, pipeline, retrieval, registry)
    if planned.issues:
        out.issues = planned.issues
        return out
    stored_entries = corpora.version_entries(conn, workspace_id, corpus_version.id)
    entries = [
        PlanEntry(
            entry,
            media_type=stored.media_type,
            traits=stored.traits,
            size_bytes=stored.size_bytes,
            refused=stored.status == "refused",
            rank=tuple(runs.version_rank(stored, stored.name)),
        )
        for entry, stored in zip(
            corpora.planner_entries(stored_entries), stored_entries, strict=True
        )
    ]
    model_envs, resolved = _revisions(request, planned.pipelines, workspace_id, principal.tenant)
    work = unique_work(
        planned.trials,
        registry,
        entries,
        model_envs=model_envs,
        lookup=lambda keys: runs.summaries(conn, workspace_id, keys),
    )
    options = spec.eval_options.dump()
    done = store.evaluated_pairs(
        conn,
        workspace_id,
        corpus_version_id=corpus_version.id,
        eval_set_version_id=eval_version.id,
        split=spec.eval_set.split,
        options=options,
    )
    evaluated = {
        trial.ordinal
        for trial in planned.trials
        if (trial.pipeline_hash, trial.retrieval_hash) in done
    }
    settings = request.app.state.settings
    queries = int(eval_version.split_counts.get(spec.eval_set.split, 0))
    halving = halving_plan(spec, len(planned.trials), queries)
    if halving is not None:  # its rungs score samples: identical full eval runs don't count
        evaluated = set()
    estimated = estimate(
        work,
        planned.trials,
        registry,
        entries,
        queries=queries,
        shard_size=spec.eval_options.shard_size,
        history=store.history(conn, workspace_id),
        prices=settings.models.prices,
        slots=settings.worker.slots,
        evaluated=evaluated,
        query_share=halving["share"] if halving is not None else 1.0,
    )
    warnings = [] if resolved else ["revisions_unresolved"]
    if len(planned.trials) > spec.budget.max_trials:
        warnings.append("trials_over_budget")
    out.plan = {
        "factors": [factor.as_dict() for factor in planned.factors],
        "strategy": spec.search.method,
        "search": halving,
        "grid": planned.grid,
        "excluded": planned.excluded,
        "trials": len(planned.trials),
        "corpus": {
            "id": str(corpus_version.corpus_id),
            "name": corpus_name,
            "version": corpus_version.version,
            "version_id": str(corpus_version.id),
            "documents": len(entries),
        },
        "eval_set": {
            "id": str(eval_version.eval_set_id),
            "name": spec.eval_set.name,
            "version": eval_version.version,
            "version_id": str(eval_version.id),
            "split": spec.eval_set.split,
            "items": queries,
        },
        "base": {
            kind: {"id": str(stored.id), "name": stored.name, "version": stored.version}
            for kind, stored in (("pipeline", base_pipeline), ("retrieval", base_retrieval))
        },
        "work": work.as_dict(),
        "estimate": estimated,
        "budget": budget_check(spec, len(planned.trials), estimated),
        "warnings": warnings,
    }
    out.trials = [
        PlannedTrialOut(
            ordinal=trial.ordinal,
            grid_index=trial.grid_index,
            levels=list(trial.levels),
            values=trial.values(planned.factors),
            pipeline_hash=trial.pipeline_hash,
            retrieval_hash=trial.retrieval_hash,
            evaluated=trial.ordinal in evaluated,
        )
        for trial in planned.trials
    ]
    out.corpus_version_id = corpus_version.id
    out.eval_set_version_id = eval_version.id
    out.pipeline_spec_id = base_pipeline.id
    out.retrieval_config_id = base_retrieval.id
    return out


def _invalid(issues: list[SpecIssue]) -> Problem:
    return Problem(
        "invalid_request",
        "The experiment is invalid.",
        code="invalid_experiment",
        errors=[issue.path for issue in issues],
        issues=[ProblemIssue(path=issue.path, code=issue.code) for issue in issues],
    )


def _experiment_or_404(
    conn: Connection, principal: Principal, experiment_id: uuid.UUID
) -> store.Experiment:
    experiment = store.get_experiment(conn, principal.workspace.id, experiment_id)
    if experiment is None:
        raise Problem("not_found", code="experiment_not_found")
    return experiment


# --- routes -----------------------------------------------------------------------------------


@router.post(
    "/experiments/plan",
    response_model=PlanOut,
    summary="Check an experiment and show its plan, without saving it",
    description="The trials, the unique work per stage (shared prefixes once; cached work "
    "counted), the time and cost estimate, and the budget check. Problems come back as "
    "issues: paths and codes.",
    responses=problem_responses(400, 401, 403, 503),
)
def plan_experiment(
    request: Request, body: ExperimentIn, principal: Principal = Depends(require_role("editor"))
) -> PlanOut:
    try:
        document = _document(body)
    except SpecInvalid as error:
        return PlanOut(
            valid=False,
            issues=[IssueOut(path=i.path, code=i.code) for i in error.issues],
            document=None,
            plan=None,
            trials=[],
        )
    with engine_of(request).connect() as conn:
        planned = _plan(request, conn, principal, document)
    return PlanOut(
        valid=not planned.issues,
        issues=[IssueOut(path=issue.path, code=issue.code) for issue in planned.issues],
        document=dump_experiment(planned.spec) if planned.spec is not None else None,
        plan=planned.plan,
        trials=planned.trials,
    )


@router.post(
    "/experiments",
    response_model=ExperimentOut,
    status_code=201,
    summary="Save an experiment with its plan; nothing runs until it's started",
    responses=problem_responses(400, 401, 403, 503),
)
def create_experiment(
    request: Request, body: ExperimentIn, principal: Principal = Depends(require_role("editor"))
) -> JSONResponse:
    try:
        document = _document(body)
    except SpecInvalid as error:
        raise _invalid(error.issues) from None
    with engine_of(request).begin() as conn:
        planned = _plan(request, conn, principal, document)
        if planned.issues or planned.spec is None or planned.plan is None:
            raise _invalid(planned.issues or [SpecIssue("(root)", "invalid")])
        resolved = dump_experiment(planned.spec)
        experiment = store.create_experiment(
            conn,
            workspace_id=principal.workspace.id,
            spec=planned.spec,
            document=resolved,
            document_hash=content_hash(resolved),
            corpus_version_id=cast(uuid.UUID, planned.corpus_version_id),
            eval_set_version_id=cast(uuid.UUID, planned.eval_set_version_id),
            pipeline_spec_id=cast(uuid.UUID, planned.pipeline_spec_id),
            retrieval_config_id=cast(uuid.UUID, planned.retrieval_config_id),
            plan=planned.plan,
            trial_rows=[
                {
                    "ordinal": trial.ordinal,
                    "grid_index": trial.grid_index,
                    "levels": trial.levels,
                    "factor_values": trial.values,
                    "pipeline_hash": trial.pipeline_hash,
                    "retrieval_hash": trial.retrieval_hash,
                }
                for trial in planned.trials
            ],
            created_by=principal.subject,
        )
        _audit(
            conn,
            principal,
            "experiment.create",
            experiment.id,
            trials=len(planned.trials),
            pipelines=planned.plan["work"]["pipelines"],
        )
    return JSONResponse(ExperimentOut.of(experiment).model_dump(mode="json"), status_code=201)


@router.get(
    "/experiments",
    response_model=Page[ExperimentOut],
    summary="The workspace's experiments, newest first",
    responses=problem_responses(400, 401, 403, 503),
)
def list_experiments(
    request: Request,
    page: PageParams = Depends(page_params),
    status: Annotated[ExperimentStatus | None, Query()] = None,
    include_archived: Annotated[bool, Query(description="Include archived ones")] = False,
    principal: Principal = Depends(require_role("viewer")),
) -> Page[ExperimentOut]:
    after = None
    if page.cursor is not None:
        created_at, experiment_ref = decode_cursor(page.cursor, size=2)
        try:
            after = (datetime.fromisoformat(created_at), uuid.UUID(experiment_ref))
        except (TypeError, ValueError, AttributeError):
            raise Problem(
                "invalid_request", "The cursor isn't valid.", code="invalid_cursor"
            ) from None
    with engine_of(request).connect() as conn:
        found = store.list_experiments(
            conn,
            principal.workspace.id,
            limit=page.limit + 1,
            after=after,
            status=status,
            include_archived=include_archived,
        )
    items = [ExperimentOut.of(experiment) for experiment in found[: page.limit]]
    cursor = None
    if len(found) > page.limit:
        last = found[page.limit - 1]
        cursor = encode_cursor([last.created_at.isoformat(), str(last.id)])
    return Page[ExperimentOut](items=items, next_cursor=cursor)


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentOut,
    summary="One experiment: its plan, progress and spending",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_experiment(
    request: Request,
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> ExperimentOut:
    with engine_of(request).connect() as conn:
        return ExperimentOut.of(_experiment_or_404(conn, principal, experiment_id))


@router.get(
    "/experiments/{experiment_id}/trials",
    response_model=TrialsOut,
    summary="An experiment's trials in plan order, with their results so far",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_trials(
    request: Request,
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> TrialsOut:
    with engine_of(request).connect() as conn:
        experiment = _experiment_or_404(conn, principal, experiment_id)
        found = store.trials_of(conn, experiment.id)
    return TrialsOut(items=[TrialOut.of(trial) for trial in found])


@router.get(
    "/experiments/{experiment_id}/work",
    response_model=WorkOut,
    summary="The plan's unique work beside what the experiment's runs did",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_work(
    request: Request,
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> WorkOut:
    with engine_of(request).connect() as conn:
        experiment = _experiment_or_404(conn, principal, experiment_id)
        done = store.stage_counts(conn, experiment.id)
    return WorkOut(planned=(experiment.plan.get("work") or {}).get("stages", {}), done=done)


@router.post(
    "/experiments/{experiment_id}/start",
    response_model=ExperimentOut,
    status_code=202,
    summary="Start a planned experiment, once its plan was reviewed",
    description="Saves each trial's variants, admits the first `budget.max_trials` trials, and "
    "queues the work. 409 `plan_stale` when the components or base documents changed since "
    "the plan: plan it again.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def start_experiment(
    request: Request,
    experiment_id: uuid.UUID,
    body: StartIn,
    principal: Principal = Depends(require_role("editor")),
) -> JSONResponse:
    with engine_of(request).begin() as conn:
        experiment = _experiment_or_404(conn, principal, experiment_id)
        try:
            experiment = store.start_experiment(
                conn, experiment, request.app.state.registry, started_by=principal.subject
            )
        except store.PlanChanged as error:
            raise Problem("conflict", code=error.code) from None
        store.enqueue_advance(
            conn,
            _queue(request),
            experiment,
            max_attempts=request.app.state.settings.worker.max_attempts,
        )
        _audit(
            conn,
            principal,
            "experiment.start",
            experiment.id,
            trials=sum(experiment.counts.get("trials", {}).values()),
        )
    return JSONResponse(ExperimentOut.of(experiment).model_dump(mode="json"), status_code=202)


@router.post(
    "/experiments/{experiment_id}/cancel",
    response_model=ExperimentOut,
    summary="Cancel an experiment: nothing new starts, and its runs in flight are cancelled",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def cancel_experiment(
    request: Request,
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_role("editor")),
) -> ExperimentOut:
    experiment = _orchestrator(request).cancel(experiment_id, principal.workspace.id)
    if experiment is None:
        raise Problem("not_found", code="experiment_not_found")
    with engine_of(request).begin() as conn:
        _audit(conn, principal, "experiment.cancel", experiment.id, status=experiment.status)
    return ExperimentOut.of(experiment)


@router.get(
    "/experiments/{experiment_id}/results",
    response_model=ResultsOut,
    summary="An experiment's results views: leaderboard, effects, interactions, Pareto, file "
    "types and where queries were lost",
    description="From the trials that have finished so far. The objective's metric, rule and "
    "checkpoint by default; `metric`, `rule` and `checkpoint` show another.",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_results(
    request: Request,
    experiment_id: uuid.UUID,
    metric: Annotated[
        str | None, Query(pattern=r"^(hit|recall|precision|mrr|ndcg)@(\d{1,3}|all)$")
    ] = None,
    rule: Annotated[
        Literal["knowledge-hit-v1", "evidence-coverage-v1", "doc-v1"] | None, Query()
    ] = None,
    checkpoint: Annotated[
        Literal["dense", "keyword", "fused", "candidates", "reranked", "final"] | None, Query()
    ] = None,
    resamples: Annotated[int, Query(ge=100, le=20_000)] = 2000,
    seed: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
    alpha: Annotated[float, Query(gt=0, lt=1, description="Significance level")] = 0.05,
    margin: Annotated[
        float, Query(ge=0, le=1, description="Practical equivalence on the objective")
    ] = 0.02,
    principal: Principal = Depends(require_role("viewer")),
) -> ResultsOut:
    with engine_of(request).connect() as conn:
        experiment = _experiment_or_404(conn, principal, experiment_id)
        objective = experiment.spec().objective
        shown = {
            "metric": metric or objective.metric,
            "rule": rule or objective.rule,
            "checkpoint": checkpoint or objective.checkpoint,
            "goal": objective.goal,
        }
        found = trial_results(conn, experiment, shown)
    results, factors, k_eval = found.results, found.factors, found.k_eval
    goal = shown["goal"]
    paired = set.intersection(*(set(result.queries) for result in results)) if results else set()
    board = leaderboard(results, goal=goal)
    return ResultsOut(
        objective=shown,
        k_eval=k_eval,
        trials=len(results),
        queries=len(paired),
        leaderboard=board,
        effects=factor_effects(results, factors, resamples=resamples, seed=seed),
        interactions=interactions(results, factors),
        pareto={
            "latency": pareto(results, cost="latency_ms.p95", goal=goal),
            "cost": pareto(results, cost="cost_per_1k_queries_usd", goal=goal),
        },
        file_types=file_types(
            results,
            factors,
            rule=shown["rule"],
            metric=slice_metric(shown["metric"], k_eval),
            goal=goal,
        ),
        where_lost=where_lost(results),
        guardrails=guardrails(
            results, goal=goal, alpha=alpha, margin=margin, resamples=resamples, seed=seed
        ),
    )


@dataclass(frozen=True)
class TrialResults:
    """The finished trials, as the results views read them."""

    trials: list[store.Trial]
    results: list[TrialResult]
    factors: list[FactorInfo]
    k_eval: int


def trial_results(
    conn: Connection, experiment: store.Experiment, shown: dict[str, str]
) -> TrialResults:
    """The experiment's finished trials with ``shown``'s metric (and rule and checkpoint) per
    query, their measures, slices and losses."""
    finished = [
        trial
        for trial in store.trials_of(conn, experiment.id)
        if trial.status == "succeeded" and trial.eval_run_id is not None
    ]
    eval_ids = [trial.eval_run_id for trial in finished if trial.eval_run_id]
    values = store.query_values(
        conn, eval_ids, rule=shown["rule"], checkpoint=shown["checkpoint"], metric=shown["metric"]
    )
    records = evalruns.eval_runs_by_id(conn, eval_ids)
    factors = [
        FactorInfo(
            path=str(factor["path"]),
            labels=tuple(factor["labels"]),
            stage=str(factor.get("stage", "query")),
            levels=tuple(factor.get("levels", ())),
        )
        for factor in experiment.plan.get("factors", [])
    ]
    k_eval = 5
    results = []
    for trial in finished:
        record = records.get(trial.eval_run_id) if trial.eval_run_id else None
        metrics = (record.metrics if record else None) or {}
        k_eval = int((record.context or {}).get("k_eval", k_eval)) if record else k_eval
        rows = values.get(trial.eval_run_id, []) if trial.eval_run_id else []
        results.append(
            TrialResult(
                ordinal=trial.ordinal,
                levels=tuple(trial.levels),
                values=trial.factor_values,
                objective=metric_estimate(
                    metrics, shown["rule"], shown["checkpoint"], shown["metric"]
                ),
                queries={row.item_id: row.value for row in rows if row.value is not None},
                measures=trial.measures,
                feasible=trial.feasible,
                constraints=trial.constraints,
                slices=(metrics.get("slices") or {}).get("file_type") or {},
                where_lost=metrics.get("where_lost") or {},
                lost={row.item_id: row.lost_at for row in rows if row.lost_at},
                eval_run_id=str(trial.eval_run_id),
            )
        )
    return TrialResults(finished, results, factors, k_eval)


def trial_documents(
    conn: Connection, workspace_id: uuid.UUID, trial: store.Trial, registry: ComponentRegistry
) -> tuple[PipelineSpec, RetrievalConfig] | None:
    """A trial's saved pipeline spec and retrieval config, resolved."""
    if trial.pipeline_spec_id is None or trial.retrieval_config_id is None:
        return None
    pipeline = specs.get_by_id(conn, "pipeline", workspace_id, trial.pipeline_spec_id)
    retrieval = specs.get_by_id(conn, "retrieval", workspace_id, trial.retrieval_config_id)
    if pipeline is None or retrieval is None:
        return None
    try:
        return (
            resolve_pipeline(pipeline.document, registry),
            resolve_retrieval(retrieval.document, registry),
        )
    except SpecInvalid:
        return None


def _confirmation_out(
    confirmation: store.Confirmation,
    trial: store.Trial,
    evaluation: evalruns.EvalRun | None,
    spec: ExperimentSpec,
) -> ConfirmationOut:
    objective = spec.objective
    test = None
    if evaluation is not None and evaluation.status == "succeeded":
        test = metric_estimate(
            evaluation.metrics, objective.rule, objective.checkpoint, objective.metric
        )
    return ConfirmationOut(
        id=confirmation.id,
        ordinal=confirmation.ordinal,
        trial_id=trial.id,
        trial_ordinal=trial.ordinal,
        values=trial.factor_values,
        eval_run_id=confirmation.eval_run_id,
        status=evaluation.status if evaluation else "queued",  # type: ignore[arg-type]
        reason=confirmation.reason,
        note=confirmation.note,
        dev=trial.objective,
        test=test,
        created_by=confirmation.created_by,
        created_at=confirmation.created_at,
    )


@router.post(
    "/experiments/{experiment_id}/trials/{trial_id}/confirm",
    response_model=ConfirmationOut,
    status_code=202,
    summary="Confirm a trial chosen on dev on the locked test split, once",
    description="Starts an eval run of the test split for the trial's pipeline run and "
    "retrieval config; the read is counted on the eval set version and audited. A "
    "second read for the experiment without a `reason` is refused with "
    "`test_split_already_read`; with one, it's logged.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def confirm_trial(
    request: Request,
    experiment_id: uuid.UUID,
    trial_id: uuid.UUID,
    body: ConfirmIn,
    principal: Principal = Depends(require_role("editor")),
) -> JSONResponse:
    with engine_of(request).begin() as conn:
        experiment = _experiment_or_404(conn, principal, experiment_id)
        trial = store.get_trial(conn, principal.workspace.id, experiment.id, trial_id)
        if trial is None:
            raise Problem("not_found", code="trial_not_found")
        try:
            confirmation, evaluation, reads = store.confirm_trial(
                conn,
                experiment,
                trial,
                _queue(request),
                reason=body.reason,
                note=body.note,
                created_by=principal.subject,
                max_attempts=request.app.state.settings.worker.max_attempts,
            )
        except store.ConfirmationRefused as error:
            raise Problem("conflict", code=error.code) from None
        record_audit_event(
            conn,
            workspace_id=principal.workspace.id,
            actor_kind=principal.actor_kind,
            actor_subject=principal.subject,
            action="eval_set.test_read",
            resource_type="eval_set_version",
            resource_id=str(confirmation.eval_set_version_id),
            request_id=current_log_context().get("request_id"),
            details={"reads": reads, "eval_run": str(evaluation.id)},
        )
        _audit(
            conn,
            principal,
            "experiment.confirm",
            experiment.id,
            trial=trial.ordinal,
            read=confirmation.ordinal,
            reason=confirmation.reason,
            eval_run=str(evaluation.id),
        )
        out = _confirmation_out(confirmation, trial, evaluation, experiment.spec())
    return JSONResponse(out.model_dump(mode="json"), status_code=202)


@router.get(
    "/experiments/{experiment_id}/confirmations",
    response_model=ConfirmationsOut,
    summary="An experiment's reads of the test split: each trial's dev and test results",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_confirmations(
    request: Request,
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> ConfirmationsOut:
    with engine_of(request).connect() as conn:
        experiment = _experiment_or_404(conn, principal, experiment_id)
        found = store.confirmations_of(conn, experiment.id)
        trials = {trial.id: trial for trial in store.trials_of(conn, experiment.id)}
        evaluations = evalruns.eval_runs_by_id(conn, [item.eval_run_id for item in found])
        version = evalsets.get_version(
            conn, principal.workspace.id, version_id=experiment.eval_set_version_id
        )
    spec = experiment.spec()
    return ConfirmationsOut(
        items=[
            _confirmation_out(item, trials[item.trial_id], evaluations.get(item.eval_run_id), spec)
            for item in found
        ],
        test_reads=version.test_reads if version else 0,
    )
