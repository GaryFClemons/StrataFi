"""Runs: planning, executing and recording a pipeline run.

A run is one pipeline spec version over one frozen corpus version, to a stage (``until``), on
a sample. Its lifecycle is ``queued`` → ``running`` → ``succeeded`` or ``failed``; cancelling
goes through ``cancelling`` to ``cancelled``.

``Orchestrator`` drives it, always holding the run's row lock, so concurrent workers advance
it one at a time:

- ``start`` (the worker's ``run.start`` job) records the run's entries and walks each through
  ``stratafi_core.planner.next_task``. A task whose artifact already exists is ``cached`` and
  the walk goes on at once; the first one that doesn't is ``queued`` with a ``stage.run`` job.
  An identical second run is therefore all cache hits, planned and finished in one transaction.
- ``complete_task``, ``fail_task`` and ``cancel_task`` (the worker's ``stage.run`` job) record
  the outcome and advance the entry to its next task.
- A pipeline with a corpus-wide clean rule (``clean/near-duplicates``) holds each entry after
  Parse until every entry has been parsed or stopped, then plans every clean task with the
  corpus: each parsed entry's parse key and rank (effective date, version, name).
- Once every entry has stopped (done, excluded, unsupported, failed or cancelled), the
  corpus-level index tasks are planned over the entries that were embedded, in entry order.
- When no task is left queued or running, the run finishes: ``failed`` if a task failed,
  ``cancelled`` if it was cancelling, else ``succeeded``. Its manifest is completed with the
  execution, cost, egress and outputs, and ``manifest_hash`` is its fingerprint.

``cancel`` marks a run ``cancelling``, cancels its waiting tasks, and returns the jobs to cancel:
queued ones never run, running ones stop at their next checkpoint. Each change notifies a
``run`` event (IDs, status and counts) for live progress.
"""

import json
import re
import threading
import uuid
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from stratafi_core.components import ComponentRegistry
from stratafi_core.manifest import RunManifest
from stratafi_core.planner import (
    ALL_STAGES,
    CorpusRef,
    EntryInput,
    StageOutput,
    Stop,
    TaskSpec,
    Wait,
    index_tasks,
    needs_corpus,
    next_task,
    stages_until,
)
from stratafi_core.specs import PipelineSpec
from stratafi_store.jobs import EVENTS_CHANNEL, JobQueue
from stratafi_store.lineage import ArtifactInput, ArtifactRecord, record_artifact
from stratafi_store.tables import (
    annotations,
    artifacts,
    corpus_version_documents,
    jobs,
    pipeline_specs,
    run_entries,
    runs,
    stage_tasks,
)

STAGE_JOB = "stage.run"
START_JOB = "run.start"
CORPUS_ORDINAL = -1
ACTIVE_TASK_STATUSES = ("pending", "queued", "running")
DONE_TASK_STATUSES = ("succeeded", "cached")
# What each stage's artifact is called.
ARTIFACT_KINDS = {
    "extract": "raw",
    "scan": "scan_report",
    "parse": "parsed",
    "clean": "cleaned",
    "chunk": "chunks",
    "enrich": "enriched",
    "embed": "vectors",
    "index": "index",
}
STOP_STATES = {"done": "done", "excluded": "excluded", "unsupported_type": "unsupported_type"}
# How often a running run's counts are refreshed: each refresh reads all its tasks.
COUNTS_SECONDS = 1
COUNTS_INTERVAL = sa.text(f"interval '{COUNTS_SECONDS} second'")
# The metrics a task records for each external provider it called: (provider, calls, tokens).
PROVIDER_METRICS = (
    ("model_gateway", "gateway.calls", "tokens"),
    ("tika", "tika.calls", None),
    # Hosted parsers: which provider received which documents.
    ("azure-document-intelligence", "azure_document_intelligence.calls", None),
    ("aws-textract", "aws_textract.calls", None),
    ("google-document-ai", "google_document_ai.calls", None),
    ("llamaparse", "llamaparse.calls", None),
    ("mistral", "mistral.calls", None),
    ("openai-compatible", "openai_compatible.calls", "tokens"),
    ("cohere", "cohere.calls", None),
    ("jina", "jina.calls", None),
    ("voyage", "voyage.calls", None),
)
FINISHED_RUN_STATUSES = ("succeeded", "failed", "cancelled")

# Called in the transaction that finishes a run or an eval run, with its kind ("run" or
# "eval_run"), its workspace and its ID: experiments learn of the runs they wait on this way
# (stratafi_store.experiments).
FinishHook = Callable[[Connection, str, uuid.UUID, uuid.UUID], None]


@dataclass(frozen=True)
class Run:
    id: uuid.UUID
    workspace_id: uuid.UUID
    kind: str
    mode: str
    status: str
    pipeline_spec_id: uuid.UUID
    corpus_version_id: uuid.UUID
    until_stage: str | None
    sample: dict[str, Any] | None
    rerun_of: uuid.UUID | None
    rerun_from: str | None
    spec_hash: str
    model_env: dict[str, Any]
    manifest: dict[str, Any] | None
    manifest_hash: str | None
    counts: dict[str, Any]
    error_code: str | None
    created_by: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def finished(self) -> bool:
        return self.status in ("succeeded", "failed", "cancelled")

    @property
    def rerun(self) -> tuple[str, str] | None:
        return (self.rerun_from, str(self.id)) if self.rerun_from else None


@dataclass(frozen=True)
class RunEntry:
    workspace_id: uuid.UUID
    run_id: uuid.UUID
    ordinal: int
    name: str
    document_id: uuid.UUID
    sha256: str
    duplicate: bool
    label: str | None
    state: str
    stop_code: str | None

    def input(self) -> EntryInput:
        return EntryInput(
            name=self.name,
            document_id=str(self.document_id),
            sha256=self.sha256,
            extension=PurePosixPath(self.name).suffix.lower(),
            duplicate=self.duplicate,
            label=self.label,
        )


@dataclass(frozen=True)
class Task:
    id: uuid.UUID
    workspace_id: uuid.UUID
    run_id: uuid.UUID
    ordinal: int
    stage: str
    component_id: str
    component_version: str
    cache_key: str
    inputs: list[str]
    params: dict[str, Any]
    env: dict[str, Any]
    route: str | None
    resource_class: str
    status: str
    cache_hit: bool
    job_id: uuid.UUID | None
    attempts: int
    error_code: str | None
    metrics: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: float | None


def version_rank(entry: Any, name: str) -> list[str]:
    """How new a corpus entry is, for near-duplicate cleaning: its effective date, then its
    version with numbers compared as numbers, then its name. Higher is newer."""
    effective = entry.effective_from.isoformat() if entry and entry.effective_from else ""
    version = str(entry.doc_version) if entry and entry.doc_version else ""
    padded = re.sub(r"\d+", lambda match: match.group().zfill(12), version)
    return [effective, padded, name]


def _run(row: Any) -> Run:
    return Run(**{column.name: getattr(row, column.name) for column in runs.columns})


def _entry(row: Any) -> RunEntry:
    return RunEntry(**{column.name: getattr(row, column.name) for column in run_entries.columns})


def _task(row: Any) -> Task:
    return Task(**{column.name: getattr(row, column.name) for column in stage_tasks.columns})


# --- reading ----------------------------------------------------------------------------------


def create_run(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    pipeline_spec_id: uuid.UUID,
    corpus_version_id: uuid.UUID,
    spec_hash: str,
    created_by: str,
    until: str | None = None,
    sample: Mapping[str, Any] | None = None,
    mode: str = "batch",
    rerun_of: uuid.UUID | None = None,
    rerun_from: str | None = None,
) -> Run:
    if until is not None and until not in ALL_STAGES:
        raise ValueError("unknown stage")
    row = conn.execute(
        sa.insert(runs)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            mode=mode,
            pipeline_spec_id=pipeline_spec_id,
            corpus_version_id=corpus_version_id,
            until_stage=until,
            sample=dict(sample) if sample is not None else None,
            rerun_of=rerun_of,
            rerun_from=rerun_from,
            spec_hash=spec_hash,
            created_by=created_by,
        )
        .returning(*runs.columns)
    ).one()
    run = _run(row)
    _notify(conn, run)
    return run


def get_run(conn: Connection, workspace_id: uuid.UUID | None, run_id: uuid.UUID) -> Run | None:
    query = sa.select(runs).where(runs.c.id == run_id)
    if workspace_id is not None:
        query = query.where(runs.c.workspace_id == workspace_id)
    row = conn.execute(query).first()
    return _run(row) if row is not None else None


def runs_by_id(conn: Connection, run_ids: Collection[uuid.UUID]) -> dict[uuid.UUID, Run]:
    if not run_ids:
        return {}
    rows = conn.execute(sa.select(runs).where(runs.c.id.in_(list(run_ids))))
    return {row.id: _run(row) for row in rows}


def list_runs(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    after: tuple[datetime, uuid.UUID] | None = None,
    status: str | None = None,
    include_archived: bool = True,
) -> list[Run]:
    """Newest first."""
    query = sa.select(runs).where(runs.c.workspace_id == workspace_id)
    if status is not None:
        query = query.where(runs.c.status == status)
    if not include_archived:
        query = query.where(
            ~sa.exists().where(
                annotations.c.run_id == runs.c.id, annotations.c.archived_at.is_not(None)
            )
        )
    if after is not None:
        query = query.where(sa.tuple_(runs.c.created_at, runs.c.id) < after)
    rows = conn.execute(query.order_by(runs.c.created_at.desc(), runs.c.id.desc()).limit(limit))
    return [_run(row) for row in rows]


def entries_of(conn: Connection, run_id: uuid.UUID) -> list[RunEntry]:
    rows = conn.execute(
        sa.select(run_entries).where(run_entries.c.run_id == run_id).order_by(run_entries.c.ordinal)
    )
    return [_entry(row) for row in rows]


def get_entry(conn: Connection, run_id: uuid.UUID, ordinal: int) -> RunEntry | None:
    row = conn.execute(
        sa.select(run_entries).where(
            run_entries.c.run_id == run_id, run_entries.c.ordinal == ordinal
        )
    ).first()
    return _entry(row) if row is not None else None


def _has_active_entries(conn: Connection, run_id: uuid.UUID) -> bool:
    return conn.execute(
        sa.select(
            sa.exists().where(run_entries.c.run_id == run_id, run_entries.c.state == "active")
        )
    ).scalar_one()


def tasks_of(
    conn: Connection,
    run_id: uuid.UUID,
    *,
    stage: str | None = None,
    ordinal: int | None = None,
    status: str | None = None,
) -> list[Task]:
    """A run's tasks, by entry and then in stage order."""
    query = sa.select(stage_tasks).where(stage_tasks.c.run_id == run_id)
    if stage is not None:
        query = query.where(stage_tasks.c.stage == stage)
    if ordinal is not None:
        query = query.where(stage_tasks.c.ordinal == ordinal)
    if status is not None:
        query = query.where(stage_tasks.c.status == status)
    order = sa.case(
        {stage: index for index, stage in enumerate(ALL_STAGES)}, value=stage_tasks.c.stage
    )
    rows = conn.execute(query.order_by(stage_tasks.c.ordinal, order, stage_tasks.c.component_id))
    tasks = [_task(row) for row in rows]
    # Corpus-level tasks (ordinal -1) come last, after every entry's.
    return [t for t in tasks if t.ordinal >= 0] + [t for t in tasks if t.ordinal < 0]


def summaries(
    conn: Connection, workspace_id: uuid.UUID, cache_keys: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """The summaries of the artifacts with these keys that exist."""
    keys = list(set(cache_keys))
    if not keys:
        return {}
    rows = conn.execute(
        sa.select(artifacts.c.cache_key, artifacts.c.summary).where(
            artifacts.c.workspace_id == workspace_id, artifacts.c.cache_key.in_(keys)
        )
    )
    return {row.cache_key: dict(row.summary) for row in rows}


def get_task(conn: Connection, task_id: uuid.UUID) -> Task | None:
    row = conn.execute(sa.select(stage_tasks).where(stage_tasks.c.id == task_id)).first()
    return _task(row) if row is not None else None


# A spec version never changes, so each is read and validated once per process: every
# task's completion plans with it.
_SPECS: OrderedDict[uuid.UUID, PipelineSpec] = OrderedDict()
_SPECS_SIZE = 64
_SPECS_LOCK = threading.Lock()


def run_spec(conn: Connection, run: Run) -> PipelineSpec:
    with _SPECS_LOCK:
        cached = _SPECS.get(run.pipeline_spec_id)
        if cached is not None:
            _SPECS.move_to_end(run.pipeline_spec_id)
            return cached
    document = conn.execute(
        sa.select(pipeline_specs.c.document).where(pipeline_specs.c.id == run.pipeline_spec_id)
    ).scalar_one()
    spec = PipelineSpec.model_validate(document)
    with _SPECS_LOCK:
        _SPECS[run.pipeline_spec_id] = spec
        while len(_SPECS) > _SPECS_SIZE:
            _SPECS.popitem(last=False)
    return spec


def counts(conn: Connection, run_id: uuid.UUID) -> dict[str, Any]:
    """Tasks by status, by stage and status, and entries by state: metadata only."""
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for row in conn.execute(
        sa.select(stage_tasks.c.stage, stage_tasks.c.status, sa.func.count())
        .where(stage_tasks.c.run_id == run_id)
        .group_by(stage_tasks.c.stage, stage_tasks.c.status)
    ):
        by_stage[row[0]][row[1]] += row[2]
        totals[row[1]] += row[2]
    entries = Counter(
        dict(
            conn.execute(
                sa.select(run_entries.c.state, sa.func.count())
                .where(run_entries.c.run_id == run_id)
                .group_by(run_entries.c.state)
            ).all()
        )
    )
    return {
        "tasks": dict(totals),
        "stages": {stage: dict(by_stage[stage]) for stage in ALL_STAGES if stage in by_stage},
        "entries": dict(entries),
    }


def _notify(conn: Connection, run: Run, run_counts: Mapping[str, Any] | None = None) -> None:
    event: dict[str, Any] = {
        "type": "run",
        "workspace_id": str(run.workspace_id),
        "run_id": str(run.id),
        "status": run.status,
    }
    if run_counts is not None:
        event["counts"] = dict(run_counts)
    conn.execute(
        sa.text("SELECT pg_notify(:channel, :payload)"),
        {"channel": EVENTS_CHANNEL, "payload": json.dumps(event, separators=(",", ":"))},
    )


# --- the orchestrator -------------------------------------------------------------------------


class Orchestrator:
    def __init__(
        self,
        engine: Engine,
        queue: JobQueue,
        registry: ComponentRegistry,
        *,
        max_attempts: int = 3,
        on_finish: FinishHook | None = None,
    ) -> None:
        self.engine = engine
        self.queue = queue
        self.registry = registry
        self.max_attempts = max_attempts
        self.on_finish = on_finish

    def _finished(self, conn: Connection, run: Run) -> None:
        if self.on_finish is not None:
            self.on_finish(conn, "run", run.workspace_id, run.id)

    # --- planning and transitions (each holds the run's row lock) -----------------------------

    def _lock(self, conn: Connection, run_id: uuid.UUID) -> Run | None:
        row = conn.execute(sa.select(runs).where(runs.c.id == run_id).with_for_update()).first()
        return _run(row) if row is not None else None

    def _share(self, conn: Connection, run_id: uuid.UUID) -> Run | None:
        """The run, locked so it can't change while tasks begin side by side."""
        row = conn.execute(
            sa.select(runs).where(runs.c.id == run_id).with_for_update(read=True)
        ).first()
        return _run(row) if row is not None else None

    def _update_run(self, conn: Connection, run: Run, **values: Any) -> Run:
        row = conn.execute(
            sa.update(runs)
            .where(runs.c.id == run.id)
            .values(updated_at=sa.func.now(), **values)
            .returning(*runs.columns)
        ).one()
        return _run(row)

    def _insert_task(
        self, conn: Connection, run: Run, ordinal: int, spec: TaskSpec, *, cached: bool
    ) -> Task:
        values: dict[str, Any] = {
            "id": uuid.uuid4(),
            "workspace_id": run.workspace_id,
            "run_id": run.id,
            "ordinal": ordinal,
            "stage": spec.stage,
            "component_id": spec.component_id,
            "component_version": spec.component_version,
            "cache_key": spec.cache_key,
            "inputs": list(spec.inputs),
            "params": spec.params,
            "env": spec.env,
            "route": spec.route,
            "resource_class": spec.resource_class,
            "status": "cached" if cached else "queued",
            "cache_hit": cached,
        }
        if cached:
            values.update(started_at=sa.func.now(), finished_at=sa.func.now(), duration_ms=0.0)
        row = conn.execute(
            sa.insert(stage_tasks).values(**values).returning(*stage_tasks.columns)
        ).one()
        task = _task(row)
        if cached:
            return task
        priority = 10 if run.mode == "interactive" else 0
        job = self.queue.enqueue_in(
            conn,
            workspace_id=run.workspace_id,
            kind=STAGE_JOB,
            payload={"run_id": str(run.id), "task_id": str(task.id)},
            resource_class=spec.resource_class,
            priority=priority,
            max_attempts=self.max_attempts,
        )
        conn.execute(
            sa.update(stage_tasks).where(stage_tasks.c.id == task.id).values(job_id=job.id)
        )
        return task

    def _summary(self, conn: Connection, run: Run, cache_key: str) -> dict[str, Any] | None:
        row = conn.execute(
            sa.select(artifacts.c.summary).where(
                artifacts.c.workspace_id == run.workspace_id, artifacts.c.cache_key == cache_key
            )
        ).first()
        return row.summary if row is not None else None

    def _outputs(self, conn: Connection, run: Run, ordinal: int) -> dict[str, StageOutput]:
        outputs = {}
        rows = conn.execute(
            sa.select(stage_tasks.c.stage, stage_tasks.c.cache_key, artifacts.c.summary)
            .join(
                artifacts,
                sa.and_(
                    artifacts.c.workspace_id == stage_tasks.c.workspace_id,
                    artifacts.c.cache_key == stage_tasks.c.cache_key,
                ),
            )
            .where(
                stage_tasks.c.run_id == run.id,
                stage_tasks.c.ordinal == ordinal,
                stage_tasks.c.status.in_(DONE_TASK_STATUSES),
            )
        )
        for row in rows:
            outputs[row.stage] = StageOutput(row.cache_key, row.summary)
        return outputs

    def _set_entry(
        self, conn: Connection, run: Run, ordinal: int, state: str, code: str | None = None
    ) -> None:
        conn.execute(
            sa.update(run_entries)
            .where(run_entries.c.run_id == run.id, run_entries.c.ordinal == ordinal)
            .values(state=state, stop_code=code)
        )

    def _advance(
        self,
        conn: Connection,
        run: Run,
        spec: PipelineSpec,
        entry: RunEntry,
        corpus: list[CorpusRef] | None = None,
    ) -> None:
        """Walk an entry forward from its finished stages: cached tasks at once, then the
        first task that has to run, the reason it stops, or a wait for the corpus."""
        if run.status != "running":
            return
        outputs = self._outputs(conn, run, entry.ordinal)
        while True:
            step = next_task(
                spec,
                self.registry,
                entry.input(),
                outputs,
                until=run.until_stage,
                model_env=run.model_env,
                rerun=run.rerun,
                corpus=corpus,
            )
            if isinstance(step, Wait):
                return  # the entry stays active; _maybe_release plans it
            if isinstance(step, Stop):
                self._set_entry(conn, run, entry.ordinal, STOP_STATES[step.reason], step.reason)
                return
            summary = self._summary(conn, run, step.cache_key)
            self._insert_task(conn, run, entry.ordinal, step, cached=summary is not None)
            if summary is None:
                return
            outputs[step.stage] = StageOutput(step.cache_key, summary)

    def _maybe_release(self, conn: Connection, run: Run, spec: PipelineSpec) -> None:
        """Plan the clean tasks of a pipeline with a corpus-wide rule once every active entry
        has been parsed."""
        if run.status != "running" or "clean" not in stages_until(run.until_stage):
            return
        if not needs_corpus(spec, self.registry):
            return
        cleaning = {task.ordinal for task in tasks_of(conn, run.id, stage="clean")}
        waiting: list[tuple[RunEntry, str]] = []
        for entry in entries_of(conn, run.id):
            if entry.state != "active" or entry.ordinal in cleaning:
                continue
            outputs = self._outputs(conn, run, entry.ordinal)
            if "parse" not in outputs:
                return  # still on its way to Parse
            waiting.append((entry, outputs["parse"].cache_key))
        if not waiting:
            return
        table = corpus_version_documents
        meta = {
            row.name: row
            for row in conn.execute(
                sa.select(table.c.name, table.c.doc_version, table.c.effective_from).where(
                    table.c.workspace_id == run.workspace_id,
                    table.c.corpus_version_id == run.corpus_version_id,
                )
            )
        }
        corpus: list[CorpusRef] = [
            (key, version_rank(meta.get(entry.name), entry.name)) for entry, key in waiting
        ]
        for entry, _ in waiting:
            self._advance(conn, run, spec, entry, corpus)

    def _maybe_index(self, conn: Connection, run: Run, spec: PipelineSpec) -> None:
        if run.status != "running" or "index" not in stages_until(run.until_stage):
            return
        if _has_active_entries(conn, run.id) or tasks_of(conn, run.id, stage="index"):
            return
        embedded = []
        for entry in entries_of(conn, run.id):
            if entry.state != "done":
                continue
            outputs = self._outputs(conn, run, entry.ordinal)
            if "enrich" in outputs and "embed" in outputs:
                embedded.append((outputs["enrich"].cache_key, outputs["embed"].cache_key))
        if not embedded:
            return
        for step in index_tasks(
            spec, self.registry, embedded, model_env=run.model_env, rerun=run.rerun
        ):
            cached = self._summary(conn, run, step.cache_key) is not None
            self._insert_task(conn, run, CORPUS_ORDINAL, step, cached=cached)

    def _maybe_finish(self, conn: Connection, run: Run) -> Run:
        if run.status not in ("running", "cancelling"):
            return run
        # EXISTS on a partial index (ix_stage_tasks_run_active): one row read, however large
        # the run.
        active = conn.execute(
            sa.select(
                sa.exists().where(
                    stage_tasks.c.run_id == run.id,
                    stage_tasks.c.status.in_(ACTIVE_TASK_STATUSES),
                )
            )
        ).scalar_one()
        if active:
            return run
        if run.status == "running" and _has_active_entries(conn, run.id):
            return run
        failed = conn.execute(
            sa.select(sa.func.count())
            .select_from(stage_tasks)
            .where(stage_tasks.c.run_id == run.id, stage_tasks.c.status == "failed")
        ).scalar_one()
        if run.status == "cancelling":
            status, code = "cancelled", None
        elif failed:
            status, code = "failed", "tasks_failed"
        else:
            status, code = "succeeded", None
        finished = self._update_run(
            conn, run, status=status, error_code=code, finished_at=sa.func.now()
        )
        manifest = final_manifest(conn, finished)
        finished = self._update_run(
            conn,
            finished,
            manifest=manifest.model_dump(mode="json", by_alias=True),
            manifest_hash=manifest.fingerprint(),
            counts=counts(conn, run.id),
        )
        self._finished(conn, finished)
        return finished

    def _settle(self, conn: Connection, run: Run) -> Run:
        """After any change: plan the index if it's due, finish if nothing is left, and tell
        listeners. The counts, which read every task of the run, are refreshed at most every
        ``COUNTS_SECONDS`` while it runs, and always when its status changes."""
        spec = run_spec(conn, run)
        self._maybe_release(conn, run, spec)
        self._maybe_index(conn, run, spec)
        status = run.status
        run = self._maybe_finish(conn, run)
        if run.status == status and not run.finished and not self._counts_due(conn, run):
            return run
        run_counts = counts(conn, run.id)
        if not run.finished:
            run = self._update_run(conn, run, counts=run_counts)
        _notify(conn, run, run_counts)
        return run

    def _counts_due(self, conn: Connection, run: Run) -> bool:
        return conn.execute(
            sa.select(runs.c.updated_at < sa.func.now() - COUNTS_INTERVAL).where(
                runs.c.id == run.id
            )
        ).scalar_one()

    # --- the run.start job --------------------------------------------------------------------

    def start(
        self,
        run_id: uuid.UUID,
        *,
        entries: list[EntryInput],
        model_env: Mapping[str, Mapping[str, Any]],
        manifest: Mapping[str, Any],
        refused: Collection[str] = (),
    ) -> Run | None:
        """Record the run's entries and plan every one as far as the cache allows. Entries
        named in ``refused`` (documents intake refused, such as an archive over its limits)
        are recorded as excluded and never planned."""
        with self.engine.begin() as conn:
            run = self._lock(conn, run_id)
            if run is None or run.status != "queued":
                return run
            run = self._update_run(
                conn,
                run,
                status="running",
                started_at=sa.func.now(),
                model_env={stage: dict(env) for stage, env in model_env.items()},
                manifest=dict(manifest),
            )
            if entries:
                conn.execute(
                    sa.insert(run_entries),
                    [
                        {
                            "workspace_id": run.workspace_id,
                            "run_id": run.id,
                            "ordinal": ordinal,
                            "name": entry.name,
                            "document_id": uuid.UUID(entry.document_id),
                            "sha256": entry.sha256,
                            "duplicate": entry.duplicate,
                            "label": entry.label,
                            "state": "excluded" if entry.name in refused else "active",
                            "stop_code": "refused" if entry.name in refused else None,
                        }
                        for ordinal, entry in enumerate(entries)
                    ],
                )
            spec = run_spec(conn, run)
            for entry in entries_of(conn, run.id):
                if entry.state == "active":
                    self._advance(conn, run, spec, entry)
            return self._settle(conn, run)

    def fail_run(self, run_id: uuid.UUID, code: str) -> Run | None:
        """A run that couldn't start: an invalid spec, an unreachable model gateway."""
        with self.engine.begin() as conn:
            run = self._lock(conn, run_id)
            if run is None or run.finished:
                return run
            run = self._update_run(
                conn, run, status="failed", error_code=code, finished_at=sa.func.now()
            )
            _notify(conn, run)
            self._finished(conn, run)
            return run

    # --- the stage.run job --------------------------------------------------------------------

    def begin_task(self, task_id: uuid.UUID, attempts: int) -> tuple[Task, Run] | None:
        """Mark a task running for this attempt, or None if it shouldn't run any more. Tasks
        of a running run begin side by side, under a shared lock on its row; a task of a
        run that's stopping is cancelled under the exclusive one."""
        with self.engine.begin() as conn:
            task = get_task(conn, task_id)
            if task is None:
                return None
            run = self._share(conn, task.run_id)
            if run is None or task.status not in ("queued", "running"):
                return None
            if run.status == "running":
                return self._begin(conn, task, attempts), run
        with self.engine.begin() as conn:
            run = self._lock(conn, task.run_id)
            task = get_task(conn, task_id)
            if run is None or task is None or task.status not in ("queued", "running"):
                return None
            if run.status == "running":
                return self._begin(conn, task, attempts), run
            self._cancel_task(conn, run, task)
            self._settle(conn, run)
            return None

    def _begin(self, conn: Connection, task: Task, attempts: int) -> Task:
        row = conn.execute(
            sa.update(stage_tasks)
            .where(stage_tasks.c.id == task.id)
            .values(
                status="running",
                attempts=attempts,
                started_at=sa.func.coalesce(stage_tasks.c.started_at, sa.func.now()),
            )
            .returning(*stage_tasks.columns)
        ).one()
        return _task(row)

    def complete_task(
        self,
        task_id: uuid.UUID,
        *,
        record: ArtifactRecord,
        inputs: Iterable[str | ArtifactInput],
        metrics: Mapping[str, Any],
        duration_ms: float,
        cache_hit: bool = False,
    ) -> Run | None:
        with self.engine.begin() as conn:
            task = get_task(conn, task_id)
            if task is None:
                return None
            run = self._lock(conn, task.run_id)
            if run is None or task.status not in ("queued", "running"):
                return run
            record_artifact(conn, record, list(inputs))
            conn.execute(
                sa.update(stage_tasks)
                .where(stage_tasks.c.id == task.id)
                .values(
                    status="succeeded",
                    cache_hit=cache_hit,
                    metrics=dict(metrics),
                    finished_at=sa.func.now(),
                    duration_ms=round(duration_ms, 3),
                    error_code=None,
                )
            )
            entry = get_entry(conn, run.id, task.ordinal) if task.ordinal >= 0 else None
            if entry is not None:
                self._advance(conn, run, run_spec(conn, run), entry)
            return self._settle(conn, run)

    def fail_task(
        self, task_id: uuid.UUID, code: str, *, duration_ms: float | None = None
    ) -> Run | None:
        with self.engine.begin() as conn:
            task = get_task(conn, task_id)
            if task is None:
                return None
            run = self._lock(conn, task.run_id)
            if run is None or task.status not in ("queued", "running"):
                return run
            conn.execute(
                sa.update(stage_tasks)
                .where(stage_tasks.c.id == task.id)
                .values(
                    status="failed",
                    error_code=code,
                    finished_at=sa.func.now(),
                    duration_ms=round(duration_ms, 3) if duration_ms is not None else None,
                )
            )
            if task.ordinal >= 0:
                self._set_entry(conn, run, task.ordinal, "failed", code)
            return self._settle(conn, run)

    def _cancel_task(self, conn: Connection, run: Run, task: Task) -> None:
        conn.execute(
            sa.update(stage_tasks)
            .where(stage_tasks.c.id == task.id, stage_tasks.c.status.in_(ACTIVE_TASK_STATUSES))
            .values(status="cancelled", finished_at=sa.func.now())
        )
        if task.ordinal >= 0:
            conn.execute(
                sa.update(run_entries)
                .where(
                    run_entries.c.run_id == run.id,
                    run_entries.c.ordinal == task.ordinal,
                    run_entries.c.state == "active",
                )
                .values(state="cancelled", stop_code="cancelled")
            )

    def cancel_task(self, task_id: uuid.UUID) -> Run | None:
        """The worker stopped a task at a checkpoint."""
        with self.engine.begin() as conn:
            task = get_task(conn, task_id)
            if task is None:
                return None
            run = self._lock(conn, task.run_id)
            if run is None:
                return None
            self._cancel_task(conn, run, task)
            return self._settle(conn, run)

    # --- the sweep ---------------------------------------------------------------------------

    def reconcile(self) -> int:
        """Settle what a job ended without recording: a stage task whose job failed or was
        cancelled outside its handler (its lease ran out on its last attempt), and a queued
        run whose start job did. Workers call it with their heartbeats. Returns how many tasks
        and runs it settled."""
        with self.engine.connect() as conn:
            orphaned = conn.execute(
                sa.select(stage_tasks.c.id, jobs.c.status, jobs.c.error_code)
                .join(jobs, jobs.c.id == stage_tasks.c.job_id)
                .where(
                    stage_tasks.c.status.in_(("queued", "running")),
                    jobs.c.status.in_(("failed", "cancelled")),
                )
            ).all()
            unstarted = conn.execute(
                sa.select(runs.c.id, jobs.c.status, jobs.c.error_code)
                .join(
                    jobs,
                    sa.and_(
                        jobs.c.kind == START_JOB,
                        jobs.c.workspace_id == runs.c.workspace_id,
                        jobs.c.payload["run_id"].astext == sa.cast(runs.c.id, sa.Text),
                    ),
                )
                .where(runs.c.status == "queued", jobs.c.status.in_(("failed", "cancelled")))
            ).all()
        for row in orphaned:
            if row.status == "cancelled":
                self.cancel_task(row.id)
            else:
                self.fail_task(row.id, row.error_code or "job_failed")
        for row in unstarted:
            self.fail_run(row.id, row.error_code or "start_cancelled")
        return len(orphaned) + len(unstarted)

    # --- the client's side --------------------------------------------------------------------

    def cancel(
        self, run_id: uuid.UUID, workspace_id: uuid.UUID
    ) -> tuple[Run, list[uuid.UUID]] | None:
        """Mark a run cancelling and cancel its waiting tasks. Returns it and the jobs to cancel
        in the queue; the run is cancelled once its running tasks stop."""
        with self.engine.begin() as conn:
            run = self._lock(conn, run_id)
            if run is None or run.workspace_id != workspace_id:
                return None
            if run.finished:
                return run, []
            jobs_to_cancel = [
                task.job_id
                for task in tasks_of(conn, run.id)
                if task.status in ACTIVE_TASK_STATUSES and task.job_id is not None
            ]
            if run.status == "queued":
                run = self._update_run(conn, run, status="cancelled", finished_at=sa.func.now())
                _notify(conn, run)
                self._finished(conn, run)
                return run, jobs_to_cancel
            run = self._update_run(conn, run, status="cancelling")
            for task in tasks_of(conn, run.id):
                if task.status in ("pending", "queued"):
                    self._cancel_task(conn, run, task)
            conn.execute(
                sa.update(run_entries)
                .where(run_entries.c.run_id == run.id, run_entries.c.state == "active")
                .values(state="cancelled", stop_code="cancelled")
            )
            return self._settle(conn, run), jobs_to_cancel


# --- the final manifest -----------------------------------------------------------------------


def final_manifest(conn: Connection, run: Run) -> RunManifest:
    """The run's manifest so far: what was recorded when it started, plus its execution, cost,
    egress and outputs."""
    tasks = tasks_of(conn, run.id)
    keys = [task.cache_key for task in tasks if task.status in DONE_TASK_STATUSES]
    made = {
        row.cache_key: StoredArtifact(row.kind, row.size_bytes, dict(row.summary))
        for row in conn.execute(
            sa.select(
                artifacts.c.cache_key, artifacts.c.kind, artifacts.c.size_bytes, artifacts.c.summary
            ).where(artifacts.c.workspace_id == run.workspace_id, artifacts.c.cache_key.in_(keys))
        )
    }
    return build_manifest(run, entries_of(conn, run.id), tasks, made)


@dataclass(frozen=True)
class StoredArtifact:
    kind: str
    size_bytes: int
    summary: dict[str, Any]


def _numbers(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def build_manifest(
    run: Run,
    entries: Iterable[RunEntry],
    tasks: list[Task],
    made: Mapping[str, StoredArtifact],
) -> RunManifest:
    """The manifest from the start record (``run.manifest``), the run's entries and tasks, and
    the artifacts its finished tasks made, by cache key. A pure function."""
    start = dict(run.manifest or {})
    by_ordinal = {entry.ordinal: entry for entry in entries}
    task_records = []
    by_stage: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_stage[task.stage].append(task)
        entry = by_ordinal.get(task.ordinal)
        task_records.append(
            {
                "stage": task.stage,
                "entry": entry.name if entry else None,
                "component": task.component_id,
                "component_version": task.component_version,
                "cache_key": task.cache_key,
                "status": task.status,
                "cache_hit": task.cache_hit,
                "attempts": task.attempts,
                "started_at": task.started_at,
                "finished_at": task.finished_at,
                "duration_ms": task.duration_ms,
                "peak_memory_bytes": task.metrics.get("peak_memory_bytes"),
                "error_code": task.error_code,
                "metrics": _numbers(task.metrics),
            }
        )
    stages = []
    for stage in ALL_STAGES:
        stage_tasks_ = by_stage.get(stage, [])
        if not stage_tasks_:
            continue
        statuses = Counter(task.status for task in stage_tasks_)
        ran = [task for task in stage_tasks_ if task.status == "succeeded"]
        duration = sum(task.duration_ms or 0.0 for task in ran)
        metrics: Counter[str] = Counter()
        for task in stage_tasks_:
            metrics.update(_numbers(task.metrics))
        stages.append(
            {
                "stage": stage,
                "tasks": len(stage_tasks_),
                "succeeded": statuses["succeeded"],
                "cached": statuses["cached"],
                "skipped": statuses["skipped"],
                "failed": statuses["failed"],
                "cancelled": statuses["cancelled"],
                "duration_ms": round(duration, 3),
                # Tasks a second of work, over the tasks this run executed.
                "throughput_per_second": (
                    round(len(ran) / (duration / 1000), 3) if ran and duration else None
                ),
                "metrics": dict(metrics),
            }
        )
    # Cost and egress count only work this run did: cache hits sent nothing anywhere.
    providers: dict[str, dict[str, int]] = {}
    egress: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        if task.status != "succeeded" or task.cache_hit:
            continue
        entry = by_ordinal.get(task.ordinal)
        for provider, calls_metric, tokens_metric in PROVIDER_METRICS:
            calls = int(task.metrics.get(calls_metric, 0))
            if not calls:
                continue
            usage = providers.setdefault(provider, {"calls": 0, "tokens": 0})
            usage["calls"] += calls
            if tokens_metric is not None:
                usage["tokens"] += int(task.metrics.get(tokens_metric, 0))
            if entry is not None:
                egress[provider].add(str(entry.document_id))
    price_table = dict(start.get("cost", {}).get("price_table", {}))
    estimated = sum(
        usage["tokens"] / 1000 * price_table.get(name, 0.0) for name, usage in providers.items()
    )
    outputs = [
        {
            "cache_key": task.cache_key,
            "kind": made[task.cache_key].kind,
            "stage": task.stage,
            "size_bytes": made[task.cache_key].size_bytes,
            "summary": made[task.cache_key].summary,
        }
        for task in tasks
        if task.cache_key in made and task.status in DONE_TASK_STATUSES
    ]
    duration_ms = (
        (run.finished_at - run.started_at).total_seconds() * 1000
        if run.finished_at and run.started_at
        else None
    )
    # What the gateway says it served, and the dimensions, come from the vectors themselves.
    models = [dict(model) for model in start.get("models", [])]
    for model in models:
        summaries = [
            made[task.cache_key].summary
            for task in by_stage.get(model["role"], [])
            if task.cache_key in made
        ]
        if summaries:
            model.setdefault("dimensions", summaries[0].get("dims"))
            model.setdefault("served_model", summaries[0].get("served_model"))
    return RunManifest.model_validate(
        {
            **start,
            "models": models,
            "execution": {
                "status": run.status,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
                "error_code": run.error_code,
                "stages": stages,
                "tasks": task_records,
            },
            "cost": {
                "providers": providers,
                "currency": "USD",
                "estimated": round(estimated, 6),
                "price_table": price_table,
            },
            "egress": [
                {"provider": name, "documents": sorted(documents)}
                for name, documents in sorted(egress.items())
            ],
            "outputs": outputs,
        }
    )
