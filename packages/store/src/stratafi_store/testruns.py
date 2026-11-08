"""Test suites and their runs.

A suite is saved as versions, like a pipeline spec: ``save_suite`` appends the resolved
document unless it matches the latest one. ``request_run`` asks for a run of a suite version.
Only one run per suite is active at a time, and at most one waits behind it: a request that
finds a run already waiting joins it (its trigger is added to the waiting run's ``triggers``,
and the waiting run takes the newer suite version), so overlapping triggers make one follow-up
run. Partial unique indexes hold both rules in the database.

``TestOrchestrator`` moves a run on, always holding its row lock:

1. ``start`` (the worker's ``test.start`` job) waits while another run of the suite is active,
   then resolves the suite's references to fixed IDs (``resolve``): the target's pipeline spec
   and retrieval config versions (a profile's documents are saved as versions of their own
   names), the corpus version, the eval set version (frozen, with dev items) and the baseline.
   It starts the pipeline runs the checks need: the target's, and a profile or pipeline
   baseline's over the same corpus version (one run when both are the same configuration). A
   pinned eval run, or the suite's last passing run, is a baseline with runs that exist.
2. ``advance`` (``test.advance``, queued by the runs' ``finish_hook``) starts the eval runs on the
   dev split once their pipeline run has succeeded with its indexes, and, when nothing is left
   to wait for, marks the run ``evaluating`` and queues ``test.evaluate``. Everything is planned
   against the cache, so a suite on an unchanged target reuses every artifact and finishes in
   seconds.
3. ``finish`` records the checks' results (the worker evaluates them,
   ``stratafi_worker.testruns``) and the run's record (``stratafi.test-run.v1``), and ends it
   ``passed`` or ``failed``. ``fail`` ends it ``error``: its inputs couldn't be resolved, or the
   evaluation itself broke. Either way the waiting run, if any, starts next, and the webhooks
   that want the run's end get a delivery (``stratafi_store.webhooks``).

``reconcile`` (the worker's heartbeat) queues whatever a run missed. Each change notifies a
``test_run`` event with the test run's ID as ``run_id``, so ``GET /events?run_id=`` follows it.
Everything stored is IDs, codes and numbers: a suite's golden queries stay in its document.
"""

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from stratafi_core.components import ComponentRegistry
from stratafi_core.evalruns import EvalOptions
from stratafi_core.experiments import parse_ref
from stratafi_core.profiles import KnowledgeProfile
from stratafi_core.specs import (
    PipelineSpec,
    RetrievalConfig,
    SpecInvalid,
    document_hash,
    dump_document,
    resolve_pipeline,
    resolve_retrieval,
    spec_hash,
)
from stratafi_core.suites import (
    BaselineRef,
    CheckResult,
    RegressionCheck,
    Summary,
    TestRunRecord,
    TestSuite,
    dump_suite,
    parse_profile_ref,
    suite_hash,
)
from stratafi_core.suites import document_hash as suite_document_hash
from stratafi_store import corpora, evalruns, evalsets, runs, specs, webhooks
from stratafi_store.jobs import EVENTS_CHANNEL, JobQueue
from stratafi_store.runs import FinishHook
from stratafi_store.tables import (
    ACTIVE_TEST_RUN_STATUSES,
    eval_runs,
    eval_set_versions,
    jobs,
    knowledge_profiles,
    test_check_results,
    test_runs,
)
from stratafi_store.tables import runs as runs_table

START_JOB = "test.start"
ADVANCE_JOB = "test.advance"
EVALUATE_JOB = "test.evaluate"
FINISHED = ("passed", "failed", "error", "cancelled")
RUN_COLUMNS = (
    "pipeline_run_id",
    "eval_run_id",
    "baseline_pipeline_run_id",
    "baseline_eval_run_id",
)


class ResolveError(Exception):
    """A suite's reference can't be resolved: a missing corpus, an unfrozen eval set... ``code``
    is a short reason, safe to show."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class TestRun:
    __test__ = False  # not a pytest class

    id: uuid.UUID
    workspace_id: uuid.UUID
    suite_id: uuid.UUID
    suite_name: str
    suite_version: int
    status: str
    trigger: str
    triggers: list[str]
    resolved: dict[str, Any]
    pipeline_run_id: uuid.UUID | None
    eval_run_id: uuid.UUID | None
    baseline_pipeline_run_id: uuid.UUID | None
    baseline_eval_run_id: uuid.UUID | None
    baseline_test_run_id: uuid.UUID | None
    summary: dict[str, Any]
    record: dict[str, Any] | None
    record_hash: str | None
    error_code: str | None
    created_by: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def finished(self) -> bool:
        return self.status in FINISHED

    @property
    def outcome(self) -> str | None:
        return self.status if self.status in ("passed", "failed") else None


@dataclass(frozen=True)
class CheckRow:
    test_run_id: uuid.UUID
    check_id: str
    ordinal: int
    type: str
    result: str
    code: str | None
    value: float | None
    baseline_value: float | None
    threshold: dict[str, Any]
    detail: dict[str, Any]
    evidence: list[dict[str, Any]]
    duration_ms: float | None

    def result_model(self) -> CheckResult:
        return CheckResult.model_validate(
            {
                "id": self.check_id,
                "type": self.type,
                "result": self.result,
                "code": self.code,
                "value": self.value,
                "threshold": self.threshold,
                "baseline_value": self.baseline_value,
                "detail": self.detail,
                "evidence": self.evidence,
                "duration_ms": self.duration_ms,
            }
        )


def _run(row: Any) -> TestRun:
    return TestRun(**{column.name: getattr(row, column.name) for column in test_runs.columns})


def _check(row: Any) -> CheckRow:
    return CheckRow(
        **{
            column.name: getattr(row, column.name)
            for column in test_check_results.columns
            if column.name != "workspace_id"
        }
    )


# --- suites ------------------------------------------------------------------------------------


def save_suite(
    conn: Connection, *, workspace_id: uuid.UUID, suite: TestSuite, created_by: str
) -> tuple[specs.StoredDocument, bool]:
    """Append ``suite`` (validated) to its name's versions; returns the version and whether it's
    new."""
    return specs.save_version(
        conn,
        "suite",
        workspace_id=workspace_id,
        document=dump_suite(suite),
        spec_hash=suite_hash(suite),
        document_hash=suite_document_hash(suite),
        created_by=created_by,
    )


def get_suite(
    conn: Connection, workspace_id: uuid.UUID, name: str, version: int | None = None
) -> specs.StoredDocument | None:
    return specs.get_version(conn, "suite", workspace_id, name, version)


def get_suite_by_id(
    conn: Connection, workspace_id: uuid.UUID, suite_id: uuid.UUID
) -> specs.StoredDocument | None:
    return specs.get_by_id(conn, "suite", workspace_id, suite_id)


def suite_of(stored: specs.StoredDocument) -> TestSuite:
    return TestSuite.model_validate(stored.document)


# --- reading runs ------------------------------------------------------------------------------


def get_test_run(
    conn: Connection, workspace_id: uuid.UUID | None, test_run_id: uuid.UUID
) -> TestRun | None:
    query = sa.select(test_runs).where(test_runs.c.id == test_run_id)
    if workspace_id is not None:
        query = query.where(test_runs.c.workspace_id == workspace_id)
    row = conn.execute(query).first()
    return _run(row) if row is not None else None


def list_test_runs(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    after: tuple[datetime, uuid.UUID] | None = None,
    suite_name: str | None = None,
    status: str | None = None,
) -> list[TestRun]:
    """Newest first; one suite's, or one status's."""
    query = sa.select(test_runs).where(test_runs.c.workspace_id == workspace_id)
    if suite_name is not None:
        query = query.where(test_runs.c.suite_name == suite_name)
    if status is not None:
        query = query.where(test_runs.c.status == status)
    if after is not None:
        query = query.where(sa.tuple_(test_runs.c.created_at, test_runs.c.id) < after)
    rows = conn.execute(
        query.order_by(test_runs.c.created_at.desc(), test_runs.c.id.desc()).limit(limit)
    )
    return [_run(row) for row in rows]


def check_results(conn: Connection, test_run_id: uuid.UUID) -> list[CheckRow]:
    rows = conn.execute(
        sa.select(test_check_results)
        .where(test_check_results.c.test_run_id == test_run_id)
        .order_by(test_check_results.c.ordinal)
    )
    return [_check(row) for row in rows]


def check_history(
    conn: Connection, workspace_id: uuid.UUID, suite_name: str, *, limit: int
) -> list[dict[str, Any]]:
    """The suite's last ``limit`` finished runs, oldest first, with each check's result and
    value: what a trend is drawn from."""
    recent = (
        sa.select(test_runs.c.id, test_runs.c.created_at, test_runs.c.status)
        .where(
            test_runs.c.workspace_id == workspace_id,
            test_runs.c.suite_name == suite_name,
            test_runs.c.status.in_(("passed", "failed")),
        )
        .order_by(test_runs.c.created_at.desc(), test_runs.c.id.desc())
        .limit(limit)
        .subquery()
    )
    rows = conn.execute(
        sa.select(
            recent.c.id,
            recent.c.created_at,
            recent.c.status,
            test_check_results.c.check_id,
            test_check_results.c.result,
            test_check_results.c.value,
        )
        .join(test_check_results, test_check_results.c.test_run_id == recent.c.id)
        .order_by(recent.c.created_at, recent.c.id, test_check_results.c.ordinal)
    )
    found: dict[uuid.UUID, dict[str, Any]] = {}
    for row in rows:
        run = found.setdefault(
            row.id,
            {
                "test_run_id": str(row.id),
                "created_at": row.created_at,
                "status": row.status,
                "checks": {},
            },
        )
        run["checks"][row.check_id] = {"result": row.result, "value": row.value}
    return list(found.values())


def _notify(conn: Connection, run: TestRun) -> None:
    event: dict[str, Any] = {
        "type": "test_run",
        "workspace_id": str(run.workspace_id),
        "run_id": str(run.id),
        "status": run.status,
        "counts": dict(run.summary or {}),
    }
    conn.execute(
        sa.text("SELECT pg_notify(:channel, :payload)"),
        {"channel": EVENTS_CHANNEL, "payload": json.dumps(event, separators=(",", ":"))},
    )


def _update(conn: Connection, run: TestRun, **values: Any) -> TestRun:
    row = conn.execute(
        sa.update(test_runs)
        .where(test_runs.c.id == run.id)
        .values(updated_at=sa.func.now(), **values)
        .returning(*test_runs.columns)
    ).one()
    updated = _run(row)
    _notify(conn, updated)
    return updated


# --- asking for a run ------------------------------------------------------------------------


def _lock_suite(conn: Connection, workspace_id: uuid.UUID, suite_name: str) -> None:
    conn.execute(
        sa.text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"stratafi:test_runs:{workspace_id}:{suite_name}"},
    )


def request_run(
    conn: Connection,
    queue: JobQueue,
    *,
    workspace_id: uuid.UUID,
    suite: specs.StoredDocument,
    trigger: str,
    created_by: str,
    max_attempts: int = 3,
) -> tuple[TestRun, bool]:
    """A run of ``suite``, queued; or the suite's waiting run, which this request joins.
    Returns the run and whether the request joined one (coalesced)."""
    _lock_suite(conn, workspace_id, suite.name)
    waiting = conn.execute(
        sa.select(test_runs)
        .where(
            test_runs.c.workspace_id == workspace_id,
            test_runs.c.suite_name == suite.name,
            test_runs.c.status == "queued",
        )
        .with_for_update()
    ).first()
    if waiting is not None:
        run = _run(waiting)
        values: dict[str, Any] = {}
        if trigger not in run.triggers:
            values["triggers"] = [*run.triggers, trigger]
        if suite.version > run.suite_version:
            values.update(suite_id=suite.id, suite_version=suite.version)
        if values:
            run = _update(conn, run, **values)
        return run, True
    row = conn.execute(
        sa.insert(test_runs)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            suite_id=suite.id,
            suite_name=suite.name,
            suite_version=suite.version,
            trigger=trigger,
            triggers=[trigger],
            created_by=created_by,
        )
        .returning(*test_runs.columns)
    ).one()
    run = _run(row)
    _enqueue(conn, queue, run, START_JOB, max_attempts=max_attempts)
    _notify(conn, run)
    return run, False


def _enqueue(
    conn: Connection, queue: JobQueue, run: TestRun, kind: str, *, max_attempts: int = 3
) -> None:
    queue.enqueue_in(
        conn,
        workspace_id=run.workspace_id,
        kind=kind,
        payload={"test_run_id": str(run.id)},
        resource_class="io",
        priority=5,
        max_attempts=max_attempts,
    )


# --- resolving references ----------------------------------------------------------------------


def _profile_version(
    conn: Connection, workspace_id: uuid.UUID, ref: str
) -> tuple[Any | None, dict[str, Any]]:
    """A profile reference's row: ``@current`` is the latest approved version, ``@previous`` the
    approved one before it, ``@latest`` the newest, ``@N`` that one."""
    name, version = parse_profile_ref(ref)
    table = knowledge_profiles
    query = sa.select(table).where(table.c.workspace_id == workspace_id, table.c.name == name)
    described = {"name": name, "selector": str(version)}
    if isinstance(version, int):
        return conn.execute(query.where(table.c.version == version)).first(), described
    if version == "latest":
        return conn.execute(query.order_by(table.c.version.desc()).limit(1)).first(), described
    approved = conn.execute(
        query.where(table.c.status == "approved").order_by(table.c.version.desc()).limit(2)
    ).all()
    index = 0 if version == "current" else 1
    return (approved[index] if len(approved) > index else None), described


def _save_document(
    conn: Connection,
    kind: specs.DocumentKind,
    workspace_id: uuid.UUID,
    resolved: PipelineSpec | RetrievalConfig,
    created_by: str,
) -> specs.StoredDocument:
    stored, _ = specs.save_version(
        conn,
        kind,
        workspace_id=workspace_id,
        document=dump_document(resolved),
        spec_hash=spec_hash(resolved),
        document_hash=document_hash(resolved),
        created_by=created_by,
    )
    return stored


def _resolve_configuration(
    conn: Connection,
    workspace_id: uuid.UUID,
    registry: ComponentRegistry,
    *,
    profile: str | None,
    pipeline: str | None,
    retrieval: str | None,
    created_by: str,
    missing: str,
) -> dict[str, Any] | None:
    """A profile's, or a pipeline's and retrieval config's, saved versions and hashes. None when
    a profile reference finds no version (a baseline that doesn't exist yet)."""
    out: dict[str, Any] = {}
    documents: dict[str, specs.StoredDocument | None] = {}
    if profile is not None:
        row, described = _profile_version(conn, workspace_id, profile)
        if row is None:
            return None
        try:
            model = KnowledgeProfile.model_validate(row.document)
            pipeline_spec = resolve_pipeline(model.pipeline, registry)
            retrieval_config = resolve_retrieval(model.retrieval, registry)
        except (SpecInvalid, ValueError):
            raise ResolveError(f"{missing}_invalid") from None
        documents["pipeline"] = _save_document(
            conn, "pipeline", workspace_id, pipeline_spec, created_by
        )
        documents["retrieval"] = _save_document(
            conn, "retrieval", workspace_id, retrieval_config, created_by
        )
        out["kind"] = "profile"
        out["profile"] = {
            "id": str(row.id),
            **described,
            "version": row.version,
            "status": row.status,
            "level": row.level,
        }
    else:
        out["kind"] = "pipeline"
        for kind, ref in (("pipeline", pipeline), ("retrieval", retrieval)):
            if ref is None:
                documents[kind] = None
                continue
            name, version = parse_ref(ref)
            stored = specs.get_version(conn, kind, workspace_id, name, version)  # type: ignore[arg-type]
            if stored is None:
                raise ResolveError(f"{missing}_{kind}_not_found")
            try:
                if kind == "pipeline":
                    resolve_pipeline(stored.document, registry)
                else:
                    resolve_retrieval(stored.document, registry)
            except SpecInvalid:
                raise ResolveError(f"{missing}_{kind}_invalid") from None
            documents[kind] = stored
    for kind, stored in documents.items():
        out[kind] = (
            {
                "id": str(stored.id),
                "name": stored.name,
                "version": stored.version,
                "hash": stored.spec_hash,
            }
            if stored is not None
            else None
        )
    return out


def _latest_frozen_eval_set(conn: Connection, workspace_id: uuid.UUID, eval_set_id: uuid.UUID):
    row = conn.execute(
        sa.select(eval_set_versions.c.id)
        .where(
            eval_set_versions.c.workspace_id == workspace_id,
            eval_set_versions.c.eval_set_id == eval_set_id,
            eval_set_versions.c.frozen.is_(True),
        )
        .order_by(eval_set_versions.c.version.desc())
        .limit(1)
    ).first()
    if row is None:
        return None
    return evalsets.get_version(conn, workspace_id, version_id=row.id)


def resolve(
    conn: Connection, run: TestRun, suite: TestSuite, registry: ComponentRegistry
) -> dict[str, Any]:
    """The suite's references as fixed IDs (the module docstring says how). Raises
    ``ResolveError``."""
    workspace_id = run.workspace_id
    target = _resolve_configuration(
        conn,
        workspace_id,
        registry,
        profile=suite.target.profile,
        pipeline=suite.target.pipeline,
        retrieval=suite.target.retrieval,
        created_by=run.created_by,
        missing="target",
    )
    if target is None:
        raise ResolveError("target_profile_not_found")
    corpus_name, corpus_number = parse_ref(suite.corpus)
    corpus = corpora.get_corpus_by_name(conn, workspace_id, corpus_name)
    if corpus is None:
        raise ResolveError("corpus_not_found")
    corpus_version = (
        corpora.get_version(conn, workspace_id, corpus_id=corpus.id, version=corpus_number)
        if corpus_number is not None
        else corpora.latest_version(conn, workspace_id, corpus.id)
    )
    if corpus_version is None:
        raise ResolveError("corpus_version_not_found")
    resolved: dict[str, Any] = {
        "target": target,
        "corpus": {
            "id": str(corpus.id),
            "name": corpus.name,
            "version": corpus_version.version,
            "version_id": str(corpus_version.id),
            "documents": corpus_version.document_count,
        },
        "eval_set": None,
        "baseline": None,
        "needs": {
            "pipeline_run": suite.needs_pipeline_run,
            "eval_run": suite.needs_eval_run,
            "baseline_eval_run": any(isinstance(c, RegressionCheck) for c in suite.checks),
        },
    }
    if suite.eval_set is not None:
        eval_set = evalsets.get_set_by_name(conn, workspace_id, suite.eval_set.name)
        if eval_set is None:
            raise ResolveError("eval_set_not_found")
        number = suite.eval_set.number
        version = (
            evalsets.get_version(conn, workspace_id, eval_set_id=eval_set.id, version=number)
            if number is not None
            else _latest_frozen_eval_set(conn, workspace_id, eval_set.id)
        )
        if version is None:
            raise ResolveError("eval_set_version_not_found")
        if not version.frozen:
            raise ResolveError("eval_set_not_frozen")
        items = int(version.split_counts.get(suite.eval_set.split, 0))
        if not items:
            raise ResolveError("eval_set_split_empty")
        resolved["eval_set"] = {
            "id": str(eval_set.id),
            "name": eval_set.name,
            "version": version.version,
            "version_id": str(version.id),
            "split": suite.eval_set.split,
            "items": items,
        }
    if suite.needs_baseline and suite.baseline_ref is not None:
        resolved["baseline"] = _resolve_baseline(conn, run, suite.baseline_ref, registry)
    return resolved


def _resolve_baseline(
    conn: Connection, run: TestRun, ref: BaselineRef, registry: ComponentRegistry
) -> dict[str, Any]:
    workspace_id = run.workspace_id
    if ref.eval_run is not None:
        pinned = evalruns.get_eval_run(conn, workspace_id, ref.eval_run)
        if pinned is None:
            raise ResolveError("baseline_eval_run_not_found")
        if pinned.status != "succeeded":
            raise ResolveError("baseline_eval_run_not_succeeded")
        return {
            "kind": "eval_run",
            "found": True,
            "eval_run_id": str(pinned.id),
            "pipeline_run_id": str(pinned.pipeline_run_id),
        }
    if ref.last_passing:
        row = conn.execute(
            sa.select(test_runs)
            .where(
                test_runs.c.workspace_id == workspace_id,
                test_runs.c.suite_name == run.suite_name,
                test_runs.c.status == "passed",
                test_runs.c.id != run.id,
            )
            .order_by(test_runs.c.finished_at.desc(), test_runs.c.id.desc())
            .limit(1)
        ).first()
        if row is None:
            return {"kind": "last_passing", "found": False}
        return {
            "kind": "last_passing",
            "found": True,
            "test_run_id": str(row.id),
            "pipeline_run_id": str(row.pipeline_run_id) if row.pipeline_run_id else None,
            "eval_run_id": str(row.eval_run_id) if row.eval_run_id else None,
        }
    configuration = _resolve_configuration(
        conn,
        workspace_id,
        registry,
        profile=ref.profile,
        pipeline=ref.pipeline,
        retrieval=ref.retrieval,
        created_by=run.created_by,
        missing="baseline",
    )
    if configuration is None:
        return {"kind": "profile", "found": False, "profile": {"reference": ref.profile}}
    return {**configuration, "found": True}


# --- the orchestrator ----------------------------------------------------------------------------


def _indexed(conn: Connection, run: runs.Run) -> bool:
    return any(
        task.status in runs.DONE_TASK_STATUSES
        for task in runs.tasks_of(conn, run.id, stage="index")
    )


def _uuid(value: Any) -> uuid.UUID | None:
    return uuid.UUID(str(value)) if value else None


Cancel = tuple[str, uuid.UUID, uuid.UUID]  # (run or eval_run, workspace, ID)


class TestOrchestrator:
    __test__ = False

    def __init__(
        self,
        engine: Engine,
        queue: JobQueue,
        registry: ComponentRegistry,
        *,
        max_attempts: int = 3,
        webhook_attempts: int = 5,
    ) -> None:
        self.engine = engine
        self.queue = queue
        self.registry = registry
        self.max_attempts = max_attempts
        self.webhook_attempts = webhook_attempts

    def _lock(self, conn: Connection, test_run_id: uuid.UUID) -> TestRun | None:
        row = conn.execute(
            sa.select(test_runs).where(test_runs.c.id == test_run_id).with_for_update()
        ).first()
        return _run(row) if row is not None else None

    def _active(self, conn: Connection, run: TestRun) -> bool:
        return (
            conn.execute(
                sa.select(test_runs.c.id).where(
                    test_runs.c.workspace_id == run.workspace_id,
                    test_runs.c.suite_name == run.suite_name,
                    test_runs.c.status.in_(ACTIVE_TEST_RUN_STATUSES),
                    test_runs.c.id != run.id,
                )
            ).first()
            is not None
        )

    def _kick_next(self, conn: Connection, run: TestRun) -> None:
        """Queue the start of the suite's waiting run, now that ``run`` has ended."""
        waiting = conn.execute(
            sa.select(test_runs).where(
                test_runs.c.workspace_id == run.workspace_id,
                test_runs.c.suite_name == run.suite_name,
                test_runs.c.status == "queued",
            )
        ).first()
        if waiting is not None:
            _enqueue(conn, self.queue, _run(waiting), START_JOB, max_attempts=self.max_attempts)

    def _start_run(
        self, conn: Connection, run: TestRun, configuration: dict[str, Any]
    ) -> uuid.UUID:
        pipeline = configuration["pipeline"]
        created = runs.create_run(
            conn,
            workspace_id=run.workspace_id,
            pipeline_spec_id=uuid.UUID(pipeline["id"]),
            corpus_version_id=uuid.UUID(run.resolved["corpus"]["version_id"]),
            spec_hash=pipeline["hash"],
            created_by=run.created_by,
        )
        self.queue.enqueue_in(
            conn,
            workspace_id=run.workspace_id,
            kind=runs.START_JOB,
            payload={"run_id": str(created.id)},
            resource_class="io",
            max_attempts=self.max_attempts,
        )
        return created.id

    def _start_eval(
        self,
        conn: Connection,
        run: TestRun,
        configuration: dict[str, Any],
        pipeline_run_id: uuid.UUID,
    ) -> uuid.UUID:
        retrieval = configuration["retrieval"]
        created = evalruns.create_eval_run(
            conn,
            workspace_id=run.workspace_id,
            pipeline_run_id=pipeline_run_id,
            retrieval_config_id=uuid.UUID(retrieval["id"]),
            eval_set_version_id=uuid.UUID(run.resolved["eval_set"]["version_id"]),
            split=run.resolved["eval_set"]["split"],
            options=EvalOptions().dump(),
            retrieval_hash=retrieval["hash"],
            test_read=False,
            created_by=run.created_by,
        )
        self.queue.enqueue_in(
            conn,
            workspace_id=run.workspace_id,
            kind=evalruns.START_JOB,
            payload={"eval_run_id": str(created.id)},
            resource_class="io",
            priority=5,
            max_attempts=self.max_attempts,
        )
        return created.id

    def start(self, test_run_id: uuid.UUID) -> TestRun | None:
        """Resolve a queued run and start the runs it needs; it waits while another run of its
        suite is active."""
        with self.engine.begin() as conn:
            run = self._lock(conn, test_run_id)
            if run is None or run.status != "queued":
                return run
            _lock_suite(conn, run.workspace_id, run.suite_name)
            if self._active(conn, run):
                return run
            stored = get_suite_by_id(conn, run.workspace_id, run.suite_id)
            if stored is None:
                return self._end(conn, run, "error", error_code="suite_not_found")
            suite = suite_of(stored)
            try:
                resolved = resolve(conn, run, suite, self.registry)
            except ResolveError as error:
                return self._end(conn, run, "error", error_code=error.code)
            run = _update(conn, run, status="running", resolved=resolved, started_at=sa.func.now())
            values: dict[str, Any] = {}
            needs = resolved["needs"]
            if needs["pipeline_run"]:
                values["pipeline_run_id"] = self._start_run(conn, run, resolved["target"])
            baseline = resolved.get("baseline") or {}
            if baseline.get("found"):
                if baseline["kind"] in ("eval_run", "last_passing"):
                    values["baseline_pipeline_run_id"] = _uuid(baseline.get("pipeline_run_id"))
                    values["baseline_eval_run_id"] = _uuid(baseline.get("eval_run_id"))
                    values["baseline_test_run_id"] = _uuid(baseline.get("test_run_id"))
                elif baseline["pipeline"]["hash"] == resolved["target"]["pipeline"]["hash"]:
                    values["baseline_pipeline_run_id"] = values.get("pipeline_run_id")
                else:
                    values["baseline_pipeline_run_id"] = self._start_run(conn, run, baseline)
            if values:
                run = _update(conn, run, **values)
            return self._advance(conn, run)

    def advance(self, test_run_id: uuid.UUID) -> TestRun | None:
        with self.engine.begin() as conn:
            run = self._lock(conn, test_run_id)
            if run is None or run.status != "running":
                return run
            return self._advance(conn, run)

    def _advance(self, conn: Connection, run: TestRun) -> TestRun:
        """Start the eval runs whose pipeline run is ready; when nothing is left to wait for,
        queue the evaluation."""
        pipelines = runs.runs_by_id(
            conn,
            {r for r in (run.pipeline_run_id, run.baseline_pipeline_run_id) if r is not None},
        )
        needs = run.resolved.get("needs") or {}
        target = run.resolved["target"]
        baseline = run.resolved.get("baseline") or {}
        values: dict[str, Any] = {}

        def ready(run_id: uuid.UUID | None) -> uuid.UUID | None:
            pipeline = pipelines.get(run_id) if run_id is not None else None
            if pipeline is None or pipeline.status != "succeeded" or not _indexed(conn, pipeline):
                return None
            return pipeline.id

        if needs.get("eval_run") and run.eval_run_id is None:
            pipeline_id = ready(run.pipeline_run_id)
            if pipeline_id is not None:
                values["eval_run_id"] = self._start_eval(conn, run, target, pipeline_id)
        wants_baseline_eval = (
            needs.get("baseline_eval_run")
            and baseline.get("found")
            and baseline.get("kind") in ("profile", "pipeline")
            and run.baseline_eval_run_id is None
        )
        if wants_baseline_eval:
            pipeline_id = ready(run.baseline_pipeline_run_id)
            same = (
                pipeline_id is not None
                and pipeline_id == run.pipeline_run_id
                and baseline["retrieval"]["hash"] == target["retrieval"]["hash"]
            )
            eval_id = values.get("eval_run_id") or run.eval_run_id
            if same and eval_id is not None:
                values["baseline_eval_run_id"] = eval_id
            elif pipeline_id is not None and not same:
                values["baseline_eval_run_id"] = self._start_eval(conn, run, baseline, pipeline_id)
        if values:
            run = _update(conn, run, **values)
        if self._waiting(conn, run, pipelines):
            return run
        run = _update(conn, run, status="evaluating")
        _enqueue(conn, self.queue, run, EVALUATE_JOB, max_attempts=self.max_attempts)
        return run

    def _waiting(
        self, conn: Connection, run: TestRun, pipelines: Mapping[uuid.UUID, runs.Run]
    ) -> bool:
        """Whether a run this test run needs hasn't finished, or an eval run is still to start."""
        if any(not pipeline.finished for pipeline in pipelines.values()):
            return True
        evals = evalruns.eval_runs_by_id(
            conn, {r for r in (run.eval_run_id, run.baseline_eval_run_id) if r is not None}
        )
        if any(not evaluation.finished for evaluation in evals.values()):
            return True
        needs = run.resolved.get("needs") or {}
        baseline = run.resolved.get("baseline") or {}

        def could_start(pipeline_run_id: uuid.UUID | None) -> bool:
            pipeline = pipelines.get(pipeline_run_id) if pipeline_run_id else None
            return (
                pipeline is not None and pipeline.status == "succeeded" and _indexed(conn, pipeline)
            )

        if needs.get("eval_run") and run.eval_run_id is None and could_start(run.pipeline_run_id):
            return True
        return bool(
            needs.get("baseline_eval_run")
            and baseline.get("found")
            and baseline.get("kind") in ("profile", "pipeline")
            and run.baseline_eval_run_id is None
            and could_start(run.baseline_pipeline_run_id)
        )

    def begin_evaluation(self, test_run_id: uuid.UUID) -> TestRun | None:
        """The run, if its checks are due for evaluation."""
        with self.engine.connect() as conn:
            run = get_test_run(conn, None, test_run_id)
        return run if run is not None and run.status == "evaluating" else None

    def finish(
        self,
        test_run_id: uuid.UUID,
        results: Sequence[CheckResult],
        record: TestRunRecord,
    ) -> TestRun | None:
        """Record the checks' results and the run's record; end it passed or failed."""
        with self.engine.begin() as conn:
            run = self._lock(conn, test_run_id)
            if run is None or run.status != "evaluating":
                return run
            if results:
                conn.execute(
                    sa.insert(test_check_results),
                    [
                        {
                            "test_run_id": run.id,
                            "workspace_id": run.workspace_id,
                            "check_id": result.id,
                            "ordinal": ordinal,
                            "type": result.type,
                            "result": result.result,
                            "code": result.code,
                            "value": result.value,
                            "baseline_value": result.baseline_value,
                            "threshold": result.threshold,
                            "detail": result.detail,
                            "evidence": [
                                link.model_dump(mode="json", exclude_none=True)
                                for link in result.evidence
                            ],
                            "duration_ms": result.duration_ms,
                        }
                        for ordinal, result in enumerate(results)
                    ],
                )
            summary = Summary.of(results)
            document = record.model_dump(mode="json", by_alias=True)
            return self._end(
                conn,
                run,
                summary.outcome,
                summary=summary.model_dump(mode="json"),
                record=document,
                record_hash=record.fingerprint(),
            )

    def fail(self, test_run_id: uuid.UUID, code: str) -> TestRun | None:
        with self.engine.begin() as conn:
            run = self._lock(conn, test_run_id)
            if run is None or run.finished:
                return run
            return self._end(conn, run, "error", error_code=code)

    def _end(self, conn: Connection, run: TestRun, status: str, **values: Any) -> TestRun:
        ended = _update(conn, run, status=status, finished_at=sa.func.now(), **values)
        self._kick_next(conn, ended)
        webhooks.queue_deliveries(conn, self.queue, ended, max_attempts=self.webhook_attempts)
        return ended

    def cancel(
        self, test_run_id: uuid.UUID, workspace_id: uuid.UUID
    ) -> tuple[TestRun | None, list[Cancel]]:
        """Cancel a run: at once, with the pipeline and eval runs it started that haven't
        finished (not a pinned baseline's or another suite's)."""
        with self.engine.begin() as conn:
            run = self._lock(conn, test_run_id)
            if run is None or run.workspace_id != workspace_id:
                return None, []
            if run.finished:
                return run, []
            created_after = run.started_at or run.created_at
            cancels: list[Cancel] = []
            pipelines = runs.runs_by_id(
                conn,
                {r for r in (run.pipeline_run_id, run.baseline_pipeline_run_id) if r is not None},
            )
            evals = evalruns.eval_runs_by_id(
                conn, {r for r in (run.eval_run_id, run.baseline_eval_run_id) if r is not None}
            )
            cancels += [
                ("run", p.workspace_id, p.id)
                for p in pipelines.values()
                if not p.finished and p.created_at >= created_after
            ]
            cancels += [
                ("eval_run", e.workspace_id, e.id)
                for e in evals.values()
                if not e.finished and e.created_at >= created_after
            ]
            return self._end(conn, run, "cancelled", error_code="cancelled"), cancels

    # --- the sweep -------------------------------------------------------------------------------

    def reconcile(self) -> int:
        """Queue what runs missed: a waiting run's start once its suite is free, a running
        one's advance once its runs ended, an evaluating one's evaluation. Returns how many."""
        finished = ("succeeded", "failed", "cancelled")

        def no_job(kind: str) -> Any:
            return ~sa.exists().where(
                jobs.c.kind == kind,
                jobs.c.status.in_(("queued", "running")),
                jobs.c.payload["test_run_id"].astext == sa.cast(test_runs.c.id, sa.Text),
            )

        other = test_runs.alias("other")
        free = ~sa.exists().where(
            other.c.workspace_id == test_runs.c.workspace_id,
            other.c.suite_name == test_runs.c.suite_name,
            other.c.status.in_(ACTIVE_TEST_RUN_STATUSES),
        )
        ended = sa.and_(
            *[
                sa.or_(
                    test_runs.c[column].is_(None),
                    sa.exists().where(
                        (runs_table.c.id if "eval" not in column else eval_runs.c.id)
                        == test_runs.c[column],
                        (runs_table.c.status if "eval" not in column else eval_runs.c.status).in_(
                            finished
                        ),
                    ),
                )
                for column in RUN_COLUMNS
            ]
        )
        wanted = [
            (START_JOB, sa.and_(test_runs.c.status == "queued", free, no_job(START_JOB))),
            (ADVANCE_JOB, sa.and_(test_runs.c.status == "running", ended, no_job(ADVANCE_JOB))),
            (EVALUATE_JOB, sa.and_(test_runs.c.status == "evaluating", no_job(EVALUATE_JOB))),
        ]
        queued = 0
        with self.engine.connect() as conn:
            found = [
                (kind, row)
                for kind, condition in wanted
                for row in conn.execute(
                    sa.select(test_runs.c.id, test_runs.c.workspace_id).where(condition)
                )
            ]
        for kind, row in found:
            self.queue.enqueue(
                workspace_id=row.workspace_id,
                kind=kind,
                payload={"test_run_id": str(row.id)},
                resource_class="io",
                priority=5,
                dedupe_key=f"{kind}:{row.id}",
                max_attempts=self.max_attempts,
            )
            queued += 1
        return queued


def finish_hook(queue: JobQueue, *, max_attempts: int = 3) -> FinishHook:
    """The run and eval-run orchestrators' ``on_finish``: queue a ``test.advance`` for each
    running test run that waits on the run."""

    def on_finish(conn: Connection, kind: str, workspace_id: uuid.UUID, run_id: uuid.UUID) -> None:
        columns = (
            (test_runs.c.pipeline_run_id, test_runs.c.baseline_pipeline_run_id)
            if kind == "run"
            else (test_runs.c.eval_run_id, test_runs.c.baseline_eval_run_id)
        )
        waiting = conn.execute(
            sa.select(test_runs.c.id)
            .where(sa.or_(*(column == run_id for column in columns)))
            .where(test_runs.c.status == "running")
        ).scalars()
        for test_run_id in waiting:
            queue.enqueue_in(
                conn,
                workspace_id=workspace_id,
                kind=ADVANCE_JOB,
                payload={"test_run_id": str(test_run_id)},
                resource_class="io",
                priority=5,
                max_attempts=max_attempts,
            )

    return on_finish


def seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round(max(0.0, (end - start).total_seconds()), 3)


def now() -> datetime:
    return datetime.now(UTC)
