"""The worker's test-run jobs: ``test.start`` resolves a run and starts the runs it
needs, ``test.advance`` moves it on as they finish, and ``test.evaluate`` judges its checks
(``stratafi_store.testruns.TestOrchestrator`` holds the transitions).

``test.evaluate`` gathers what the checks read, for the target and for the baseline:

- the pipeline run: its record, each corpus entry with its detected type, scan outcome and
  parse status, and each stage task with its artifact's summary
- the eval run: its aggregates and record, and each query's scores
- for ``no_findings`` checks: the stage's output text, scanned in process by the PII detector
  (``stratafi_core.pii``); only counts by category and the IDs of the chunks with findings
  are kept
- for golden queries: the eval run's retrieval engine (its config, indexes, models and as-of
  date), searched as the check's asker; an asker given by name only is the eval set's
- for custom checks: the run's chunks, documents and query scores, written once as a
  bundle that each check's sandbox reads (``stratafi_worker.sandbox``), and deleted afterwards

then evaluates each check (``stratafi_core.suites``) and records the results and the run's
``stratafi.test-run.v1``. A check that can't run is an ``error`` with a code; it never stops
the others. Golden queries, chunks and documents never reach a log, an event or a result.

A custom check runs only where the deployment's sandbox is configured and its security review
signed off, and the workspace's owner turned ``custom_checks`` on; otherwise it's an error,
``custom_checks_disabled``, whose ``reason`` says which is missing.
"""

import logging
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from datetime import date
from typing import Any, cast

from sqlalchemy.engine import Connection, Engine

from stratafi_components.embed import EmbedComponent
from stratafi_components.retrieval import RetrievalCache, section_of
from stratafi_core.artifacts import Chunk, ParsedDocument
from stratafi_core.components import ComponentError, ComponentRegistry, StageContext
from stratafi_core.evalsets import Asker
from stratafi_core.pii import DEFAULT_CATEGORIES, count_by_category, find_pii
from stratafi_core.settings import Settings
from stratafi_core.suites import (
    AnyCheck,
    CheckResult,
    CustomCheck,
    DataChangeCheck,
    EntryState,
    EvidenceLink,
    GoldenCheck,
    GoldenPassage,
    MetricCheck,
    QueryRow,
    RegressionCheck,
    RevisionGuardCheck,
    RunData,
    StageCheck,
    SuiteRef,
    Summary,
    TaskState,
    TestRunInfo,
    TestRunRecord,
    TestSuite,
    code_manifest,
    custom_result,
    data_change,
    errored,
    evaluate_data_change,
    evaluate_metric,
    evaluate_regression,
    evaluate_revision,
    evaluate_stage,
    judge_golden,
)
from stratafi_store import evalruns, evalsets, features, runs, testruns
from stratafi_store.artifacts import ArtifactNotFound, ArtifactStore, artifact_key
from stratafi_store.jobs import JobQueue
from stratafi_store.testruns import (
    ADVANCE_JOB,
    EVALUATE_JOB,
    START_JOB,
    TestOrchestrator,
    TestRun,
    seconds_between,
)
from stratafi_worker.evals import IndexCache, _engine_for, _metas
from stratafi_worker.runs import ServicesFor
from stratafi_worker.sandbox import Bundle, Sandbox, SandboxError, sandbox_from_settings
from stratafi_worker.worker import HandlerRegistry, JobContext

logger = logging.getLogger("stratafi.worker.testruns")

TEXT_STAGES = {"parse": "parsed", "clean": "cleaned", "chunk": "chunks", "enrich": "chunks"}


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 3)


def _asker(conn: Connection, evaluation: Any, check: GoldenCheck) -> Asker:
    """A golden check's asker. One given by name alone (no roles or groups) is that name in the
    eval run's eval set's ``askers`` table, with the check's user if it gives one; otherwise,
    or for a name the table doesn't have, it's what the check says."""
    spec = check.asker_spec
    if spec.name is not None and not (spec.roles or spec.groups):
        version = evalsets.get_version(
            conn, evaluation.workspace_id, version_id=evaluation.eval_set_version_id
        )
        named = (version.document.get("askers") or {}).get(spec.name) if version else None
        if isinstance(named, dict):
            asker = Asker.model_validate({**named, "name": spec.name})
            return asker.model_copy(update={"user": spec.user}) if spec.user else asker
    return Asker(name=spec.name, user=spec.user, roles=spec.roles, groups=spec.groups)


class _CustomChecks:
    """A test run's custom checks: whether they may run here, and the bundle they read, made
    when the first one runs."""

    def __init__(
        self,
        evaluator: "TestEvaluator",
        conn: Connection,
        run: TestRun,
        suite: TestSuite,
        stored: Any,
        target: RunData,
    ) -> None:
        self.evaluator = evaluator
        self.conn = conn
        self.run = run
        self.suite = suite
        self.stored = stored
        self.target = target
        self.bundle: Bundle | None = None
        self._checked = False
        self._reason: str | None = None

    def blocked(self) -> str | None:
        if not self._checked:
            self._reason = self.evaluator.custom_checks_blocked(self.conn, self.run.workspace_id)
            self._checked = True
        return self._reason

    def _bundle(self) -> Bundle:
        if self.bundle is None:
            bundle = Bundle(self.evaluator.settings.suites.sandbox.workdir)
            try:
                bundle.write_code(self.suite.code)
                counts = bundle.write_data(
                    chunks=self.evaluator.chunks_of(self.conn, self.run),
                    documents=self.target.entries,
                    queries=self.target.queries,
                )
            except BaseException:
                bundle.close()
                raise
            logger.info(
                "custom-check bundle written",
                extra={
                    "test_run_id": str(self.run.id),
                    **{f"{name}_count": count for name, count in counts.items()},
                },
            )
            self.bundle = bundle
        return self.bundle

    def judge(self, check: CustomCheck) -> CheckResult:
        reason = self.blocked()
        if reason is not None:
            return errored(check, "custom_checks_disabled", reason=reason)
        sandbox = self.evaluator.sandbox
        assert sandbox is not None  # noqa: S101 - custom_checks_blocked said it's there
        run = self.run
        evidence = [
            EvidenceLink(kind=kind, id=str(identifier))
            for kind, identifier in (("run", run.pipeline_run_id), ("eval_run", run.eval_run_id))
            if identifier is not None
        ]
        try:
            bundle = self._bundle()
            directory = bundle.for_check(
                {
                    "check_id": check.id,
                    "suite": {"name": self.stored.name, "version": self.stored.version},
                    "runs": {
                        "test_run": str(run.id),
                        "pipeline_run": str(run.pipeline_run_id) if run.pipeline_run_id else None,
                        "eval_run": str(run.eval_run_id) if run.eval_run_id else None,
                    },
                    "metrics": dict(self.target.metrics or {}),
                }
            )
            payload = sandbox.run(
                directory, bundle.code, check.path, check.function, check.timeout_s
            )
        except SandboxError as error:
            return errored(check, error.code)
        return custom_result(check, payload, evidence=evidence)

    def close(self) -> None:
        if self.bundle is not None:
            self.bundle.close()


class TestEvaluator:
    """The ``test.evaluate`` job."""

    __test__ = False

    def __init__(
        self,
        engine: Engine,
        store: ArtifactStore,
        registry: ComponentRegistry,
        orchestrator: TestOrchestrator,
        services: ServicesFor,
        settings: Settings,
        indexes: IndexCache,
        cache: RetrievalCache,
        sandbox: Sandbox | None = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.registry = registry
        self.orchestrator = orchestrator
        self.services = services
        self.settings = settings
        self.indexes = indexes
        self.cache = cache
        self.sandbox = sandbox

    def __call__(self, job: JobContext) -> Mapping[str, Any]:
        test_run_id = uuid.UUID(str(job.payload["test_run_id"]))
        run = self.orchestrator.begin_evaluation(test_run_id)
        if run is None:
            return {"skipped": True}
        try:
            results, record = self._evaluate(run, job)
        except Exception as error:
            if job.job.attempts < job.job.max_attempts:
                raise
            logger.warning(
                "test run evaluation failed",
                extra={"test_run_id": str(test_run_id), "error": type(error).__name__},
            )
            self.orchestrator.fail(test_run_id, "evaluation_failed")
            return {"failed": "evaluation_failed"}
        finished = self.orchestrator.finish(test_run_id, results, record)
        return {"status": finished.status if finished else "missing", "checks": len(results)}

    # --- gathering ------------------------------------------------------------------------------

    def _run_data(
        self, conn: Connection, run: TestRun, pipeline_run_id: Any, eval_run_id: Any
    ) -> RunData:
        data = RunData()
        workspace_id = run.workspace_id
        if pipeline_run_id is not None:
            pipeline = runs.get_run(conn, workspace_id, pipeline_run_id)
            if pipeline is not None:
                data.run_id = str(pipeline.id)
                data.run_status = pipeline.status
                data.run_manifest = pipeline.manifest
                data.run_seconds = seconds_between(pipeline.started_at, pipeline.finished_at)
                data.embedding = self._embedding(conn, pipeline)
                data.entries, data.tasks = self._entries_and_tasks(conn, pipeline)
        if eval_run_id is not None:
            evaluation = evalruns.get_eval_run(conn, workspace_id, eval_run_id)
            if evaluation is not None:
                data.eval_run_id = str(evaluation.id)
                data.eval_status = evaluation.status
                data.eval_set_version_id = str(evaluation.eval_set_version_id)
                data.metrics = evaluation.metrics
                data.eval_manifest = evaluation.manifest
                data.eval_seconds = seconds_between(evaluation.started_at, evaluation.finished_at)
                data.queries = [
                    QueryRow(
                        item_id=row.item_id,
                        has_evidence=row.has_evidence,
                        tags=list(row.tags),
                        file_types=list(row.file_types),
                        score=row.score,
                    )
                    for row in evalruns.query_results(conn, evaluation.id)
                ]
        return data

    def _embedding(self, conn: Connection, pipeline: runs.Run) -> dict[str, Any] | None:
        try:
            spec = runs.run_spec(conn, pipeline)
            ref = spec.stages.embed
            embedder = cast(EmbedComponent, self.registry.get(ref.component)())
            model = embedder.model_name(embedder.resolve_params(ref.params))
        except Exception:  # an unknown embedder leaves the revision guard nothing to compare
            return None
        return {
            "component": ref.component,
            "model": model,
            "revision": (pipeline.model_env.get("embed") or {}).get("revision"),
        }

    def _entries_and_tasks(
        self, conn: Connection, pipeline: runs.Run
    ) -> tuple[list[EntryState], list[TaskState]]:
        entries = runs.entries_of(conn, pipeline.id)
        names = {entry.ordinal: entry.name for entry in entries}
        task_list = runs.tasks_of(conn, pipeline.id)
        found = runs.summaries(
            conn,
            pipeline.workspace_id,
            [task.cache_key for task in task_list if task.status in runs.DONE_TASK_STATUSES],
        )
        tasks = [
            TaskState(
                stage=task.stage,
                status=task.status,
                entry=names.get(task.ordinal),
                summary=found.get(task.cache_key, {}),
                metrics=task.metrics or {},
            )
            for task in task_list
        ]
        by_entry: dict[tuple[str, str], TaskState] = {}
        for task in tasks:
            if task.entry is not None:
                by_entry[(task.entry, task.stage)] = task
        states = []
        for entry in entries:
            extract = by_entry.get((entry.name, "extract"))
            scan = by_entry.get((entry.name, "scan"))
            parse = by_entry.get((entry.name, "parse"))
            states.append(
                EntryState(
                    name=entry.name,
                    sha256=entry.sha256,
                    document_id=str(entry.document_id),
                    media_type=(extract.summary.get("media_type") if extract else None),
                    state=entry.state,
                    stop_code=entry.stop_code,
                    scan_outcome=(scan.summary.get("outcome") if scan else None),
                    scan_decision=(scan.summary.get("decision") if scan else None),
                    parse_status=(parse.status if parse else None),
                )
            )
        return states, tasks

    # --- checks that need more than the records --------------------------------------------------

    def _read(self, workspace_id: uuid.UUID, cache_key: str) -> bytes:
        try:
            return self.store.get(artifact_key(workspace_id, cache_key))
        except ArtifactNotFound:
            raise ComponentError("artifact_missing") from None

    def _texts(
        self, conn: Connection, run: TestRun, pipeline_run_id: uuid.UUID, stage: str
    ) -> list[tuple[str, str]]:
        """The stage's output text, as (ID, text) pairs: each chunk, or each document's text.
        For ``index``, what the index holds: the embedded chunks' index text."""
        source = "enrich" if stage == "index" else stage
        texts: list[tuple[str, str]] = []
        for task in runs.tasks_of(conn, pipeline_run_id, stage=source):
            if task.status not in runs.DONE_TASK_STATUSES:
                continue
            data = self._read(run.workspace_id, task.cache_key)
            if TEXT_STAGES.get(source) == "chunks":
                for line in data.splitlines():
                    if line.strip():
                        chunk = Chunk.model_validate_json(line)
                        text = chunk.index_text if stage == "index" else chunk.text
                        texts.append((chunk.chunk_id, text))
            else:
                document = ParsedDocument.model_validate_json(data)
                texts.append((task.cache_key, document.text))
        return texts

    def _findings(
        self, conn: Connection, run: TestRun, check: StageCheck, pipeline_run_id: uuid.UUID | None
    ) -> dict[str, Any]:
        """A no_findings check's scan: each text of the stage's output through the PII
        detector, for the check's categories or, when it names none, the detector's defaults."""
        if pipeline_run_id is None:
            return {"error": "run_missing"}
        texts = self._texts(conn, run, pipeline_run_id, check.stage)
        categories = check.categories or DEFAULT_CATEGORIES
        by_category: dict[str, int] = {}
        flagged: list[str] = []
        for identifier, text in texts:
            counts = count_by_category(find_pii(text, categories))
            for category, count in counts.items():
                by_category[category] = by_category.get(category, 0) + count
            if counts and identifier not in flagged:
                flagged.append(identifier)
        return {
            "findings": sum(by_category.values()),
            "by_category": by_category,
            "scanned": len(texts),
            "chunks": flagged[:50],
        }

    def _golden(
        self,
        conn: Connection,
        run: TestRun,
        check: GoldenCheck,
        context: StageContext,
    ) -> CheckResult:
        if run.eval_run_id is None:
            return errored(check, "eval_run_missing")
        evaluation = evalruns.get_eval_run(conn, run.workspace_id, run.eval_run_id)
        if evaluation is None or evaluation.status != "succeeded":
            return errored(check, f"eval_run_{evaluation.status if evaluation else 'missing'}")
        corpus_version = uuid.UUID(evaluation.context["pipeline_run"]["corpus_version_id"])
        metas = _metas(conn, run.workspace_id, corpus_version)
        engine, chunks = _engine_for(
            evaluation,
            entries=metas,
            registry=self.registry,
            indexes=self.indexes,
            context=context,
            cache=self.cache,
        )
        asker = _asker(conn, evaluation, check)
        day = check.as_of or (
            date.fromisoformat(evaluation.context["day"]) if evaluation.context.get("day") else None
        )
        result = engine.search(check.query, asker=asker, day=day)
        by_document = {}
        for meta in sorted(metas, key=lambda entry: entry.name):
            by_document.setdefault(meta.document_id, meta)
        passages = []
        for chunk_id, _ in result.checkpoints.get("final", []):
            chunk = chunks[chunk_id]
            meta = by_document.get(chunk.document_id)
            passages.append(
                GoldenPassage(
                    chunk_id=chunk_id,
                    doc_key=meta.doc_key if meta else None,
                    version=meta.version if meta else None,
                    section=section_of(chunk),
                    text=chunk.text,
                )
            )
        return judge_golden(check, passages, run_id=str(run.pipeline_run_id))

    def custom_checks_blocked(self, conn: Connection, workspace_id: uuid.UUID) -> str | None:
        """Why the workspace's custom checks can't run here, or None: the deployment's sandbox
        (configured, its review signed off), then the workspace's ``custom_checks`` feature."""
        reason = self.settings.suites.custom_checks_blocked()
        if reason is None and self.sandbox is None:
            reason = "sandbox_not_configured"
        if reason is None and not features.enabled(conn, workspace_id, features.CUSTOM_CHECKS):
            reason = "feature_off"
        return reason

    def chunks_of(self, conn: Connection, run: TestRun) -> Iterator[Chunk]:
        """The pipeline run's final chunks: its enrich stage's output, else its chunk stage's."""
        if run.pipeline_run_id is None:
            return
        for stage in ("enrich", "chunk"):
            tasks = [
                task
                for task in runs.tasks_of(conn, run.pipeline_run_id, stage=stage)
                if task.status in runs.DONE_TASK_STATUSES
            ]
            if not tasks:
                continue
            for task in tasks:
                for line in self._read(run.workspace_id, task.cache_key).splitlines():
                    if line.strip():
                        yield Chunk.model_validate_json(line)
            return

    # --- evaluating -----------------------------------------------------------------------------

    def _check(
        self,
        conn: Connection,
        run: TestRun,
        check: AnyCheck,
        target: RunData,
        baseline: RunData | None,
        context: StageContext,
        custom: _CustomChecks,
    ) -> CheckResult:
        if isinstance(check, MetricCheck):
            return evaluate_metric(check, target)
        if isinstance(check, RegressionCheck):
            return evaluate_regression(check, target, baseline)
        if isinstance(check, StageCheck):
            if check.assert_ == "no_findings":
                target.findings = {
                    **target.findings,
                    check.id: self._findings(conn, run, check, run.pipeline_run_id),
                }
            return evaluate_stage(check, target)
        if isinstance(check, GoldenCheck):
            return self._golden(conn, run, check, context)
        if isinstance(check, DataChangeCheck):
            return evaluate_data_change(check, target, baseline)
        if isinstance(check, RevisionGuardCheck):
            return evaluate_revision(check, target, baseline)
        assert isinstance(check, CustomCheck)  # noqa: S101 - the union's last member
        return custom.judge(check)

    def _evaluate(self, run: TestRun, job: JobContext) -> tuple[list[CheckResult], TestRunRecord]:
        context = StageContext(
            workspace_id=str(run.workspace_id),
            services=self.services(run.workspace_id),
            cancelled=job.stop,
            rate_limit=lambda name, tokens: job.provider(name, tokens=tokens),
        )
        results: list[CheckResult] = []
        custom: _CustomChecks | None = None
        try:
            with self.engine.connect() as conn:
                stored = testruns.get_suite_by_id(conn, run.workspace_id, run.suite_id)
                if stored is None:
                    raise ComponentError("suite_not_found")
                suite = TestSuite.model_validate(stored.document)
                target = self._run_data(conn, run, run.pipeline_run_id, run.eval_run_id)
                baseline_found = bool((run.resolved.get("baseline") or {}).get("found"))
                baseline = (
                    self._run_data(
                        conn, run, run.baseline_pipeline_run_id, run.baseline_eval_run_id
                    )
                    if baseline_found
                    else None
                )
                custom = _CustomChecks(self, conn, run, suite, stored, target)
                for check in suite.checks:
                    job.checkpoint()
                    started = time.monotonic()
                    try:
                        result = self._check(conn, run, check, target, baseline, context, custom)
                    except ComponentError as error:
                        result = errored(check, error.code)
                    result.duration_ms = _ms(started)
                    results.append(result)
        finally:
            if custom is not None:
                custom.close()
            context.close()
        record = self._record(run, stored, results, target, baseline)
        record.code = code_manifest(suite)
        return results, record

    def _record(
        self,
        run: TestRun,
        stored: Any,
        results: Sequence[CheckResult],
        target: RunData,
        baseline: RunData | None,
    ) -> TestRunRecord:
        summary = Summary.of(results)
        finished = testruns.now()
        started = run.started_at or run.created_at
        change = (
            data_change(target.entries, baseline.entries)
            if baseline is not None and baseline.run_status and target.run_status
            else None
        )
        return TestRunRecord(
            test_run=TestRunInfo(
                id=str(run.id),
                suite=SuiteRef(
                    id=str(stored.id),
                    name=stored.name,
                    version=stored.version,
                    document_hash=stored.document_hash,
                ),
                trigger=run.trigger,  # type: ignore[arg-type]
                triggers=list(run.triggers),
                status=summary.outcome,
                created_by=run.created_by,
                created_at=run.created_at,
                started_at=run.started_at,
                finished_at=finished,
                duration_ms=round((finished - started).total_seconds() * 1000, 3),
            ),
            resolved=run.resolved,
            runs={
                "pipeline_run": str(run.pipeline_run_id) if run.pipeline_run_id else None,
                "eval_run": str(run.eval_run_id) if run.eval_run_id else None,
                "baseline_pipeline_run": (
                    str(run.baseline_pipeline_run_id) if run.baseline_pipeline_run_id else None
                ),
                "baseline_eval_run": (
                    str(run.baseline_eval_run_id) if run.baseline_eval_run_id else None
                ),
            },
            summary=summary,
            outcome=summary.outcome,
            checks=list(results),
            data_change=change,
        )


def register_test_handlers(
    handlers: HandlerRegistry,
    *,
    engine: Engine,
    queue: JobQueue,
    store: ArtifactStore,
    registry: ComponentRegistry,
    services: ServicesFor,
    settings: Settings,
) -> TestOrchestrator:
    """Register ``test.start``, ``test.advance`` and ``test.evaluate``; returns the
    orchestrator, whose ``reconcile`` the worker runs with its heartbeats."""
    orchestrator = TestOrchestrator(
        engine,
        queue,
        registry,
        max_attempts=settings.worker.max_attempts,
        webhook_attempts=settings.suites.webhook_attempts,
    )

    def start(job: JobContext) -> Mapping[str, Any]:
        run = orchestrator.start(uuid.UUID(str(job.payload["test_run_id"])))
        return {"status": run.status if run else "missing"}

    def advance(job: JobContext) -> Mapping[str, Any]:
        run = orchestrator.advance(uuid.UUID(str(job.payload["test_run_id"])))
        return {"status": run.status if run else "missing"}

    handlers.register(START_JOB, start)
    handlers.register(ADVANCE_JOB, advance)
    handlers.register(
        EVALUATE_JOB,
        TestEvaluator(
            engine,
            store,
            registry,
            orchestrator,
            services,
            settings,
            IndexCache(store, engine=engine),
            RetrievalCache(),
            sandbox_from_settings(settings.suites.sandbox),
        ),
    )
    return orchestrator
