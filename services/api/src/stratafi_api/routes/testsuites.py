"""Test suites and test runs.

``POST /test-suites/validate`` checks a ``stratafi.test-suite.v1`` without saving it; ``POST
/test-suites`` saves it as the next version of its name (the same document again adds none).
``POST /test-runs`` asks for a run of a saved suite, or of a suite sent with the request (saved
first, as a CI pipeline does with the suite in its repository). One run per suite is active at a
time; a request that finds a run already waiting joins it (``coalesced``). A worker resolves the
run, starts or reuses the pipeline and eval runs it needs, and judges its checks
(``stratafi_store.testruns``); ``test_run`` events on ``GET /events?run_id=`` follow it.

A finished run has its record, ``stratafi.test-run.v1``, and ``GET /test-runs/{id}/report``
renders it as JSON, JUnit XML or Markdown. Results, reports and audit events carry IDs, codes and
numbers only: a golden check's query stays in its suite.
"""

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, ProblemIssue, problem_responses
from stratafi_core.evalsets import load_yaml
from stratafi_core.logs import current_log_context
from stratafi_core.specs import SpecInvalid, SpecIssue
from stratafi_core.suites import (
    SuiteRef,
    TestRunInfo,
    TestRunRecord,
    dump_suite,
    junit_xml,
    markdown_summary,
    parse_suite,
)
from stratafi_store import automation, specs
from stratafi_store import testruns as store
from stratafi_store.audit import record_audit_event
from stratafi_store.jobs import JobQueue

router = APIRouter(tags=["tests"])

TestRunStatus = Literal[
    "queued", "running", "evaluating", "passed", "failed", "error", "cancelling", "cancelled"
]
Trigger = Literal["api", "schedule", "corpus_version_frozen", "target_changed", "ci"]
MAX_CONTENT_CHARS = 200_000
SUITE_REF = r"^[a-z0-9][a-z0-9._-]{0,63}(@([1-9][0-9]{0,8}|latest))?$"


CODE_DESCRIPTION = (
    "Custom checks' Python files by the path the suite names: added to the suite's "
    "code, replacing a file of the same path"
)


class SuiteIn(BaseModel):
    """A suite as a document, or as YAML text: one of the two."""

    model_config = ConfigDict(extra="forbid")

    document: dict[str, Any] | None = Field(
        default=None, description="The stratafi.test-suite.v1 document"
    )
    content: str | None = Field(
        default=None, min_length=1, max_length=MAX_CONTENT_CHARS, description="The same, as YAML"
    )
    code: dict[str, str] | None = Field(default=None, description=CODE_DESCRIPTION)

    @model_validator(mode="after")
    def _one(self) -> "SuiteIn":
        if (self.document is None) == (self.content is None):
            raise ValueError("give either document or content")
        return self


class IssueOut(BaseModel):
    path: str
    code: str


class ValidationOut(BaseModel):
    valid: bool
    issues: list[IssueOut]
    document: dict[str, Any] | None = Field(description="The suite, every default filled in")
    needs: dict[str, bool] = Field(
        description="What its checks need: a pipeline run, an eval run, a baseline, and the "
        "workspace's custom_checks feature"
    )


class SuiteOut(BaseModel):
    id: uuid.UUID
    name: str
    version: int
    document: dict[str, Any] = Field(description="The stratafi.test-suite.v1, resolved")
    spec_hash: str = Field(description="Its configuration, without name and description")
    document_hash: str
    checks: int
    created_by: str
    created_at: datetime

    @classmethod
    def of(cls, stored: specs.StoredDocument) -> "SuiteOut":
        return cls(
            id=stored.id,
            name=stored.name,
            version=stored.version,
            document=stored.document,
            spec_hash=stored.spec_hash,
            document_hash=stored.document_hash,
            checks=len(stored.document.get("checks") or []),
            created_by=stored.created_by,
            created_at=stored.created_at,
        )


class CheckOut(BaseModel):
    id: str
    type: str
    result: Literal["pass", "fail", "error", "skipped"]
    code: str | None
    value: float | None
    threshold: dict[str, Any]
    baseline_value: float | None
    detail: dict[str, Any]
    evidence: list[dict[str, Any]]
    duration_ms: float | None


class CodeFileOut(BaseModel):
    path: str
    sha256: str
    bytes: int


class TestRunOut(BaseModel):
    __test__ = False

    id: uuid.UUID
    suite_id: uuid.UUID
    suite_name: str
    suite_version: int
    status: TestRunStatus
    outcome: Literal["passed", "failed"] | None = Field(
        description="passed when no check failed or errored"
    )
    trigger: Trigger
    triggers: list[str] = Field(description="Every trigger combined into this run")
    resolved: dict[str, Any] = Field(
        description="Target, corpus version, eval set version and baseline, as fixed IDs"
    )
    pipeline_run_id: uuid.UUID | None
    eval_run_id: uuid.UUID | None
    baseline_pipeline_run_id: uuid.UUID | None
    baseline_eval_run_id: uuid.UUID | None
    baseline_test_run_id: uuid.UUID | None
    summary: dict[str, Any] = Field(description="Checks by result")
    checks: list[CheckOut] | None = Field(
        default=None, description="Each check's result, once the run has finished"
    )
    record_hash: str | None
    code: list[CodeFileOut] = Field(
        default_factory=list,
        description="The custom checks' code the run used: each file's path, SHA-256 and size",
    )
    error_code: str | None
    created_by: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def of(cls, run: store.TestRun, checks: list[store.CheckRow] | None = None) -> "TestRunOut":
        return cls(
            id=run.id,
            suite_id=run.suite_id,
            suite_name=run.suite_name,
            suite_version=run.suite_version,
            status=run.status,  # type: ignore[arg-type]
            outcome=run.outcome,  # type: ignore[arg-type]
            trigger=run.trigger,  # type: ignore[arg-type]
            triggers=run.triggers,
            resolved=run.resolved,
            pipeline_run_id=run.pipeline_run_id,
            eval_run_id=run.eval_run_id,
            baseline_pipeline_run_id=run.baseline_pipeline_run_id,
            baseline_eval_run_id=run.baseline_eval_run_id,
            baseline_test_run_id=run.baseline_test_run_id,
            summary=run.summary or {},
            checks=(
                [
                    CheckOut(
                        id=row.check_id,
                        type=row.type,
                        result=row.result,  # type: ignore[arg-type]
                        code=row.code,
                        value=row.value,
                        threshold=row.threshold,
                        baseline_value=row.baseline_value,
                        detail=row.detail,
                        evidence=row.evidence,
                        duration_ms=row.duration_ms,
                    )
                    for row in checks
                ]
                if checks is not None
                else None
            ),
            record_hash=run.record_hash,
            code=[
                CodeFileOut.model_validate(item)
                for item in (run.record or {}).get("code") or []
                if isinstance(item, dict)
            ],
            error_code=run.error_code,
            created_by=run.created_by,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )


class StartedOut(TestRunOut):
    coalesced: bool = Field(description="The request joined a run already waiting for the suite")


class TestRunIn(BaseModel):
    __test__ = False
    model_config = ConfigDict(extra="forbid")

    suite: str | None = Field(
        default=None, pattern=SUITE_REF, description="A saved suite: name, name@N or name@latest"
    )
    document: dict[str, Any] | None = Field(
        default=None, description="A suite to save (as its name's next version) and run"
    )
    content: str | None = Field(
        default=None, min_length=1, max_length=MAX_CONTENT_CHARS, description="The same, as YAML"
    )
    code: dict[str, str] | None = Field(default=None, description=CODE_DESCRIPTION)
    trigger: Literal["api", "ci"] = Field(default="api", description="Who asked: a person or CI")

    @model_validator(mode="after")
    def _one(self) -> "TestRunIn":
        if sum(value is not None for value in (self.suite, self.document, self.content)) != 1:
            raise ValueError("give one of suite, document or content")
        if self.suite is not None and self.code is not None:
            raise ValueError("code comes with a suite's document or content")
        return self


class HistoryOut(BaseModel):
    suite: str
    runs: list[dict[str, Any]] = Field(
        description="Finished runs, oldest first, with each check's result and value"
    )


# --- helpers ------------------------------------------------------------------------------------


def _queue(request: Request) -> JobQueue:
    queue = request.app.state.jobs
    if queue is None:
        raise Problem("unavailable", code="database_not_configured")
    return queue


def _audit(
    conn: Connection,
    principal: Principal,
    action: str,
    resource_type: str,
    resource_id: str,
    **details: Any,
) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        request_id=current_log_context().get("request_id"),
        details=details,
    )


def _invalid(issues: list[SpecIssue]) -> Problem:
    return Problem(
        "invalid_request",
        "The test suite is invalid.",
        code="invalid_suite",
        errors=[issue.path for issue in issues],
        issues=[ProblemIssue(path=issue.path, code=issue.code) for issue in issues],
    )


def _document(
    document: dict[str, Any] | None, content: str | None, code: dict[str, str] | None = None
) -> dict[str, Any]:
    if document is not None:
        loaded: Any = dict(document)
    else:
        loaded = load_yaml(content or "")
        if not isinstance(loaded, dict):
            raise SpecInvalid([SpecIssue("(root)", "not_an_object")])
    if code:
        inline = loaded.get("code")
        if inline is not None and not isinstance(inline, dict):
            raise SpecInvalid([SpecIssue("code", "dict_type")])
        loaded["code"] = {**(inline or {}), **code}
    return loaded


def _save(
    conn: Connection, principal: Principal, document: dict[str, Any]
) -> tuple[specs.StoredDocument, bool]:
    try:
        suite = parse_suite(document)
    except SpecInvalid as error:
        raise _invalid(error.issues) from None
    stored, created = store.save_suite(
        conn, workspace_id=principal.workspace.id, suite=suite, created_by=principal.subject
    )
    if created:
        _audit(
            conn,
            principal,
            "test_suite.save",
            "test_suite",
            str(stored.id),
            version=stored.version,
            checks=len(suite.checks),
        )
        # The latest version's schedules are the suite's.
        automation.sync_schedules(conn, principal.workspace.id, suite)
    return stored, created


def _run_or_404(conn: Connection, principal: Principal, test_run_id: uuid.UUID) -> store.TestRun:
    run = store.get_test_run(conn, principal.workspace.id, test_run_id)
    if run is None:
        raise Problem("not_found", code="test_run_not_found")
    return run


def record_of(run: store.TestRun) -> TestRunRecord:
    """A finished run's record; a run without one (it couldn't evaluate, or hasn't finished)
    gets a record of what's known, with no checks."""
    if run.record is not None:
        return TestRunRecord.model_validate(run.record)
    return TestRunRecord(
        test_run=TestRunInfo(
            id=str(run.id),
            suite=SuiteRef(
                id=str(run.suite_id),
                name=run.suite_name,
                version=run.suite_version,
                document_hash="",
            ),
            trigger=run.trigger,  # type: ignore[arg-type]
            triggers=list(run.triggers),
            status=run.status,
            error_code=run.error_code,
            created_by=run.created_by,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
        ),
        resolved=run.resolved,
        runs={
            "pipeline_run": str(run.pipeline_run_id) if run.pipeline_run_id else None,
            "eval_run": str(run.eval_run_id) if run.eval_run_id else None,
        },
    )


# --- suites ---------------------------------------------------------------------------------------


@router.post(
    "/test-suites/validate",
    response_model=ValidationOut,
    summary="Check a test suite without saving it",
    description="Problems come back as issues: paths and codes, never values. References "
    "(the target, corpus, eval set and baseline) are resolved when a run starts.",
    responses=problem_responses(400, 401, 403),
)
def validate_suite(
    body: SuiteIn, principal: Principal = Depends(require_role("viewer"))
) -> ValidationOut:
    try:
        suite = parse_suite(_document(body.document, body.content, body.code))
    except SpecInvalid as error:
        return ValidationOut(
            valid=False,
            issues=[IssueOut(path=issue.path, code=issue.code) for issue in error.issues],
            document=None,
            needs={},
        )
    return ValidationOut(
        valid=True,
        issues=[],
        document=dump_suite(suite),
        needs={
            "pipeline_run": suite.needs_pipeline_run,
            "eval_run": suite.needs_eval_run,
            "baseline": suite.needs_baseline,
            "custom_checks": bool(suite.custom_checks),
        },
    )


@router.post(
    "/test-suites",
    response_model=SuiteOut,
    status_code=201,
    summary="Save a test suite as its name's next version",
    description="201 with the new version; 200 with the latest one when it's the same document.",
    responses=problem_responses(400, 401, 403, 503),
)
def save_suite(
    request: Request, body: SuiteIn, principal: Principal = Depends(require_role("editor"))
) -> JSONResponse:
    try:
        document = _document(body.document, body.content, body.code)
    except SpecInvalid as error:
        raise _invalid(error.issues) from None
    with engine_of(request).begin() as conn:
        stored, created = _save(conn, principal, document)
    return JSONResponse(
        SuiteOut.of(stored).model_dump(mode="json"), status_code=201 if created else 200
    )


@router.get(
    "/test-suites",
    response_model=Page[SuiteOut],
    summary="The workspace's test suites: each name's latest version, by name",
    responses=problem_responses(400, 401, 403, 503),
)
def list_suites(
    request: Request,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("viewer")),
) -> Page[SuiteOut]:
    after = decode_cursor(page.cursor, size=1)[0] if page.cursor is not None else None
    with engine_of(request).connect() as conn:
        found = specs.list_latest(
            conn, "suite", principal.workspace.id, limit=page.limit + 1, after_name=after
        )
    cursor = encode_cursor([found[page.limit - 1].name]) if len(found) > page.limit else None
    return Page[SuiteOut](
        items=[SuiteOut.of(item) for item in found[: page.limit]], next_cursor=cursor
    )


@router.get(
    "/test-suites/{name}",
    response_model=SuiteOut,
    summary="A test suite's latest version, or the one asked for",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_suite(
    request: Request,
    name: str,
    version: Annotated[int | None, Query(ge=1)] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> SuiteOut:
    with engine_of(request).connect() as conn:
        stored = store.get_suite(conn, principal.workspace.id, name, version)
    if stored is None:
        raise Problem("not_found", code="test_suite_not_found")
    return SuiteOut.of(stored)


@router.get(
    "/test-suites/{name}/versions",
    response_model=Page[SuiteOut],
    summary="A test suite's versions, newest first",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_suite_versions(
    request: Request,
    name: str,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("viewer")),
) -> Page[SuiteOut]:
    before = None
    if page.cursor is not None:
        try:
            before = int(decode_cursor(page.cursor, size=1)[0])
        except (TypeError, ValueError):
            raise Problem(
                "invalid_request", "The cursor isn't valid.", code="invalid_cursor"
            ) from None
    with engine_of(request).connect() as conn:
        found = specs.list_versions(
            conn, "suite", principal.workspace.id, name, limit=page.limit + 1, before_version=before
        )
    if not found and before is None:
        raise Problem("not_found", code="test_suite_not_found")
    cursor = encode_cursor([found[page.limit - 1].version]) if len(found) > page.limit else None
    return Page[SuiteOut](
        items=[SuiteOut.of(item) for item in found[: page.limit]], next_cursor=cursor
    )


@router.get(
    "/test-suites/{name}/history",
    response_model=HistoryOut,
    summary="A suite's finished runs with each check's result and value, for trends",
    responses=problem_responses(400, 401, 403, 503),
)
def suite_history(
    request: Request,
    name: str,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    principal: Principal = Depends(require_role("viewer")),
) -> HistoryOut:
    with engine_of(request).connect() as conn:
        runs = store.check_history(conn, principal.workspace.id, name, limit=limit)
    for run in runs:
        run["created_at"] = run["created_at"].isoformat()
    return HistoryOut(suite=name, runs=runs)


class ScheduleOut(BaseModel):
    cron: str
    time_zone: str
    next_fire_at: datetime
    last_fired_at: datetime | None
    last_test_run_id: uuid.UUID | None


class AutomationOut(BaseModel):
    suite: str
    version: int
    events: list[str] = Field(description="The events that run it: corpus_version_frozen...")
    schedules: list[ScheduleOut] = Field(description="Its cron schedules and their next times")
    time_zone: str


@router.get(
    "/test-suites/{name}/automation",
    response_model=AutomationOut,
    summary="What runs a suite on its own: its events, and its schedules' next times",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def suite_automation(
    request: Request, name: str, principal: Principal = Depends(require_role("viewer"))
) -> AutomationOut:
    with engine_of(request).connect() as conn:
        stored = store.get_suite(conn, principal.workspace.id, name)
        if stored is None:
            raise Problem("not_found", code="test_suite_not_found")
        schedules = automation.schedules_of(conn, principal.workspace.id, name)
    suite = store.suite_of(stored)
    return AutomationOut(
        suite=name,
        version=stored.version,
        events=sorted({trigger.on for trigger in suite.triggers if trigger.on is not None}),
        schedules=[
            ScheduleOut(
                cron=schedule.cron,
                time_zone=schedule.time_zone,
                next_fire_at=schedule.next_fire_at,
                last_fired_at=schedule.last_fired_at,
                last_test_run_id=schedule.last_test_run_id,
            )
            for schedule in schedules
        ],
        time_zone=suite.time_zone,
    )


# --- runs -----------------------------------------------------------------------------------------


@router.post(
    "/test-runs",
    response_model=StartedOut,
    status_code=202,
    summary="Run a test suite: a saved one, or one sent with the request",
    description="202 with the queued run. While a run of the suite is active, one more waits "
    "behind it, and requests meanwhile join that one (`coalesced`). Follow it with "
    "`GET /events?run_id=`. An API token needs the `tests:run` scope.",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def start_test_run(
    request: Request,
    body: TestRunIn,
    principal: Principal = Depends(require_role("editor", scope="tests:run")),
) -> JSONResponse:
    queue = _queue(request)
    with engine_of(request).begin() as conn:
        if body.suite is not None:
            name, _, number = body.suite.partition("@")
            version = int(number) if number and number != "latest" else None
            stored = store.get_suite(conn, principal.workspace.id, name, version)
            if stored is None:
                raise Problem("not_found", code="test_suite_not_found", errors=["suite"])
        else:
            try:
                document = _document(body.document, body.content, body.code)
            except SpecInvalid as error:
                raise _invalid(error.issues) from None
            stored, _ = _save(conn, principal, document)
        run, coalesced = store.request_run(
            conn,
            queue,
            workspace_id=principal.workspace.id,
            suite=stored,
            trigger=body.trigger,
            created_by=principal.subject,
            max_attempts=request.app.state.settings.worker.max_attempts,
        )
        _audit(
            conn,
            principal,
            "test_run.start",
            "test_run",
            str(run.id),
            suite_version=stored.version,
            trigger=body.trigger,
            coalesced=coalesced,
        )
    out = StartedOut(**TestRunOut.of(run).model_dump(), coalesced=coalesced)
    return JSONResponse(out.model_dump(mode="json"), status_code=202)


@router.get(
    "/test-runs",
    response_model=Page[TestRunOut],
    summary="The workspace's test runs, newest first",
    responses=problem_responses(400, 401, 403, 503),
)
def list_test_runs(
    request: Request,
    page: PageParams = Depends(page_params),
    suite: Annotated[str | None, Query(max_length=64)] = None,
    status: Annotated[TestRunStatus | None, Query()] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> Page[TestRunOut]:
    after = None
    if page.cursor is not None:
        created_at, run_ref = decode_cursor(page.cursor, size=2)
        try:
            after = (datetime.fromisoformat(created_at), uuid.UUID(run_ref))
        except (TypeError, ValueError, AttributeError):
            raise Problem(
                "invalid_request", "The cursor isn't valid.", code="invalid_cursor"
            ) from None
    with engine_of(request).connect() as conn:
        found = store.list_test_runs(
            conn,
            principal.workspace.id,
            limit=page.limit + 1,
            after=after,
            suite_name=suite,
            status=status,
        )
    cursor = None
    if len(found) > page.limit:
        last = found[page.limit - 1]
        cursor = encode_cursor([last.created_at.isoformat(), str(last.id)])
    return Page[TestRunOut](
        items=[TestRunOut.of(run) for run in found[: page.limit]], next_cursor=cursor
    )


@router.get(
    "/test-runs/{test_run_id}",
    response_model=TestRunOut,
    summary="One test run: its status, resolved inputs, runs and each check's result",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_test_run(
    request: Request, test_run_id: uuid.UUID, principal: Principal = Depends(require_role("viewer"))
) -> TestRunOut:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, test_run_id)
        checks = store.check_results(conn, run.id) if run.finished else None
    return TestRunOut.of(run, checks)


@router.get(
    "/test-runs/{test_run_id}/report",
    summary="A finished test run's report: JSON (stratafi.test-run.v1), JUnit XML or Markdown",
    responses={
        200: {
            "description": "The report",
            "content": {
                "application/json": {},
                "application/xml": {},
                "text/markdown": {},
            },
        },
        **problem_responses(400, 401, 403, 404, 409, 503),
    },
)
def get_test_run_report(
    request: Request,
    test_run_id: uuid.UUID,
    format: Annotated[Literal["json", "junit", "markdown"], Query()] = "json",
    link_base: Annotated[
        str | None,
        Query(max_length=300, pattern=r"^https?://[^\s]+$", description="Markdown links' base"),
    ] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> Response:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, test_run_id)
    if not run.finished:
        raise Problem("conflict", "The run hasn't finished.", code="test_run_not_finished")
    record = record_of(run)
    if format == "junit":
        return Response(junit_xml(record), media_type="application/xml")
    if format == "markdown":
        return Response(
            markdown_summary(record, link_base=link_base), media_type="text/markdown; charset=utf-8"
        )
    return JSONResponse(record.model_dump(mode="json", by_alias=True))


@router.post(
    "/test-runs/{test_run_id}/cancel",
    response_model=TestRunOut,
    summary="Cancel a test run, with the pipeline and eval runs it started",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def cancel_test_run(
    request: Request,
    test_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("editor", scope="tests:run")),
) -> TestRunOut:
    orchestrator: store.TestOrchestrator | None = request.app.state.tests
    if orchestrator is None:
        raise Problem("unavailable", code="database_not_configured")
    run, cancels = orchestrator.cancel(test_run_id, principal.workspace.id)
    if run is None:
        raise Problem("not_found", code="test_run_not_found")
    for kind, workspace_id, run_id in cancels:
        other = request.app.state.runs if kind == "run" else request.app.state.evals
        cancelled = other.cancel(run_id, workspace_id)
        if cancelled is not None:
            for job_id in cancelled[1]:
                _queue(request).cancel(job_id, workspace_id)
    with engine_of(request).begin() as conn:
        _audit(conn, principal, "test_run.cancel", "test_run", str(run.id))
    return TestRunOut.of(run)
