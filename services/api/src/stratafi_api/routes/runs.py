"""Runs: start, list, read, cancel and re-run from a stage.

``POST /runs`` records a queued run and its ``run.start`` job in one transaction and answers
202; a worker plans it (``stratafi_worker.runs``). Progress arrives as ``run`` events on
``GET /events?run_id=...``; the run, its entries and its tasks are read here. A finished run's
manifest is its record; while it runs, ``/manifest`` shows the record so far.
"""

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of, store_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, problem_responses
from stratafi_core.logs import current_log_context
from stratafi_core.manifest import RunManifest
from stratafi_core.planner import ALL_STAGES
from stratafi_store import corpora, specs
from stratafi_store import runs as store
from stratafi_store.artifacts import ArtifactNotFound, quarantine_key
from stratafi_store.audit import record_audit_event
from stratafi_store.jobs import JobQueue

router = APIRouter(tags=["runs"])

Stage = Literal["extract", "scan", "parse", "clean", "chunk", "enrich", "embed", "index"]
RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]
TaskStatus = Literal[
    "pending", "queued", "running", "succeeded", "cached", "skipped", "failed", "cancelled"
]
EntryState = Literal["active", "done", "excluded", "unsupported_type", "failed", "cancelled"]
INTERACTIVE_PRIORITY = 10


class SampleIn(BaseModel):
    """Which entries a run works on: ``size`` chosen at random by ``seed``, or ``entries`` by
    name. Without a sample, a run works on every entry."""

    model_config = ConfigDict(extra="forbid")

    size: int | None = Field(default=None, ge=1, le=100_000)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1, description="Defaults to 0")
    entries: list[str] | None = Field(default=None, min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def _one_way(self) -> "SampleIn":
        if (self.size is None) == (self.entries is None):
            raise ValueError("give either size or entries")
        if self.entries is not None and self.seed is not None:
            raise ValueError("seed goes with size")
        return self


class RunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_spec_id: uuid.UUID = Field(description="A saved pipeline spec version")
    corpus_version_id: uuid.UUID = Field(description="A frozen corpus version")
    until: Stage | None = Field(default=None, description="The last stage; every stage if none")
    sample: SampleIn | None = None
    mode: Literal["interactive", "batch"] = Field(
        default="batch", description="Interactive runs' jobs go ahead of batch runs' jobs"
    )


class RerunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_stage: Stage = Field(description="This stage and every later one run again")


class RunCounts(BaseModel):
    tasks: dict[str, int] = Field(default_factory=dict, description="Tasks by status")
    stages: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="Tasks by stage and status"
    )
    entries: dict[str, int] = Field(default_factory=dict, description="Entries by state")


class RunOut(BaseModel):
    id: uuid.UUID
    kind: Literal["pipeline"]
    mode: Literal["interactive", "batch"]
    status: RunStatus
    pipeline_spec_id: uuid.UUID
    corpus_version_id: uuid.UUID
    until: Stage | None
    sample: dict[str, Any] | None
    rerun_of: uuid.UUID | None
    rerun_from: Stage | None
    spec_hash: str
    manifest_hash: str | None = Field(description="The finished run's fingerprint")
    counts: RunCounts
    error_code: str | None
    created_by: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def of(cls, run: store.Run) -> "RunOut":
        return cls(
            id=run.id,
            kind=run.kind,  # type: ignore[arg-type]
            mode=run.mode,  # type: ignore[arg-type]
            status=run.status,  # type: ignore[arg-type]
            pipeline_spec_id=run.pipeline_spec_id,
            corpus_version_id=run.corpus_version_id,
            until=run.until_stage,  # type: ignore[arg-type]
            sample=run.sample,
            rerun_of=run.rerun_of,
            rerun_from=run.rerun_from,  # type: ignore[arg-type]
            spec_hash=run.spec_hash,
            manifest_hash=run.manifest_hash,
            counts=RunCounts.model_validate(run.counts or {}),
            error_code=run.error_code,
            created_by=run.created_by,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )


class RunEntryOut(BaseModel):
    ordinal: int
    name: str
    document_id: uuid.UUID
    sha256: str
    duplicate: bool
    state: EntryState
    stop_code: str | None

    @classmethod
    def of(cls, entry: store.RunEntry) -> "RunEntryOut":
        return cls(
            ordinal=entry.ordinal,
            name=entry.name,
            document_id=entry.document_id,
            sha256=entry.sha256,
            duplicate=entry.duplicate,
            state=entry.state,  # type: ignore[arg-type]
            stop_code=entry.stop_code,
        )


class TaskOut(BaseModel):
    id: uuid.UUID
    ordinal: int = Field(description="The entry's ordinal; -1 for corpus-level tasks")
    entry: str | None = Field(description="The entry's name; none for corpus-level tasks")
    stage: Stage
    component_id: str
    component_version: str
    cache_key: str = Field(description="The key of the artifact it makes")
    inputs: list[str] = Field(description="Input artifacts' keys (extract: the document's hash)")
    params: dict[str, Any]
    env: dict[str, Any]
    route: str | None
    resource_class: Literal["cpu", "gpu", "io"]
    status: TaskStatus
    cache_hit: bool
    job_id: uuid.UUID | None
    attempts: int
    error_code: str | None
    metrics: dict[str, Any]
    summary: dict[str, Any] | None = Field(
        description="Its artifact's summary, once it has one: metadata only"
    )
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: float | None

    @classmethod
    def of(
        cls, task: store.Task, entry: str | None, summary: dict[str, Any] | None = None
    ) -> "TaskOut":
        return cls(
            id=task.id,
            ordinal=task.ordinal,
            entry=entry,
            stage=task.stage,  # type: ignore[arg-type]
            component_id=task.component_id,
            component_version=task.component_version,
            cache_key=task.cache_key,
            inputs=list(task.inputs),
            params=task.params,
            env=task.env,
            route=task.route,
            resource_class=task.resource_class,  # type: ignore[arg-type]
            status=task.status,  # type: ignore[arg-type]
            cache_hit=task.cache_hit,
            job_id=task.job_id,
            attempts=task.attempts,
            error_code=task.error_code,
            metrics=task.metrics,
            summary=summary,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
            duration_ms=task.duration_ms,
        )


def _queue(request: Request) -> JobQueue:
    queue = request.app.state.jobs
    if queue is None:
        raise Problem("unavailable", code="database_not_configured")
    return queue


def _orchestrator(request: Request) -> store.Orchestrator:
    orchestrator = request.app.state.runs
    if orchestrator is None:
        raise Problem("unavailable", code="database_not_configured")
    return orchestrator


def _run_or_404(conn: Connection, principal: Principal, run_id: uuid.UUID) -> store.Run:
    run = store.get_run(conn, principal.workspace.id, run_id)
    if run is None:
        raise Problem("not_found", code="run_not_found")
    return run


def _audit(conn: Connection, principal: Principal, action: str, run_id: uuid.UUID, **details: Any):
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type="run",
        resource_id=str(run_id),
        request_id=current_log_context().get("request_id"),
        details=details,
    )


def _enqueue(
    request: Request,
    conn: Connection,
    principal: Principal,
    *,
    pipeline_spec_id: uuid.UUID,
    corpus_version_id: uuid.UUID,
    until: str | None,
    sample: dict[str, Any] | None,
    mode: str,
    rerun_of: uuid.UUID | None = None,
    rerun_from: str | None = None,
) -> store.Run:
    spec = specs.get_by_id(conn, "pipeline", principal.workspace.id, pipeline_spec_id)
    if spec is None:
        raise Problem("not_found", code="pipeline_spec_not_found", errors=["pipeline_spec_id"])
    version = corpora.get_version(conn, principal.workspace.id, version_id=corpus_version_id)
    if version is None:
        raise Problem("not_found", code="corpus_version_not_found", errors=["corpus_version_id"])
    if sample is not None and sample.get("entries"):
        names = {e.name for e in corpora.version_entries(conn, principal.workspace.id, version.id)}
        unknown = [
            f"sample.entries.{index}"
            for index, name in enumerate(sample["entries"])
            if name not in names
        ]
        if unknown:
            raise Problem("invalid_request", code="unknown_entries", errors=unknown[:20])
    run = store.create_run(
        conn,
        workspace_id=principal.workspace.id,
        pipeline_spec_id=spec.id,
        corpus_version_id=version.id,
        spec_hash=spec.spec_hash,
        created_by=principal.subject,
        until=until,
        sample=sample,
        mode=mode,
        rerun_of=rerun_of,
        rerun_from=rerun_from,
    )
    _queue(request).enqueue_in(
        conn,
        workspace_id=principal.workspace.id,
        kind=store.START_JOB,
        payload={"run_id": str(run.id)},
        resource_class="io",
        priority=INTERACTIVE_PRIORITY if mode == "interactive" else 0,
        max_attempts=request.app.state.settings.worker.max_attempts,
    )
    return run


def _accepted(run: store.Run) -> JSONResponse:
    return JSONResponse(RunOut.of(run).model_dump(mode="json"), status_code=202)


@router.post(
    "/runs",
    response_model=RunOut,
    status_code=202,
    summary="Start a run: a pipeline spec version over a corpus version",
    description="202 with the queued run. A worker plans it: cached work is skipped, and the "
    "rest runs as stage jobs. Follow it with `GET /events?run_id=`.",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def start_run(
    request: Request, body: RunIn, principal: Principal = Depends(require_role("editor"))
) -> JSONResponse:
    sample = body.sample.model_dump(exclude_none=True) if body.sample else None
    with engine_of(request).begin() as conn:
        run = _enqueue(
            request,
            conn,
            principal,
            pipeline_spec_id=body.pipeline_spec_id,
            corpus_version_id=body.corpus_version_id,
            until=body.until,
            sample=sample,
            mode=body.mode,
        )
        _audit(conn, principal, "run.start", run.id, until=body.until, mode=body.mode)
    return _accepted(run)


@router.get(
    "/runs",
    response_model=Page[RunOut],
    summary="The workspace's runs, newest first",
    responses=problem_responses(400, 401, 403, 503),
)
def list_runs(
    request: Request,
    page: PageParams = Depends(page_params),
    status: Annotated[RunStatus | None, Query(description="Only runs in this state")] = None,
    include_archived: Annotated[bool, Query(description="Include archived ones")] = False,
    principal: Principal = Depends(require_role("viewer")),
) -> Page[RunOut]:
    after = None
    if page.cursor is not None:
        created_at, run_id = decode_cursor(page.cursor, size=2)
        try:
            after = (datetime.fromisoformat(created_at), uuid.UUID(run_id))
        except (TypeError, ValueError, AttributeError):
            raise Problem(
                "invalid_request", "The cursor isn't valid.", code="invalid_cursor"
            ) from None
    with engine_of(request).connect() as conn:
        found = store.list_runs(
            conn,
            principal.workspace.id,
            limit=page.limit + 1,
            after=after,
            status=status,
            include_archived=include_archived,
        )
    items = [RunOut.of(run) for run in found[: page.limit]]
    next_cursor = None
    if len(found) > page.limit:
        last = found[page.limit - 1]
        next_cursor = encode_cursor([last.created_at.isoformat(), str(last.id)])
    return Page[RunOut](items=items, next_cursor=next_cursor)


@router.get(
    "/runs/{run_id}",
    response_model=RunOut,
    summary="One run, with its counts",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_run(
    request: Request, run_id: uuid.UUID, principal: Principal = Depends(require_role("viewer"))
) -> RunOut:
    with engine_of(request).connect() as conn:
        return RunOut.of(_run_or_404(conn, principal, run_id))


def _offset(page: PageParams) -> int:
    if page.cursor is None:
        return 0
    (offset,) = decode_cursor(page.cursor, size=1)
    if not isinstance(offset, int) or offset < 0:
        raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
    return offset


@router.get(
    "/runs/{run_id}/entries",
    response_model=Page[RunEntryOut],
    summary="The entries a run works on, in order, with where each stopped",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_run_entries(
    request: Request,
    run_id: uuid.UUID,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("viewer")),
) -> Page[RunEntryOut]:
    offset = _offset(page)
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, run_id)
        found = store.entries_of(conn, run.id)[offset : offset + page.limit + 1]
    items = [RunEntryOut.of(entry) for entry in found[: page.limit]]
    cursor = encode_cursor([offset + page.limit]) if len(found) > page.limit else None
    return Page[RunEntryOut](items=items, next_cursor=cursor)


@router.get(
    "/runs/{run_id}/tasks",
    response_model=Page[TaskOut],
    summary="A run's stage tasks: by entry in stage order, then the corpus-level ones",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_run_tasks(
    request: Request,
    run_id: uuid.UUID,
    page: PageParams = Depends(page_params),
    stage: Annotated[Stage | None, Query(description="Only this stage's tasks")] = None,
    status: Annotated[TaskStatus | None, Query(description="Only tasks in this state")] = None,
    ordinal: Annotated[int | None, Query(ge=-1, description="Only this entry's tasks")] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> Page[TaskOut]:
    offset = _offset(page)
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, run_id)
        names = {entry.ordinal: entry.name for entry in store.entries_of(conn, run.id)}
        found = store.tasks_of(conn, run.id, stage=stage, ordinal=ordinal, status=status)
        found = found[offset : offset + page.limit + 1]
        summaries = store.summaries(conn, run.workspace_id, [task.cache_key for task in found])
    items = [
        TaskOut.of(task, names.get(task.ordinal), summaries.get(task.cache_key))
        for task in found[: page.limit]
    ]
    cursor = encode_cursor([offset + page.limit]) if len(found) > page.limit else None
    return Page[TaskOut](items=items, next_cursor=cursor)


@router.get(
    "/runs/{run_id}/manifest",
    response_model=RunManifest,
    summary="A run's record: final once it has finished, the record so far while it runs",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def get_run_manifest(
    request: Request, run_id: uuid.UUID, principal: Principal = Depends(require_role("viewer"))
) -> JSONResponse:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, run_id)
        if run.manifest is None:
            raise Problem("conflict", "The run hasn't started.", code="run_not_started")
        manifest = (
            RunManifest.model_validate(run.manifest)
            if run.finished and run.manifest_hash is not None
            else store.final_manifest(conn, run)
        )
    return JSONResponse(manifest.model_dump(mode="json", by_alias=True))


@router.post(
    "/runs/{run_id}/cancel",
    response_model=RunOut,
    summary="Cancel a run: waiting tasks at once, running ones at their next checkpoint",
    description="The run is `cancelling` until its running tasks stop, then `cancelled`. "
    "Finished artifacts stay; a later run reuses them.",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def cancel_run(
    request: Request, run_id: uuid.UUID, principal: Principal = Depends(require_role("editor"))
) -> RunOut:
    cancelled = _orchestrator(request).cancel(run_id, principal.workspace.id)
    if cancelled is None:
        raise Problem("not_found", code="run_not_found")
    run, job_ids = cancelled
    queue = _queue(request)
    for job_id in job_ids:
        queue.cancel(job_id, principal.workspace.id)
    with engine_of(request).begin() as conn:
        _audit(conn, principal, "run.cancel", run.id, status=run.status, jobs=len(job_ids))
        run = store.get_run(conn, principal.workspace.id, run.id) or run
    return RunOut.of(run)


@router.post(
    "/runs/{run_id}/rerun",
    response_model=RunOut,
    status_code=202,
    summary="Re-run a run from a stage: earlier stages reuse the cache, later ones run again",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def rerun(
    request: Request,
    run_id: uuid.UUID,
    body: RerunIn,
    principal: Principal = Depends(require_role("editor")),
) -> JSONResponse:
    with engine_of(request).begin() as conn:
        original = _run_or_404(conn, principal, run_id)
        last = original.until_stage or ALL_STAGES[-1]
        if ALL_STAGES.index(body.from_stage) > ALL_STAGES.index(last):
            raise Problem("invalid_request", code="stage_after_until", errors=["from_stage"])
        run = _enqueue(
            request,
            conn,
            principal,
            pipeline_spec_id=original.pipeline_spec_id,
            corpus_version_id=original.corpus_version_id,
            until=original.until_stage,
            sample=original.sample,
            mode=original.mode,
            rerun_of=original.id,
            rerun_from=body.from_stage,
        )
        _audit(conn, principal, "run.rerun", run.id, rerun_of=str(original.id))
    return _accepted(run)


# The types the Parse screen renders beside the parsed elements: PDFs with pdf.js, images as is.
VIEWABLE_TYPES = ("application/pdf", "image/png", "image/jpeg")


@router.get(
    "/runs/{run_id}/entries/{ordinal}/original",
    summary="An entry's original PDF or image, for the Parse screen's viewer",
    description="Served only after the run's scan let the document through (its decision isn't "
    "exclude), and only for PDFs and images, as a download with a sandboxing CSP. The viewer "
    "renders it with pdf.js in a worker; nothing in it is fetched.",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"application/octet-stream": {"schema": {"type": "string"}}}},
        **problem_responses(400, 401, 403, 404, 409, 503),
    },
)
def get_entry_original(
    request: Request,
    run_id: uuid.UUID,
    ordinal: int,
    principal: Principal = Depends(require_role("viewer")),
) -> StreamingResponse:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, run_id)
        entry = store.get_entry(conn, run.id, ordinal)
        if entry is None:
            raise Problem("not_found", code="entry_not_found")
        tasks = store.tasks_of(conn, run.id, ordinal=ordinal)
        keys = [task.cache_key for task in tasks if task.stage in ("extract", "scan")]
        summaries = store.summaries(conn, run.workspace_id, keys)
    by_stage = {
        task.stage: summaries.get(task.cache_key) or {}
        for task in tasks
        if task.status in store.DONE_TASK_STATUSES
    }
    if "scan" not in by_stage or by_stage["scan"].get("decision") == "exclude":
        raise Problem("conflict", "The run's scan hasn't passed this document.", code="not_scanned")
    media_type = str(by_stage.get("extract", {}).get("media_type", ""))
    if media_type not in VIEWABLE_TYPES:
        raise Problem("conflict", "Only PDFs and images are shown.", code="not_viewable")
    try:
        stream = store_of(request).open(quarantine_key(run.workspace_id, entry.sha256))
    except ArtifactNotFound:
        raise Problem("not_found", code="document_missing") from None

    def body():
        with stream:
            while chunk := stream.read(1024 * 1024):
                yield chunk

    headers = {
        "Content-Disposition": 'attachment; filename="document"',
        "Content-Security-Policy": "sandbox; default-src 'none'",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, max-age=3600",
    }
    return StreamingResponse(body(), media_type=media_type, headers=headers)
