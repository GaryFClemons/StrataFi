"""Jobs in the workspace: list, read and cancel. Runs create jobs; clients don't."""

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, problem_responses
from stratafi_core.logs import current_log_context
from stratafi_store.audit import record_audit_event
from stratafi_store.jobs import Job, JobQueue

router = APIRouter(tags=["jobs"])

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class JobOut(BaseModel):
    id: uuid.UUID
    kind: str
    status: JobStatus
    resource_class: Literal["cpu", "gpu", "io"]
    priority: int
    attempts: int
    max_attempts: int
    cancel_requested: bool
    progress: dict[str, Any] | None
    error_code: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def of(cls, job: Job) -> "JobOut":
        return cls(
            id=job.id,
            kind=job.kind,
            status=job.status,  # type: ignore[arg-type]
            resource_class=job.resource_class,  # type: ignore[arg-type]
            priority=job.priority,
            attempts=job.attempts,
            max_attempts=job.max_attempts,
            cancel_requested=job.cancel_requested,
            progress=dict(job.progress) if job.progress is not None else None,
            error_code=job.error_code,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )


class CancelOut(BaseModel):
    outcome: Literal["cancelled", "requested", "finished"]
    job: JobOut


def _queue(request: Request) -> JobQueue:
    queue = request.app.state.jobs
    if queue is None:
        raise Problem("unavailable", code="database_not_configured")
    return queue


@router.get(
    "/jobs",
    response_model=Page[JobOut],
    summary="The workspace's jobs, newest first",
    responses=problem_responses(400, 401, 403, 503),
)
def list_jobs(
    request: Request,
    page: PageParams = Depends(page_params),
    status: Annotated[JobStatus | None, Query(description="Only jobs in this state")] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> Page[JobOut]:
    after = None
    if page.cursor is not None:
        created_at, job_id = decode_cursor(page.cursor, size=2)
        try:
            after = (datetime.fromisoformat(created_at), uuid.UUID(job_id))
        except (TypeError, ValueError, AttributeError):
            raise Problem(
                "invalid_request", "The cursor isn't valid.", code="invalid_cursor"
            ) from None
    found = _queue(request).list_jobs(
        principal.workspace.id, limit=page.limit + 1, after=after, status=status
    )
    items = [JobOut.of(job) for job in found[: page.limit]]
    next_cursor = None
    if len(found) > page.limit:
        last = found[page.limit - 1]
        next_cursor = encode_cursor([last.created_at.isoformat(), str(last.id)])
    return Page[JobOut](items=items, next_cursor=next_cursor)


@router.get(
    "/jobs/{job_id}",
    response_model=JobOut,
    summary="One job",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_job(
    request: Request, job_id: uuid.UUID, principal: Principal = Depends(require_role("viewer"))
) -> JobOut:
    job = _queue(request).get(job_id, principal.workspace.id)
    if job is None:
        raise Problem("not_found", code="job_not_found")
    return JobOut.of(job)


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=CancelOut,
    summary="Cancel a job: at once if queued, at its next checkpoint if running",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def cancel_job(
    request: Request, job_id: uuid.UUID, principal: Principal = Depends(require_role("editor"))
) -> CancelOut:
    queue = _queue(request)
    cancelled = queue.cancel(job_id, principal.workspace.id)
    if cancelled is None:
        raise Problem("not_found", code="job_not_found")
    job, outcome = cancelled
    with queue.engine.begin() as conn:
        record_audit_event(
            conn,
            workspace_id=principal.workspace.id,
            actor_kind=principal.actor_kind,
            actor_subject=principal.subject,
            action="job.cancel",
            resource_type="job",
            resource_id=str(job.id),
            request_id=current_log_context().get("request_id"),
            details={"outcome": outcome.value},
        )
    return CancelOut(outcome=outcome.value, job=JobOut.of(job))  # type: ignore[arg-type]
