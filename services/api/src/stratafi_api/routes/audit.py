"""The workspace's audit events, newest first. Owners only."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, problem_responses
from stratafi_store.audit import list_audit_events

router = APIRouter(tags=["audit"])


class AuditEventOut(BaseModel):
    id: int
    occurred_at: datetime
    actor_kind: str
    actor_subject: str
    action: str
    outcome: str
    resource_type: str | None
    resource_id: str | None
    request_id: str | None
    details: dict[str, Any]


@router.get(
    "/audit-events",
    response_model=Page[AuditEventOut],
    summary="The workspace's audit events, newest first",
    responses=problem_responses(400, 401, 403),
)
def audit_events(
    request: Request,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("owner")),
) -> Page[AuditEventOut]:
    after = None
    if page.cursor is not None:
        occurred_at, event_id = decode_cursor(page.cursor, size=2)
        try:
            after = (datetime.fromisoformat(occurred_at), int(event_id))
        except (TypeError, ValueError):
            raise Problem(
                "invalid_request", "The cursor isn't valid.", code="invalid_cursor"
            ) from None
    engine = request.app.state.engine
    if engine is None:
        raise Problem("unavailable", code="database_not_configured")
    with engine.connect() as conn:
        events = list_audit_events(conn, principal.workspace.id, limit=page.limit + 1, after=after)
    items = [AuditEventOut(**vars(event)) for event in events[: page.limit]]
    next_cursor = None
    if len(events) > page.limit:
        last = events[page.limit - 1]
        next_cursor = encode_cursor([last.occurred_at.isoformat(), last.id])
    return Page[AuditEventOut](items=items, next_cursor=next_cursor)
