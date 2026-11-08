"""Audit events: who did what, with IDs and counts only.

``details`` must pass ``check_metadata``: numbers, booleans and token-shaped strings, never
document, chunk or query text.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from stratafi_core.logs import check_metadata
from stratafi_store.tables import audit_events


@dataclass(frozen=True)
class AuditEvent:
    id: int
    workspace_id: uuid.UUID
    occurred_at: datetime
    actor_kind: str
    actor_subject: str
    action: str
    outcome: str
    resource_type: str | None = None
    resource_id: str | None = None
    request_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


def record_audit_event(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    actor_kind: str,
    actor_subject: str,
    action: str,
    outcome: str = "success",
    resource_type: str | None = None,
    resource_id: str | None = None,
    request_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> int:
    details = dict(details or {})
    check_metadata(details, where="audit details")
    return conn.execute(
        sa.insert(audit_events)
        .values(
            workspace_id=workspace_id,
            actor_kind=actor_kind,
            actor_subject=actor_subject,
            action=action,
            outcome=outcome,
            resource_type=resource_type,
            resource_id=resource_id,
            request_id=request_id,
            details=details,
        )
        .returning(audit_events.c.id)
    ).scalar_one()


def list_audit_events(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    after: tuple[datetime, int] | None = None,
) -> list[AuditEvent]:
    """Newest first. ``after`` is the (occurred_at, id) of the last event of the previous page."""
    query = sa.select(audit_events).where(audit_events.c.workspace_id == workspace_id)
    if after is not None:
        occurred_at, event_id = after
        query = query.where(
            sa.tuple_(audit_events.c.occurred_at, audit_events.c.id) < (occurred_at, event_id)
        )
    rows = conn.execute(
        query.order_by(audit_events.c.occurred_at.desc(), audit_events.c.id.desc()).limit(limit)
    )
    return [
        AuditEvent(
            id=row.id,
            workspace_id=row.workspace_id,
            occurred_at=row.occurred_at,
            actor_kind=row.actor_kind,
            actor_subject=row.actor_subject,
            action=row.action,
            outcome=row.outcome,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            request_id=row.request_id,
            details=row.details,
        )
        for row in rows
    ]
