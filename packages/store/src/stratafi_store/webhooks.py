"""Webhooks: where finished test runs are announced, signed, with retries.

A webhook is a URL, the events it wants (``test_run.finished``), and optionally one suite's runs
only. When a test run ends, ``queue_deliveries`` writes one delivery per matching webhook, with
the JSON body it will send, and queues a ``webhook.deliver`` job for it; the worker sends it
signed (``stratafi_core.webhooks``), and the job queue retries a failed send with backoff up to
``suites.webhook_attempts``. Every retry carries the same body and delivery ID.

The body is IDs, codes, counts and links: the test run, its suite and version, its status and
outcome, its summary, and its triggers. The secret is derived from the deployment's key
(``suites.webhook_secret_key``) or, without one, stored to sign with; shown once when the
webhook is made, and never returned or logged again.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from stratafi_core.webhooks import derive_secret, new_secret
from stratafi_store.jobs import JobQueue
from stratafi_store.tables import webhook_deliveries, webhooks

DELIVER_JOB = "webhook.deliver"
FINISHED_EVENT = "test_run.finished"


@dataclass(frozen=True)
class Webhook:
    id: uuid.UUID
    workspace_id: uuid.UUID
    url: str
    suite_name: str | None
    events: list[str]
    enabled: bool
    created_by: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Delivery:
    id: uuid.UUID
    workspace_id: uuid.UUID
    webhook_id: uuid.UUID
    event: str
    test_run_id: uuid.UUID | None
    body: str
    status: str
    attempts: int
    response_status: int | None
    error_code: str | None
    created_at: datetime
    delivered_at: datetime | None


def _webhook(row: Any) -> Webhook:
    return Webhook(
        **{
            column.name: getattr(row, column.name)
            for column in webhooks.columns
            if column.name != "secret"
        }
    )


def _delivery(row: Any) -> Delivery:
    return Delivery(
        **{column.name: getattr(row, column.name) for column in webhook_deliveries.columns}
    )


def create_webhook(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    url: str,
    suite_name: str | None,
    events: list[str],
    created_by: str,
    secret_key: str | None = None,
) -> tuple[Webhook, str]:
    """A webhook and its secret, which is returned this once. With the deployment's
    ``secret_key`` the secret is derived from it and not stored; without one, it's a
    random secret stored to sign with."""
    webhook_id = uuid.uuid4()
    secret = derive_secret(secret_key, webhook_id) if secret_key else new_secret()
    row = conn.execute(
        sa.insert(webhooks)
        .values(
            id=webhook_id,
            workspace_id=workspace_id,
            url=url,
            secret=None if secret_key else secret,
            suite_name=suite_name,
            events=events,
            created_by=created_by,
        )
        .returning(*webhooks.columns)
    ).one()
    return _webhook(row), secret


def list_webhooks(conn: Connection, workspace_id: uuid.UUID) -> list[Webhook]:
    rows = conn.execute(
        sa.select(webhooks)
        .where(webhooks.c.workspace_id == workspace_id)
        .order_by(webhooks.c.created_at, webhooks.c.id)
    )
    return [_webhook(row) for row in rows]


def get_webhook(conn: Connection, workspace_id: uuid.UUID, webhook_id: uuid.UUID) -> Webhook | None:
    row = conn.execute(
        sa.select(webhooks).where(
            webhooks.c.workspace_id == workspace_id, webhooks.c.id == webhook_id
        )
    ).first()
    return _webhook(row) if row is not None else None


def delete_webhook(conn: Connection, workspace_id: uuid.UUID, webhook_id: uuid.UUID) -> bool:
    deleted = conn.execute(
        sa.delete(webhooks).where(
            webhooks.c.workspace_id == workspace_id, webhooks.c.id == webhook_id
        )
    )
    return deleted.rowcount > 0


def deliveries_of(
    conn: Connection, workspace_id: uuid.UUID, webhook_id: uuid.UUID, *, limit: int
) -> list[Delivery]:
    """A webhook's deliveries, newest first."""
    rows = conn.execute(
        sa.select(webhook_deliveries)
        .where(
            webhook_deliveries.c.workspace_id == workspace_id,
            webhook_deliveries.c.webhook_id == webhook_id,
        )
        .order_by(webhook_deliveries.c.created_at.desc(), webhook_deliveries.c.id.desc())
        .limit(limit)
    )
    return [_delivery(row) for row in rows]


def payload(run: Any, delivery_id: uuid.UUID) -> dict[str, Any]:
    """What a ``test_run.finished`` delivery says: IDs, codes, counts and links."""
    return {
        "id": str(delivery_id),
        "type": FINISHED_EVENT,
        "created_at": datetime.now(UTC).isoformat(),
        "data": {
            "test_run_id": str(run.id),
            "workspace_id": str(run.workspace_id),
            "suite": {"name": run.suite_name, "version": run.suite_version},
            "status": run.status,
            "outcome": run.status if run.status in ("passed", "failed") else None,
            "error_code": run.error_code,
            "summary": dict(run.summary or {}),
            "trigger": run.trigger,
            "triggers": list(run.triggers),
            "links": {
                "test_run": f"/test-runs/{run.id}",
                "report": f"/test-runs/{run.id}/report",
            },
        },
    }


def queue_deliveries(conn: Connection, queue: JobQueue, run: Any, *, max_attempts: int = 5) -> int:
    """A delivery, and its job, for each enabled webhook that wants ``run``'s end."""
    rows = conn.execute(
        sa.select(webhooks.c.id).where(
            webhooks.c.workspace_id == run.workspace_id,
            webhooks.c.enabled.is_(True),
            sa.or_(webhooks.c.suite_name.is_(None), webhooks.c.suite_name == run.suite_name),
            webhooks.c.events.contains([FINISHED_EVENT]),
        )
    ).scalars()
    queued = 0
    for webhook_id in rows:
        delivery_id = uuid.uuid4()
        body = json.dumps(payload(run, delivery_id), separators=(",", ":"), sort_keys=True)
        conn.execute(
            sa.insert(webhook_deliveries).values(
                id=delivery_id,
                workspace_id=run.workspace_id,
                webhook_id=webhook_id,
                event=FINISHED_EVENT,
                test_run_id=run.id,
                body=body,
            )
        )
        queue.enqueue_in(
            conn,
            workspace_id=run.workspace_id,
            kind=DELIVER_JOB,
            payload={"delivery_id": str(delivery_id)},
            resource_class="io",
            max_attempts=max_attempts,
        )
        queued += 1
    return queued


def begin_delivery(
    engine: Engine, delivery_id: uuid.UUID, *, secret_key: str | None = None
) -> tuple[Delivery, str, str] | None:
    """A pending delivery with its webhook's URL and secret (stored, or derived from
    ``secret_key``), its attempt counted; None if it's no longer pending, its webhook is gone,
    or its secret can't be derived here (``webhook_key_missing``)."""
    with engine.begin() as conn:
        row = conn.execute(
            sa.update(webhook_deliveries)
            .where(webhook_deliveries.c.id == delivery_id, webhook_deliveries.c.status == "pending")
            .values(attempts=webhook_deliveries.c.attempts + 1)
            .returning(*webhook_deliveries.columns)
        ).first()
        if row is None:
            return None
        hook = conn.execute(
            sa.select(webhooks.c.url, webhooks.c.secret, webhooks.c.enabled).where(
                webhooks.c.id == row.webhook_id
            )
        ).first()
    if hook is None or not hook.enabled:
        finish_delivery(engine, delivery_id, delivered=False, error_code="webhook_disabled")
        return None
    if hook.secret is not None:
        return _delivery(row), hook.url, hook.secret
    if not secret_key:
        finish_delivery(engine, delivery_id, delivered=False, error_code="webhook_key_missing")
        return None
    return _delivery(row), hook.url, derive_secret(secret_key, row.webhook_id)


def finish_delivery(
    engine: Engine,
    delivery_id: uuid.UUID,
    *,
    delivered: bool,
    response_status: int | None = None,
    error_code: str | None = None,
    final: bool = True,
) -> None:
    """Record an attempt: delivered, or failed (``final``), or still pending a retry."""
    values: dict[str, Any] = {"response_status": response_status, "error_code": error_code}
    if delivered:
        values.update(status="delivered", delivered_at=sa.func.now(), error_code=None)
    elif final:
        values["status"] = "failed"
    with engine.begin() as conn:
        conn.execute(
            sa.update(webhook_deliveries)
            .where(webhook_deliveries.c.id == delivery_id)
            .values(**values)
        )
