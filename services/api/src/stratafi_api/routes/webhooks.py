"""Webhooks: where finished test runs are announced.

An owner registers a URL, optionally for one suite's runs. The answer carries the webhook's
signing secret once; it's never returned again. Each delivery is signed
(``stratafi_core.webhooks``: ``Stratafi-Signature: t=...,v1=...``) and retried with backoff
(``stratafi_worker.automation``). The URL must pass the SSRF guard URL import uses when it's
registered and again at every delivery. URLs and secrets never reach a log or an audit event.
"""

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_components.fetch import FetchRefused, UrlGuard
from stratafi_core.logs import current_log_context
from stratafi_store import webhooks as store
from stratafi_store.audit import record_audit_event

router = APIRouter(tags=["webhooks"])

Event = Literal["test_run.finished"]


class WebhookIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=8, max_length=2048, description="An http or https URL")
    suite: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$",
        description="Only this suite's runs; every suite's if none",
    )
    events: list[Event] = Field(default_factory=lambda: ["test_run.finished"], min_length=1)


class WebhookOut(BaseModel):
    id: uuid.UUID
    url: str
    suite: str | None
    events: list[str]
    enabled: bool
    created_by: str
    created_at: datetime

    @classmethod
    def of(cls, webhook: store.Webhook) -> "WebhookOut":
        return cls(
            id=webhook.id,
            url=webhook.url,
            suite=webhook.suite_name,
            events=webhook.events,
            enabled=webhook.enabled,
            created_by=webhook.created_by,
            created_at=webhook.created_at,
        )


class CreatedWebhookOut(WebhookOut):
    secret: str = Field(description="Signs every delivery. Shown this once: keep it now")


class WebhooksOut(BaseModel):
    items: list[WebhookOut]


class DeliveryOut(BaseModel):
    id: uuid.UUID
    event: str
    test_run_id: uuid.UUID | None
    status: Literal["pending", "delivered", "failed"]
    attempts: int
    response_status: int | None
    error_code: str | None
    created_at: datetime
    delivered_at: datetime | None


class DeliveriesOut(BaseModel):
    items: list[DeliveryOut]


def _audit(
    principal: Principal, conn: Connection, action: str, webhook_id: uuid.UUID, **details: Any
) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type="webhook",
        resource_id=str(webhook_id),
        request_id=current_log_context().get("request_id"),
        details=details,
    )


@router.post(
    "/webhooks",
    response_model=CreatedWebhookOut,
    status_code=201,
    summary="Register a webhook; its signing secret comes back this once",
    responses=problem_responses(400, 401, 403, 503),
)
def create_webhook(
    request: Request, body: WebhookIn, principal: Principal = Depends(require_role("owner"))
) -> JSONResponse:
    suites = request.app.state.settings.suites
    guard = UrlGuard(allowed_ports=suites.webhook_ports, allow_private=suites.webhook_allow_private)
    try:
        guard.check(body.url)
    except FetchRefused as refused:
        raise Problem(
            "invalid_request", "The URL can't receive webhooks.", code=refused.code, errors=["url"]
        ) from None
    with engine_of(request).begin() as conn:
        webhook, secret = store.create_webhook(
            conn,
            workspace_id=principal.workspace.id,
            url=body.url,
            suite_name=body.suite,
            events=list(body.events),
            created_by=principal.subject,
            secret_key=(
                suites.webhook_secret_key.get_secret_value()
                if suites.webhook_secret_key is not None
                else None
            ),
        )
        _audit(principal, conn, "webhook.create", webhook.id, events=len(body.events))
    out = CreatedWebhookOut(**WebhookOut.of(webhook).model_dump(), secret=secret)
    return JSONResponse(out.model_dump(mode="json"), status_code=201)


@router.get(
    "/webhooks",
    response_model=WebhooksOut,
    summary="The workspace's webhooks, without their secrets",
    responses=problem_responses(400, 401, 403, 503),
)
def list_webhooks(
    request: Request, principal: Principal = Depends(require_role("owner"))
) -> WebhooksOut:
    with engine_of(request).connect() as conn:
        found = store.list_webhooks(conn, principal.workspace.id)
    return WebhooksOut(items=[WebhookOut.of(webhook) for webhook in found])


@router.delete(
    "/webhooks/{webhook_id}",
    status_code=204,
    summary="Remove a webhook and its deliveries",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def delete_webhook(
    request: Request, webhook_id: uuid.UUID, principal: Principal = Depends(require_role("owner"))
) -> Response:
    with engine_of(request).begin() as conn:
        if not store.delete_webhook(conn, principal.workspace.id, webhook_id):
            raise Problem("not_found", code="webhook_not_found")
        _audit(principal, conn, "webhook.delete", webhook_id)
    return Response(status_code=204)


@router.get(
    "/webhooks/{webhook_id}/deliveries",
    response_model=DeliveriesOut,
    summary="A webhook's deliveries, newest first, with their attempts and answers",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_deliveries(
    request: Request,
    webhook_id: uuid.UUID,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    principal: Principal = Depends(require_role("owner")),
) -> DeliveriesOut:
    with engine_of(request).connect() as conn:
        if store.get_webhook(conn, principal.workspace.id, webhook_id) is None:
            raise Problem("not_found", code="webhook_not_found")
        found = store.deliveries_of(conn, principal.workspace.id, webhook_id, limit=limit)
    return DeliveriesOut(
        items=[
            DeliveryOut(
                id=delivery.id,
                event=delivery.event,
                test_run_id=delivery.test_run_id,
                status=delivery.status,  # type: ignore[arg-type]
                attempts=delivery.attempts,
                response_status=delivery.response_status,
                error_code=delivery.error_code,
                created_at=delivery.created_at,
                delivered_at=delivery.delivered_at,
            )
            for delivery in found
        ]
    )
