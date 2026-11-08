"""Workspace features: what a workspace's owner turned on.

``GET /workspace/features`` lists them, on or off, and says why one can't be turned on here.
``PUT /workspace/features/{feature}`` turns one on or off: an owner's call, never an API
token's. ``custom_checks`` runs test suites' custom checks in the sandbox. It can't be
turned on until this deployment configures the sandbox and its security review is signed off
(``suites.sandbox.review``); the change keeps the review's reference, and an audit event
records it. Turning it off always works.
"""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_core.logs import current_log_context
from stratafi_core.settings import Settings
from stratafi_store import features as store
from stratafi_store.audit import record_audit_event

router = APIRouter(tags=["workspace"])

FeatureName = Literal["custom_checks"]


class FeatureOut(BaseModel):
    name: FeatureName
    enabled: bool
    blocked: str | None = Field(
        description="Why it can't be turned on here: sandbox_not_configured or "
        "sandbox_not_reviewed; null when it can"
    )
    review: str | None = Field(description="The security review it was turned on under")
    changed_by: str | None
    changed_at: datetime | None


class FeaturesOut(BaseModel):
    items: list[FeatureOut]


class FeatureIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


def _blocked(settings: Settings, name: str) -> str | None:
    return settings.suites.custom_checks_blocked() if name == store.CUSTOM_CHECKS else None


def _out(settings: Settings, feature: store.Feature) -> FeatureOut:
    return FeatureOut(
        name=feature.name,  # type: ignore[arg-type]
        enabled=feature.enabled,
        blocked=_blocked(settings, feature.name),
        review=feature.review,
        changed_by=feature.changed_by,
        changed_at=feature.changed_at,
    )


@router.get(
    "/workspace/features",
    response_model=FeaturesOut,
    summary="The workspace's features, on or off",
    responses=problem_responses(401, 403, 503),
)
def list_features(
    request: Request, principal: Principal = Depends(require_role("viewer"))
) -> FeaturesOut:
    settings: Settings = request.app.state.settings
    with engine_of(request).connect() as conn:
        found = store.features_of(conn, principal.workspace.id)
    return FeaturesOut(items=[_out(settings, feature) for feature in found.values()])


@router.put(
    "/workspace/features/{feature}",
    response_model=FeatureOut,
    summary="Turn a workspace feature on or off",
    description="Owners only. custom_checks can't be turned on (409) until the sandbox is "
    "configured and its security review is signed off.",
    responses=problem_responses(400, 401, 403, 409, 503),
)
def set_feature(
    request: Request,
    feature: FeatureName,
    body: FeatureIn,
    principal: Principal = Depends(require_role("owner")),
) -> FeatureOut:
    settings: Settings = request.app.state.settings
    blocked = _blocked(settings, feature)
    if body.enabled and blocked is not None:
        detail = (
            "The custom-check sandbox isn't configured here."
            if blocked == "sandbox_not_configured"
            else "The custom-check sandbox's security review isn't signed off."
        )
        raise Problem("conflict", detail, code=blocked)
    review = settings.suites.sandbox.review if body.enabled else None
    with engine_of(request).begin() as conn:
        changed = store.set_feature(
            conn,
            principal.workspace.id,
            feature,
            on=body.enabled,
            changed_by=principal.subject,
            review=review,
        )
        record_audit_event(
            conn,
            workspace_id=principal.workspace.id,
            actor_kind=principal.actor_kind,
            actor_subject=principal.subject,
            action="workspace.feature_on" if body.enabled else "workspace.feature_off",
            resource_type="workspace_feature",
            resource_id=feature,
            request_id=current_log_context().get("request_id"),
        )
    return _out(settings, changed)
