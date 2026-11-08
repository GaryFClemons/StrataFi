"""API tokens for scripts and CI.

An owner makes a token with a name, scopes and a lifetime (at most a year); the answer carries
the token once. Only its hash is stored. ``read`` reads the workspace, ``runs:write`` does what
an editor does, and ``tests:run`` starts and cancels test runs, which is all CI needs with
``read``. A token can't manage tokens or webhooks, or review profiles: those are an owner's.
Revoking is immediate. Audit events name the token by its ID, and calls made with it are the
token's (``actor_kind`` ``token``).
"""

import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_core.logs import current_log_context
from stratafi_store import tokens as store
from stratafi_store.audit import record_audit_event

router = APIRouter(tags=["tokens"])

Scope = Literal["read", "runs:write", "tests:run"]


class TokenIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$", description="Such as ci-main")
    scopes: list[Scope] = Field(min_length=1, max_length=3)
    expires_in_days: int = Field(default=90, ge=1, le=store.MAX_DAYS)


class TokenOut(BaseModel):
    id: uuid.UUID
    name: str
    prefix: str = Field(description="The token's first characters, to tell tokens apart")
    scopes: list[str]
    active: bool
    created_by: str
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    last_used_at: datetime | None

    @classmethod
    def of(cls, token: store.ApiToken) -> "TokenOut":
        return cls(
            id=token.id,
            name=token.name,
            prefix=token.prefix,
            scopes=token.scopes,
            active=token.active,
            created_by=token.created_by,
            created_at=token.created_at,
            expires_at=token.expires_at,
            revoked_at=token.revoked_at,
            last_used_at=token.last_used_at,
        )


class CreatedTokenOut(TokenOut):
    token: str = Field(description="The token. Shown this once: keep it in a secret store")


class TokensOut(BaseModel):
    items: list[TokenOut]


def _audit(
    conn: Connection, principal: Principal, action: str, token_id: uuid.UUID, **details: Any
) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type="api_token",
        resource_id=str(token_id),
        request_id=current_log_context().get("request_id"),
        details=details,
    )


@router.post(
    "/api-tokens",
    response_model=CreatedTokenOut,
    status_code=201,
    summary="Make an API token; the token comes back this once",
    responses=problem_responses(400, 401, 403, 503),
)
def create_token(
    request: Request, body: TokenIn, principal: Principal = Depends(require_role("owner"))
) -> JSONResponse:
    with engine_of(request).begin() as conn:
        token, value = store.create_token(
            conn,
            workspace_id=principal.workspace.id,
            name=body.name,
            scopes=list(body.scopes),
            days=body.expires_in_days,
            created_by=principal.subject,
        )
        _audit(
            conn,
            principal,
            "api_token.create",
            token.id,
            scopes=",".join(token.scopes),
            days=body.expires_in_days,
        )
    out = CreatedTokenOut(**TokenOut.of(token).model_dump(), token=value)
    return JSONResponse(out.model_dump(mode="json"), status_code=201)


@router.get(
    "/api-tokens",
    response_model=TokensOut,
    summary="The workspace's API tokens, newest first, without the tokens themselves",
    responses=problem_responses(400, 401, 403, 503),
)
def list_tokens(
    request: Request, principal: Principal = Depends(require_role("owner"))
) -> TokensOut:
    with engine_of(request).connect() as conn:
        found = store.list_tokens(conn, principal.workspace.id)
    return TokensOut(items=[TokenOut.of(token) for token in found])


@router.delete(
    "/api-tokens/{token_id}",
    response_model=TokenOut,
    summary="Revoke an API token, at once",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def revoke_token(
    request: Request, token_id: uuid.UUID, principal: Principal = Depends(require_role("owner"))
) -> TokenOut:
    with engine_of(request).begin() as conn:
        token = store.revoke_token(conn, principal.workspace.id, token_id, principal.subject)
        if token is None:
            raise Problem("not_found", code="api_token_not_found")
        _audit(conn, principal, "api_token.revoke", token.id)
    return TokenOut.of(token)
