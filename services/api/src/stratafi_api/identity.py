"""Identity behind one interface.

``IdentityProvider`` turns a request into verified claims (a ``Caller``): an identity token in
``X-Stratafi-Identity``, signed with the development key (``scripts/dev_setup.py``) and minted by
the development sign-in. An API token, sent as ``Authorization: Bearer sft_...``, stands
for a script or CI in one workspace: its scopes decide what it may do. Then:

- the workspace comes from the caller's tenant (``WorkspaceDirectory``), created on first sight
  when ``identity.auto_provision_workspaces`` is on
- workspace roles are the caller's roles mapped through ``identity.role_map`` and its groups
  through ``identity.group_map``, plus a local membership if one exists
- ``require_role("editor")`` is a FastAPI dependency; roles nest: owner ⊃ editor ⊃ viewer. An
  API token needs the role's scope instead (``read`` for viewer routes, ``runs:write`` for
  editor routes, or the scope a route names, such as ``tests:run``); owner routes refuse tokens

Refusals are 401 problems whose ``code`` is the identity error's reason (``missing``,
``expired``, ``unknown_key``, ``missing_tenant``, ``token_revoked``...), or 403
``missing_role``, ``missing_scope`` or ``token_not_allowed``.
"""

import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from fastapi import Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.engine import Engine

from stratafi_api.problems import Problem
from stratafi_core.identity import HEADER, Caller, IdentityError, TokenVerifier
from stratafi_core.logs import current_log_context
from stratafi_store.audit import record_audit_event
from stratafi_store.scope import enter_workspace, system_scope
from stratafi_store.tokens import ApiToken, TokenRefused, authenticate
from stratafi_store.workspaces import (
    Workspace,
    membership_role,
    provision_workspace,
    workspace_for_tenant,
)

logger = logging.getLogger("stratafi.api.identity")

WorkspaceRole = Literal["owner", "editor", "viewer"]
ROLE_RANK: dict[str, int] = {"viewer": 1, "editor": 2, "owner": 3}
Scope = Literal["read", "runs:write", "tests:run"]
# The scope an API token needs for a role's routes; owner routes refuse tokens.
ROLE_SCOPES: dict[str, str | None] = {"viewer": "read", "editor": "runs:write", "owner": None}

# Documents the header in the OpenAPI contract; verification happens in the provider.
identity_header = APIKeyHeader(
    name=HEADER,
    auto_error=False,
    description="An identity token (an EdDSA JWT). Mint one with "
    "`uv run python scripts/dev_setup.py token --tenant local --user owner`.",
)
bearer = HTTPBearer(
    auto_error=False,
    description="An API token (`sft_...`) for scripts and CI, with scopes.",
)


def _one_header(request: Request) -> str:
    values = request.headers.getlist(HEADER)
    if len(values) != 1:
        raise IdentityError("missing" if not values else "ambiguous")
    return values[0]


class IdentityProvider(Protocol):
    def authenticate(self, request: Request) -> Caller:
        """The request's verified claims, or ``IdentityError`` with a reason code."""
        ...


class DevIdentityProvider:
    """One ``X-Stratafi-Identity`` header, verified against the development key."""

    def __init__(self, verifier: TokenVerifier) -> None:
        self.verifier = verifier

    def authenticate(self, request: Request) -> Caller:
        return self.verifier.verify(_one_header(request)).caller()


class WorkspaceDirectory(Protocol):
    def resolve(self, identity: Caller) -> tuple[Workspace, frozenset[str]] | None:
        """The tenant's workspace and the subject's local membership roles, or None."""
        ...


class SqlWorkspaceDirectory:
    def __init__(self, engine: Engine, *, auto_provision: bool) -> None:
        self.engine = engine
        self.auto_provision = auto_provision

    def resolve(self, identity: Caller) -> tuple[Workspace, frozenset[str]] | None:
        # Before the workspace is known, so across them (row-level security).
        with system_scope(), self.engine.begin() as conn:
            workspace = workspace_for_tenant(conn, identity.tid)
            if workspace is None:
                if not self.auto_provision:
                    return None
                workspace, created = provision_workspace(conn, identity.tid)
                if created:
                    record_audit_event(
                        conn,
                        workspace_id=workspace.id,
                        actor_kind="service" if identity.is_service else "user",
                        actor_subject=identity.sub,
                        action="workspace.provision",
                        resource_type="workspace",
                        resource_id=str(workspace.id),
                        request_id=current_log_context().get("request_id"),
                    )
                    logger.info("workspace provisioned", extra={"workspace_id": str(workspace.id)})
            role = membership_role(conn, workspace.id, identity.sub)
        return workspace, frozenset({role} if role else ())


class TokenDirectory(Protocol):
    def authenticate(self, value: str) -> tuple[Workspace, ApiToken]:
        """The live token's workspace and record, or ``TokenRefused``."""
        ...


class SqlTokens:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def authenticate(self, value: str) -> tuple[Workspace, ApiToken]:
        # A token is found by its hash before its workspace is known.
        with system_scope(), self.engine.begin() as conn:
            return authenticate(conn, value)


@dataclass
class StaticWorkspaces:
    """Workspaces held in memory: for unit tests, and for exporting the contract offline."""

    workspaces: dict[str, Workspace] = field(default_factory=dict)
    memberships: dict[tuple[str, str], str] = field(default_factory=dict)
    auto_provision: bool = True

    def resolve(self, identity: Caller) -> tuple[Workspace, frozenset[str]] | None:
        workspace = self.workspaces.get(identity.tid)
        if workspace is None:
            if not self.auto_provision:
                return None
            workspace = Workspace(uuid.uuid4(), identity.tid, identity.tid, identity.tid)
            self.workspaces[identity.tid] = workspace
        role = self.memberships.get((identity.tid, identity.sub))
        return workspace, frozenset({role} if role else ())


@dataclass(frozen=True)
class Principal:
    """Who is calling, in which workspace, with which workspace roles: a person or a service
    (a verified identity token), or an API token with its scopes."""

    identity: Caller | None
    workspace: Workspace
    roles: frozenset[str]
    token: ApiToken | None = None

    @property
    def subject(self) -> str:
        if self.token is not None:
            return f"token:{self.token.id}"
        assert self.identity is not None  # noqa: S101 - a principal has one or the other
        return self.identity.sub

    @property
    def tenant(self) -> str:
        if self.identity is not None:
            return self.identity.tid
        return self.workspace.external_tenant_id or self.workspace.slug

    @property
    def actor_kind(self) -> str:
        if self.token is not None:
            return "token"
        return "service" if self.identity is not None and self.identity.is_service else "user"

    @property
    def scopes(self) -> frozenset[str] | None:
        """An API token's scopes; None for people and services, whose roles decide."""
        return frozenset(self.token.scopes) if self.token is not None else None

    def has_role(self, role: str) -> bool:
        needed = ROLE_RANK[role]
        return any(ROLE_RANK.get(held, 0) >= needed for held in self.roles)


def workspace_roles(
    identity: Caller,
    role_map: Mapping[str, str],
    local: frozenset[str],
    group_map: Mapping[str, str] | None = None,
) -> frozenset[str]:
    mapped = {role_map[role] for role in identity.roles if role in role_map}
    groups = group_map or {}
    mapped |= {groups[group] for group in identity.groups if group in groups}
    return frozenset(mapped | set(local))


@dataclass
class IdentityService:
    provider: IdentityProvider
    directory: WorkspaceDirectory
    role_map: Mapping[str, str]
    tokens: TokenDirectory | None = None
    group_map: Mapping[str, str] = field(default_factory=dict)

    def _token(self, value: str) -> Principal:
        if self.tokens is None:
            raise Problem("unauthorized", code="token_invalid")
        try:
            workspace, token = self.tokens.authenticate(value)
        except TokenRefused as refused:
            logger.warning("api token refused", extra={"reason": refused.reason})
            raise Problem("unauthorized", code=f"token_{refused.reason}") from None
        writes = {"runs:write", "tests:run"} & set(token.scopes)
        return Principal(None, workspace, frozenset({"editor" if writes else "viewer"}), token)

    def principal(self, request: Request) -> Principal:
        authorization = request.headers.get("authorization", "")
        scheme, _, value = authorization.partition(" ")
        value = value.strip()
        if scheme.lower() == "bearer" and value:
            return self._token(value)
        try:
            identity = self.provider.authenticate(request)
        except IdentityError as exc:
            logger.warning("identity token refused", extra={"reason": exc.reason})
            raise Problem("unauthorized", code=exc.reason) from None
        return self.place(identity)

    def place(self, identity: Caller) -> Principal:
        """A verified caller in its tenant's workspace, with its workspace roles."""
        resolved = self.directory.resolve(identity)
        if resolved is None:
            raise Problem(
                "forbidden", "This tenant has no Stratafi workspace.", code="unknown_workspace"
            )
        workspace, local_roles = resolved
        roles = workspace_roles(identity, self.role_map, local_roles, self.group_map)
        return Principal(identity, workspace, roles)


def bind_workspace(principal: Principal) -> Principal:
    """Scope the rest of the request to the principal's workspace: its transactions see only
    that workspace's rows (``stratafi_store.scope``). Call it from the request's own task, an
    async function, so the scope reaches the handler and the threads it runs in."""
    enter_workspace(principal.workspace.id)
    return principal


async def get_principal(
    request: Request,
    _token: str | None = Depends(identity_header),
    _bearer: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> Principal:
    """The caller, authenticated and placed in a workspace. Any workspace role, or none. From
    here on, the request sees only that workspace's rows."""
    cached = getattr(request.state, "principal", None)
    if cached is None:
        service: IdentityService = request.app.state.identity
        cached = await run_in_threadpool(service.principal, request)
        request.state.principal = cached
    return bind_workspace(cached)


def require_role(role: WorkspaceRole, scope: Scope | None = None) -> Callable[..., Principal]:
    """A dependency that returns the principal if it holds ``role`` (or a higher one). An API
    token needs ``scope`` instead, or the role's scope (``ROLE_SCOPES``)."""
    if role not in ROLE_RANK:
        raise ValueError("unknown workspace role")
    needed = scope or ROLE_SCOPES[role]

    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if principal.scopes is not None:
            if needed is None:
                raise Problem("forbidden", "API tokens can't do this.", code="token_not_allowed")
            if needed not in principal.scopes:
                raise Problem("forbidden", f"This needs the {needed} scope.", code="missing_scope")
            return principal
        if not principal.has_role(role):
            logger.info(
                "role refused",
                extra={"needed": role, "workspace_id": str(principal.workspace.id)},
            )
            raise Problem("forbidden", f"This needs the {role} role.", code="missing_role")
        return principal

    return dependency
