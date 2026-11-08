"""API tokens.

A token is ``sft_`` and 43 random URL-safe characters (256 bits). Only its SHA-256 is stored, so
a database read can't use one; the token itself is returned once, when it's made. Each token
belongs to one workspace, has scopes (``read``, ``runs:write``, ``tests:run``), expires (at most
a year out), and can be revoked. ``authenticate`` finds a live token by its hash and notes when
it was last used, at most once a minute.
"""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from stratafi_store.tables import api_tokens, workspaces
from stratafi_store.workspaces import Workspace

TOKEN_PREFIX = "sft_"  # noqa: S105 - the prefix that marks a token, not one
MAX_DAYS = 365
TOKEN_LENGTH = len(TOKEN_PREFIX) + 43


@dataclass(frozen=True)
class ApiToken:
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    prefix: str
    scopes: list[str]
    created_by: str
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    revoked_by: str | None
    last_used_at: datetime | None

    @property
    def active(self) -> bool:
        return self.revoked_at is None and self.expires_at > datetime.now(UTC)


def _token(row: Any) -> ApiToken:
    return ApiToken(
        **{
            column.name: getattr(row, column.name)
            for column in api_tokens.columns
            if column.name != "token_hash"
        }
    )


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def looks_like_token(value: str) -> bool:
    return value.startswith(TOKEN_PREFIX) and len(value) == TOKEN_LENGTH


def create_token(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    name: str,
    scopes: list[str],
    days: int,
    created_by: str,
) -> tuple[ApiToken, str]:
    """A new token and its value, which is returned this once."""
    if not 1 <= days <= MAX_DAYS:
        raise ValueError("a token lives 1 to 365 days")
    value = TOKEN_PREFIX + secrets.token_urlsafe(32)
    now = datetime.now(UTC)
    row = conn.execute(
        sa.insert(api_tokens)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            name=name,
            prefix=value[: len(TOKEN_PREFIX) + 6],
            token_hash=token_hash(value),
            scopes=sorted(set(scopes)),
            created_by=created_by,
            created_at=now,
            expires_at=now + timedelta(days=days),
        )
        .returning(*api_tokens.columns)
    ).one()
    return _token(row), value


def list_tokens(conn: Connection, workspace_id: uuid.UUID) -> list[ApiToken]:
    rows = conn.execute(
        sa.select(api_tokens)
        .where(api_tokens.c.workspace_id == workspace_id)
        .order_by(api_tokens.c.created_at.desc(), api_tokens.c.id.desc())
    )
    return [_token(row) for row in rows]


def revoke_token(
    conn: Connection, workspace_id: uuid.UUID, token_id: uuid.UUID, revoked_by: str
) -> ApiToken | None:
    row = conn.execute(
        sa.update(api_tokens)
        .where(api_tokens.c.workspace_id == workspace_id, api_tokens.c.id == token_id)
        .values(
            revoked_at=sa.func.coalesce(api_tokens.c.revoked_at, sa.func.now()),
            revoked_by=sa.func.coalesce(api_tokens.c.revoked_by, revoked_by),
        )
        .returning(*api_tokens.columns)
    ).first()
    return _token(row) if row is not None else None


class TokenRefused(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def authenticate(conn: Connection, value: str) -> tuple[Workspace, ApiToken]:
    """The live token ``value`` is, with its workspace. Raises ``TokenRefused``: ``invalid``
    (no such token), ``expired`` or ``revoked``."""
    if not looks_like_token(value):
        raise TokenRefused("invalid")
    row = conn.execute(
        sa.select(api_tokens, workspaces.c.slug, workspaces.c.name.label("workspace_name"))
        .join(workspaces, workspaces.c.id == api_tokens.c.workspace_id)
        .add_columns(workspaces.c.external_tenant_id)
        .where(api_tokens.c.token_hash == token_hash(value))
    ).first()
    if row is None:
        raise TokenRefused("invalid")
    token = _token(row)
    if token.revoked_at is not None:
        raise TokenRefused("revoked")
    if token.expires_at <= datetime.now(UTC):
        raise TokenRefused("expired")
    conn.execute(
        sa.update(api_tokens)
        .where(
            api_tokens.c.id == token.id,
            sa.or_(
                api_tokens.c.last_used_at.is_(None),
                api_tokens.c.last_used_at < sa.func.now() - sa.text("interval '1 minute'"),
            ),
        )
        .values(last_used_at=sa.func.now())
    )
    workspace = Workspace(row.workspace_id, row.slug, row.workspace_name, row.external_tenant_id)
    return workspace, token
