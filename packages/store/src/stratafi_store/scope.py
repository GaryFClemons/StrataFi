"""Row-level security's scope: which workspace's rows a transaction may see.

The baseline migration turns row-level security on, and forces it on the tables' owner, for
every table that carries a ``workspace_id`` (and for ``workspaces`` by its ``id``). Each table's
policy lets a row through when

    current_setting('stratafi.scope', true) = 'system'
    OR workspace_id = NULLIF(current_setting('stratafi.workspace_id', true), '')::uuid

so a transaction that set neither sees no rows and can write none. Every transaction an engine
begins sets both, transaction-locally (``set_config(..., true)``), so a pooled connection never
carries one request's workspace into the next. The scope comes from:

1. the innermost ``workspace_scope(id)`` or ``system_scope()`` in the current context
   (``contextvars``: a request's task, or a thread), else
2. the engine's default, its ``stratafi_default_scope`` execution option: ``system`` when unset
   (the worker, the CLI, migrations and tests, which handle every workspace), and ``none`` for
   the API (``api_engine``), which sees nothing until a request's principal names its
   workspace.

This guards against a query that forgets its ``workspace_id`` filter; it isn't a boundary
against code that sets its own scope, which any process holding the role can do (the threat
model, ``docs/security/threat-model.md``). Superusers and ``BYPASSRLS`` roles skip policies, so
services connect as the bootstrap role, never as a superuser.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Final, Literal

from sqlalchemy import event
from sqlalchemy.engine import Connection, Engine

SCOPE_SETTING: Final = "stratafi.scope"
WORKSPACE_SETTING: Final = "stratafi.workspace_id"
DEFAULT_OPTION: Final = "stratafi_default_scope"

System = Literal["system"]
NoScope = Literal["none"]
Scope = uuid.UUID | System | NoScope
SYSTEM: Final[System] = "system"
NONE: Final[NoScope] = "none"

_current: ContextVar[Scope | None] = ContextVar("stratafi_scope", default=None)

_SET = (
    "SELECT set_config('stratafi.scope', %s, true), set_config('stratafi.workspace_id', %s, true)"
)


def current_scope() -> Scope | None:
    """The scope set in this context, if any (not the engine's default)."""
    return _current.get()


@contextmanager
def workspace_scope(workspace_id: uuid.UUID) -> Iterator[None]:
    """Transactions begun inside see only ``workspace_id``'s rows."""
    token = _current.set(workspace_id)
    try:
        yield
    finally:
        _current.reset(token)


@contextmanager
def system_scope() -> Iterator[None]:
    """Transactions begun inside see every workspace: for work that must look across them
    before a workspace is known (identity, the job queue, the scheduler). Keep the block small."""
    token = _current.set(SYSTEM)
    try:
        yield
    finally:
        _current.reset(token)


def enter_workspace(workspace_id: uuid.UUID) -> Token[Scope | None]:
    """Set the scope for the rest of this context, such as a request's task once its principal
    is known. Returns the token that ``leave`` takes."""
    return _current.set(workspace_id)


def leave(token: Token[Scope | None]) -> None:
    _current.reset(token)


def _values(scope: Scope) -> tuple[str, str]:
    if scope == SYSTEM:
        return SYSTEM, ""
    if isinstance(scope, uuid.UUID):
        return "workspace", str(scope)
    return "", ""


def _on_begin(conn: Connection) -> None:
    scope = _current.get()
    if scope is None:
        scope = conn.get_execution_options().get(DEFAULT_OPTION, SYSTEM)
    dbapi = conn.connection.dbapi_connection
    if dbapi is None:  # pragma: no cover - a connection SQLAlchemy just checked out has one
        return
    # psycopg starts the transaction with this statement; SQLAlchemy's own BEGIN is implicit.
    with dbapi.cursor() as cursor:
        cursor.execute(_SET, _values(scope))


def install(engine: Engine) -> None:
    """Set the scope at the start of each of ``engine``'s transactions (``create_engine`` does)."""
    if not event.contains(engine, "begin", _on_begin):
        event.listen(engine, "begin", _on_begin)


def api_engine(engine: Engine) -> Engine:
    """``engine`` (sharing its pool) whose transactions see nothing unless a scope is set."""
    install(engine)
    return engine.execution_options(**{DEFAULT_OPTION: NONE})
