"""Idempotent Postgres bootstrap: Stratafi's database and its least-privilege role.

    stratafi-store bootstrap

reads ``STRATAFI_PG_ADMIN_DSN`` (a superuser) and ``STRATAFI_DATABASE_URL`` (the role, password
and database the services use). It creates the role (LOGIN, nothing else), a database owned by
it, and the ``vector`` extension, which needs a superuser. Running it again is safe; it resets
the role's password to the URL's, so a regenerated ``.env`` works. Services then connect as the
role, and migrations run as it, since it owns the database.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]{0,62}")

Connect = Callable[..., Any]


class BootstrapError(ValueError):
    pass


@dataclass(frozen=True)
class BootstrapResult:
    database: str
    role: str
    created_role: bool
    created_database: bool


def _identifier(value: object, what: str) -> sql.Identifier:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise BootstrapError(f"invalid {what} name")
    return sql.Identifier(value)


def target_from_url(database_url: str) -> tuple[str, str, str | None]:
    """The database, role and password a services URL names."""
    try:
        params = conninfo_to_dict(database_url)
    except psycopg.ProgrammingError:
        raise BootstrapError("the database URL can't be parsed") from None
    database, role = params.get("dbname"), params.get("user")
    if not database or not role:
        raise BootstrapError("the database URL must name a user and a database")
    password = params.get("password")
    return str(database), str(role), str(password) if password else None


def bootstrap(
    admin_dsn: str,
    database: str,
    role: str,
    *,
    password: str | None = None,
    extensions: Sequence[str] = ("vector",),
    connect: Connect = psycopg.connect,
) -> BootstrapResult:
    """Create ``role`` (LOGIN, no other privileges) and ``database`` owned by it, if missing,
    then the ``extensions`` in it, with the admin connection."""
    db_ident = _identifier(database, "database")
    role_ident = _identifier(role, "role")
    ext_idents = [_identifier(name, "extension") for name in extensions]

    with connect(admin_dsn, autocommit=True) as conn:
        created_role = _ensure_role(conn, role, role_ident, password)
        created_database = _ensure_database(conn, database, db_ident, role_ident)
        conn.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(db_ident))
        conn.execute(
            sql.SQL("GRANT CONNECT, TEMPORARY ON DATABASE {} TO {}").format(db_ident, role_ident)
        )

    if ext_idents:
        with connect(make_conninfo(admin_dsn, dbname=database), autocommit=True) as conn:
            for ext in ext_idents:
                conn.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS {}").format(ext))

    return BootstrapResult(database, role, created_role, created_database)


def _ensure_role(conn: Any, role: str, role_ident: sql.Identifier, password: str | None) -> bool:
    password_clause = (
        sql.SQL(" PASSWORD {}").format(sql.Literal(password))
        if password is not None
        else sql.SQL("")
    )
    exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
    if exists:
        if password is not None:
            conn.execute(sql.SQL("ALTER ROLE {} WITH LOGIN{}").format(role_ident, password_clause))
        return False
    try:
        conn.execute(
            sql.SQL("CREATE ROLE {} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE{}").format(
                role_ident, password_clause
            )
        )
    except psycopg.errors.DuplicateObject:
        return False  # created concurrently by another bootstrap
    return True


def _ensure_database(
    conn: Any, database: str, db_ident: sql.Identifier, role_ident: sql.Identifier
) -> bool:
    exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,)).fetchone()
    if exists:
        return False
    try:
        conn.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(db_ident, role_ident))
    except psycopg.errors.DuplicateDatabase:
        return False
    return True
