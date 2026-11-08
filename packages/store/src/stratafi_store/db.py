"""The database engine: SQLAlchemy 2 Core over psycopg 3, synchronous.

Connections are synchronous and used from worker threads: psycopg's async connections don't
work on Windows' default event loop, and Stratafi's core packages must install and run on
Windows too.

Every transaction sets its row-level security scope (``stratafi_store.scope``).
"""

from collections.abc import Iterator
from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine, make_url

from stratafi_store import scope

APPLICATION_NAME = "stratafi"


def sqlalchemy_url(url: str) -> str:
    """A libpq URL (``postgresql://...``) as SQLAlchemy's psycopg 3 URL
    (``postgresql+psycopg://...``). Other drivers are refused."""
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql":
        raise ValueError("Stratafi needs a PostgreSQL URL")
    if parsed.drivername not in ("postgresql", "postgresql+psycopg"):
        raise ValueError("Stratafi uses the psycopg 3 driver")
    return parsed.set(drivername="postgresql+psycopg").render_as_string(hide_password=False)


def libpq_url(engine: Engine) -> str:
    """The engine's URL as plain libpq takes it (``postgresql://...``), for psycopg directly."""
    return engine.url.set(drivername="postgresql").render_as_string(hide_password=False)


def create_engine(
    url: str,
    *,
    pool_size: int = 5,
    max_overflow: int = 5,
    application_name: str = APPLICATION_NAME,
) -> Engine:
    """An engine for ``url``. Nothing connects until the first query. Its transactions see every
    workspace unless a scope says otherwise (``stratafi_store.scope``)."""
    engine = sa.create_engine(
        sqlalchemy_url(url),
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        connect_args={"application_name": application_name},
    )
    scope.install(engine)
    return engine


@contextmanager
def transaction(engine: Engine) -> Iterator[Connection]:
    """One transaction: committed when the block ends, rolled back if it raises."""
    with engine.begin() as conn:
        yield conn


def ping(engine: Engine) -> None:
    """Raise unless the database answers."""
    with engine.connect() as conn:
        conn.execute(sa.text("SELECT 1"))
