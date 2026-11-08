"""Forward-only Alembic migrations for the ``stratafi`` schema.

    stratafi-store migrate              # apply every migration
    stratafi-store revision -m "..."    # start a new one (development)

Migrations are written by hand, never edited after they merge, and never downgraded: a fix is a
new migration. ``tests/test_migrations.py`` checks that they build exactly the metadata in
``stratafi_store.tables``.
"""

import threading
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Connection, Engine

from stratafi_store.tables import SCHEMA

MIGRATIONS_DIR = Path(__file__).resolve().parent
# Alembic keeps the running migration's context in a module global, so two upgrades in one
# process must not overlap. Across processes, env.py's advisory lock serializes them.
_IN_PROCESS = threading.Lock()


def alembic_config(connection: Connection | None = None) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def upgrade(engine: Engine, revision: str = "head") -> None:
    """Apply migrations up to ``revision`` in one transaction. Safe to run again, and safe to
    run from several processes at once."""
    with _IN_PROCESS, engine.begin() as conn:
        command.upgrade(alembic_config(conn), revision)


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"version_table_schema": SCHEMA})
        return context.get_current_revision()


def head_revision() -> str:
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    if head is None:
        raise RuntimeError("there are no migrations")
    return head


def new_revision(message: str, revision_id: str) -> Path:
    """Write a new, empty migration after the current head. For development."""
    script = command.revision(alembic_config(), message=message, rev_id=revision_id)
    if script is None or isinstance(script, list):
        raise RuntimeError("alembic didn't write one migration")
    return Path(script.path)
