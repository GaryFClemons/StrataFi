"""Alembic environment for Stratafi's forward-only migrations.

Run migrations through ``stratafi_store.migrations.upgrade`` (or ``stratafi-store migrate``),
which hands this module an open transaction. An advisory lock serializes concurrent runs, and the
version table lives in the ``stratafi`` schema with everything else.
"""

import sqlalchemy as sa
from alembic import context

from stratafi_store.tables import SCHEMA, metadata

# pg_advisory_xact_lock key for migrations: "STRF".
MIGRATION_LOCK = 0x5354_5246


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    if type_ == "schema":
        return name == SCHEMA
    return True


def run_migrations_online() -> None:
    connection = context.config.attributes.get("connection")
    if connection is None:
        raise RuntimeError("run migrations through stratafi_store.migrations.upgrade()")
    connection.execute(sa.text("SELECT pg_advisory_xact_lock(:key)"), {"key": MIGRATION_LOCK})
    connection.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
    context.configure(
        connection=connection,
        target_metadata=metadata,
        version_table_schema=SCHEMA,
        include_schemas=True,
        include_name=include_name,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("Stratafi's migrations run online only")
run_migrations_online()
