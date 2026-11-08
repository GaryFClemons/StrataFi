"""Migrations and bootstrap: forward-only, idempotent, and matching the metadata."""

import importlib.util
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from stratafi_store.bootstrap import BootstrapError, bootstrap, target_from_url
from stratafi_store.db import create_engine, sqlalchemy_url
from stratafi_store.migrations import (
    MIGRATIONS_DIR,
    alembic_config,
    current_revision,
    head_revision,
    upgrade,
)
from stratafi_store.tables import SCHEMA, metadata

# --- unit tier ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://u:p@h:5432/d", "postgresql+psycopg://u:p@h:5432/d"),
        ("postgresql+psycopg://u:p@h/d", "postgresql+psycopg://u:p@h/d"),
    ],
)
def test_libpq_urls_become_psycopg_urls(url: str, expected: str) -> None:
    assert sqlalchemy_url(url) == expected


@pytest.mark.parametrize("url", ["mysql://u:p@h/d", "postgresql+psycopg2://u:p@h/d"])
def test_other_databases_and_drivers_are_refused(url: str) -> None:
    with pytest.raises(ValueError, match="Stratafi"):
        sqlalchemy_url(url)


def test_the_migration_history_is_one_line_of_numbered_revisions() -> None:
    script = ScriptDirectory.from_config(alembic_config())
    assert script.get_heads() == [head_revision()]
    revisions = list(reversed(list(script.walk_revisions())))
    assert [r.revision for r in revisions] == [f"{n:04d}" for n in range(1, len(revisions) + 1)]


def test_every_migration_refuses_to_downgrade() -> None:
    for path in sorted((MIGRATIONS_DIR / "versions").glob("[0-9]*.py")):
        spec = importlib.util.spec_from_file_location(f"migration_{path.stem}", path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(NotImplementedError, match="forward-only"):
            module.downgrade()


def test_bootstrap_reads_its_target_from_the_services_url() -> None:
    assert target_from_url("postgresql://stratafi:pw@127.0.0.1:5442/stratafi") == (
        "stratafi",
        "stratafi",
        "pw",
    )
    with pytest.raises(BootstrapError):
        target_from_url("postgresql://127.0.0.1:5442/stratafi")


@pytest.mark.parametrize("name", ["Stratafi", "drop table", "a" * 64, ""])
def test_bootstrap_refuses_bad_identifiers(name: str) -> None:
    def no_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not connect")

    with pytest.raises(BootstrapError):
        bootstrap("postgresql://unused", name, "stratafi", connect=no_connect)
    with pytest.raises(BootstrapError):
        bootstrap("postgresql://unused", "stratafi", name, connect=no_connect)


# --- integration tier ---------------------------------------------------------------------


def schema_snapshot(engine: sa.Engine) -> list[tuple]:
    """Tables, columns, constraints and indexes of the stratafi schema."""
    query = sa.text(
        """
        SELECT 'column', table_name, column_name || ' ' || data_type || ' ' || is_nullable
               || ' ' || coalesce(column_default, '')
          FROM information_schema.columns WHERE table_schema = :schema
        UNION ALL
        SELECT 'constraint', conrelid::regclass::text, conname || ' ' || pg_get_constraintdef(oid)
          FROM pg_constraint WHERE connamespace = CAST(:schema AS regnamespace)
        UNION ALL
        SELECT 'index', tablename, indexdef FROM pg_indexes WHERE schemaname = :schema
        ORDER BY 1, 2, 3
        """
    )
    with engine.connect() as conn:
        return [tuple(row) for row in conn.execute(query, {"schema": SCHEMA})]


@pytest.mark.integration
def test_migrations_apply_to_an_empty_database_and_rerunning_changes_nothing(
    fresh_database: Callable[[], str],
) -> None:
    engine = create_engine(fresh_database())
    try:
        assert current_revision(engine) is None
        upgrade(engine)
        assert current_revision(engine) == head_revision()
        first = schema_snapshot(engine)
        assert {table for kind, table, _ in first if kind == "column"} >= {
            "workspaces",
            "memberships",
            "audit_events",
            "alembic_version",
        }
        upgrade(engine)
        assert schema_snapshot(engine) == first
        assert current_revision(engine) == head_revision()
    finally:
        engine.dispose()


@pytest.mark.integration
def test_the_migrations_build_exactly_the_metadata(migrated_engine: sa.Engine) -> None:
    with migrated_engine.connect() as conn:
        context = MigrationContext.configure(
            conn,
            opts={
                "include_schemas": True,
                "include_name": lambda name, type_, parents: (
                    name == SCHEMA if type_ == "schema" else True
                ),
                "compare_type": True,
                "compare_server_default": True,
                "version_table_schema": SCHEMA,
            },
        )
        differences = compare_metadata(context, metadata)
    assert differences == []


@pytest.mark.integration
def test_concurrent_upgrades_serialize(fresh_database: Callable[[], str]) -> None:
    """Separate processes, as when the API and the worker start together; and threads."""
    url = fresh_database()
    script = "import os, stratafi_store.db as d, stratafi_store.migrations as m; " + (
        "m.upgrade(d.create_engine(os.environ['MIGRATE_URL']))"
    )
    processes = [
        subprocess.Popen(  # noqa: S603 - fixed arguments
            [sys.executable, "-c", script],
            env={**os.environ, "MIGRATE_URL": url},
            stderr=subprocess.PIPE,
        )
        for _ in range(3)
    ]
    for process in processes:
        _, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr.decode()[-2000:]
    engine = create_engine(url, pool_size=4)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            upgrade(engine)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert errors == []
        assert current_revision(engine) == head_revision()
    finally:
        engine.dispose()


@pytest.mark.integration
def test_bootstrap_creates_a_least_privilege_owner_and_runs_twice(compose_services) -> None:
    admin = compose_services("postgres").pg_admin_dsn()
    database, role = "t_bootstrap_check", "t_bootstrap_role"
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {database} WITH (FORCE)")
        conn.execute(f"DROP ROLE IF EXISTS {role}")
    try:
        first = bootstrap(admin, database, role, password="first-password")
        second = bootstrap(admin, database, role, password="second-password")
        assert (first.created_role, first.created_database) == (True, True)
        assert (second.created_role, second.created_database) == (False, False)
        parts = urlsplit(admin)
        role_url = urlunsplit(
            parts._replace(
                netloc=f"{role}:second-password@{parts.hostname}:{parts.port}",
                path=f"/{database}",
            )
        )
        with psycopg.connect(role_url) as conn:
            flags = conn.execute(
                "SELECT rolsuper, rolcreatedb, rolcreaterole FROM pg_roles WHERE rolname = %s",
                (role,),
            ).fetchone()
            assert flags == (False, False, False)
            assert conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            ).fetchone() == (1,)
        engine = create_engine(role_url)
        try:
            upgrade(engine)
            assert current_revision(engine) == head_revision()
        finally:
            engine.dispose()
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {database} WITH (FORCE)")
            conn.execute(f"DROP ROLE IF EXISTS {role}")
