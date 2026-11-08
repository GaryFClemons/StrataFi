"""Row-level security by workspace: every workspace table has a forced policy, and
a transaction sees only the workspace its scope names."""

import tempfile
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError

from stratafi_store.corpora import create_corpus
from stratafi_store.scope import (
    NONE,
    SYSTEM,
    _values,
    api_engine,
    current_scope,
    system_scope,
    workspace_scope,
)
from stratafi_store.tables import SCHEMA, corpora, metadata, workspaces
from stratafi_store.workspaces import provision_workspace

# --- unit tier ----------------------------------------------------------------------------


def test_scopes_set_the_settings_the_policies_read() -> None:
    workspace = uuid.uuid4()
    assert _values(SYSTEM) == ("system", "")
    assert _values(workspace) == ("workspace", str(workspace))
    assert _values(NONE) == ("", "")


def test_scopes_nest_and_unwind() -> None:
    first, second = uuid.uuid4(), uuid.uuid4()
    assert current_scope() is None
    with workspace_scope(first):
        assert current_scope() == first
        with system_scope():
            assert current_scope() == SYSTEM
            with workspace_scope(second):
                assert current_scope() == second
            assert current_scope() == SYSTEM
        assert current_scope() == first
    assert current_scope() is None


# --- integration tier ---------------------------------------------------------------------

PROTECTED = sorted(
    [table.name for table in metadata.sorted_tables if "workspace_id" in table.c]
    + ["workspaces", "answer_results"]
)


@pytest.mark.integration
def test_every_workspace_table_forces_its_policy(migrated_engine) -> None:
    with migrated_engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, "
                "(SELECT count(*) FROM pg_policies p WHERE p.schemaname = n.nspname "
                "AND p.tablename = c.relname) AS policies "
                "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = :schema AND c.relkind = 'r'"
            ),
            {"schema": SCHEMA},
        ).all()
        superuser = conn.execute(
            sa.text("SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user")
        ).scalar_one()
    protected = {row.relname: row for row in rows if row.relrowsecurity}
    assert sorted(protected) == PROTECTED
    assert all(row.relforcerowsecurity and row.policies == 1 for row in protected.values())
    # The services' role is bound by the policies; a superuser would skip them.
    assert superuser is False


def _two_workspaces(engine) -> tuple[uuid.UUID, uuid.UUID]:
    with system_scope(), engine.begin() as conn:
        first, _ = provision_workspace(conn, "tenant-a")
        second, _ = provision_workspace(conn, "tenant-b")
        create_corpus(conn, first.id, "alpha", "owner")
        create_corpus(conn, second.id, "beta", "owner")
    return first.id, second.id


@pytest.mark.integration
def test_row_level_security_blocks_reads_across_workspaces(migrated_engine) -> None:
    first, second = _two_workspaces(migrated_engine)
    query = sa.select(corpora.c.name).order_by(corpora.c.name)

    with workspace_scope(first), migrated_engine.connect() as conn:
        # No workspace filter in the query: the policy is the filter.
        assert conn.execute(query).scalars().all() == ["alpha"]
        assert (
            conn.execute(sa.select(corpora.c.name).where(corpora.c.workspace_id == second)).all()
            == []
        )
        assert conn.execute(sa.select(workspaces.c.id)).scalars().all() == [first]
    with workspace_scope(second), migrated_engine.connect() as conn:
        assert conn.execute(query).scalars().all() == ["beta"]
    with system_scope(), migrated_engine.connect() as conn:
        assert conn.execute(query).scalars().all() == ["alpha", "beta"]

    # The API's engine sees nothing until a request names its workspace.
    scoped = api_engine(migrated_engine)
    with scoped.connect() as conn:
        assert conn.execute(query).all() == []
        assert conn.execute(sa.select(workspaces.c.id)).all() == []
    with workspace_scope(first), scoped.connect() as conn:
        assert conn.execute(query).scalars().all() == ["alpha"]

    # Updates and deletes reach only the scope's own rows.
    with workspace_scope(first), migrated_engine.begin() as conn:
        assert conn.execute(sa.update(corpora).values(description="changed")).rowcount == 1
        assert (
            conn.execute(sa.delete(corpora).where(corpora.c.workspace_id == second)).rowcount == 0
        )
    with system_scope(), migrated_engine.connect() as conn:
        assert (
            conn.execute(
                sa.select(corpora.c.description).where(corpora.c.workspace_id == second)
            ).scalar_one()
            == ""
        )


@pytest.mark.integration
def test_row_level_security_refuses_writes_into_another_workspace(migrated_engine) -> None:
    first, second = _two_workspaces(migrated_engine)
    with (
        pytest.raises(DBAPIError, match="row-level security"),
        workspace_scope(first),
        migrated_engine.begin() as conn,
    ):
        create_corpus(conn, second, "planted", "owner")
    with (
        pytest.raises(DBAPIError, match="row-level security"),
        api_engine(migrated_engine).begin() as conn,
    ):
        create_corpus(conn, first, "unscoped", "owner")


@pytest.mark.integration
def test_a_scope_ends_with_its_transaction(migrated_engine) -> None:
    first, _ = _two_workspaces(migrated_engine)
    scoped = api_engine(migrated_engine)
    # One pooled connection serves both: the first transaction's scope doesn't carry over.
    with workspace_scope(first), scoped.connect() as conn:
        assert conn.execute(sa.select(sa.func.count()).select_from(corpora)).scalar_one() == 1
        conn.commit()
        assert conn.execute(
            sa.text("SELECT current_setting('stratafi.workspace_id', true)")
        ).scalar_one() == str(first)
    with scoped.connect() as conn:
        assert conn.execute(sa.select(sa.func.count()).select_from(corpora)).scalar_one() == 0
        assert (
            conn.execute(
                sa.text("SELECT current_setting('stratafi.workspace_id', true)")
            ).scalar_one()
            == ""
        )


@pytest.mark.integration
def test_a_refused_write_and_a_purge_carry_no_text(migrated_engine, canary) -> None:
    # Row-level security's refusal names the table, never the row; a workspace purge logs IDs
    # and counts only.
    from stratafi_store import purge
    from stratafi_store.artifacts import LocalArtifactStore

    first, second = _two_workspaces(migrated_engine)
    with (
        pytest.raises(DBAPIError) as refused,
        workspace_scope(first),
        migrated_engine.begin() as conn,
    ):
        create_corpus(conn, second, "planted", "owner", description=canary.value)
    canary.assert_absent(str(refused.value.orig))
    store = LocalArtifactStore(Path(tempfile.mkdtemp()))
    with workspace_scope(second), migrated_engine.begin() as conn:
        conn.execute(
            sa.update(corpora)
            .where(corpora.c.workspace_id == second)
            .values(description=canary.value)
        )
        plan = purge.plan_workspace_purge(conn, store, second)
    assert plan is not None
    purge.remove(migrated_engine, store, plan)
    canary.assert_not_logged()
