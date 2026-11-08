"""Artifact records and lineage in Postgres agree with the in-memory graph."""

import hashlib
import importlib.util
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from stratafi_core.lineage import LineageGraph
from stratafi_core.logs import TextFieldError
from stratafi_store.lineage import (
    ArtifactInput,
    ArtifactRecord,
    ancestors,
    descendants,
    get_artifact,
    inputs_of,
    lineage_edges,
    record_artifact,
    unreferenced_artifacts,
)
from stratafi_store.tables import artifacts, workspaces

pytestmark = pytest.mark.integration


def _edges() -> list[tuple[str, str]]:
    spec = importlib.util.spec_from_file_location(
        "core_lineage_cases",
        Path(__file__).resolve().parents[2] / "core" / "tests" / "test_lineage.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EDGES


def key_of(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def new_workspace(conn: sa.Connection, slug: str) -> uuid.UUID:
    workspace_id = uuid.uuid4()
    conn.execute(sa.insert(workspaces).values(id=workspace_id, slug=slug, name=slug))
    return workspace_id


def record(conn: sa.Connection, workspace_id: uuid.UUID, name: str, inputs=()) -> bool:
    return record_artifact(
        conn,
        ArtifactRecord(
            workspace_id=workspace_id,
            cache_key=key_of(name),
            kind="raw_document" if name.startswith("doc") else "stage_output",
            storage_uri=f"local:{name}",
            size_bytes=10,
            content_sha256=key_of(f"content of {name}"),
            summary={"elements": 3, "language": "en"},
        ),
        [key_of(source) for source in inputs],
    )


def load_graph(conn: sa.Connection, workspace_id: uuid.UUID) -> LineageGraph:
    edges = _edges()
    inputs: dict[str, list[str]] = {}
    for artifact, source in edges:
        inputs.setdefault(artifact, []).append(source)
    done: set[str] = set()

    def ensure(name: str) -> None:
        if name in done:
            return
        for source in inputs.get(name, []):
            ensure(source)
        record(conn, workspace_id, name, inputs.get(name, []))
        done.add(name)

    for name in {node for edge in edges for node in edge}:
        ensure(name)
    return LineageGraph.from_edges(edges)


def test_descendants_and_ancestors_match_the_in_memory_graph(migrated_engine: sa.Engine) -> None:
    with migrated_engine.begin() as conn:
        workspace_id = new_workspace(conn, "lineage")
        graph = load_graph(conn, workspace_id)
        names = {node for edge in _edges() for node in edge}
        for name in names:
            expected_down = {key_of(n) for n in graph.descendants(name)}
            expected_up = {key_of(n) for n in graph.ancestors(name)}
            assert descendants(conn, workspace_id, key_of(name)) == expected_down, name
            assert ancestors(conn, workspace_id, key_of(name)) == expected_up, name
        everything_from_a = descendants(conn, workspace_id, key_of("doc-a"))
        assert len(everything_from_a) == 8
        stored = LineageGraph.from_edges(lineage_edges(conn, workspace_id))
        assert {key_of(n) for n in graph.roots()} == stored.roots()


def test_lineage_stays_inside_its_workspace(migrated_engine: sa.Engine) -> None:
    with migrated_engine.begin() as conn:
        first = new_workspace(conn, "first")
        second = new_workspace(conn, "second")
        load_graph(conn, first)
        record(conn, second, "doc-a")
        record(conn, second, "parsed-a", ["doc-a"])
        assert descendants(conn, second, key_of("doc-a")) == {key_of("parsed-a")}
        assert len(descendants(conn, first, key_of("doc-a"))) == 8


def test_recording_is_idempotent_and_keeps_inputs_in_order(migrated_engine: sa.Engine) -> None:
    with migrated_engine.begin() as conn:
        workspace_id = new_workspace(conn, "idempotent")
        for name in ("doc-a", "doc-b"):
            record(conn, workspace_id, name)
        assert record(conn, workspace_id, "joined", ["doc-b", "doc-a"]) is True
        assert record(conn, workspace_id, "joined", ["doc-a"]) is False
        assert inputs_of(conn, workspace_id, key_of("joined")) == [
            ArtifactInput(key_of("doc-b"), "input", 0),
            ArtifactInput(key_of("doc-a"), "input", 1),
        ]
        stored = get_artifact(conn, workspace_id, key_of("joined"))
        assert stored is not None
        assert stored.summary == {"elements": 3, "language": "en"}
        assert get_artifact(conn, workspace_id, key_of("missing")) is None


def test_an_input_with_dependents_cannot_be_deleted(migrated_engine: sa.Engine) -> None:
    with migrated_engine.begin() as conn:
        workspace_id = new_workspace(conn, "guarded")
        record(conn, workspace_id, "doc-a")
        record(conn, workspace_id, "parsed-a", ["doc-a"])
    with pytest.raises(IntegrityError), migrated_engine.begin() as conn:
        conn.execute(sa.delete(artifacts).where(artifacts.c.cache_key == key_of("doc-a")))
    with migrated_engine.begin() as conn:
        # Deleting the dependent first, then the input, works: purges follow lineage.
        conn.execute(sa.delete(artifacts).where(artifacts.c.cache_key == key_of("parsed-a")))
        conn.execute(sa.delete(artifacts).where(artifacts.c.cache_key == key_of("doc-a")))
        # Deleting a workspace removes all of its artifacts and edges at once.
        load_graph(conn, workspace_id)
        conn.execute(sa.delete(workspaces).where(workspaces.c.id == workspace_id))
        assert lineage_edges(conn, workspace_id) == []


def test_unreferenced_artifacts_are_the_leaves(migrated_engine: sa.Engine) -> None:
    with migrated_engine.begin() as conn:
        workspace_id = new_workspace(conn, "leaves")
        graph = load_graph(conn, workspace_id)
        later = datetime.now(UTC) + timedelta(minutes=1)
        found = {
            item.cache_key
            for item in unreferenced_artifacts(conn, workspace_id, created_before=later)
        }
        assert found == {key_of(name) for name in graph.leaves()}
        earlier = datetime.now(UTC) - timedelta(days=1)
        assert unreferenced_artifacts(conn, workspace_id, created_before=earlier) == []


def test_summaries_hold_no_text(migrated_engine: sa.Engine) -> None:
    with migrated_engine.begin() as conn:
        workspace_id = new_workspace(conn, "summaries")
        with pytest.raises(TextFieldError, match="title"):
            record_artifact(
                conn,
                ArtifactRecord(
                    workspace_id=workspace_id,
                    cache_key=key_of("x"),
                    kind="parsed_document",
                    storage_uri="local:x",
                    size_bytes=1,
                    content_sha256=key_of("x"),
                    summary={"title": "Overdraft policy"},
                ),
            )


def test_bad_keys_are_refused_by_the_database(migrated_engine: sa.Engine) -> None:
    with migrated_engine.begin() as conn:
        workspace_id = new_workspace(conn, "shapes")
    bad = sa.insert(artifacts).values(
        workspace_id=workspace_id,
        cache_key="not-a-hash",
        kind="raw_document",
        storage_uri="local:x",
        size_bytes=1,
        content_sha256=key_of("x"),
        media_type="text/plain",
    )
    with pytest.raises(IntegrityError), migrated_engine.begin() as conn:
        conn.execute(bad)
