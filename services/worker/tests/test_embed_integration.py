"""The indexes and the Embed screen's API against Postgres: a pgvector HNSW index builds and
reports ANN recall@10 against exact search; Postgres full-text with the english configuration;
runs with both, searched through the API; the projection and nearest neighbours."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlalchemy as sa

from stratafi_components.embed import Vectors
from stratafi_components.index import IndexInput
from stratafi_components.pgindex import PgDenseIndex, PgKeywordIndex, PgvectorHnsw, PostgresFts
from stratafi_core.artifacts import Chunk, VectorsSidecar
from stratafi_core.components import StageContext
from stratafi_testing.documents import Block, markdown_text
from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import FIXTURES, running_lab

pytestmark = pytest.mark.integration


def chunks_and_vectors(rows: int, dims: int = 32) -> tuple[list[Chunk], Vectors]:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(rows, dims)).astype(np.float32)
    chunks = [
        Chunk(
            chunk_id=f"{i:032x}",
            document_id=f"d-{i % 7}",
            ordinal=i,
            text=f"Chunk {i} about wires and holds number {i % 13}",
            index_text=f"Policy\nWires\nChunk {i} about wires and holds number {i % 13}",
            char_start=0,
            char_end=10,
        )
        for i in range(rows)
    ]
    sidecar = VectorsSidecar(
        chunk_ids=[c.chunk_id for c in chunks],
        component="embed/fake-hash",
        model="m",
        revision="r1",
        dims=dims,
        normalized=False,
    )
    return chunks, Vectors(matrix, sidecar)


def test_a_pgvector_hnsw_index_builds_and_reports_ann_recall(migrated_engine: sa.Engine) -> None:
    """A pgvector HNSW index builds and reports ANN recall@10 against exact search."""
    chunks, vectors = chunks_and_vectors(600)
    context = StageContext(services={"database": migrated_engine})
    params = PgvectorHnsw.resolve_params({"m": 16, "ef_construction": 64, "ef_search": 100})
    built = PgvectorHnsw().build([IndexInput("c" * 64, chunks, vectors)], params, context)
    recall = built.manifest.stats["ann_recall_at_10"]
    assert 0.8 <= recall <= 1.0, recall
    assert context.metrics["ann_recall_at_10"] == recall
    assert built.manifest.backend == "postgres"
    index = PgDenseIndex(built.manifest, migrated_engine)
    hits = index.search(vectors.matrix[5], 10, revision="r1")
    assert hits[0][0] == chunks[5].chunk_id
    allowed = np.array([c.document_id == "d-3" for c in chunks])
    filtered = index.search(vectors.matrix[3], 5, revision="r1", allowed=allowed)
    assert filtered
    assert all(chunks[int(cid, 16)].document_id == "d-3" for cid, _ in filtered)
    # A low ef_search costs recall: the number moves with the setting.
    low = PgvectorHnsw.resolve_params({"m": 2, "ef_construction": 4, "ef_search": 1})
    weak = PgvectorHnsw().build([IndexInput("d" * 64, chunks, vectors)], low, context)
    assert weak.manifest.stats["ann_recall_at_10"] < recall


def test_postgres_full_text_ors_stemmed_words(migrated_engine: sa.Engine) -> None:
    chunks, _ = chunks_and_vectors(40)
    context = StageContext(services={"database": migrated_engine})
    built = PostgresFts().build(
        [IndexInput("c" * 64, chunks)], PostgresFts.resolve_params({}), context
    )
    index = PgKeywordIndex(built.manifest, migrated_engine)
    hits = index.search("holds number 7", 5)
    assert hits
    assert all(score > 0 for _, score in hits)
    # Words are OR-ed and the english configuration stems them: every chunk has "Wires", and
    # "policies" meets "Policy".
    assert len(index.search("wire policies", 50)) == 40
    # Apostrophes split a word for to_tsquery, and stop words alone find nothing.
    assert len(index.search("the guest's wires", 50)) == 40
    assert index.search("the and of", 5) == []
    allowed = np.array([c.document_id == "d-2" for c in chunks])
    filtered = index.search("wire policies", 50, allowed=allowed)
    assert filtered
    assert all(chunks[int(cid, 16)].document_id == "d-2" for cid, _ in filtered)


def test_runs_with_postgres_indexes_search_and_project(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    files = {
        "wires.md": markdown_text(
            "Wire Transfers",
            [Block("paragraph", f"Confirm wire {i} by callback.") for i in range(30)],
        ).encode(),
        "tips.txt": (FIXTURES / "formats" / "tips.txt").read_bytes(),
    }
    spec = {
        "schema": "stratafi.pipeline.v1",
        "name": "pg-indexes",
        "stages": {
            "chunk": {"component": "chunk/recursive", "params": {"size": 64, "overlap": 8}},
            "embed": {
                "component": "embed/fake-hash",
                "params": {"dimensions": 64, "dtype": "int8"},
            },
            "index": {
                "dense": {"component": "index/pgvector-hnsw"},
                "keyword": {"component": "index/postgres-fts"},
            },
        },
    }
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        version = lab.upload_corpus("pg", files)
        run = lab.wait_run(lab.start_run(lab.save_spec(spec), version)["id"])
        assert run["status"] == "succeeded", run
        indexes = {t["component_id"]: t for t in lab.tasks(run["id"], "index")}
        assert 0 < indexes["index/pgvector-hnsw"]["metrics"]["ann_recall_at_10"] <= 1
        embeds = lab.tasks(run["id"], "embed")
        assert all(t["summary"]["dtype"] == "int8" for t in embeds)
        for mode in ("dense", "keyword", "hybrid"):
            found = lab.post(
                "/search", {"run_id": run["id"], "query": "confirm wire by callback", "mode": mode}
            )
            assert found["results"], mode
            assert found["results"][0]["entry"] == "wires.md", mode
        catalog = lab.get("/models/catalog")["models"]
        assert {m["where"] for m in catalog} >= {"gateway", "in_process"}
        # Projections read in-memory indexes; this run's dense index is in Postgres.
        refused = lab.client.get(f"/api/v1/runs/{run['id']}/projection", headers=lab.editor)
        assert refused.json()["code"] == "index_not_in_memory"
        exact = dict(spec, name="exact")
        exact["stages"] = dict(
            spec["stages"], index={"dense": "index/exact", "keyword": "index/bm25"}
        )
        run = lab.wait_run(lab.start_run(lab.save_spec(exact), version)["id"])
        projection = lab.get(f"/runs/{run['id']}/projection")
        assert projection["method"] == "pca"
        assert len(projection["points"]) == projection["total"] > 5
        assert {p["entry"] for p in projection["points"]} == {"wires.md", "tips.txt"}
        first = projection["points"][0]["chunk_id"]
        near = lab.get(f"/runs/{run['id']}/neighbours", chunk_id=first, k=5)["neighbours"]
        assert len(near) == 5
        assert all(item["chunk_id"] != first for item in near)
