"""Deletion: a purge removes every derived artifact and row, a workspace's
purge removes all it has, and retention purges what expired."""

import uuid
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from stratafi_core.settings import RetentionSettings
from stratafi_store import lineage, purge
from stratafi_store.artifacts import artifact_key, quarantine_key, workspace_prefix
from stratafi_store.retention import sweep
from stratafi_store.scope import system_scope, workspace_scope
from stratafi_store.tables import metadata
from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import FIXTURES, Lab, running_lab

pytestmark = pytest.mark.integration

HANDBOOK = FIXTURES / "demo-corpus" / "corpus" / "handbook"
KEPT, PURGED = "tips-v3.md", "dress-code-v2.md"

PIPELINE = {
    "schema": "stratafi.pipeline.v1",
    "name": "purge",
    "stages": {
        "scan": {
            "checks": [
                {"component": "scan/content-safety"},
                {"component": "scan/clamav", "params": {"max_signature_age_hours": 72}},
                {"component": "scan/dedupe"},
            ]
        },
        "parse": {"route": {"default": {"component": "parse/markdown"}}},
        "clean": [],
        "chunk": {"component": "chunk/sections", "index_text": "{text}"},
        "embed": {"component": "embed/gateway", "params": {"alias": "embed-default"}},
        "index": {
            "dense": {"component": "index/exact"},
            "keyword": {"component": "index/postgres-fts"},
        },
    },
}


def index_tables(engine: sa.Engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'stratafi_index'"
            )
        )
        return {row.table_name for row in rows}


def prepared(lab: Lab) -> tuple[dict[str, Any], str]:
    """Two documents through a pipeline with a Postgres index; the documents by name."""
    version = lab.upload_corpus(
        "handbook", {name: (HANDBOOK / name).read_bytes() for name in (KEPT, PURGED)}
    )
    run = lab.wait_run(lab.start_run(lab.save_spec(PIPELINE), version)["id"])
    assert run["status"] == "succeeded"
    documents = {item["name"]: item for item in lab.get("/documents")["items"]}
    return documents, run["id"]


def draft_candidate(engine: sa.Engine, lab: Lab, run_id: str, document_id: str) -> None:
    """A question drafted from the document, still waiting for review."""
    jsonl = (
        '{"askers": {"staff": {"roles": ["staff"], "groups": ["all-staff"]}}}\n'
        '{"id": "q1", "query": "dress code", "asker": "staff", '
        '"expected": [{"doc_key": "dress-code", "version": "2"}]}\n'
    )
    eval_set = lab.import_eval_set("drafts", jsonl, format="jsonl", freeze=False)["eval_set"]["id"]
    batch = uuid.uuid4()
    with workspace_scope(lab.workspace_id), engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO stratafi.question_batches "
                "(id, workspace_id, eval_set_id, pipeline_run_id, config, created_by) "
                "VALUES (:id, :ws, :es, :run, '{}', 'owner')"
            ),
            {"id": batch, "ws": lab.workspace_id, "es": eval_set, "run": run_id},
        )
        conn.execute(
            sa.text(
                "INSERT INTO stratafi.question_candidates (id, workspace_id, eval_set_id, "
                "batch_id, ordinal, document_id, doc_key, query, evidence, drafted_query) "
                "VALUES (:id, :ws, :es, :batch, 0, :doc, 'k', 'q', 'quoted text', 'q')"
            ),
            {
                "id": uuid.uuid4(),
                "ws": lab.workspace_id,
                "es": eval_set,
                "batch": batch,
                "doc": document_id,
            },
        )


def test_a_purge_removes_every_derived_artifact_and_row(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        documents, run_id = prepared(lab)
        gone, kept = documents[PURGED], documents[KEPT]
        workspace = lab.workspace_id
        draft_candidate(migrated_engine, lab, run_id, gone["id"])
        with workspace_scope(workspace), migrated_engine.connect() as conn:
            derived = lineage.descendants(conn, workspace, gone["sha256"])
            derived_kept = lineage.descendants(conn, workspace, kept["sha256"])
        tables_before = index_tables(migrated_engine)
        assert derived
        assert tables_before

        path = f"/api/v1/documents/{gone['id']}"
        refused = lab.client.delete(path, headers=lab.editor)
        assert (refused.status_code, refused.json()["code"]) == (403, "missing_role")
        response = lab.client.delete(path, headers=dev_identity.headers(role="owner"))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "withdrawn"
        assert body["artifacts"] == len(derived) + 1  # its bytes' root artifact too
        assert (body["working_sets"], body["candidates"]) == (1, 1)
        assert body["tables"] >= 1
        assert body["objects"] > len(derived)

        with workspace_scope(workspace), migrated_engine.connect() as conn:
            left = conn.execute(
                sa.text("SELECT count(*) FROM stratafi.artifacts WHERE cache_key = ANY(:keys)"),
                {"keys": sorted(derived | {gone["sha256"]})},
            ).scalar_one()
            kept_left = conn.execute(
                sa.text("SELECT count(*) FROM stratafi.artifacts WHERE cache_key = ANY(:keys)"),
                {"keys": sorted(derived_kept - derived)},
            ).scalar_one()
            edges = conn.execute(
                sa.text(
                    "SELECT count(*) FROM stratafi.artifact_inputs "
                    "WHERE artifact_key = ANY(:keys) OR input_key = ANY(:keys)"
                ),
                {"keys": sorted(derived | {gone["sha256"]})},
            ).scalar_one()
            working = conn.execute(
                sa.text("SELECT count(*) FROM stratafi.corpus_documents WHERE document_id = :d"),
                {"d": gone["id"]},
            ).scalar_one()
            candidates = conn.execute(
                sa.text("SELECT count(*) FROM stratafi.question_candidates WHERE document_id = :d"),
                {"d": gone["id"]},
            ).scalar_one()
            audited = conn.execute(
                sa.text("SELECT details FROM stratafi.audit_events WHERE action = 'document.purge'")
            ).scalar_one()
            orphans = purge.orphans(conn, lab.store, workspace)
        assert (left, edges, working, candidates) == (0, 0, 0, 0)
        assert kept_left > 0  # what only the other document made stays
        assert orphans == []
        assert audited["reason"] == "deleted"
        for key in derived | {gone["sha256"]}:
            assert not list(lab.store.keys(artifact_key(workspace, key)))
        assert not lab.store.exists(quarantine_key(workspace, gone["sha256"]))
        assert lab.store.exists(quarantine_key(workspace, kept["sha256"]))
        assert not index_tables(migrated_engine) & tables_before  # its index went with it

        tombstone = lab.get(f"/documents/{gone['id']}")
        assert (tombstone["status"], tombstone["reason_code"]) == ("withdrawn", "deleted")
        # A second purge finds nothing left; another workspace's owner finds no document.
        again = lab.client.delete(path, headers=dev_identity.headers(role="owner"))
        assert again.status_code == 200
        assert again.json()["artifacts"] == 0
        other = lab.client.delete(path, headers=dev_identity.headers(tenant="halcyon"))
        assert (other.status_code, other.json()["code"]) == (404, "document_not_found")


def test_a_workspace_purge_removes_every_row_and_object(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepared(lab)
        workspace = lab.workspace_id
        other = lab.client.post(
            "/api/v1/corpora", json={"name": "kept"}, headers=dev_identity.headers("halcyon")
        )
        assert other.status_code == 201
        tables_before = index_tables(migrated_engine)
        with workspace_scope(workspace), migrated_engine.begin() as conn:
            plan = purge.plan_workspace_purge(conn, lab.store, workspace)
        assert plan is not None
        assert plan.rows > 0
        assert plan.tables
        purge.remove(migrated_engine, lab.store, plan)

        with system_scope(), migrated_engine.connect() as conn:
            for table in metadata.sorted_tables:
                if "workspace_id" in table.c:
                    count = conn.execute(
                        sa.select(sa.func.count())
                        .select_from(table)
                        .where(table.c.workspace_id == workspace)
                    ).scalar_one()
                    assert count == 0, table.name
            assert (
                conn.execute(
                    sa.text("SELECT count(*) FROM stratafi.workspaces WHERE id = :w"),
                    {"w": workspace},
                ).scalar_one()
                == 0
            )
            # Another workspace's rows stay.
            assert (
                conn.execute(
                    sa.text("SELECT count(*) FROM stratafi.corpora WHERE name = 'kept'")
                ).scalar_one()
                == 1
            )
        assert list(lab.store.keys(workspace_prefix(workspace))) == []
        assert not index_tables(migrated_engine) & tables_before


def test_retention_purges_what_expired(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        documents, _ = prepared(lab)
        gone, kept = documents[PURGED], documents[KEPT]
        workspace = lab.workspace_id
        with workspace_scope(workspace), migrated_engine.begin() as conn:
            derived_kept = lineage.descendants(conn, workspace, kept["sha256"])
            derived_gone = lineage.descendants(conn, workspace, gone["sha256"])
            conn.execute(
                sa.text(
                    "UPDATE stratafi.documents SET created_at = now() - interval '40 days' "
                    "WHERE id = :d"
                ),
                {"d": gone["id"]},
            )
            conn.execute(
                sa.text(
                    "UPDATE stratafi.audit_events SET occurred_at = now() - interval '400 days'"
                )
            )
            old_events = conn.execute(
                sa.text("SELECT count(*) FROM stratafi.audit_events")
            ).scalar_one()
        assert old_events > 0

        report = sweep(
            migrated_engine,
            lab.store,
            RetentionSettings(document_days=30, audit_days=365),
        )
        assert (report.documents, report.audit_events) == (1, old_events)
        expired = lab.get(f"/documents/{gone['id']}")
        assert (expired["status"], expired["reason_code"]) == ("withdrawn", "retention")
        assert lab.get(f"/documents/{kept['id']}")["status"] == "stored"
        with workspace_scope(workspace), migrated_engine.connect() as conn:
            events = conn.execute(
                sa.text("SELECT action, actor_subject FROM stratafi.audit_events")
            ).all()
            left = conn.execute(
                sa.text("SELECT count(*) FROM stratafi.artifacts WHERE cache_key = ANY(:keys)"),
                {"keys": sorted(derived_gone - derived_kept)},
            ).scalar_one()
        assert [tuple(event) for event in events] == [("document.purge", "system:retention")]
        assert left == 0

        # Derived artifacts expire with what was derived from them; the document's bytes stay.
        with workspace_scope(workspace), migrated_engine.begin() as conn:
            # The index built with both went with the purged document.
            derived_kept = lineage.descendants(conn, workspace, kept["sha256"])
            conn.execute(
                sa.text(
                    "UPDATE stratafi.artifacts SET created_at = now() - interval '40 days' "
                    "WHERE cache_key = ANY(:keys)"
                ),
                {"keys": sorted(derived_kept)},
            )
        report = sweep(migrated_engine, lab.store, RetentionSettings(artifact_days=30))
        assert report.artifacts == len(derived_kept)
        with workspace_scope(workspace), migrated_engine.connect() as conn:
            remaining = set(
                conn.execute(sa.text("SELECT cache_key FROM stratafi.artifacts")).scalars()
            )
        assert kept["sha256"] in remaining
        assert not remaining & derived_kept
        assert lab.store.exists(quarantine_key(workspace, kept["sha256"]))
