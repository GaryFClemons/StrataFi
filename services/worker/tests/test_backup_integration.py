"""The restore drill: back up a lab, lose the database and the store, restore both into new ones,
migrate, and find the lab as it was: the same rows, the same objects byte for byte, the same
search results, under row-level security."""

import hashlib
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from stratafi_core.settings import StorageSettings
from stratafi_store.artifacts import (
    LocalArtifactStore,
    artifact_store_from_settings,
    quarantine_key,
)
from stratafi_store.backup import BackupError, backup, restore, row_counts, verify
from stratafi_store.db import create_engine
from stratafi_store.migrations import current_revision, head_revision, upgrade
from stratafi_store.scope import system_scope
from stratafi_store.workspaces import provision_workspace
from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import FIXTURES, Lab, running_lab

pytestmark = pytest.mark.integration

HANDBOOK = FIXTURES / "demo-corpus" / "corpus" / "handbook"
NAMES = ("tips-v3.md", "dress-code-v2.md", "scheduling-v4.md")
PIPELINE = {
    "schema": "stratafi.pipeline.v1",
    "name": "drill",
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
QUERY = {"query": "who counts the tip jar at close", "mode": "keyword", "top_k": 5}


def objects(store: Any, workspace: uuid.UUID) -> dict[str, str]:
    return {
        key: hashlib.sha256(store.get(key)).hexdigest() for key in store.keys(f"ws/{workspace}/")
    }


def hits(lab: Lab, run_id: str) -> list[tuple[str, float]]:
    found = lab.post("/search", {"run_id": run_id, **QUERY})
    return [(hit["chunk_id"], round(hit["score"], 6)) for hit in found["results"]]


def test_the_restore_drill(
    fresh_database: Callable[[], str],
    dev_identity: DevIdentity,
    tmp_path: Path,
    serve_app: Any,
) -> None:
    before = create_engine(fresh_database(), pool_size=10, max_overflow=10)
    upgrade(before)
    owner = dev_identity.headers(role="owner")
    with running_lab(before, dev_identity, tmp_path / "before", serve_app) as lab:
        version = lab.upload_corpus(
            "drill", {name: (HANDBOOK / name).read_bytes() for name in NAMES}
        )
        run = lab.wait_run(lab.start_run(lab.save_spec(PIPELINE), version)["id"])
        assert run["status"] == "succeeded"
        hook = lab.client.post(
            "/api/v1/webhooks", json={"url": "http://127.0.0.1:9/hook"}, headers=owner
        )
        assert hook.status_code == 201
        workspace = lab.workspace_id
        documents_before = lab.get("/documents")["items"]
        hits_before = hits(lab, run["id"])
        objects_before = objects(lab.store, workspace)
        rows_before = row_counts(before)
        manifest = backup(before, lab.store, tmp_path / "backup")
    before.dispose()
    assert hits_before
    assert manifest.revision == head_revision()
    assert manifest.objects["count"] == len(objects_before)
    assert manifest.rows == rows_before
    assert any(table.startswith("stratafi_index.") for table in rows_before)
    assert verify(tmp_path / "backup") == []

    # The disaster: a new, empty database and an empty store. Restore, then the upgrade.
    after = create_engine(fresh_database(), pool_size=10, max_overflow=10)
    store = LocalArtifactStore(tmp_path / "after" / "store")
    report = restore(after, store, tmp_path / "backup")
    assert (report.revision_from, report.revision_to) == (head_revision(), head_revision())
    assert current_revision(after) == head_revision()
    assert report.rows == rows_before
    assert objects(store, workspace) == objects_before

    with running_lab(after, dev_identity, tmp_path / "after", serve_app) as lab:
        assert lab.workspace_id == workspace
        assert lab.get("/documents")["items"] == documents_before
        assert hits(lab, run["id"]) == hits_before
        chunks = lab.tasks(run["id"], "chunk")
        assert lab.artifact(chunks[0]["cache_key"])
        assert len(lab.client.get("/api/v1/webhooks", headers=owner).json()["items"]) == 1
        # Row-level security holds in the restored database: another tenant sees none of it.
        other = lab.client.get("/api/v1/documents", headers=dev_identity.headers("halcyon"))
        assert other.json()["items"] == []
    after.dispose()


def test_a_restore_refuses_a_damaged_backup_or_a_database_in_use(
    migrated_engine: sa.Engine, fresh_database: Callable[[], str], tmp_path: Path
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    with system_scope(), migrated_engine.begin() as conn:
        workspace, _ = provision_workspace(conn, "tenant0")
    store.put_quarantined(workspace.id, b"synthetic bytes")
    backup(migrated_engine, store, tmp_path / "backup")
    with pytest.raises(BackupError) as busy:
        restore(migrated_engine, store, tmp_path / "backup")
    assert busy.value.code == "database_not_empty"
    with pytest.raises(BackupError) as not_empty:
        backup(migrated_engine, store, tmp_path / "backup")
    assert not_empty.value.code == "backup_directory_not_empty"
    damaged = (
        tmp_path
        / "backup"
        / "objects"
        / quarantine_key(workspace.id, hashlib.sha256(b"synthetic bytes").hexdigest())
    )
    damaged.write_bytes(b"tampered")
    assert verify(tmp_path / "backup")[0].startswith("object_damaged:")
    fresh = create_engine(fresh_database())
    with pytest.raises(BackupError) as refused:
        restore(fresh, store, tmp_path / "backup")
    assert refused.value.code == "object_damaged"
    fresh.dispose()


def test_objects_round_trip_through_seaweedfs(
    migrated_engine: sa.Engine,
    fresh_database: Callable[[], str],
    compose_services: Any,
    tmp_path: Path,
) -> None:
    services = compose_services("seaweedfs")
    store = artifact_store_from_settings(
        StorageSettings(
            driver="s3",
            endpoint=services.s3_endpoint(),
            access_key=services.s3_access_key(),
            secret_key=services.s3_secret_key(),
            bucket=services.s3_bucket(),
        )
    )
    with system_scope(), migrated_engine.begin() as conn:
        workspace, _ = provision_workspace(conn, f"drill-{uuid.uuid4().hex[:8]}")
    for index in range(5):
        store.put_quarantined(workspace.id, f"synthetic document {index}".encode() * 1000)
    kept = objects(store, workspace.id)
    backup(migrated_engine, store, tmp_path / "backup")
    for key in kept:
        store.delete(key)
    assert objects(store, workspace.id) == {}
    after = create_engine(fresh_database())
    report = restore(after, store, tmp_path / "backup")
    after.dispose()
    assert report.objects == 5
    assert objects(store, workspace.id) == kept
    for key in kept:
        store.delete(key)
