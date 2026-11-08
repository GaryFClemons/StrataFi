"""Eval sets and manifest imports against Postgres (integration tier).

- the Halcyon demo set imports through the API as a frozen version of 88 items; importing it
  again returns that version; a draft can be split again, then frozen, and a frozen one can't
  change
- its corpus imports through ``extract/manifest`` as a corpus version whose entries carry each
  document's key, version, collection, ACL, effective dates and status
- the same metadata arrives with an upload that carries the manifest
- a frozen version refuses changes in SQL too, except its test-split lock and read counter
"""

import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
import yaml
from fastapi.testclient import TestClient

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, SqlWorkspaceDirectory
from stratafi_core.settings import OfflineSettings
from stratafi_store import evalsets
from stratafi_store.artifacts import LocalArtifactStore
from stratafi_store.tables import eval_items, eval_set_versions
from stratafi_testing.canary import CanaryCheck
from stratafi_testing.components import stub_registry
from stratafi_testing.identity import DevIdentity

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"
HALCYON = FIXTURES / "demo-corpus"


@pytest.fixture
def client(migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path) -> TestClient:
    app = create_app(
        OfflineSettings(
            intake={"spool_dir": tmp_path / "spool", "folder_roots": {"fixtures": FIXTURES}}
        ),
        identity=DevIdentityProvider(dev_identity.verifier),
        workspaces=SqlWorkspaceDirectory(migrated_engine, auto_provision=True),
        engine=migrated_engine,
        store=LocalArtifactStore(tmp_path / "store"),
        registry=stub_registry(),
        connect=False,
    )
    return TestClient(app)


def import_halcyon(client: TestClient, headers: dict[str, str], **body: object):
    return client.post(
        "/api/v1/eval-sets/import",
        json={
            "name": "halcyon-retrieval",
            "content": (HALCYON / "retrieval.yaml").read_text(encoding="utf-8"),
            **body,
        },
        headers=headers,
    )


def test_the_halcyon_set_imports_as_a_frozen_version(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    editor = dev_identity.headers(role="editor")
    first = import_halcyon(client, editor, seed=3)
    assert first.status_code == 201, first.text
    body = first.json()
    assert body["created"] is True
    assert body["report"] == {
        "items": 88,
        "derived_tags": 88,
        "assigned_splits": 88,
        "splits": body["version"]["split_counts"],
    }
    version = body["version"]
    assert (version["version"], version["item_count"], version["frozen"]) == (1, 88, True)
    assert version["split_counts"] == {"dev": 62, "test": 26}
    assert version["test_locked"] is True
    assert version["test_reads"] == 0
    assert version["document"]["as_of"] == "2026-11-01"
    # The set names its own askers: their groups are what document ACLs are checked against.
    assert version["document"]["askers"]["manager"]["groups"] == ["all-staff", "managers"]
    assert version["split_params"] == {"test_fraction": 0.3, "seed": 3, "stratified_by": ["tags"]}

    again = import_halcyon(client, editor, seed=3)
    assert again.status_code == 200
    assert again.json()["version"]["id"] == version["id"]

    set_id = body["eval_set"]["id"]
    items = client.get(
        f"/api/v1/eval-sets/{set_id}/versions/1/items",
        params={"limit": 200},
        headers=editor,
    ).json()["items"]
    source = yaml.safe_load((HALCYON / "retrieval.yaml").read_text(encoding="utf-8"))["queries"]
    assert [item["id"] for item in items] == [query["id"] for query in source]
    assert [item["expected"] for item in items] == [query["expected"] for query in source]
    test = client.get(
        f"/api/v1/eval-sets/{set_id}/versions/1/items",
        params={"split": "test", "limit": 200},
        headers=editor,
    ).json()["items"]
    assert len(test) == 26
    tagged = client.get(
        f"/api/v1/eval-sets/{set_id}/versions/1/items",
        params={"tag": "pto-leave", "limit": 200},
        headers=editor,
    ).json()["items"]
    assert tagged
    assert all("pto-leave" in item["tags"] for item in tagged)

    listed = client.get("/api/v1/eval-sets", headers=dev_identity.headers(role="viewer")).json()
    assert [(s["name"], s["latest_version"]["version"]) for s in listed["items"]] == [
        ("halcyon-retrieval", 1)
    ]
    frozen = client.post(
        f"/api/v1/eval-sets/{set_id}/versions/1/split", json={"seed": 4}, headers=editor
    )
    assert (frozen.status_code, frozen.json()["code"]) == (409, "version_frozen")


def test_a_draft_can_be_split_again_then_frozen(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    editor = dev_identity.headers(role="editor")
    draft = import_halcyon(client, editor, freeze=False).json()
    set_id = draft["eval_set"]["id"]
    assert draft["version"]["frozen"] is False
    split = client.post(
        f"/api/v1/eval-sets/{set_id}/versions/1/split",
        json={"seed": 11, "test_fraction": 0.5},
        headers=editor,
    )
    assert split.status_code == 200, split.text
    assert split.json()["split_counts"] == {"dev": 44, "test": 44}
    assert split.json()["document_hash"] != draft["version"]["document_hash"]
    frozen = client.post(f"/api/v1/eval-sets/{set_id}/versions/1/freeze", headers=editor)
    assert frozen.json()["frozen"] is True
    second = import_halcyon(client, editor, seed=5)
    assert second.json()["version"]["version"] == 2
    versions = client.get(f"/api/v1/eval-sets/{set_id}/versions", headers=editor).json()
    assert [v["version"] for v in versions["items"]] == [2, 1]


def test_invalid_imports_name_paths_and_codes_never_text(
    client: TestClient, dev_identity: DevIdentity, canary: CanaryCheck
) -> None:
    editor = dev_identity.headers(role="editor")
    response = client.post(
        "/api/v1/eval-sets/import",
        json={
            "name": "bad",
            "content": f'{{"items": [{{"id": "q", "query": "Q", "asker": "{canary.value}"}}]}}',
        },
        headers=editor,
    )
    assert response.status_code == 400
    assert response.json()["issues"] == [
        {"path": "items.0.asker", "code": "string_pattern_mismatch"}
    ]
    canary.assert_absent(response.text)
    viewer = import_halcyon(client, dev_identity.headers(role="viewer"))
    assert viewer.status_code == 403
    canary.assert_not_logged()


def test_the_halcyon_corpus_imports_as_a_corpus_version_with_metadata(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    """Its corpus imports as a corpus version with metadata."""
    editor = dev_identity.headers(role="editor")
    corpus = client.post("/api/v1/corpora", json={"name": "halcyon"}, headers=editor).json()
    imported = client.post(
        "/api/v1/documents/import-manifest",
        json={"root": "fixtures", "path": "demo-corpus/corpus", "corpus_id": corpus["id"]},
        headers=editor,
    )
    assert imported.status_code == 201, imported.text
    assert imported.json()["added_to_corpus"] == 24
    version = client.post(f"/api/v1/corpora/{corpus['id']}/versions", headers=editor).json()
    assert version["document_count"] == 24
    entries = client.get(
        f"/api/v1/corpora/{corpus['id']}/versions/1/documents",
        params={"limit": 200},
        headers=editor,
    ).json()["items"]
    manifest = yaml.safe_load((HALCYON / "corpus" / "manifest.yaml").read_text(encoding="utf-8"))
    by_path = {document["path"]: document for document in manifest["documents"]}
    assert {entry["name"] for entry in entries} == set(by_path)
    for entry in entries:
        source = by_path[entry["name"]]
        assert entry["doc_key"] == source["doc_key"]
        assert entry["doc_version"] == source["version"]
        assert entry["collection"] == source["collection"]
        assert entry["acl"] == {"groups": source["acl"]["groups"], "users": []}
        assert entry["effective_from"] == source["effective_from"].isoformat()
        to = source.get("effective_to")
        assert entry["effective_to"] == (to.isoformat() if to else None)
        assert entry["metadata"]["status"] == source["status"]
        assert entry["metadata"]["title"] == source["title"]
    memo = next(entry for entry in entries if entry["doc_key"] == "expansion-2027")
    assert memo["acl"]["groups"] == ["executives"]
    trap = next(entry for entry in entries if entry["name"].endswith("loyalty-v4.md"))
    assert trap["effective_from"] == "2027-01-01"


def test_an_upload_can_carry_its_manifest(client: TestClient, dev_identity: DevIdentity) -> None:
    editor = dev_identity.headers(role="editor")
    corpus = client.post("/api/v1/corpora", json={"name": "two"}, headers=editor).json()
    manifest = {
        "documents": [
            {
                "path": "handbook/tips-v3.md",
                "collection": "handbook",
                "doc_key": "tips",
                "version": 3,
                "effective_from": "2025-07-01",
                "acl": {"groups": ["all-staff"]},
                "title": "Tip Policy",
            }
        ]
    }
    data = (HALCYON / "corpus" / "handbook" / "tips-v3.md").read_bytes()
    response = client.post(
        "/api/v1/documents",
        files=[
            ("file", ("handbook/tips-v3.md", data, "text/markdown")),
            ("manifest", ("manifest.yaml", yaml.safe_dump(manifest).encode(), "text/yaml")),
        ],
        params={"corpus_id": corpus["id"]},
        headers=editor,
    )
    assert response.status_code == 201, response.text
    entry = client.get(f"/api/v1/corpora/{corpus['id']}/documents", headers=editor).json()
    (item,) = entry["items"]
    assert (item["doc_key"], item["doc_version"], item["collection"]) == (
        "tips",
        "3",
        "handbook",
    )
    missing = client.post(
        "/api/v1/documents",
        files=[
            ("file", ("other.md", b"# Other\n", "text/markdown")),
            ("manifest", ("manifest.yaml", yaml.safe_dump(manifest).encode(), "text/yaml")),
        ],
        params={"corpus_id": corpus["id"]},
        headers=editor,
    )
    assert (missing.status_code, missing.json()["code"]) == (400, "manifest_file_missing")
    bad = client.post(
        "/api/v1/documents",
        files=[
            ("file", ("other.md", b"# Other\n", "text/markdown")),
            ("manifest", ("manifest.yaml", b"documents: [{path: other.md}]", "text/yaml")),
        ],
        params={"corpus_id": corpus["id"]},
        headers=editor,
    )
    assert (bad.status_code, bad.json()["code"]) == (400, "invalid_manifest")
    assert "documents.0.doc_key" in bad.json()["errors"]


def test_a_frozen_version_refuses_changes_but_counts_test_reads(
    client: TestClient, dev_identity: DevIdentity, migrated_engine: sa.Engine
) -> None:
    body = import_halcyon(client, dev_identity.headers(role="editor")).json()
    version_id = uuid.UUID(body["version"]["id"])
    with migrated_engine.begin() as conn:
        read = evalsets.record_test_read(conn, version_id)
    assert read.test_reads == 1
    assert read.last_test_read_at is not None
    for statement in (
        sa.update(eval_items)
        .where(eval_items.c.eval_set_version_id == version_id)
        .values(split="dev"),
        sa.delete(eval_items).where(eval_items.c.eval_set_version_id == version_id),
        sa.update(eval_set_versions)
        .where(eval_set_versions.c.id == version_id)
        .values(item_count=1),
        sa.delete(eval_set_versions).where(eval_set_versions.c.id == version_id),
    ):
        with (
            pytest.raises(sa.exc.DBAPIError, match="frozen eval set version"),
            migrated_engine.begin() as conn,
        ):
            conn.execute(statement)
    with migrated_engine.begin() as conn:
        workspace = conn.execute(
            sa.select(eval_set_versions.c.workspace_id).where(eval_set_versions.c.id == version_id)
        ).scalar_one()
        conn.execute(sa.text("DELETE FROM stratafi.workspaces WHERE id = :id"), {"id": workspace})
        assert conn.execute(sa.select(sa.func.count()).select_from(eval_items)).scalar_one() == 0
