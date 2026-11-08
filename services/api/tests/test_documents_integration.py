"""Intake and corpora against Postgres (integration tier): uploads deduplicate,
archives unpack with provenance, a 200 MB upload streams, and a frozen version can't change.

The artifact store is the local driver; packages/store/tests/test_artifacts.py covers the S3
driver's streaming."""

import hashlib
import tracemalloc
import uuid
from pathlib import Path

import httpx2
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, SqlWorkspaceDirectory
from stratafi_core.settings import OfflineSettings
from stratafi_store.artifacts import LocalArtifactStore, quarantine_key
from stratafi_store.lineage import get_artifact
from stratafi_store.tables import corpus_version_documents, corpus_versions
from stratafi_testing.canary import CanaryCheck
from stratafi_testing.components import stub_registry
from stratafi_testing.documents import Block, docx_bytes, pdf_bytes, zip_bytes
from stratafi_testing.identity import DevIdentity

pytestmark = pytest.mark.integration

MiB = 1024 * 1024


def build(engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, **settings: object):
    store = LocalArtifactStore(tmp_path / "store")
    app = create_app(
        OfflineSettings(**{"intake": {"spool_dir": tmp_path / "spool"}, **settings}),
        identity=DevIdentityProvider(dev_identity.verifier),
        workspaces=SqlWorkspaceDirectory(engine, auto_provision=True),
        engine=engine,
        store=store,
        registry=stub_registry(),
        connect=False,
    )
    return app, store


@pytest.fixture
def lab(migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path):
    app, store = build(migrated_engine, dev_identity, tmp_path)
    return TestClient(app), store


def upload(client: TestClient, headers: dict, files: list[tuple[str, bytes]], **params: str):
    return client.post(
        "/api/v1/documents",
        files=[("file", (name, data, "application/octet-stream")) for name, data in files],
        params=params,
        headers=headers,
    )


def test_uploads_are_stored_in_quarantine_and_deduplicated(lab, dev_identity: DevIdentity) -> None:
    client, store = lab
    editor = dev_identity.headers(role="editor")
    pdf = pdf_bytes("Retention", [Block("paragraph", "Keep records for seven years.")])
    first = upload(client, editor, [("retention.pdf", pdf), ("notes.md", b"# Notes\n")])
    assert first.status_code == 201, first.text
    body = first.json()
    assert body["counts"] == {"files": 2, "documents": 2, "new": 2, "deduplicated": 0, "refused": 0}
    document = body["files"][0]["document"]
    assert document["media_type"] == "application/pdf"
    assert document["traits"] == {"pages": 1, "scanned": False, "partly_scanned": False}
    assert document["sha256"] == hashlib.sha256(pdf).hexdigest()
    workspace = uuid.UUID(client.get("/api/v1/me", headers=editor).json()["workspace"]["id"])
    assert store.get(quarantine_key(workspace, document["sha256"])) == pdf

    again = upload(client, editor, [("copy-of-retention.pdf", pdf)])
    assert again.json()["files"][0]["deduplicated"] is True
    assert again.json()["files"][0]["document"]["id"] == document["id"]
    detail = client.get(f"/api/v1/documents/{document['id']}", headers=editor).json()
    assert [origin["name"] for origin in detail["origins"]] == [
        "retention.pdf",
        "copy-of-retention.pdf",
    ]
    listed = client.get("/api/v1/documents", headers=editor).json()
    assert len(listed["items"]) == 2


def test_raw_documents_are_root_artifacts(lab, migrated_engine, dev_identity: DevIdentity) -> None:
    client, _ = lab
    editor = dev_identity.headers(role="editor")
    document = upload(client, editor, [("a.md", b"# A\n")]).json()["files"][0]["document"]
    workspace = uuid.UUID(client.get("/api/v1/me", headers=editor).json()["workspace"]["id"])
    with migrated_engine.connect() as conn:
        artifact = get_artifact(conn, workspace, document["sha256"])
    assert artifact is not None
    assert (artifact.kind, artifact.size_bytes) == ("document", 4)


def test_archives_unpack_into_members_with_provenance(lab, dev_identity: DevIdentity) -> None:
    client, _ = lab
    editor = dev_identity.headers(role="editor")
    docx = docx_bytes("Fees", [Block("paragraph", "Fees apply.")])
    archive = zip_bytes({"policies/fees.docx": docx, "../escape.txt": b"labels only"})
    corpus = client.post("/api/v1/corpora", json={"name": "bundle"}, headers=editor).json()
    result = upload(client, editor, [("bundle.zip", archive)], corpus_id=corpus["id"]).json()
    (item,) = result["files"]
    assert item["refused"] is None
    assert [member["name"] for member in item["members"]] == [
        "bundle.zip/policies/fees.docx",
        "bundle.zip/escape.txt",
    ]
    assert result["added_to_corpus"] == 2
    member = item["members"][0]["document"]
    origins = client.get(f"/api/v1/documents/{member['id']}", headers=editor).json()["origins"]
    assert origins[0]["parent_document_id"] == item["document"]["id"]
    assert origins[0]["member_path"] == "policies/fees.docx"
    entries = client.get(f"/api/v1/corpora/{corpus['id']}/documents", headers=editor).json()
    assert [entry["name"] for entry in entries["items"]] == [
        "bundle.zip/escape.txt",
        "bundle.zip/policies/fees.docx",
    ]


def test_refused_archives_are_recorded_with_their_reason(lab, dev_identity: DevIdentity) -> None:
    client, _ = lab
    editor = dev_identity.headers(role="editor")
    bomb = zip_bytes({"zeros.bin": b"\0" * (16 * MiB)})
    nested = zip_bytes({"inner.zip": zip_bytes({"a.md": b"# a"})})
    corpus = client.post("/api/v1/corpora", json={"name": "risky"}, headers=editor).json()
    result = upload(
        client, editor, [("bomb.zip", bomb), ("nested.zip", nested)], corpus_id=corpus["id"]
    ).json()
    assert [item["refused"] for item in result["files"]] == [
        {"outcome": "limit_exceeded", "reason": "archive_expansion_limit"},
        {"outcome": "limit_exceeded", "reason": "archive_nested"},
    ]
    assert [item["document"]["status"] for item in result["files"]] == ["refused", "refused"]
    assert result["counts"]["refused"] == 2
    # The refused archives join the corpus, so the scan can show why they're excluded.
    entries = client.get(f"/api/v1/corpora/{corpus['id']}/documents", headers=editor).json()
    assert [(e["name"], e["reason_code"]) for e in entries["items"]] == [
        ("bomb.zip", "archive_expansion_limit"),
        ("nested.zip", "archive_nested"),
    ]


def test_a_frozen_version_cannot_change(lab, migrated_engine, dev_identity: DevIdentity) -> None:
    client, _ = lab
    editor = dev_identity.headers(role="editor")
    corpus = client.post("/api/v1/corpora", json={"name": "policies"}, headers=editor).json()
    upload(client, editor, [("a.md", b"# A\n"), ("b.md", b"# B\n")], corpus_id=corpus["id"])
    first = client.post(f"/api/v1/corpora/{corpus['id']}/versions", headers=editor)
    assert first.status_code == 201
    assert (first.json()["version"], first.json()["document_count"]) == (1, 2)
    unchanged = client.post(f"/api/v1/corpora/{corpus['id']}/versions", headers=editor)
    assert unchanged.status_code == 200
    assert unchanged.json()["id"] == first.json()["id"]

    # Editing the working set leaves version 1 as it was.
    client.post(
        f"/api/v1/corpora/{corpus['id']}/documents/remove", json={"names": ["a.md"]}, headers=editor
    )
    upload(client, editor, [("c.md", b"# C\n")], corpus_id=corpus["id"])
    second = client.post(f"/api/v1/corpora/{corpus['id']}/versions", headers=editor).json()
    assert second["version"] == 2
    assert second["manifest_hash"] != first.json()["manifest_hash"]
    v1 = client.get(f"/api/v1/corpora/{corpus['id']}/versions/1/documents", headers=editor).json()
    assert [entry["name"] for entry in v1["items"]] == ["a.md", "b.md"]
    v2 = client.get(f"/api/v1/corpora/{corpus['id']}/versions/2/documents", headers=editor).json()
    assert [entry["name"] for entry in v2["items"]] == ["b.md", "c.md"]

    # The database itself refuses every change to a sealed version.
    version_id = uuid.UUID(first.json()["id"])
    attempts = [
        sa.update(corpus_versions)
        .where(corpus_versions.c.id == version_id)
        .values(document_count=9),
        sa.delete(corpus_versions).where(corpus_versions.c.id == version_id),
        sa.update(corpus_version_documents)
        .where(corpus_version_documents.c.corpus_version_id == version_id)
        .values(size_bytes=0),
        sa.delete(corpus_version_documents).where(
            corpus_version_documents.c.corpus_version_id == version_id
        ),
    ]
    for attempt in attempts:
        with pytest.raises(sa.exc.IntegrityError, match="frozen"), migrated_engine.begin() as conn:
            conn.execute(attempt)
    with migrated_engine.begin() as conn:
        row = conn.execute(
            sa.select(corpus_version_documents).where(
                corpus_version_documents.c.corpus_version_id == version_id
            )
        ).first()
        assert row is not None
        values = {**row._asdict(), "name": "smuggled.md"}
    with pytest.raises(sa.exc.IntegrityError, match="frozen"), migrated_engine.begin() as conn:
        conn.execute(sa.insert(corpus_version_documents).values(**values))
    after = client.get(f"/api/v1/corpora/{corpus['id']}/versions/1", headers=editor).json()
    assert after == first.json()


def test_deleting_a_corpus_takes_its_frozen_versions(lab, migrated_engine, dev_identity) -> None:
    client, _ = lab
    editor = dev_identity.headers(role="editor")
    corpus = client.post("/api/v1/corpora", json={"name": "gone"}, headers=editor).json()
    upload(client, editor, [("a.md", b"# A\n")], corpus_id=corpus["id"])
    client.post(f"/api/v1/corpora/{corpus['id']}/versions", headers=editor)
    from stratafi_store.tables import corpora

    with migrated_engine.begin() as conn:
        conn.execute(sa.delete(corpora).where(corpora.c.id == uuid.UUID(corpus["id"])))
        assert conn.execute(sa.select(sa.func.count()).select_from(corpus_versions)).scalar() == 0


def test_the_working_set_shows_duplicate_groups(lab, dev_identity: DevIdentity) -> None:
    client, _ = lab
    editor = dev_identity.headers(role="editor")
    corpus = client.post("/api/v1/corpora", json={"name": "dupes"}, headers=editor).json()
    result = upload(client, editor, [("a.md", b"same"), ("b.md", b"other")], corpus_id=corpus["id"])
    same = result.json()["files"][0]["document"]["id"]
    added = client.post(
        f"/api/v1/corpora/{corpus['id']}/documents",
        json={"entries": [{"document_id": same, "name": "copies/a-again.md"}]},
        headers=editor,
    )
    assert added.json() == {"count": 1}
    detail = client.get(f"/api/v1/corpora/{corpus['id']}", headers=editor).json()
    assert detail["entries"] == 3
    assert detail["duplicate_groups"] == [["a.md", "copies/a-again.md"]]
    missing = client.post(
        f"/api/v1/corpora/{corpus['id']}/documents",
        json={"entries": [{"document_id": str(uuid.uuid4())}]},
        headers=editor,
    )
    assert missing.status_code == 404
    assert missing.json()["errors"] == ["entries.0.document_id"]


def test_corpus_names_are_unique_per_workspace(lab, dev_identity: DevIdentity) -> None:
    client, _ = lab
    editor = dev_identity.headers(role="editor")
    assert client.post("/api/v1/corpora", json={"name": "one"}, headers=editor).status_code == 201
    taken = client.post("/api/v1/corpora", json={"name": "one"}, headers=editor)
    assert taken.status_code == 409
    other = dev_identity.headers("halcyon", "editor")
    assert client.post("/api/v1/corpora", json={"name": "one"}, headers=other).status_code == 201
    assert len(client.get("/api/v1/corpora", headers=other).json()["items"]) == 1


def test_folder_imports_read_only_named_roots(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path
) -> None:
    root = tmp_path / "fixtures"
    (root / "corpus").mkdir(parents=True)
    (root / "corpus" / "a.md").write_bytes(b"# A\n")
    (root / "corpus" / "b.txt").write_bytes(b"B\n")
    app, _ = build(
        migrated_engine,
        dev_identity,
        tmp_path,
        intake={"folder_roots": {"fixtures": root}, "spool_dir": tmp_path / "spool"},
    )
    client = TestClient(app)
    editor = dev_identity.headers(role="editor")
    corpus = client.post("/api/v1/corpora", json={"name": "fixtures"}, headers=editor).json()
    imported = client.post(
        "/api/v1/documents/import-folder",
        json={"root": "fixtures", "path": "corpus", "corpus_id": corpus["id"]},
        headers=editor,
    )
    assert imported.status_code == 201, imported.text
    assert [item["name"] for item in imported.json()["files"]] == ["a.md", "b.txt"]
    assert imported.json()["added_to_corpus"] == 2
    unknown = client.post("/api/v1/documents/import-folder", json={"root": "etc"}, headers=editor)
    assert (unknown.status_code, unknown.json()["code"]) == (404, "unknown_root")
    escape = client.post(
        "/api/v1/documents/import-folder",
        json={"root": "fixtures", "path": "../.."},
        headers=editor,
    )
    assert (escape.status_code, escape.json()["code"]) == (403, "path_outside_root")


def test_uploads_need_the_editor_role_and_log_no_names(
    lab, dev_identity: DevIdentity, canary: CanaryCheck
) -> None:
    client, _ = lab
    viewer = upload(client, dev_identity.headers(role="viewer"), [("a.md", b"# a")])
    assert viewer.status_code == 403
    response = upload(
        client, dev_identity.headers(role="editor"), [(f"{canary.value}.md", canary.value.encode())]
    )
    assert response.status_code == 201
    canary.assert_absent(canary.captured_text())
    events = client.get("/api/v1/audit-events", headers=dev_identity.headers(role="owner")).text
    canary.assert_absent(events)


def test_a_200_mb_upload_streams_without_being_loaded_into_memory(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app
) -> None:
    app, store = build(migrated_engine, dev_identity, tmp_path)
    url = serve_app(app)
    boundary = "stratafi-big-upload"
    piece = b"stratafi streams uploads to disk. " * 30_841  # just over 1 MiB
    pieces = 200
    expected = hashlib.sha256()
    for _ in range(pieces):
        expected.update(piece)

    def body():
        yield (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n"
        ).encode()
        for _ in range(pieces):
            yield piece
        yield f"\r\n--{boundary}--\r\n".encode()

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        **dev_identity.headers(role="editor"),
    }
    tracemalloc.start()
    try:
        response = httpx2.post(
            f"{url}/api/v1/documents", content=body(), headers=headers, timeout=600
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert response.status_code == 201, response.text
    document = response.json()["files"][0]["document"]
    assert document["size_bytes"] == len(piece) * pieces > 200 * MiB
    assert document["sha256"] == expected.hexdigest()
    # The whole process (server and client) peaked far below the file's size.
    print(f"200 MB upload: peak traced memory {peak / MiB:.1f} MiB")
    assert peak < 32 * MiB, f"peak traced memory was {peak / MiB:.1f} MiB"
    workspace = uuid.UUID(httpx2.get(f"{url}/api/v1/me", headers=headers).json()["workspace"]["id"])
    assert store.exists(quarantine_key(workspace, document["sha256"]))
    assert list((tmp_path / "spool").glob("upload-*")) == []
