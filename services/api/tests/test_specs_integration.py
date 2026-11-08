"""Spec versions and the registry snapshot against Postgres (integration tier)."""

import uuid

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, SqlWorkspaceDirectory
from stratafi_core.settings import OfflineSettings
from stratafi_store.components import list_component_rows, sync_components
from stratafi_store.specs import get_version, list_latest
from stratafi_store.tables import corpora, corpus_versions, workspaces
from stratafi_testing.components import stub, stub_registry
from stratafi_testing.identity import DevIdentity

pytestmark = pytest.mark.integration


@pytest.fixture
def client(migrated_engine: sa.Engine, dev_identity: DevIdentity) -> TestClient:
    app = create_app(
        OfflineSettings(),
        identity=DevIdentityProvider(dev_identity.verifier),
        workspaces=SqlWorkspaceDirectory(migrated_engine, auto_provision=True),
        engine=migrated_engine,
        registry=stub_registry(),
        connect=False,
    )
    return TestClient(app)


def spec(**stages: object) -> dict:
    return {"schema": "stratafi.pipeline.v1", "name": "lab", "stages": stages}


def test_spec_versions_persist(
    client: TestClient, migrated_engine: sa.Engine, dev_identity: DevIdentity
) -> None:
    editor = dev_identity.headers(role="editor")
    first = client.post("/api/v1/pipeline-specs", json=spec(), headers=editor)
    assert first.status_code == 201, first.text
    assert first.json()["version"] == 1
    # The default chunker's own default: the same configuration, so no new version.
    same = client.post(
        "/api/v1/pipeline-specs",
        json=spec(chunk={"params": {"max_tokens": 400}}),
        headers=editor,
    )
    assert same.status_code == 200
    assert same.json()["id"] == first.json()["id"]
    second = client.post(
        "/api/v1/pipeline-specs", json=spec(chunk={"params": {"max_tokens": 600}}), headers=editor
    )
    assert second.status_code == 201
    assert second.json()["version"] == 2
    assert second.json()["spec_hash"] != first.json()["spec_hash"]

    # A fresh connection reads what was written: the versions persisted.
    workspace_id = uuid.UUID(client.get("/api/v1/me", headers=editor).json()["workspace"]["id"])
    with migrated_engine.connect() as conn:
        stored = get_version(conn, "pipeline", workspace_id, "lab", 1)
        assert stored is not None
        assert stored.document["stages"]["chunk"]["params"]["max_tokens"] == 400
        assert [s.version for s in list_latest(conn, "pipeline", workspace_id, limit=10)] == [2]

    viewer = dev_identity.headers(role="viewer")
    latest = client.get("/api/v1/pipeline-specs/lab", headers=viewer).json()
    assert latest["version"] == 2
    old = client.get("/api/v1/pipeline-specs/lab/versions/1", headers=viewer).json()
    assert old["document"]["stages"]["chunk"]["params"]["max_tokens"] == 400
    versions = client.get("/api/v1/pipeline-specs/lab/versions", headers=viewer).json()
    assert [item["version"] for item in versions["items"]] == [2, 1]
    paged = client.get(
        "/api/v1/pipeline-specs/lab/versions", params={"limit": 1}, headers=viewer
    ).json()
    assert [item["version"] for item in paged["items"]] == [2]
    rest = client.get(
        "/api/v1/pipeline-specs/lab/versions",
        params={"limit": 1, "cursor": paged["next_cursor"]},
        headers=viewer,
    ).json()
    assert [item["version"] for item in rest["items"]] == [1]
    assert rest["next_cursor"] is None

    diff = client.get(
        "/api/v1/pipeline-specs/lab/diff",
        params={"from_version": 1, "to_version": 2},
        headers=viewer,
    ).json()
    assert diff["changes"] == [
        {"path": "stages.chunk.params.max_tokens", "op": "changed", "before": 400, "after": 600}
    ]
    assert client.get("/api/v1/pipeline-specs/lab/versions/9", headers=viewer).status_code == 404

    events = client.get("/api/v1/audit-events", headers=dev_identity.headers(role="owner")).json()
    saves = [item for item in events["items"] if item["action"] == "pipeline_spec.save"]
    assert [item["details"] for item in saves] == [{"version": 2}, {"version": 1}]


def test_specs_stay_inside_their_workspace(client: TestClient, dev_identity: DevIdentity) -> None:
    client.post(
        "/api/v1/pipeline-specs", json=spec(), headers=dev_identity.headers("tenant0", "editor")
    )
    other = client.get("/api/v1/pipeline-specs", headers=dev_identity.headers("halcyon", "owner"))
    assert other.json()["items"] == []
    assert (
        client.get(
            "/api/v1/pipeline-specs/lab", headers=dev_identity.headers("halcyon", "owner")
        ).status_code
        == 404
    )


def test_names_list_their_latest_versions_in_pages(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    editor = dev_identity.headers(role="editor")
    for name in ("c-spec", "a-spec", "b-spec"):
        client.post("/api/v1/pipeline-specs", json={**spec(), "name": name}, headers=editor)
    client.post(
        "/api/v1/pipeline-specs",
        json={**spec(clean=[]), "name": "a-spec"},
        headers=editor,
    )
    first = client.get("/api/v1/pipeline-specs", params={"limit": 2}, headers=editor).json()
    assert [(item["name"], item["version"]) for item in first["items"]] == [
        ("a-spec", 2),
        ("b-spec", 1),
    ]
    rest = client.get(
        "/api/v1/pipeline-specs", params={"cursor": first["next_cursor"]}, headers=editor
    ).json()
    assert [item["name"] for item in rest["items"]] == ["c-spec"]


def test_retrieval_configs_version_the_same_way(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    editor = dev_identity.headers(role="editor")
    config = {"schema": "stratafi.retrieval.v1", "name": "hybrid"}
    assert client.post("/api/v1/retrieval-configs", json=config, headers=editor).status_code == 201
    assert client.post("/api/v1/retrieval-configs", json=config, headers=editor).status_code == 200
    changed = client.post("/api/v1/retrieval-configs", json={**config, "top_k": 5}, headers=editor)
    assert changed.json()["version"] == 2
    assert changed.json()["document"]["top_k"] == 5


def test_the_registry_snapshot_keeps_retired_versions(migrated_engine: sa.Engine) -> None:
    registry = stub_registry()
    with migrated_engine.begin() as conn:
        assert sync_components(conn, registry.descriptions()) == len(registry)
    newer = [d for d in registry.descriptions() if d.info.id != "clean/unicode"]
    bumped = stub("clean/unicode", "clean", version="1.1.0")
    with migrated_engine.begin() as conn:
        sync_components(conn, [*newer, *type(registry)([bumped]).descriptions()])
        rows = {(row.component_id, row.version): row for row in list_component_rows(conn)}
    assert rows[("clean/unicode", "1.0.0")].available is False
    assert rows[("clean/unicode", "1.1.0")].available is True
    assert rows[("chunk/sections", "1.0.0")].params_schema["properties"]["max_tokens"]


def test_the_app_records_its_registry_when_it_starts(
    migrated_engine: sa.Engine, dev_identity: DevIdentity
) -> None:
    app = create_app(
        OfflineSettings(),
        identity=DevIdentityProvider(dev_identity.verifier),
        engine=migrated_engine,
        registry=stub_registry(),
    )
    with TestClient(app):
        pass
    with migrated_engine.connect() as conn:
        assert len(list_component_rows(conn, available_only=True)) == len(stub_registry())


def test_corpus_versions_belong_to_a_corpus_in_their_workspace(migrated_engine: sa.Engine) -> None:
    one, two = uuid.uuid4(), uuid.uuid4()
    corpus = uuid.uuid4()
    with migrated_engine.begin() as conn:
        conn.execute(
            sa.insert(workspaces),
            [{"id": one, "slug": "one", "name": "one"}, {"id": two, "slug": "two", "name": "two"}],
        )
        conn.execute(
            sa.insert(corpora).values(id=corpus, workspace_id=one, name="policies", created_by="u")
        )
    version = {
        "id": uuid.uuid4(),
        "corpus_id": corpus,
        "version": 1,
        "manifest_hash": "0" * 64,
        "document_count": 0,
        "total_bytes": 0,
        "created_by": "u",
    }
    with pytest.raises(sa.exc.IntegrityError), migrated_engine.begin() as conn:
        conn.execute(sa.insert(corpus_versions).values(**version, workspace_id=two))
    with migrated_engine.begin() as conn:
        conn.execute(sa.insert(corpus_versions).values(**version, workspace_id=one))
    with migrated_engine.begin() as conn:
        conn.execute(sa.delete(workspaces).where(workspaces.c.id == one))
        assert conn.execute(sa.select(sa.func.count()).select_from(corpus_versions)).scalar() == 0
