"""The API against Postgres and SeaweedFS (integration tier)."""

import uuid

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, SqlWorkspaceDirectory
from stratafi_core.settings import OfflineSettings, StorageSettings
from stratafi_store.artifacts import artifact_store_from_settings
from stratafi_store.audit import record_audit_event
from stratafi_store.db import create_engine
from stratafi_store.jobs import JobQueue
from stratafi_store.workspaces import grant_role, workspace_for_tenant
from stratafi_testing.identity import DevIdentity

pytestmark = pytest.mark.integration


@pytest.fixture
def store(compose_services):
    services = compose_services("seaweedfs")
    return artifact_store_from_settings(
        StorageSettings(
            driver="s3",
            endpoint=services.s3_endpoint(),
            access_key=services.s3_access_key(),
            secret_key=services.s3_secret_key(),
            bucket=services.s3_bucket(),
        )
    )


@pytest.fixture
def client(migrated_engine: sa.Engine, store, dev_identity: DevIdentity) -> TestClient:
    app = create_app(
        OfflineSettings(),
        identity=DevIdentityProvider(dev_identity.verifier),
        workspaces=SqlWorkspaceDirectory(migrated_engine, auto_provision=True),
        engine=migrated_engine,
        store=store,
        connect=False,
    )
    return TestClient(app)


def test_health_reports_the_database_and_storage(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"] == {"database": {"status": "ok"}, "storage": {"status": "ok"}}


def test_health_fails_when_the_database_is_down(store, dev_identity: DevIdentity) -> None:
    dead = create_engine("postgresql://nobody:nothing@127.0.0.1:9/none")
    app = create_app(
        OfflineSettings(),
        identity=DevIdentityProvider(dev_identity.verifier),
        engine=dead,
        store=store,
        connect=False,
    )
    response = TestClient(app).get("/api/v1/health")
    assert response.status_code == 503
    assert response.json()["checks"]["database"]["status"] == "failed"
    assert response.json()["checks"]["storage"] == {"status": "ok"}
    dead.dispose()


def test_a_first_token_provisions_its_workspace_once_and_audits_it(
    client: TestClient, migrated_engine: sa.Engine, dev_identity: DevIdentity
) -> None:
    first = client.get("/api/v1/me", headers=dev_identity.headers("halcyon", "owner"))
    second = client.get("/api/v1/me", headers=dev_identity.headers("halcyon", "viewer"))
    assert first.status_code == second.status_code == 200
    assert first.json()["workspace"] == second.json()["workspace"]
    events = client.get("/api/v1/audit-events", headers=dev_identity.headers("halcyon", "owner"))
    assert events.status_code == 200
    actions = [item["action"] for item in events.json()["items"]]
    assert actions == ["workspace.provision"]
    assert events.json()["items"][0]["actor_subject"] == "u-halcyon-owner"
    with migrated_engine.connect() as conn:
        workspace = workspace_for_tenant(conn, "halcyon")
    assert workspace is not None
    assert str(workspace.id) == first.json()["workspace"]["id"]


def test_audit_events_page_newest_first(
    client: TestClient, migrated_engine: sa.Engine, dev_identity: DevIdentity
) -> None:
    headers = dev_identity.headers("tenant0", "owner")
    workspace_id = uuid.UUID(client.get("/api/v1/me", headers=headers).json()["workspace"]["id"])
    with migrated_engine.begin() as conn:
        for number in range(7):
            record_audit_event(
                conn,
                workspace_id=workspace_id,
                actor_kind="system",
                actor_subject="test",
                action="test.event",
                details={"number": number},
            )
    seen: list[int] = []
    cursor = None
    pages = 0
    while True:
        params = {"limit": 3, **({"cursor": cursor} if cursor else {})}
        body = client.get("/api/v1/audit-events", params=params, headers=headers).json()
        pages += 1
        seen += [item["id"] for item in body["items"]]
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert pages == 3  # 8 events: the provisioning one and 7 more
    assert len(seen) == len(set(seen)) == 8
    assert seen == sorted(seen, reverse=True)


def test_a_membership_grants_a_role(
    client: TestClient, migrated_engine: sa.Engine, dev_identity: DevIdentity
) -> None:
    headers = dev_identity.headers("tenant0", "guest")
    me = client.get("/api/v1/me", headers=headers).json()
    assert me["roles"] == []
    assert client.get("/api/v1/audit-events", headers=headers).status_code == 403
    with migrated_engine.begin() as conn:
        grant_role(conn, uuid.UUID(me["workspace"]["id"]), "u-tenant0-guest", "owner")
    assert client.get("/api/v1/me", headers=headers).json()["roles"] == ["owner"]
    assert client.get("/api/v1/audit-events", headers=headers).status_code == 200


def test_jobs_can_be_listed_read_and_cancelled_by_editors(
    client: TestClient, migrated_engine: sa.Engine, dev_identity: DevIdentity
) -> None:
    editor = dev_identity.headers("tenant0", "editor")
    viewer = dev_identity.headers("tenant0", "viewer")
    workspace_id = uuid.UUID(client.get("/api/v1/me", headers=editor).json()["workspace"]["id"])
    queue = JobQueue(migrated_engine)
    made = [queue.enqueue(workspace_id=workspace_id, kind="system.selftest")[0] for _ in range(3)]

    first = client.get("/api/v1/jobs", params={"limit": 2}, headers=viewer).json()
    rest = client.get(
        "/api/v1/jobs", params={"limit": 2, "cursor": first["next_cursor"]}, headers=viewer
    ).json()
    listed = [item["id"] for item in first["items"] + rest["items"]]
    assert listed == [str(job.id) for job in reversed(made)]
    assert rest["next_cursor"] is None

    target = made[0]
    assert client.get(f"/api/v1/jobs/{target.id}", headers=viewer).json()["status"] == "queued"
    refused = client.post(f"/api/v1/jobs/{target.id}/cancel", headers=viewer)
    assert refused.status_code == 403
    cancelled = client.post(f"/api/v1/jobs/{target.id}/cancel", headers=editor)
    assert cancelled.status_code == 200
    assert cancelled.json()["outcome"] == "cancelled"
    assert cancelled.json()["job"]["status"] == "cancelled"
    again = client.post(f"/api/v1/jobs/{target.id}/cancel", headers=editor).json()
    assert again["outcome"] == "finished"

    owner = dev_identity.headers("tenant0", "owner")
    actions = [
        e["action"] for e in client.get("/api/v1/audit-events", headers=owner).json()["items"]
    ]
    assert actions.count("job.cancel") == 2


def test_jobs_stay_inside_their_workspace(
    client: TestClient, migrated_engine: sa.Engine, dev_identity: DevIdentity
) -> None:
    demo = dev_identity.headers("halcyon", "owner")
    demo_workspace = uuid.UUID(client.get("/api/v1/me", headers=demo).json()["workspace"]["id"])
    job, _ = JobQueue(migrated_engine).enqueue(workspace_id=demo_workspace, kind="system.selftest")
    other = dev_identity.headers("tenant0", "owner")
    assert client.get(f"/api/v1/jobs/{job.id}", headers=other).status_code == 404
    assert client.post(f"/api/v1/jobs/{job.id}/cancel", headers=other).status_code == 404
    assert client.get("/api/v1/jobs", headers=other).json()["items"] == []
