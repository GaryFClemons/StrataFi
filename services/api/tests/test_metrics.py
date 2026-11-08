"""Prometheus metrics: requests by route template, a token in prod, the queue
read from Postgres when scraped, and the worker's counters. No label carries text."""

import uuid

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, StaticWorkspaces
from stratafi_core.observability import JOBS_FINISHED, PROVIDER_WAIT, STAGE_TASKS, exposition
from stratafi_core.settings import OfflineSettings
from stratafi_store.jobs import JobQueue
from stratafi_store.scope import system_scope
from stratafi_store.workspaces import provision_workspace
from stratafi_testing.canary import CanaryCheck
from stratafi_testing.identity import DevIdentity


def build(dev_identity: DevIdentity, engine=None, **settings) -> TestClient:
    app = create_app(
        OfflineSettings(**settings),
        identity=DevIdentityProvider(dev_identity.verifier),
        workspaces=StaticWorkspaces(),
        engine=engine,
        connect=False,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_requests_are_counted_by_route_template(
    dev_identity: DevIdentity, canary: CanaryCheck
) -> None:
    client = build(dev_identity)
    headers = dev_identity.headers(role="viewer")
    document_id = uuid.uuid4()
    client.get(f"/api/v1/documents/{document_id}?q={canary.value}", headers=headers)
    body = client.get("/api/v1/metrics").text
    assert 'route="/documents/{document_id}"' in body
    assert "stratafi_http_request_duration_seconds_bucket" in body
    canary.assert_absent(body)
    assert str(document_id) not in body


def test_a_metrics_token_guards_the_endpoint(dev_identity: DevIdentity) -> None:
    client = build(dev_identity, observability={"metrics_token": "scrape-secret"})
    refused = client.get("/api/v1/metrics")
    assert (refused.status_code, refused.json()["code"]) == (401, "metrics_token")
    wrong = client.get("/api/v1/metrics", headers={"Authorization": "Bearer other"})
    assert wrong.status_code == 401
    allowed = client.get("/api/v1/metrics", headers={"Authorization": "Bearer scrape-secret"})
    assert allowed.status_code == 200
    assert allowed.headers["content-type"].startswith("text/plain")
    off = build(dev_identity, observability={"metrics": False}).get("/api/v1/metrics")
    assert off.status_code == 404


def test_a_prod_deployment_needs_a_metrics_token() -> None:
    prod = OfflineSettings(env="prod", limits={"parse_confinement": "required"})
    assert any("metrics_token" in problem for problem in prod.deployment_problems())
    tokened = OfflineSettings(
        env="prod",
        limits={"parse_confinement": "required"},
        observability={"metrics_token": "x"},
    )
    assert tokened.deployment_problems() == []
    # Development needs neither.
    assert OfflineSettings().deployment_problems() == []


def test_the_workers_counters_are_in_the_registry() -> None:
    JOBS_FINISHED.labels("stage.run", "succeeded").inc()
    STAGE_TASKS.labels("parse", "succeeded").inc()
    PROVIDER_WAIT.labels("model_gateway").observe(0.0)
    body = exposition().decode()
    assert 'stratafi_jobs_finished_total{kind="stage.run",outcome="succeeded"}' in body
    assert 'stratafi_stage_tasks_total{outcome="succeeded",stage="parse"}' in body
    assert 'stratafi_provider_wait_seconds_count{provider="model_gateway"}' in body


@pytest.mark.integration
def test_the_queue_is_read_when_scraped(
    migrated_engine: sa.Engine, dev_identity: DevIdentity
) -> None:
    with system_scope(), migrated_engine.begin() as conn:
        workspace, _ = provision_workspace(conn, "tenant0")
    queue = JobQueue(migrated_engine)
    for _ in range(3):
        queue.enqueue(workspace_id=workspace.id, kind="stage.run", resource_class="cpu")
    queue.enqueue(workspace_id=workspace.id, kind="webhook.deliver", resource_class="io")
    body = build(dev_identity, migrated_engine).get("/api/v1/metrics").text
    assert 'stratafi_queue_jobs{kind="stage.run",resource_class="cpu",status="queued"} 3.0' in body
    assert 'stratafi_queue_oldest_seconds{resource_class="io"}' in body
