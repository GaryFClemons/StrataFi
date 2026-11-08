"""Server-sent events deliver job progress (integration tier).

A real HTTP server streams ``/api/v1/events`` while a worker runs a job that reports progress.
"""

import json
import threading
import uuid

import httpx2 as httpx
import pytest
import sqlalchemy as sa

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, SqlWorkspaceDirectory
from stratafi_core.settings import OfflineSettings
from stratafi_store.jobs import JobQueue
from stratafi_testing.identity import DevIdentity
from stratafi_worker.handlers import builtin_handlers
from stratafi_worker.worker import Worker

pytestmark = pytest.mark.integration


def read_events(response: httpx.Response, until_status: str, events: list[dict]) -> None:
    """Collect job events until one reaches ``until_status`` or the stream closes."""
    kind = None
    try:
        for line in response.iter_lines():
            if line.startswith("event: "):
                kind = line.removeprefix("event: ")
            elif line.startswith("data: ") and kind == "job":
                event = json.loads(line.removeprefix("data: "))
                events.append(event)
                if event["status"] == until_status:
                    return
    except (httpx.ReadError, httpx.StreamClosed):
        return  # the test closed the stream


def test_server_sent_events_deliver_progress(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, serve_app
) -> None:
    app = create_app(
        OfflineSettings(),
        identity=DevIdentityProvider(dev_identity.verifier),
        workspaces=SqlWorkspaceDirectory(migrated_engine, auto_provision=True),
        engine=migrated_engine,
        connect=False,
    )
    url = serve_app(app)
    headers = dev_identity.headers("tenant0", "viewer")
    workspace_id = uuid.UUID(
        httpx.get(f"{url}/api/v1/me", headers=headers).json()["workspace"]["id"]
    )
    queue = JobQueue(migrated_engine, lease_seconds=5)
    job, _ = queue.enqueue(
        workspace_id=workspace_id,
        kind="system.selftest",
        payload={"steps": 5, "step_seconds": 0.05},
    )
    other, _ = queue.enqueue(workspace_id=workspace_id, kind="system.selftest")
    worker = Worker(
        queue,
        builtin_handlers(),
        slots={"cpu": 1},
        poll_seconds=0.05,
        heartbeat_seconds=0.5,
        progress_interval=0.0,
    )
    events: list[dict] = []
    with httpx.stream(
        "GET",
        f"{url}/api/v1/events",
        params={"job_id": str(job.id)},
        headers=headers,
        timeout=30,
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        reader = threading.Thread(target=read_events, args=(response, "succeeded", events))
        reader.start()
        assert app.state.events.listening.wait(10), "the hub didn't start listening"
        worker.start()
        reader.join(30)
        worker.stop(grace_seconds=1.0)
    assert events, "no events arrived"
    assert {event["job_id"] for event in events} == {str(job.id)}, "the job_id filter holds"
    progress = [event["progress"]["done"] for event in events if "progress" in event]
    assert progress[:5] == [1, 2, 3, 4, 5]
    assert events[-1]["status"] == "succeeded"
    assert all(
        set(event)
        <= {
            "type",
            "workspace_id",
            "job_id",
            "kind",
            "status",
            "attempts",
            "progress",
            "error_code",
        }
        for event in events
    )
    assert other.id != job.id


def test_events_need_a_viewer_and_stay_in_the_workspace(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, serve_app
) -> None:
    app = create_app(
        OfflineSettings(),
        identity=DevIdentityProvider(dev_identity.verifier),
        workspaces=SqlWorkspaceDirectory(migrated_engine, auto_provision=True),
        engine=migrated_engine,
        connect=False,
    )
    url = serve_app(app)
    guest = httpx.get(f"{url}/api/v1/events", headers=dev_identity.headers("tenant0", "guest"))
    assert guest.status_code == 403
    tenant0 = dev_identity.headers("tenant0", "viewer")
    demo = dev_identity.headers("halcyon", "owner")
    demo_workspace = uuid.UUID(
        httpx.get(f"{url}/api/v1/me", headers=demo).json()["workspace"]["id"]
    )
    queue = JobQueue(migrated_engine)
    events: list[dict] = []
    with httpx.stream("GET", f"{url}/api/v1/events", headers=tenant0, timeout=10) as response:
        reader = threading.Thread(
            target=read_events, args=(response, "queued", events), daemon=True
        )
        reader.start()
        assert app.state.events.listening.wait(10)
        queue.enqueue(workspace_id=demo_workspace, kind="system.selftest")
        reader.join(2)
    assert events == [], "another workspace's events never arrive"
