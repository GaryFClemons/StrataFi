"""The job queue against Postgres (integration tier).

Leases and backoff are short here, so expiry and retries happen within the test.
"""

import threading
import time
import uuid
from datetime import timedelta

import pytest
import sqlalchemy as sa

from stratafi_store.db import libpq_url
from stratafi_store.events import EventListener
from stratafi_store.jobs import CancelOutcome, JobQueue
from stratafi_store.tables import jobs, workspaces

pytestmark = pytest.mark.integration


@pytest.fixture
def workspace(migrated_engine: sa.Engine) -> uuid.UUID:
    workspace_id = uuid.uuid4()
    with migrated_engine.begin() as conn:
        conn.execute(sa.insert(workspaces).values(id=workspace_id, slug="jobs", name="jobs"))
    return workspace_id


@pytest.fixture
def queue(migrated_engine: sa.Engine) -> JobQueue:
    return JobQueue(
        migrated_engine,
        lease_seconds=1,
        backoff_base_seconds=0.2,
        backoff_max_seconds=1.0,
        jitter=False,
    )


def test_a_dedupe_key_is_unique_among_active_jobs(queue: JobQueue, workspace: uuid.UUID) -> None:
    first, created = queue.enqueue(workspace_id=workspace, kind="t.job", dedupe_key="parse:doc-1")
    again, created_again = queue.enqueue(
        workspace_id=workspace, kind="t.job", dedupe_key="parse:doc-1"
    )
    other, _ = queue.enqueue(workspace_id=workspace, kind="t.job", dedupe_key="parse:doc-2")
    assert (created, created_again) == (True, False)
    assert again.id == first.id
    assert other.id != first.id
    (claimed,) = [job for job in queue.claim("w1", "cpu", limit=5) if job.id == first.id]
    assert queue.complete(claimed.id, "w1", {"chunks": 3})
    after, created_after = queue.enqueue(
        workspace_id=workspace, kind="t.job", dedupe_key="parse:doc-1"
    )
    assert created_after is True
    assert after.id != first.id


def test_claims_take_the_highest_priority_of_their_class(
    queue: JobQueue, workspace: uuid.UUID
) -> None:
    low, _ = queue.enqueue(workspace_id=workspace, kind="t.job", priority=0)
    high, _ = queue.enqueue(workspace_id=workspace, kind="t.job", priority=5)
    io_job, _ = queue.enqueue(workspace_id=workspace, kind="t.job", resource_class="io")
    claimed = queue.claim("w1", "cpu", limit=10)
    assert [job.id for job in claimed] == [high.id, low.id]
    assert all(job.status == "running" and job.attempts == 1 for job in claimed)
    assert all(job.lease_owner == "w1" and job.lease_expires_at for job in claimed)
    assert queue.claim("w2", "cpu") == []
    assert [job.id for job in queue.claim("w2", "io")] == [io_job.id]
    with pytest.raises(ValueError, match="resource class"):
        queue.enqueue(workspace_id=workspace, kind="t.job", resource_class="tpu")


def test_concurrent_claims_never_share_a_job(queue: JobQueue, workspace: uuid.UUID) -> None:
    for _ in range(40):
        queue.enqueue(workspace_id=workspace, kind="t.job")
    claimed: list[uuid.UUID] = []
    lock = threading.Lock()

    def claim_all(owner: str) -> None:
        while batch := queue.claim(owner, "cpu", limit=3):
            with lock:
                claimed.extend(job.id for job in batch)

    threads = [threading.Thread(target=claim_all, args=(f"w{n}",)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(claimed) == len(set(claimed)) == 40


def test_an_expired_lease_is_reclaimed(queue: JobQueue, workspace: uuid.UUID) -> None:
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.job")
    assert len(queue.claim("dead-worker", "cpu")) == 1
    assert queue.reclaim_expired() == 0, "the lease is still valid"
    time.sleep(1.2)
    assert queue.reclaim_expired() == 1
    (again,) = queue.claim("live-worker", "cpu")
    assert again.id == job.id
    assert again.attempts == 2
    assert not queue.complete(job.id, "dead-worker"), "the old owner lost its lease"
    assert not queue.heartbeat(job.id, "dead-worker").alive
    assert queue.complete(job.id, "live-worker")


def test_heartbeats_keep_a_lease(queue: JobQueue, workspace: uuid.UUID) -> None:
    queue.enqueue(workspace_id=workspace, kind="t.job")
    (job,) = queue.claim("w1", "cpu")
    for _ in range(4):
        time.sleep(0.4)
        assert queue.heartbeat(job.id, "w1").alive
    assert queue.reclaim_expired() == 0


def test_retries_back_off_and_stop_at_max_attempts(queue: JobQueue, workspace: uuid.UUID) -> None:
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.job", max_attempts=3)
    delays = []
    for attempt in range(1, 4):
        deadline = time.monotonic() + 5
        while not (claimed := queue.claim("w1", "cpu")):
            assert time.monotonic() < deadline, "the retry never came due"
            time.sleep(0.02)
        assert claimed[0].attempts == attempt
        failed = queue.fail(job.id, "w1", "timeout")
        assert failed is not None
        if attempt < 3:
            assert failed.status == "queued"
            delays.append(failed.run_after - failed.updated_at)
        else:
            assert failed.status == "failed"
            assert failed.error_code == "timeout"
            assert failed.finished_at is not None
    assert [round(d / timedelta(seconds=1), 1) for d in delays] == [0.2, 0.4]
    assert queue.claim("w1", "cpu") == []


def test_backoff_doubles_up_to_its_cap_with_equal_jitter(migrated_engine: sa.Engine) -> None:
    exact = JobQueue(migrated_engine, backoff_base_seconds=5, backoff_max_seconds=30, jitter=False)
    assert [exact.backoff_seconds(n) for n in range(1, 6)] == [5, 10, 20, 30, 30]
    low = JobQueue(migrated_engine, backoff_base_seconds=8, random_fraction=lambda: 0.0)
    high = JobQueue(migrated_engine, backoff_base_seconds=8, random_fraction=lambda: 1.0)
    assert (low.backoff_seconds(2), high.backoff_seconds(2)) == (8, 16)


def test_a_permanent_failure_does_not_retry(queue: JobQueue, workspace: uuid.UUID) -> None:
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.job", max_attempts=5)
    queue.claim("w1", "cpu")
    failed = queue.fail(job.id, "w1", "invalid_payload", retryable=False)
    assert failed is not None
    assert (failed.status, failed.attempts) == ("failed", 1)


def test_cancelling_a_queued_job_stops_it_running(queue: JobQueue, workspace: uuid.UUID) -> None:
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.job")
    result = queue.cancel(job.id, workspace)
    assert result is not None
    cancelled, outcome = result
    assert (outcome, cancelled.status) == (CancelOutcome.CANCELLED, "cancelled")
    assert queue.claim("w1", "cpu") == []
    again = queue.cancel(job.id, workspace)
    assert again is not None
    assert again[1] == CancelOutcome.FINISHED


def test_cancelling_a_running_job_asks_it_to_stop(queue: JobQueue, workspace: uuid.UUID) -> None:
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.job")
    queue.claim("w1", "cpu")
    result = queue.cancel(job.id, workspace)
    assert result is not None
    running, outcome = result
    assert (outcome, running.status, running.cancel_requested) == (
        CancelOutcome.REQUESTED,
        "running",
        True,
    )
    beat = queue.heartbeat(job.id, "w1")
    assert (beat.alive, beat.cancel_requested) == (True, True)
    assert queue.mark_cancelled(job.id, "w1")
    final = queue.get(job.id)
    assert final is not None
    assert (final.status, final.lease_owner) == ("cancelled", None)


def test_a_cancelled_job_whose_worker_died_ends_cancelled(
    queue: JobQueue, workspace: uuid.UUID
) -> None:
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.job")
    queue.claim("dead-worker", "cpu")
    queue.cancel(job.id, workspace)
    time.sleep(1.2)
    assert queue.reclaim_expired() == 1
    final = queue.get(job.id)
    assert final is not None
    assert final.status == "cancelled"


def test_cancel_is_scoped_to_the_workspace(queue: JobQueue, workspace: uuid.UUID) -> None:
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.job")
    assert queue.cancel(job.id, uuid.uuid4()) is None
    assert queue.get(job.id, uuid.uuid4()) is None
    assert queue.get(job.id, workspace) is not None


def test_release_hands_a_job_back_without_using_an_attempt(
    queue: JobQueue, workspace: uuid.UUID
) -> None:
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.job")
    queue.claim("w1", "cpu")
    assert queue.release(job.id, "w1")
    (again,) = queue.claim("w2", "cpu")
    assert again.attempts == 1


def test_progress_is_metadata_only(queue: JobQueue, workspace: uuid.UUID) -> None:
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.job")
    queue.claim("w1", "cpu")
    assert queue.report_progress(job.id, "w1", {"done": 2, "total": 5})
    with pytest.raises(ValueError, match="text"):
        queue.report_progress(job.id, "w1", {"text": "a chunk"})
    assert not queue.report_progress(job.id, "someone-else", {"done": 3})
    current = queue.get(job.id)
    assert current is not None
    assert current.progress == {"done": 2, "total": 5}


def test_listing_pages_newest_first(queue: JobQueue, workspace: uuid.UUID) -> None:
    created = [queue.enqueue(workspace_id=workspace, kind="t.job")[0] for _ in range(5)]
    first = queue.list_jobs(workspace, limit=3)
    rest = queue.list_jobs(workspace, limit=3, after=(first[-1].created_at, first[-1].id))
    assert [job.id for job in first + rest] == [job.id for job in reversed(created)]
    assert queue.list_jobs(workspace, limit=10, status="running") == []


def test_every_change_is_announced_after_commit(
    queue: JobQueue, workspace: uuid.UUID, migrated_engine: sa.Engine
) -> None:
    events: list[dict] = []
    listener = EventListener(libpq_url(migrated_engine), events.append, poll_seconds=0.1)
    listener.start()
    try:
        assert listener.listening.wait(5)
        job, _ = queue.enqueue(workspace_id=workspace, kind="t.job")
        queue.claim("w1", "cpu")
        queue.report_progress(job.id, "w1", {"done": 1, "total": 2})
        queue.complete(job.id, "w1")
        with pytest.raises(sa.exc.IntegrityError):
            notify_then_roll_back(migrated_engine, job.id, workspace)
        deadline = time.monotonic() + 5
        while len(events) < 4 and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        listener.stop()
    assert [(e["status"], e.get("progress")) for e in events] == [
        ("queued", None),
        ("running", None),
        ("running", {"done": 1, "total": 2}),
        ("succeeded", {"done": 1, "total": 2}),
    ]
    assert all(e["job_id"] == str(job.id) and e["workspace_id"] == str(workspace) for e in events)


def notify_then_roll_back(engine: sa.Engine, job_id: uuid.UUID, workspace: uuid.UUID) -> None:
    """Notify, then fail the transaction: a rolled-back change is never announced."""
    with engine.begin() as conn:
        conn.execute(sa.text("SELECT pg_notify('stratafi_events', '{\"type\":\"ghost\"}')"))
        conn.execute(sa.insert(jobs).values(id=job_id, workspace_id=workspace, kind="t.job"))
