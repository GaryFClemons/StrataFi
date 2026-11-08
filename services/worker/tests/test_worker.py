"""The worker: provider limits in the unit tier; the worker loop against Postgres
in the integration tier."""

import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from itertools import pairwise

import pytest
import sqlalchemy as sa

from stratafi_core.settings import ProviderLimit
from stratafi_store.jobs import Job, JobQueue
from stratafi_store.tables import workspaces
from stratafi_worker.handlers import builtin_handlers
from stratafi_worker.limits import ProviderLimiter, TokenBucket, WaitCancelled
from stratafi_worker.worker import (
    HandlerRegistry,
    JobCancelled,
    JobContext,
    PermanentJobError,
    Worker,
)

# --- provider limits (unit tier) ----------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_a_token_bucket_allows_a_burst_then_its_rate() -> None:
    clock = FakeClock()
    bucket = TokenBucket(3, 1.0, clock=clock)
    assert [bucket.take(1) for _ in range(3)] == [0, 0, 0]
    assert bucket.take(1) == pytest.approx(1.0)
    clock.now += 0.5
    assert bucket.take(1) == pytest.approx(0.5)
    clock.now += 0.5
    assert bucket.take(1) == 0


def test_requests_per_minute_are_enforced() -> None:
    clock = FakeClock()
    limiter = ProviderLimiter(
        {"gateway": ProviderLimit(requests_per_minute=60)}, clock=clock, sleep=clock.sleep
    )
    waits = []
    for _ in range(62):
        with limiter.acquire("gateway") as waited:
            waits.append(waited)
    assert sum(waits[:60]) == 0, "a full bucket allows a burst of 60"
    assert sum(waits[60:]) == pytest.approx(2.0, abs=0.01), "then one a second"


def test_tokens_per_minute_are_enforced() -> None:
    clock = FakeClock()
    limiter = ProviderLimiter(
        {"gateway": ProviderLimit(tokens_per_minute=6000)}, clock=clock, sleep=clock.sleep
    )
    with limiter.acquire("gateway", tokens=6000) as first:
        assert first == 0
    with limiter.acquire("gateway", tokens=3000) as second:
        assert second == pytest.approx(30.0, abs=0.01)


def test_concurrency_is_capped() -> None:
    limiter = ProviderLimiter({"gateway": ProviderLimit(concurrency=2)})
    active, peak = 0, 0
    lock = threading.Lock()

    def call() -> None:
        nonlocal active, peak
        with limiter.acquire("gateway"):
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1

    threads = [threading.Thread(target=call) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak == 2


def test_unknown_providers_are_unlimited_and_waits_can_be_cancelled() -> None:
    limiter = ProviderLimiter({"slow": ProviderLimit(requests_per_minute=1)})
    with limiter.acquire("anything", tokens=10**9) as waited:
        assert waited == 0
    with limiter.acquire("slow"):
        pass
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(WaitCancelled), limiter.acquire("slow", cancelled=cancelled):
        pass


def test_handler_kinds_register_once() -> None:
    registry = builtin_handlers()
    assert registry.kinds() == ["system.selftest"]
    with pytest.raises(ValueError, match="already"):
        registry.register("system.selftest", lambda context: None)


# --- the worker loop (integration tier) ---------------------------------------------------


@pytest.fixture
def workspace(migrated_engine: sa.Engine) -> uuid.UUID:
    workspace_id = uuid.uuid4()
    with migrated_engine.begin() as conn:
        conn.execute(sa.insert(workspaces).values(id=workspace_id, slug="work", name="work"))
    return workspace_id


@pytest.fixture
def queue(migrated_engine: sa.Engine) -> JobQueue:
    return JobQueue(
        migrated_engine,
        lease_seconds=1,
        backoff_base_seconds=0.1,
        backoff_max_seconds=0.5,
        jitter=False,
    )


@pytest.fixture
def make_worker(queue: JobQueue) -> Iterator[Callable[..., Worker]]:
    started: list[Worker] = []

    def make(handlers: HandlerRegistry, **options) -> Worker:
        settings = {
            "slots": {"cpu": 4},
            "poll_seconds": 0.02,
            "heartbeat_seconds": 0.25,
            "shutdown_grace_seconds": 5.0,
            "progress_interval": 0.0,
            **options,
        }
        worker = Worker(queue, handlers, **settings)
        worker.start()
        started.append(worker)
        return worker

    yield make
    for worker in started:
        if worker._threads:
            worker.stop(grace_seconds=1.0)


def wait_for(predicate: Callable[[], bool], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def job_now(queue: JobQueue, job: Job) -> Job:
    current = queue.get(job.id)
    assert current is not None
    return current


@pytest.mark.integration
def test_two_workers_run_200_jobs_each_exactly_once(
    queue: JobQueue, workspace: uuid.UUID, make_worker, migrated_engine: sa.Engine
) -> None:
    runs: Counter[uuid.UUID] = Counter()
    by_worker: Counter[str] = Counter()
    lock = threading.Lock()

    def record(context: JobContext) -> dict:
        with lock:
            runs[context.job.id] += 1
            by_worker[context.owner] += 1
        time.sleep(0.005)
        return {"ok": True}

    handlers = HandlerRegistry()
    handlers.register("t.record", record)
    created = [queue.enqueue(workspace_id=workspace, kind="t.record")[0] for _ in range(200)]
    make_worker(handlers, owner="worker-a")
    make_worker(handlers, owner="worker-b")

    def all_done() -> bool:
        with migrated_engine.connect() as conn:
            pending = conn.execute(
                sa.text(
                    "SELECT count(*) FROM stratafi.jobs WHERE status <> 'succeeded' "
                    "AND workspace_id = :w"
                ),
                {"w": workspace},
            ).scalar_one()
        return pending == 0

    wait_for(all_done, timeout=60)
    assert set(runs) == {job.id for job in created}
    assert set(runs.values()) == {1}, "each job ran exactly once"
    assert by_worker["worker-a"] > 0
    assert by_worker["worker-b"] > 0
    assert sum(by_worker.values()) == 200


@pytest.mark.integration
def test_a_job_whose_worker_died_is_reclaimed_and_finished(
    queue: JobQueue, workspace: uuid.UUID, make_worker
) -> None:
    handlers = HandlerRegistry()
    handlers.register("t.ok", lambda context: {"owner": context.owner})
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.ok")
    assert len(queue.claim("dead-worker", "cpu")) == 1  # claimed, then its worker vanishes
    make_worker(handlers, owner="live-worker")
    wait_for(lambda: job_now(queue, job).status == "succeeded")
    final = job_now(queue, job)
    assert final.attempts == 2
    assert final.result == {"owner": "live-worker"}


@pytest.mark.integration
def test_retries_back_off_and_stop_at_max_attempts(
    queue: JobQueue, workspace: uuid.UUID, make_worker
) -> None:
    attempts: list[float] = []

    def always_fails(context: JobContext) -> None:
        attempts.append(time.monotonic())
        raise RuntimeError("the document said something private")

    handlers = HandlerRegistry()
    handlers.register("t.fail", always_fails)
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.fail", max_attempts=3)
    make_worker(handlers)
    wait_for(lambda: job_now(queue, job).status == "failed")
    final = job_now(queue, job)
    assert (final.attempts, final.error_code) == (3, "RuntimeError")
    assert len(attempts) == 3
    gaps = [later - earlier for earlier, later in pairwise(attempts)]
    assert gaps[0] >= 0.1
    assert gaps[1] >= 0.2, "the second retry waits twice as long"


@pytest.mark.integration
def test_permanent_errors_and_unknown_kinds_fail_at_once(
    queue: JobQueue, workspace: uuid.UUID, make_worker
) -> None:
    def refuse(context: JobContext) -> None:
        raise PermanentJobError("invalid_payload")

    handlers = HandlerRegistry()
    handlers.register("t.refuse", refuse)
    refused, _ = queue.enqueue(workspace_id=workspace, kind="t.refuse", max_attempts=5)
    unknown, _ = queue.enqueue(workspace_id=workspace, kind="t.nobody_handles_this")
    make_worker(handlers)
    wait_for(lambda: job_now(queue, refused).finished and job_now(queue, unknown).finished)
    assert (job_now(queue, refused).status, job_now(queue, refused).error_code) == (
        "failed",
        "invalid_payload",
    )
    assert job_now(queue, refused).attempts == 1
    assert job_now(queue, unknown).error_code == "unknown_kind"


@pytest.mark.integration
def test_cancelling_queued_and_running_jobs(
    queue: JobQueue, workspace: uuid.UUID, make_worker
) -> None:
    started = threading.Event()
    steps: Counter[uuid.UUID] = Counter()

    def long_job(context: JobContext) -> None:
        started.set()
        while True:
            steps[context.job.id] += 1
            context.checkpoint()
            time.sleep(0.02)

    handlers = HandlerRegistry()
    handlers.register("t.long", long_job)
    queued, _ = queue.enqueue(workspace_id=workspace, kind="t.long", resource_class="io")
    queue.cancel(queued.id, workspace)
    running, _ = queue.enqueue(workspace_id=workspace, kind="t.long")
    make_worker(handlers, slots={"cpu": 1, "io": 1})
    assert started.wait(10)
    queue.cancel(running.id, workspace)
    wait_for(lambda: job_now(queue, running).status == "cancelled")
    assert job_now(queue, queued).status == "cancelled"
    assert steps[queued.id] == 0, "a job cancelled while queued never runs"
    assert steps[running.id] > 0


@pytest.mark.integration
def test_a_graceful_stop_hands_running_jobs_back(
    queue: JobQueue, workspace: uuid.UUID, make_worker
) -> None:
    started = threading.Event()

    def long_job(context: JobContext) -> None:
        started.set()
        while True:
            context.checkpoint()
            time.sleep(0.02)

    handlers = HandlerRegistry()
    handlers.register("t.long", long_job)
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.long")
    worker = make_worker(handlers)
    assert started.wait(10)
    worker.stop(grace_seconds=0.2)
    final = job_now(queue, job)
    assert (final.status, final.attempts, final.lease_owner) == ("queued", 0, None)


@pytest.mark.integration
def test_the_selftest_job_reports_progress(
    queue: JobQueue, workspace: uuid.UUID, make_worker
) -> None:
    job, _ = queue.enqueue(
        workspace_id=workspace, kind="system.selftest", payload={"steps": 4, "step_seconds": 0.01}
    )
    bad, _ = queue.enqueue(
        workspace_id=workspace, kind="system.selftest", payload={"steps": "many"}
    )
    make_worker(builtin_handlers())
    wait_for(lambda: job_now(queue, job).finished and job_now(queue, bad).finished)
    final = job_now(queue, job)
    assert (final.status, final.progress, final.result) == (
        "succeeded",
        {"done": 4, "total": 4},
        {"steps": 4},
    )
    assert (job_now(queue, bad).status, job_now(queue, bad).error_code) == (
        "failed",
        "invalid_payload",
    )


@pytest.mark.integration
def test_a_lost_lease_stops_the_job_without_writing_results(
    queue: JobQueue, workspace: uuid.UUID, make_worker
) -> None:
    started = threading.Event()
    stopped = threading.Event()

    def slow(context: JobContext) -> dict:
        started.set()
        try:
            while True:
                context.checkpoint()
                time.sleep(0.02)
        except JobCancelled:
            stopped.set()
            raise

    handlers = HandlerRegistry()
    handlers.register("t.slow", slow)
    job, _ = queue.enqueue(workspace_id=workspace, kind="t.slow")
    worker = make_worker(handlers, owner="w-original")
    assert started.wait(10)
    with queue.engine.begin() as conn:  # another worker takes the job over
        conn.execute(
            sa.text("UPDATE stratafi.jobs SET lease_owner = 'w-other' WHERE id = :id"),
            {"id": job.id},
        )
    assert stopped.wait(10)
    worker.stop(grace_seconds=1.0)
    final = job_now(queue, job)
    assert (final.status, final.lease_owner) == ("running", "w-other")
