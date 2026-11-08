"""The worker loop.

``Worker`` runs a number of slots per resource class (``worker.slots``). Each slot is a thread
that claims one job at a time and runs its handler. Besides the slots:

- a heartbeat thread renews the lease of every running job, and passes on cancellation
  requests; it also returns jobs whose lease expired elsewhere (a dead worker) to the queue,
  then runs the ``maintenance`` callbacks (the run orchestrator's reconcile sweep)
- ``stop()`` is graceful. Slots stop claiming, running jobs get ``shutdown_grace_seconds`` to
  finish, and then they're asked to stop at their next checkpoint and handed back to the queue
  without using up an attempt.

Handlers cooperate through ``JobContext``: ``checkpoint()`` raises ``JobCancelled`` once the job
is cancelled, its lease is lost, or the worker is shutting down; ``progress(...)`` reports
progress (throttled); ``provider(...)`` waits for a provider's rate and concurrency limits.
"""

import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stratafi_core.logs import log_context
from stratafi_core.observability import (
    JOB_DURATION,
    JOBS_FINISHED,
    JOBS_RUNNING,
    PROVIDER_ERRORS,
    PROVIDER_WAIT,
)
from stratafi_store.jobs import Job, JobQueue
from stratafi_store.scope import workspace_scope
from stratafi_worker.limits import ProviderLimiter, WaitCancelled

logger = logging.getLogger("stratafi.worker")


class JobCancelled(Exception):
    """Raised at a checkpoint: stop the job now."""


class PermanentJobError(Exception):
    """A failure that retrying can't fix. ``code`` is a short, text-free reason."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


Handler = Callable[["JobContext"], Mapping[str, Any] | None]


class HandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, kind: str, handler: Handler) -> None:
        if kind in self._handlers:
            raise ValueError("a handler for this kind is already registered")
        self._handlers[kind] = handler

    def get(self, kind: str) -> Handler | None:
        return self._handlers.get(kind)

    def kinds(self) -> list[str]:
        return sorted(self._handlers)


@dataclass
class JobContext:
    job: Job
    queue: JobQueue
    owner: str
    limiter: ProviderLimiter
    progress_interval: float = 0.5
    stop: threading.Event = field(default_factory=threading.Event)
    lease_lost: bool = False
    shutting_down: bool = False
    cancel_requested: bool = False
    _last_progress: float = 0.0

    @property
    def payload(self) -> Mapping[str, Any]:
        return self.job.payload

    def checkpoint(self) -> None:
        """Raise ``JobCancelled`` if the job must stop. Call it between units of work."""
        if self.stop.is_set():
            raise JobCancelled(str(self.job.id))

    def progress(self, *, force: bool = False, **values: Any) -> None:
        """Report progress (numbers and IDs only). Throttled unless ``force``."""
        now = time.monotonic()
        if not force and now - self._last_progress < self.progress_interval:
            return
        self._last_progress = now
        if not self.queue.report_progress(self.job.id, self.owner, values):
            self._lose_lease()
        self.checkpoint()

    @contextmanager
    def provider(self, name: str, *, tokens: int = 0) -> Iterator[float]:
        """Wait for ``name``'s limits; yields the seconds spent waiting."""
        try:
            with self.limiter.acquire(name, tokens=tokens, cancelled=self.stop) as waited:
                PROVIDER_WAIT.labels(name).observe(waited)
                try:
                    yield waited
                except Exception as error:
                    # The code only: a component's error codes are short and hold no text.
                    code = getattr(error, "code", None)
                    PROVIDER_ERRORS.labels(name, code if isinstance(code, str) else "error").inc()
                    raise
        except WaitCancelled:
            raise JobCancelled(str(self.job.id)) from None

    def _lose_lease(self) -> None:
        self.lease_lost = True
        self.stop.set()


def default_owner() -> str:
    """A worker's lease owner name: host, process and a random suffix."""
    return f"{socket.gethostname()[:40]}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class Worker:
    def __init__(
        self,
        queue: JobQueue,
        handlers: HandlerRegistry,
        *,
        slots: Mapping[str, int],
        limiter: ProviderLimiter | None = None,
        owner: str | None = None,
        poll_seconds: float = 1.0,
        heartbeat_seconds: float = 15.0,
        shutdown_grace_seconds: float = 30.0,
        progress_interval: float = 0.5,
        health_file: Path | None = None,
        maintenance: Sequence[Callable[[], object]] = (),
    ) -> None:
        self.queue = queue
        self.handlers = handlers
        self.slots = {name: count for name, count in slots.items() if count > 0}
        self.limiter = limiter or ProviderLimiter({})
        self.owner = owner or default_owner()
        self.poll_seconds = poll_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.shutdown_grace_seconds = shutdown_grace_seconds
        self.progress_interval = progress_interval
        self.health_file = health_file
        self.maintenance = list(maintenance)
        self._stopping = threading.Event()
        self._threads: list[threading.Thread] = []
        self._running: dict[uuid.UUID, JobContext] = {}
        self._lock = threading.Lock()

    # --- lifecycle -----------------------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("the worker is already running")
        for resource_class, count in sorted(self.slots.items()):
            for number in range(count):
                thread = threading.Thread(
                    target=self._slot,
                    args=(resource_class,),
                    name=f"slot-{resource_class}-{number}",
                    daemon=True,
                )
                self._threads.append(thread)
        self._threads.append(
            threading.Thread(target=self._heartbeats, name="heartbeat", daemon=True)
        )
        self._touch_health_file()
        for thread in self._threads:
            thread.start()
        logger.info(
            "worker started",
            extra={"slots": ",".join(f"{k}={v}" for k, v in sorted(self.slots.items()))},
        )

    def stop(self, *, grace_seconds: float | None = None) -> None:
        """Stop claiming, let running jobs finish within the grace period, then interrupt them
        and hand them back to the queue."""
        grace = self.shutdown_grace_seconds if grace_seconds is None else grace_seconds
        self._stopping.set()
        deadline = time.monotonic() + grace
        while self.running_jobs() and time.monotonic() < deadline:
            time.sleep(0.05)
        with self._lock:
            interrupted = list(self._running.values())
        for context in interrupted:
            context.shutting_down = True
            context.stop.set()
        for thread in self._threads:
            thread.join(timeout=max(grace, 5.0))
        self._threads = []
        logger.info("worker stopped", extra={"interrupted": len(interrupted)})

    def running_jobs(self) -> list[uuid.UUID]:
        with self._lock:
            return list(self._running)

    # --- threads -------------------------------------------------------------------------

    def _slot(self, resource_class: str) -> None:
        while not self._stopping.is_set():
            try:
                claimed = self.queue.claim(self.owner, resource_class, limit=1)
            except Exception:
                logger.exception("claiming failed")
                self._stopping.wait(self.poll_seconds)
                continue
            if not claimed:
                self._stopping.wait(self.poll_seconds)
                continue
            self._run(claimed[0])

    def _heartbeats(self) -> None:
        while not self._stopping.wait(self.heartbeat_seconds):
            self._beat()
        self._beat()

    def _beat(self) -> None:
        self._touch_health_file()
        with self._lock:
            contexts = list(self._running.values())
        for context in contexts:
            try:
                beat = self.queue.heartbeat(context.job.id, self.owner)
            except Exception:
                logger.exception("heartbeat failed", extra={"job_id": str(context.job.id)})
                continue
            if not beat.alive:
                context._lose_lease()
            elif beat.cancel_requested:
                context.cancel_requested = True
                context.stop.set()
        try:
            reclaimed = self.queue.reclaim_expired()
        except Exception:
            logger.exception("reclaiming expired leases failed")
        else:
            if reclaimed:
                logger.info("reclaimed expired jobs", extra={"jobs": reclaimed})
        for callback in self.maintenance:
            try:
                callback()
            except Exception:
                logger.exception("maintenance failed")

    def _touch_health_file(self) -> None:
        if self.health_file is not None:
            try:
                self.health_file.touch()
            except OSError:
                logger.warning("can't touch the health file")

    # --- one job -------------------------------------------------------------------------

    def _run(self, job: Job) -> None:
        context = JobContext(
            job, self.queue, self.owner, self.limiter, progress_interval=self.progress_interval
        )
        with self._lock:
            self._running[job.id] = context
        started = time.monotonic()
        running = JOBS_RUNNING.labels(job.resource_class)
        running.inc()
        with log_context(workspace_id=str(job.workspace_id), job_id=str(job.id)):
            try:
                outcome = self._execute(context)
            finally:
                running.dec()
                with self._lock:
                    self._running.pop(job.id, None)
            JOBS_FINISHED.labels(job.kind, outcome).inc()
            JOB_DURATION.labels(job.kind).observe(time.monotonic() - started)
            logger.info(
                "job finished",
                extra={
                    "kind": job.kind,
                    "outcome": outcome,
                    "attempt": job.attempts,
                    "duration_ms": round((time.monotonic() - started) * 1000, 1),
                },
            )

    def _execute(self, context: JobContext) -> str:
        job = context.job
        handler = self.handlers.get(job.kind)
        if handler is None:
            self.queue.fail(job.id, self.owner, "unknown_kind", retryable=False)
            return "unknown_kind"
        try:
            context.checkpoint()
            # Row-level security: a job sees only its own workspace's rows. Threads it
            # starts, and the queue's own calls, see every workspace (the engine's default).
            with workspace_scope(job.workspace_id):
                result = handler(context)
        except JobCancelled:
            if context.lease_lost:
                return "lease_lost"
            if context.shutting_down and not context.cancel_requested:
                self.queue.release(job.id, self.owner)
                return "released"
            self.queue.mark_cancelled(job.id, self.owner)
            return "cancelled"
        except PermanentJobError as exc:
            self.queue.fail(job.id, self.owner, exc.code, retryable=False)
            return "failed"
        except Exception as exc:
            # The JSON formatter records the type and frames, never the message.
            logger.warning("job attempt failed", exc_info=exc)
            failed = self.queue.fail(job.id, self.owner, type(exc).__name__)
            return "retrying" if failed is not None and failed.status == "queued" else "failed"
        if context.lease_lost:
            return "lease_lost"
        if not self.queue.complete(job.id, self.owner, result):
            return "lease_lost"
        return "succeeded"
