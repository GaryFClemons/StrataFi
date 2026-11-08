"""The Postgres job queue.

- ``enqueue`` with an optional dedupe key: while a job with the same key is queued or running in
  the workspace, enqueueing it again returns that job.
- ``claim`` takes the next jobs of one resource class, highest priority first, with
  ``FOR UPDATE SKIP LOCKED``, so concurrent workers never claim the same job. Claiming counts an
  attempt and starts a lease that ``heartbeat`` renews.
- ``complete``, ``fail`` (retry with exponential backoff and jitter until ``max_attempts``),
  ``release`` (a graceful shutdown hands the job back without using up an attempt) and
  ``mark_cancelled`` finish a claim; each acts only while the caller still holds the lease.
- ``cancel`` cancels a queued job at once and asks a running one to stop at its next
  checkpoint; the worker learns of it from ``heartbeat``.
- ``reclaim_expired`` returns jobs whose lease ran out (their worker died) to the queue.

Every change notifies ``stratafi_events`` in the same transaction, so listeners hear about it
when it commits (``stratafi_store.events``). Times come from the database's clock, never a
worker's. Payloads hold IDs and parameters; progress and error codes, which travel to clients,
must be metadata only (``check_metadata``).
"""

import json
import logging
import random
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from stratafi_core.logs import check_metadata
from stratafi_store.scope import system_scope
from stratafi_store.tables import ACTIVE_JOB_STATUSES, RESOURCE_CLASSES, jobs

logger = logging.getLogger("stratafi.store.jobs")

EVENTS_CHANNEL = "stratafi_events"
MAX_PROGRESS_BYTES = 2048


@dataclass(frozen=True)
class Job:
    id: uuid.UUID
    workspace_id: uuid.UUID
    kind: str
    payload: Mapping[str, Any]
    priority: int
    resource_class: str
    status: str
    dedupe_key: str | None
    attempts: int
    max_attempts: int
    run_after: datetime
    lease_owner: str | None
    lease_expires_at: datetime | None
    cancel_requested: bool
    progress: Mapping[str, Any] | None
    result: Mapping[str, Any] | None
    error_code: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def finished(self) -> bool:
        return self.status not in ACTIVE_JOB_STATUSES


def _job(row: Any) -> Job:
    return Job(**{column.name: getattr(row, column.name) for column in jobs.columns})


class CancelOutcome(StrEnum):
    CANCELLED = "cancelled"  # it was queued, and now never runs
    REQUESTED = "requested"  # it is running, and stops at its next checkpoint
    FINISHED = "finished"  # it had already finished


@dataclass(frozen=True)
class Heartbeat:
    alive: bool  # False: the lease was lost, so stop without writing results
    cancel_requested: bool


def event_payload(job: Job) -> dict[str, Any]:
    """What listeners hear about a job: IDs, status and counts, never payloads or results."""
    event: dict[str, Any] = {
        "type": "job",
        "workspace_id": str(job.workspace_id),
        "job_id": str(job.id),
        "kind": job.kind,
        "status": job.status,
        "attempts": job.attempts,
    }
    if job.progress is not None:
        event["progress"] = dict(job.progress)
    if job.error_code is not None:
        event["error_code"] = job.error_code
    return event


def _notify(conn: Connection, job: Job) -> None:
    conn.execute(
        sa.text("SELECT pg_notify(:channel, :payload)"),
        {
            "channel": EVENTS_CHANNEL,
            "payload": json.dumps(event_payload(job), separators=(",", ":")),
        },
    )


def _lease(seconds: float) -> Any:
    return sa.func.now() + sa.func.make_interval(0, 0, 0, 0, 0, 0, seconds)


_RETURNING = [*jobs.columns]


class JobQueue:
    def __init__(
        self,
        engine: Engine,
        *,
        lease_seconds: float = 60,
        backoff_base_seconds: float = 5,
        backoff_max_seconds: float = 600,
        jitter: bool = True,
        random_fraction: Callable[[], float] = random.random,
    ) -> None:
        self.engine = engine
        self.lease_seconds = lease_seconds
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.jitter = jitter
        self._random = random_fraction

    # --- creating and reading ------------------------------------------------------------

    def enqueue(
        self,
        *,
        workspace_id: uuid.UUID,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        resource_class: str = "cpu",
        priority: int = 0,
        dedupe_key: str | None = None,
        max_attempts: int = 3,
    ) -> tuple[Job, bool]:
        """Queue a job. Returns it and whether it's new; with a dedupe key, an active job with
        the same key is returned instead of a new one."""
        if resource_class not in RESOURCE_CLASSES:
            raise ValueError("unknown resource class")
        values = {
            "id": uuid.uuid4(),
            "workspace_id": workspace_id,
            "kind": kind,
            "payload": dict(payload or {}),
            "resource_class": resource_class,
            "priority": priority,
            "dedupe_key": dedupe_key,
            "max_attempts": max_attempts,
        }
        for _ in range(5):
            with self.engine.begin() as conn:
                statement = insert(jobs).values(**values)
                if dedupe_key is not None:
                    statement = statement.on_conflict_do_nothing(
                        index_elements=["workspace_id", "dedupe_key"],
                        index_where=sa.and_(
                            jobs.c.dedupe_key.is_not(None), jobs.c.status.in_(ACTIVE_JOB_STATUSES)
                        ),
                    )
                row = conn.execute(statement.returning(*_RETURNING)).first()
                if row is not None:
                    job = _job(row)
                    _notify(conn, job)
                    return job, True
                existing = conn.execute(
                    sa.select(jobs).where(
                        jobs.c.workspace_id == workspace_id,
                        jobs.c.dedupe_key == dedupe_key,
                        jobs.c.status.in_(ACTIVE_JOB_STATUSES),
                    )
                ).first()
                if existing is not None:
                    return _job(existing), False
            # The active job finished between the two statements: try again.
        raise RuntimeError("couldn't enqueue: the dedupe key kept changing hands")

    def enqueue_in(
        self,
        conn: Connection,
        *,
        workspace_id: uuid.UUID,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        resource_class: str = "cpu",
        priority: int = 0,
        max_attempts: int = 3,
    ) -> Job:
        """Queue a job in the caller's transaction: it exists, and listeners hear of it, only
        if that transaction commits. No dedupe key."""
        if resource_class not in RESOURCE_CLASSES:
            raise ValueError("unknown resource class")
        row = conn.execute(
            insert(jobs)
            .values(
                id=uuid.uuid4(),
                workspace_id=workspace_id,
                kind=kind,
                payload=dict(payload or {}),
                resource_class=resource_class,
                priority=priority,
                max_attempts=max_attempts,
            )
            .returning(*_RETURNING)
        ).one()
        job = _job(row)
        _notify(conn, job)
        return job

    def get(self, job_id: uuid.UUID, workspace_id: uuid.UUID | None = None) -> Job | None:
        query = sa.select(jobs).where(jobs.c.id == job_id)
        if workspace_id is not None:
            query = query.where(jobs.c.workspace_id == workspace_id)
        with self.engine.connect() as conn:
            row = conn.execute(query).first()
        return _job(row) if row is not None else None

    def list_jobs(
        self,
        workspace_id: uuid.UUID,
        *,
        limit: int,
        after: tuple[datetime, uuid.UUID] | None = None,
        status: str | None = None,
    ) -> list[Job]:
        """Newest first. ``after`` is the (created_at, id) of the previous page's last job."""
        query = sa.select(jobs).where(jobs.c.workspace_id == workspace_id)
        if status is not None:
            query = query.where(jobs.c.status == status)
        if after is not None:
            query = query.where(sa.tuple_(jobs.c.created_at, jobs.c.id) < after)
        query = query.order_by(jobs.c.created_at.desc(), jobs.c.id.desc()).limit(limit)
        with self.engine.connect() as conn:
            return [_job(row) for row in conn.execute(query)]

    # --- the worker's side ---------------------------------------------------------------

    def claim(self, owner: str, resource_class: str, *, limit: int = 1) -> list[Job]:
        """Claim up to ``limit`` due jobs of ``resource_class``, highest priority first."""
        due = (
            sa.select(jobs.c.id)
            .where(
                jobs.c.status == "queued",
                jobs.c.resource_class == resource_class,
                jobs.c.run_after <= sa.func.now(),
            )
            .order_by(jobs.c.priority.desc(), jobs.c.run_after, jobs.c.created_at, jobs.c.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
            .cte("due")
        )
        statement = (
            sa.update(jobs)
            .where(jobs.c.id == due.c.id)
            .values(
                status="running",
                attempts=jobs.c.attempts + 1,
                lease_owner=owner,
                lease_expires_at=_lease(self.lease_seconds),
                started_at=sa.func.coalesce(jobs.c.started_at, sa.func.now()),
                updated_at=sa.func.now(),
            )
            .returning(*_RETURNING)
        )
        with self.engine.begin() as conn:
            claimed = [_job(row) for row in conn.execute(statement)]
            for job in claimed:
                _notify(conn, job)
        return sorted(claimed, key=lambda job: (-job.priority, job.run_after, job.created_at))

    def heartbeat(self, job_id: uuid.UUID, owner: str) -> Heartbeat:
        statement = (
            sa.update(jobs)
            .where(jobs.c.id == job_id, jobs.c.lease_owner == owner, jobs.c.status == "running")
            .values(lease_expires_at=_lease(self.lease_seconds), updated_at=sa.func.now())
            .returning(jobs.c.cancel_requested)
        )
        with self.engine.begin() as conn:
            row = conn.execute(statement).first()
        if row is None:
            return Heartbeat(alive=False, cancel_requested=False)
        return Heartbeat(alive=True, cancel_requested=row.cancel_requested)

    def report_progress(self, job_id: uuid.UUID, owner: str, progress: Mapping[str, Any]) -> bool:
        check_metadata(progress, where="job progress")
        if len(json.dumps(progress)) > MAX_PROGRESS_BYTES:
            raise ValueError("job progress is too large")
        return self._finish_or_update(job_id, owner, {"progress": dict(progress)}) is not None

    def complete(
        self, job_id: uuid.UUID, owner: str, result: Mapping[str, Any] | None = None
    ) -> bool:
        values = {"status": "succeeded", "result": dict(result) if result is not None else None}
        return self._finish_or_update(job_id, owner, values, finish=True) is not None

    def fail(
        self, job_id: uuid.UUID, owner: str, error_code: str, *, retryable: bool = True
    ) -> Job | None:
        """Record a failed attempt. A retryable failure with attempts left goes back to the queue
        after a backoff; otherwise the job fails. Returns the job, or None if the lease was lost.
        """
        check_metadata({"error_code": error_code}, where="job error")
        with self.engine.begin() as conn:
            row = conn.execute(
                sa.select(jobs.c.attempts, jobs.c.max_attempts)
                .where(jobs.c.id == job_id, jobs.c.lease_owner == owner, jobs.c.status == "running")
                .with_for_update()
            ).first()
            if row is None:
                return None
            if retryable and row.attempts < row.max_attempts:
                values: dict[str, Any] = {
                    "status": "queued",
                    "run_after": _lease(self.backoff_seconds(row.attempts)),
                    "lease_owner": None,
                    "lease_expires_at": None,
                }
            else:
                values = {
                    "status": "failed",
                    "finished_at": sa.func.now(),
                    "lease_owner": None,
                    "lease_expires_at": None,
                }
            updated = conn.execute(
                sa.update(jobs)
                .where(jobs.c.id == job_id)
                .values(error_code=error_code, updated_at=sa.func.now(), **values)
                .returning(*_RETURNING)
            ).one()
            job = _job(updated)
            _notify(conn, job)
            return job

    def release(self, job_id: uuid.UUID, owner: str) -> bool:
        """Hand a running job back to the queue without using up an attempt: for a graceful
        shutdown that interrupted it."""
        values = {
            "status": "queued",
            "attempts": sa.func.greatest(jobs.c.attempts - 1, 0),
            "run_after": sa.func.now(),
            "lease_owner": None,
            "lease_expires_at": None,
        }
        return self._finish_or_update(job_id, owner, values) is not None

    def mark_cancelled(self, job_id: uuid.UUID, owner: str) -> bool:
        """The worker stopped a job whose cancellation was requested."""
        values = {"status": "cancelled", "error_code": None}
        return self._finish_or_update(job_id, owner, values, finish=True) is not None

    def reclaim_expired(self) -> int:
        """Return running jobs whose lease ran out to the queue. A job with no attempts left
        fails (``lease_expired``); one whose cancellation was requested is cancelled."""
        expired = (
            sa.select(jobs.c.id)
            .where(jobs.c.status == "running", jobs.c.lease_expires_at < sa.func.now())
            .with_for_update(skip_locked=True)
            .cte("expired")
        )
        exhausted = jobs.c.attempts >= jobs.c.max_attempts
        statement = (
            sa.update(jobs)
            .where(jobs.c.id == expired.c.id)
            .values(
                status=sa.case(
                    (jobs.c.cancel_requested, "cancelled"),
                    (exhausted, "failed"),
                    else_="queued",
                ),
                error_code=sa.case(
                    (jobs.c.cancel_requested, None),
                    (exhausted, "lease_expired"),
                    else_=jobs.c.error_code,
                ),
                finished_at=sa.case(
                    (sa.or_(jobs.c.cancel_requested, exhausted), sa.func.now()), else_=None
                ),
                run_after=sa.func.now(),
                lease_owner=None,
                lease_expires_at=None,
                updated_at=sa.func.now(),
            )
            .returning(*_RETURNING)
        )
        with self.engine.begin() as conn:
            reclaimed = [_job(row) for row in conn.execute(statement)]
            for job in reclaimed:
                _notify(conn, job)
        return len(reclaimed)

    def backoff_seconds(self, attempts: int) -> float:
        """The wait before attempt ``attempts + 1``: doubling from the base, capped, with "equal
        jitter" (between half and all of it), so failed jobs don't all retry at once."""
        delay = min(self.backoff_max_seconds, self.backoff_base_seconds * 2 ** max(attempts - 1, 0))
        if not self.jitter:
            return delay
        return delay / 2 + self._random() * delay / 2

    # --- the client's side ---------------------------------------------------------------

    def cancel(
        self, job_id: uuid.UUID, workspace_id: uuid.UUID
    ) -> tuple[Job, CancelOutcome] | None:
        """Cancel a job: at once if queued, at its next checkpoint if running. None if the job
        isn't in the workspace."""
        scoped = sa.and_(jobs.c.id == job_id, jobs.c.workspace_id == workspace_id)
        with self.engine.begin() as conn:
            row = conn.execute(
                sa.update(jobs)
                .where(scoped, jobs.c.status == "queued")
                .values(
                    status="cancelled",
                    cancel_requested=True,
                    finished_at=sa.func.now(),
                    updated_at=sa.func.now(),
                )
                .returning(*_RETURNING)
            ).first()
            if row is not None:
                job = _job(row)
                _notify(conn, job)
                return job, CancelOutcome.CANCELLED
            row = conn.execute(
                sa.update(jobs)
                .where(scoped, jobs.c.status == "running")
                .values(cancel_requested=True, updated_at=sa.func.now())
                .returning(*_RETURNING)
            ).first()
            if row is not None:
                return _job(row), CancelOutcome.REQUESTED
            row = conn.execute(sa.select(jobs).where(scoped)).first()
        return (_job(row), CancelOutcome.FINISHED) if row is not None else None

    # --- helpers -------------------------------------------------------------------------

    def _finish_or_update(
        self,
        job_id: uuid.UUID,
        owner: str,
        values: Mapping[str, Any],
        *,
        finish: bool = False,
    ) -> Job | None:
        extra: dict[str, Any] = {"updated_at": sa.func.now()}
        if finish:
            extra.update(finished_at=sa.func.now(), lease_owner=None, lease_expires_at=None)
        statement = (
            sa.update(jobs)
            .where(jobs.c.id == job_id, jobs.c.lease_owner == owner, jobs.c.status == "running")
            .values(**values, **extra)
            .returning(*_RETURNING)
        )
        with self.engine.begin() as conn:
            row = conn.execute(statement).first()
            if row is None:
                return None
            job = _job(row)
            _notify(conn, job)
            return job


class QueueCollector:
    """The queue for Prometheus, read when scraped: active jobs by status, resource class
    and kind, and the age of the oldest job each class could run now. Counts and seconds only.
    Register it with ``stratafi_core.observability.REGISTRY``."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def describe(self) -> list[Any]:
        return []

    def collect(self) -> Iterator[Any]:
        from prometheus_client.core import GaugeMetricFamily

        depth = GaugeMetricFamily(
            "stratafi_queue_jobs",
            "Queued and running jobs, by status, resource class and kind",
            labels=["status", "resource_class", "kind"],
        )
        age = GaugeMetricFamily(
            "stratafi_queue_oldest_seconds",
            "Age of the oldest job each resource class could run now",
            labels=["resource_class"],
        )
        try:
            with system_scope(), self.engine.connect() as conn:
                for row in conn.execute(
                    sa.select(jobs.c.status, jobs.c.resource_class, jobs.c.kind, sa.func.count())
                    .where(jobs.c.status.in_(ACTIVE_JOB_STATUSES))
                    .group_by(jobs.c.status, jobs.c.resource_class, jobs.c.kind)
                ):
                    depth.add_metric([row[0], row[1], row[2]], row[3])
                for row in conn.execute(
                    sa.select(
                        jobs.c.resource_class,
                        sa.func.extract("epoch", sa.func.now() - sa.func.min(jobs.c.run_after)),
                    )
                    .where(jobs.c.status == "queued", jobs.c.run_after <= sa.func.now())
                    .group_by(jobs.c.resource_class)
                ):
                    age.add_metric([row[0]], float(row[1]))
        except sa.exc.SQLAlchemyError:
            logger.warning("the queue's metrics couldn't be read")
        yield depth
        yield age
