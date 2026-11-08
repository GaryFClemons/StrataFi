"""When suites run on their own.

- **Events.** ``fire_event`` runs every suite whose latest version has the event among its
  triggers and refers to what changed without pinning a version: freezing a corpus version
  (``corpus_version_frozen``) runs the suites on that corpus at ``name`` or ``name@latest``; a
  new version of a pipeline spec or retrieval config, or a profile's new version or review
  (``target_changed``), runs the suites whose target names it. Runs coalesce as every request
  does (``testruns.request_run``), so one change makes one run.
- **Schedules.** ``sync_schedules`` keeps ``test_schedules`` in step with a suite's latest
  version: one row per cron schedule, with its next time in UTC, computed in the suite's time
  zone. ``SuiteScheduler.tick`` runs with the worker's heartbeat. Only the worker that takes the
  scheduler's transaction-scoped advisory lock fires schedules, so one leader fires each due
  schedule once, however many workers run; a schedule missed while no worker ran fires once,
  then moves to its next time after now.
"""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from stratafi_core import cron
from stratafi_core.experiments import parse_ref
from stratafi_core.suites import TestSuite, parse_profile_ref
from stratafi_store import specs, testruns
from stratafi_store.jobs import JobQueue
from stratafi_store.tables import test_schedules

SCHEDULER_LOCK = "stratafi:test-scheduler"
SCHEDULER = "system:scheduler"
PAGE = 200


@dataclass(frozen=True)
class Schedule:
    workspace_id: uuid.UUID
    suite_name: str
    cron: str
    time_zone: str
    next_fire_at: datetime
    last_fired_at: datetime | None
    last_test_run_id: uuid.UUID | None
    created_at: datetime


def _schedule(row: Any) -> Schedule:
    return Schedule(**{column.name: getattr(row, column.name) for column in test_schedules.columns})


def sync_schedules(
    conn: Connection,
    workspace_id: uuid.UUID,
    suite: TestSuite,
    *,
    now: datetime | None = None,
) -> list[Schedule]:
    """Make the suite's schedules its latest version's: drop the ones it no longer has, add the
    new ones, and recompute a kept one's next time when its time zone changed."""
    moment = now or datetime.now(UTC)
    wanted = {trigger.schedule for trigger in suite.triggers if trigger.schedule is not None}
    conn.execute(
        sa.delete(test_schedules).where(
            test_schedules.c.workspace_id == workspace_id,
            test_schedules.c.suite_name == suite.name,
            test_schedules.c.cron.not_in(sorted(wanted)) if wanted else sa.true(),
        )
    )
    for text in sorted(wanted):
        upcoming = cron.parse(text).next_after(moment, suite.time_zone)
        statement = insert(test_schedules).values(
            workspace_id=workspace_id,
            suite_name=suite.name,
            cron=text,
            time_zone=suite.time_zone,
            next_fire_at=upcoming,
        )
        conn.execute(
            statement.on_conflict_do_update(
                index_elements=["workspace_id", "suite_name", "cron"],
                set_={"time_zone": suite.time_zone, "next_fire_at": upcoming},
                where=test_schedules.c.time_zone != statement.excluded.time_zone,
            )
        )
    return schedules_of(conn, workspace_id, suite.name)


def schedules_of(conn: Connection, workspace_id: uuid.UUID, suite_name: str) -> list[Schedule]:
    rows = conn.execute(
        sa.select(test_schedules)
        .where(
            test_schedules.c.workspace_id == workspace_id,
            test_schedules.c.suite_name == suite_name,
        )
        .order_by(test_schedules.c.cron)
    )
    return [_schedule(row) for row in rows]


def _latest_suites(conn: Connection, workspace_id: uuid.UUID) -> Iterator[specs.StoredDocument]:
    after: str | None = None
    while True:
        page = specs.list_latest(conn, "suite", workspace_id, limit=PAGE, after_name=after)
        yield from page
        if len(page) < PAGE:
            return
        after = page[-1].name


def _unpinned(reference: str | None, name: str) -> bool:
    """Whether a pipeline, retrieval or corpus reference names ``name`` at its latest."""
    if reference is None:
        return False
    found, version = parse_ref(reference)
    return found == name and version is None


def _unpinned_profile(reference: str | None, name: str) -> bool:
    if reference is None:
        return False
    found, version = parse_profile_ref(reference)
    return found == name and not isinstance(version, int)


def wants(
    suite: TestSuite,
    event: str,
    *,
    corpus: str | None = None,
    pipeline: str | None = None,
    retrieval: str | None = None,
    profile: str | None = None,
) -> bool:
    """Whether ``event`` about what changed runs ``suite``."""
    if not any(trigger.on == event for trigger in suite.triggers):
        return False
    if event == "corpus_version_frozen":
        return corpus is not None and _unpinned(suite.corpus, corpus)
    target = suite.target
    return bool(
        (pipeline is not None and _unpinned(target.pipeline, pipeline))
        or (retrieval is not None and _unpinned(target.retrieval, retrieval))
        or (profile is not None and _unpinned_profile(target.profile, profile))
    )


def fire_event(
    conn: Connection,
    queue: JobQueue,
    *,
    workspace_id: uuid.UUID,
    event: str,
    created_by: str,
    max_attempts: int = 3,
    corpus: str | None = None,
    pipeline: str | None = None,
    retrieval: str | None = None,
    profile: str | None = None,
) -> list[testruns.TestRun]:
    """Request a run of every suite the event concerns (the module docstring says which)."""
    started = []
    for stored in list(_latest_suites(conn, workspace_id)):
        suite = TestSuite.model_validate(stored.document)
        if wants(
            suite, event, corpus=corpus, pipeline=pipeline, retrieval=retrieval, profile=profile
        ):
            run, _ = testruns.request_run(
                conn,
                queue,
                workspace_id=workspace_id,
                suite=stored,
                trigger=event,
                created_by=created_by,
                max_attempts=max_attempts,
            )
            started.append(run)
    return started


class SuiteScheduler:
    """Fires due schedules; one worker at a time, through an advisory lock."""

    def __init__(self, engine: Engine, queue: JobQueue, *, max_attempts: int = 3) -> None:
        self.engine = engine
        self.queue = queue
        self.max_attempts = max_attempts

    def tick(self, now: datetime | None = None) -> int:
        """Fire every due schedule; returns how many fired (0 when another worker leads)."""
        moment = now or datetime.now(UTC)
        fired = 0
        with self.engine.begin() as conn:
            leader = conn.execute(
                sa.text("SELECT pg_try_advisory_xact_lock(hashtextextended(:key, 0))"),
                {"key": SCHEDULER_LOCK},
            ).scalar_one()
            if not leader:
                return 0
            due = conn.execute(
                sa.select(test_schedules)
                .where(test_schedules.c.next_fire_at <= moment)
                .order_by(test_schedules.c.next_fire_at)
                .with_for_update(skip_locked=True)
            ).all()
            for row in due:
                key = (
                    test_schedules.c.workspace_id == row.workspace_id,
                    test_schedules.c.suite_name == row.suite_name,
                    test_schedules.c.cron == row.cron,
                )
                stored = testruns.get_suite(conn, row.workspace_id, row.suite_name)
                suite = TestSuite.model_validate(stored.document) if stored is not None else None
                if suite is None or row.cron not in {t.schedule for t in suite.triggers}:
                    conn.execute(sa.delete(test_schedules).where(*key))
                    continue
                run, _ = testruns.request_run(
                    conn,
                    self.queue,
                    workspace_id=row.workspace_id,
                    suite=stored,  # type: ignore[arg-type]
                    trigger="schedule",
                    created_by=SCHEDULER,
                    max_attempts=self.max_attempts,
                )
                conn.execute(
                    sa.update(test_schedules)
                    .where(*key)
                    .values(
                        next_fire_at=cron.parse(row.cron).next_after(moment, row.time_zone),
                        last_fired_at=moment,
                        last_test_run_id=run.id,
                    )
                )
                fired += 1
        return fired
