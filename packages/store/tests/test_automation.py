"""Schedules against Postgres, without a worker (integration tier): a suite's
schedules follow its latest version, and only the scheduler that holds the advisory lock fires a
due schedule, once."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa

from stratafi_core.suites import parse_suite
from stratafi_store import automation, testruns
from stratafi_store.automation import SCHEDULER_LOCK, SuiteScheduler
from stratafi_store.jobs import JobQueue
from stratafi_store.tables import test_runs, test_schedules, workspaces

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)  # a Monday


@pytest.fixture
def workspace(migrated_engine: sa.Engine) -> uuid.UUID:
    workspace_id = uuid.uuid4()
    with migrated_engine.begin() as conn:
        conn.execute(sa.insert(workspaces).values(id=workspace_id, slug="sched", name="sched"))
    return workspace_id


def suite(*schedules: str, zone: str = "UTC"):
    return parse_suite(
        {
            "schema": "stratafi.test-suite.v1",
            "name": "nightly",
            "target": {"pipeline": "p"},
            "corpus": "c",
            "time_zone": zone,
            "triggers": [{"schedule": text} for text in schedules],
            "checks": [{"id": "parsed", "stage": "parse", "assert": "all_succeeded"}],
        }
    )


def save(engine: sa.Engine, workspace: uuid.UUID, version) -> None:
    with engine.begin() as conn:
        testruns.save_suite(conn, workspace_id=workspace, suite=version, created_by="u")
        automation.sync_schedules(conn, workspace, version, now=NOW)


def test_schedules_follow_the_latest_version(
    migrated_engine: sa.Engine, workspace: uuid.UUID
) -> None:
    save(migrated_engine, workspace, suite("0 2 * * *", "30 7 * * mon"))
    with migrated_engine.connect() as conn:
        found = automation.schedules_of(conn, workspace, "nightly")
    assert [(s.cron, s.next_fire_at) for s in found] == [
        ("0 2 * * *", datetime(2026, 11, 3, 2, 0, tzinfo=UTC)),
        ("30 7 * * mon", datetime(2026, 11, 9, 7, 30, tzinfo=UTC)),
    ]
    save(migrated_engine, workspace, suite("0 2 * * *", zone="America/Chicago"))
    with migrated_engine.connect() as conn:
        found = automation.schedules_of(conn, workspace, "nightly")
    # 02:00 in Chicago, in standard time after Nov 1, is 08:00 UTC.
    assert [(s.cron, s.time_zone, s.next_fire_at) for s in found] == [
        ("0 2 * * *", "America/Chicago", datetime(2026, 11, 3, 8, 0, tzinfo=UTC))
    ]


def test_one_leader_fires_a_due_schedule_once(
    migrated_engine: sa.Engine, workspace: uuid.UUID
) -> None:
    save(migrated_engine, workspace, suite("0 2 * * *"))
    with migrated_engine.begin() as conn:
        conn.execute(sa.update(test_schedules).values(next_fire_at=NOW - timedelta(minutes=5)))
    queue = JobQueue(migrated_engine)
    first, second = SuiteScheduler(migrated_engine, queue), SuiteScheduler(migrated_engine, queue)
    # Another worker leads: holding the lock, it keeps this one from firing.
    with migrated_engine.begin() as conn:
        conn.execute(
            sa.text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": SCHEDULER_LOCK},
        )
        assert first.tick(NOW) == 0
    assert first.tick(NOW) == 1
    assert second.tick(NOW) == 0
    with migrated_engine.connect() as conn:
        runs = conn.execute(sa.select(test_runs.c.trigger, test_runs.c.created_by)).all()
        schedule = conn.execute(sa.select(test_schedules)).one()
    assert [tuple(run) for run in runs] == [("schedule", "system:scheduler")]
    assert schedule.next_fire_at == datetime(2026, 11, 3, 2, 0, tzinfo=UTC)
    assert schedule.last_fired_at == NOW
    # A schedule the suite no longer has is dropped instead of fired.
    save(migrated_engine, workspace, suite())
    with migrated_engine.connect() as conn:
        assert conn.execute(sa.select(sa.func.count()).select_from(test_schedules)).scalar() == 0
