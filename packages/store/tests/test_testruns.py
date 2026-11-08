"""Test runs against Postgres, without a worker (integration tier): one run per
suite is active, one waits behind it, and requests meanwhile join the waiting one."""

import uuid

import pytest
import sqlalchemy as sa

from stratafi_components import default_registry
from stratafi_core.suites import parse_suite
from stratafi_store import testruns
from stratafi_store.jobs import JobQueue
from stratafi_store.tables import jobs, test_runs, workspaces

pytestmark = pytest.mark.integration


@pytest.fixture
def workspace(migrated_engine: sa.Engine) -> uuid.UUID:
    workspace_id = uuid.uuid4()
    with migrated_engine.begin() as conn:
        conn.execute(sa.insert(workspaces).values(id=workspace_id, slug="tests", name="tests"))
    return workspace_id


def suite(name: str = "guard") -> object:
    return parse_suite(
        {
            "schema": "stratafi.test-suite.v1",
            "name": name,
            "target": {"pipeline": "p"},
            "corpus": "c",
            "checks": [{"id": "parsed", "stage": "parse", "assert": "all_succeeded"}],
        }
    )


def start_jobs(engine: sa.Engine, run_id: uuid.UUID) -> int:
    with engine.connect() as conn:
        return conn.execute(
            sa.select(sa.func.count()).where(
                jobs.c.kind == testruns.START_JOB,
                jobs.c.payload["test_run_id"].astext == str(run_id),
            )
        ).scalar_one()


def test_one_run_is_active_and_requests_join_the_waiting_one(
    migrated_engine: sa.Engine, workspace: uuid.UUID
) -> None:
    queue = JobQueue(migrated_engine)
    orchestrator = testruns.TestOrchestrator(migrated_engine, queue, default_registry())

    def request(trigger: str = "api", version: object = None) -> tuple[testruns.TestRun, bool]:
        with migrated_engine.begin() as conn:
            stored, _ = testruns.save_suite(
                conn, workspace_id=workspace, suite=version or suite(), created_by="u"
            )
            return testruns.request_run(
                conn, queue, workspace_id=workspace, suite=stored, trigger=trigger, created_by="u"
            )

    first, joined = request()
    assert (first.status, joined, first.triggers) == ("queued", False, ["api"])
    same, joined = request("ci")
    assert (same.id, joined, same.triggers) == (first.id, True, ["api", "ci"])
    # The first run becomes the active one; the next request waits behind it.
    with migrated_engine.begin() as conn:
        conn.execute(
            sa.update(test_runs).where(test_runs.c.id == first.id).values(status="running")
        )
    second, joined = request("schedule")
    assert (second.status, joined) == ("queued", False)
    newer = suite().model_copy(update={"description": "v2"})
    third, joined = request("corpus_version_frozen", newer)
    assert (third.id, joined, third.suite_version) == (second.id, True, 2)
    assert third.triggers == ["schedule", "corpus_version_frozen"]
    # While the first is active, the waiting run doesn't start; when it ends, the waiting one's
    # start is queued.
    assert orchestrator.start(second.id).status == "queued"
    before = start_jobs(migrated_engine, second.id)
    assert orchestrator.fail(first.id, "evaluation_failed").status == "error"
    assert start_jobs(migrated_engine, second.id) == before + 1
    # Its references don't resolve here (there's no pipeline spec p), so it ends as an error.
    ended = orchestrator.start(second.id)
    assert (ended.status, ended.error_code) == ("error", "target_pipeline_not_found")
    # A second active run of the suite is refused by the database itself.
    with pytest.raises(sa.exc.IntegrityError), migrated_engine.begin() as conn:
        conn.execute(
            sa.update(test_runs)
            .where(test_runs.c.id.in_([first.id, second.id]))
            .values(status="running")
        )
