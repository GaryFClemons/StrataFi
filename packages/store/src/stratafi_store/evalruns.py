"""Eval runs: planning, executing and recording an evaluation.

An eval run searches a finished pipeline run's indexes with a retrieval config, for every item of
a frozen eval set version's split, and scores the results. Its lifecycle is a pipeline run's:
``queued`` → ``running`` → ``succeeded`` or ``failed``, or ``cancelling`` → ``cancelled``.

``EvalOrchestrator`` drives it, always holding the eval run's row lock:

- ``start`` (the worker's ``eval.start`` job) records the shards it planned: a shard whose
  ranked lists exist already is ``cached``; the others are ``queued`` with ``eval.shard`` jobs.
  When nothing is left to search, ``eval.finish`` is queued at once.
- ``complete_shard``, ``fail_shard`` and ``cancel_shard`` (``eval.shard``) record a shard; the
  last one to finish queues ``eval.finish``, and a failed one fails the run.
- ``finish`` (``eval.finish``) writes each query's result, the aggregates and the record.

Each change notifies an ``eval_run`` event (IDs, status and counts), with the eval run's ID as
``run_id``, so ``GET /events?run_id=`` follows eval runs too.
"""

import json
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from stratafi_store.jobs import EVENTS_CHANNEL, JobQueue
from stratafi_store.lineage import ArtifactInput, ArtifactRecord, record_artifact
from stratafi_store.runs import FinishHook
from stratafi_store.tables import (
    annotations,
    artifacts,
    eval_query_results,
    eval_runs,
    eval_shards,
    jobs,
)

START_JOB = "eval.start"
SHARD_JOB = "eval.shard"
FINISH_JOB = "eval.finish"
ACTIVE_SHARD_STATUSES = ("queued", "running")
DONE_SHARD_STATUSES = ("succeeded", "cached")
FINISHED = ("succeeded", "failed", "cancelled")


@dataclass(frozen=True)
class EvalRun:
    id: uuid.UUID
    workspace_id: uuid.UUID
    status: str
    pipeline_run_id: uuid.UUID
    retrieval_config_id: uuid.UUID
    eval_set_version_id: uuid.UUID
    split: str
    options: dict[str, Any]
    retrieval_hash: str
    results_key: str | None
    results_cached: bool
    test_read: bool
    context: dict[str, Any]
    manifest: dict[str, Any] | None
    manifest_hash: str | None
    metrics: dict[str, Any] | None
    counts: dict[str, Any]
    error_code: str | None
    created_by: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def finished(self) -> bool:
        return self.status in FINISHED


@dataclass(frozen=True)
class EvalShard:
    id: uuid.UUID
    workspace_id: uuid.UUID
    eval_run_id: uuid.UUID
    ordinal: int
    cache_key: str
    item_ids: list[str]
    status: str
    cache_hit: bool
    job_id: uuid.UUID | None
    attempts: int
    error_code: str | None
    metrics: dict[str, Any]
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: float | None


@dataclass(frozen=True)
class QueryResult:
    item_id: str
    ordinal: int
    shard: int
    split: str
    tags: list[str]
    file_types: list[str]
    has_evidence: bool
    hit: bool | None
    first_hit_rank: int | None
    lost_at: str | None
    avoid_violation: bool | None
    must_not_return_violation: bool
    score: dict[str, Any]


@dataclass(frozen=True)
class ShardPlan:
    cache_key: str
    item_ids: list[str]
    cached: bool


def _run(row: Any) -> EvalRun:
    return EvalRun(**{column.name: getattr(row, column.name) for column in eval_runs.columns})


def _shard(row: Any) -> EvalShard:
    return EvalShard(**{column.name: getattr(row, column.name) for column in eval_shards.columns})


def _result(row: Any) -> QueryResult:
    return QueryResult(
        **{
            column.name: getattr(row, column.name)
            for column in eval_query_results.columns
            if column.name not in ("workspace_id", "eval_run_id")
        }
    )


# --- reading ----------------------------------------------------------------------------------


def create_eval_run(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    pipeline_run_id: uuid.UUID,
    retrieval_config_id: uuid.UUID,
    eval_set_version_id: uuid.UUID,
    split: str,
    options: Mapping[str, Any],
    retrieval_hash: str,
    test_read: bool,
    created_by: str,
) -> EvalRun:
    row = conn.execute(
        sa.insert(eval_runs)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            pipeline_run_id=pipeline_run_id,
            retrieval_config_id=retrieval_config_id,
            eval_set_version_id=eval_set_version_id,
            split=split,
            options=dict(options),
            retrieval_hash=retrieval_hash,
            test_read=test_read,
            created_by=created_by,
        )
        .returning(*eval_runs.columns)
    ).one()
    run = _run(row)
    _notify(conn, run)
    return run


def get_eval_run(
    conn: Connection, workspace_id: uuid.UUID | None, eval_run_id: uuid.UUID
) -> EvalRun | None:
    query = sa.select(eval_runs).where(eval_runs.c.id == eval_run_id)
    if workspace_id is not None:
        query = query.where(eval_runs.c.workspace_id == workspace_id)
    row = conn.execute(query).first()
    return _run(row) if row is not None else None


def eval_runs_by_id(
    conn: Connection, eval_run_ids: Iterable[uuid.UUID]
) -> dict[uuid.UUID, EvalRun]:
    ids = list(eval_run_ids)
    if not ids:
        return {}
    rows = conn.execute(sa.select(eval_runs).where(eval_runs.c.id.in_(ids)))
    return {row.id: _run(row) for row in rows}


def list_eval_runs(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    after: tuple[datetime, uuid.UUID] | None = None,
    pipeline_run_id: uuid.UUID | None = None,
    eval_set_version_id: uuid.UUID | None = None,
    status: str | None = None,
    include_archived: bool = True,
) -> list[EvalRun]:
    """Newest first."""
    query = sa.select(eval_runs).where(eval_runs.c.workspace_id == workspace_id)
    if not include_archived:
        query = query.where(
            ~sa.exists().where(
                annotations.c.eval_run_id == eval_runs.c.id, annotations.c.archived_at.is_not(None)
            )
        )
    if pipeline_run_id is not None:
        query = query.where(eval_runs.c.pipeline_run_id == pipeline_run_id)
    if eval_set_version_id is not None:
        query = query.where(eval_runs.c.eval_set_version_id == eval_set_version_id)
    if status is not None:
        query = query.where(eval_runs.c.status == status)
    if after is not None:
        query = query.where(sa.tuple_(eval_runs.c.created_at, eval_runs.c.id) < after)
    rows = conn.execute(
        query.order_by(eval_runs.c.created_at.desc(), eval_runs.c.id.desc()).limit(limit)
    )
    return [_run(row) for row in rows]


def shards_of(conn: Connection, eval_run_id: uuid.UUID) -> list[EvalShard]:
    rows = conn.execute(
        sa.select(eval_shards)
        .where(eval_shards.c.eval_run_id == eval_run_id)
        .order_by(eval_shards.c.ordinal)
    )
    return [_shard(row) for row in rows]


def get_shard(conn: Connection, shard_id: uuid.UUID) -> EvalShard | None:
    row = conn.execute(sa.select(eval_shards).where(eval_shards.c.id == shard_id)).first()
    return _shard(row) if row is not None else None


def query_results(
    conn: Connection,
    eval_run_id: uuid.UUID,
    *,
    limit: int | None = None,
    after_ordinal: int | None = None,
    outcome: str | None = None,
    lost_at: str | None = None,
    tag: str | None = None,
) -> list[QueryResult]:
    """An eval run's per-query results, in the eval set's order. ``outcome`` is ``hit``,
    ``miss`` or ``abstain`` (no evidence)."""
    table = eval_query_results
    query = sa.select(table).where(table.c.eval_run_id == eval_run_id)
    if outcome == "hit":
        query = query.where(table.c.hit.is_(True))
    elif outcome == "miss":
        query = query.where(table.c.hit.is_(False))
    elif outcome == "abstain":
        query = query.where(table.c.has_evidence.is_(False))
    if lost_at is not None:
        query = query.where(table.c.lost_at == lost_at)
    if tag is not None:
        query = query.where(table.c.tags.contains([tag]))
    if after_ordinal is not None:
        query = query.where(table.c.ordinal > after_ordinal)
    query = query.order_by(table.c.ordinal)
    if limit is not None:
        query = query.limit(limit)
    return [_result(row) for row in conn.execute(query)]


def query_result(conn: Connection, eval_run_id: uuid.UUID, item_id: str) -> QueryResult | None:
    row = conn.execute(
        sa.select(eval_query_results).where(
            eval_query_results.c.eval_run_id == eval_run_id,
            eval_query_results.c.item_id == item_id,
        )
    ).first()
    return _result(row) if row is not None else None


def counts(conn: Connection, eval_run_id: uuid.UUID) -> dict[str, Any]:
    statuses = Counter(
        dict(
            conn.execute(
                sa.select(eval_shards.c.status, sa.func.count())
                .where(eval_shards.c.eval_run_id == eval_run_id)
                .group_by(eval_shards.c.status)
            ).all()
        )
    )
    queries = conn.execute(
        sa.select(
            sa.func.coalesce(sa.func.sum(sa.func.jsonb_array_length(eval_shards.c.item_ids)), 0)
        ).where(eval_shards.c.eval_run_id == eval_run_id)
    ).scalar_one()
    return {"shards": dict(statuses), "queries": int(queries)}


def _notify(conn: Connection, run: EvalRun, run_counts: Mapping[str, Any] | None = None) -> None:
    event: dict[str, Any] = {
        "type": "eval_run",
        "workspace_id": str(run.workspace_id),
        "run_id": str(run.id),
        "status": run.status,
    }
    if run_counts is not None:
        event["counts"] = dict(run_counts)
    conn.execute(
        sa.text("SELECT pg_notify(:channel, :payload)"),
        {"channel": EVENTS_CHANNEL, "payload": json.dumps(event, separators=(",", ":"))},
    )


def artifact_exists(conn: Connection, workspace_id: uuid.UUID, cache_key: str) -> bool:
    return conn.execute(
        sa.select(
            sa.exists().where(
                artifacts.c.workspace_id == workspace_id, artifacts.c.cache_key == cache_key
            )
        )
    ).scalar_one()


# --- the orchestrator -------------------------------------------------------------------------


class EvalOrchestrator:
    def __init__(
        self,
        engine: Engine,
        queue: JobQueue,
        *,
        max_attempts: int = 3,
        on_finish: FinishHook | None = None,
    ) -> None:
        self.engine = engine
        self.queue = queue
        self.max_attempts = max_attempts
        self.on_finish = on_finish

    def _finished(self, conn: Connection, run: EvalRun) -> None:
        if self.on_finish is not None:
            self.on_finish(conn, "eval_run", run.workspace_id, run.id)

    def _lock(self, conn: Connection, eval_run_id: uuid.UUID) -> EvalRun | None:
        row = conn.execute(
            sa.select(eval_runs).where(eval_runs.c.id == eval_run_id).with_for_update()
        ).first()
        return _run(row) if row is not None else None

    def _update(self, conn: Connection, run: EvalRun, **values: Any) -> EvalRun:
        row = conn.execute(
            sa.update(eval_runs)
            .where(eval_runs.c.id == run.id)
            .values(updated_at=sa.func.now(), **values)
            .returning(*eval_runs.columns)
        ).one()
        return _run(row)

    def _enqueue(
        self, conn: Connection, run: EvalRun, kind: str, payload: dict[str, Any]
    ) -> uuid.UUID:
        job = self.queue.enqueue_in(
            conn,
            workspace_id=run.workspace_id,
            kind=kind,
            payload={"eval_run_id": str(run.id), **payload},
            resource_class="io",
            priority=5,
            max_attempts=self.max_attempts,
        )
        return job.id

    def _settle(self, conn: Connection, run: EvalRun) -> EvalRun:
        run_counts = counts(conn, run.id)
        run = self._update(conn, run, counts=run_counts)
        _notify(conn, run, run_counts)
        return run

    def _after_shard(self, conn: Connection, run: EvalRun) -> EvalRun:
        """Queue the finish once no shard is left to run, or settle a cancelled run."""
        shard_list = shards_of(conn, run.id)
        if any(shard.status in ACTIVE_SHARD_STATUSES for shard in shard_list):
            return self._settle(conn, run)
        if run.status == "cancelling":
            run = self._update(conn, run, status="cancelled", finished_at=sa.func.now())
            self._finished(conn, run)
        elif run.status == "running":
            if any(shard.status == "failed" for shard in shard_list):
                run = self._update(
                    conn,
                    run,
                    status="failed",
                    error_code="shards_failed",
                    finished_at=sa.func.now(),
                )
                self._finished(conn, run)
            else:
                self._enqueue(conn, run, FINISH_JOB, {})
        return self._settle(conn, run)

    # --- eval.start ---------------------------------------------------------------------------

    def start(
        self,
        eval_run_id: uuid.UUID,
        *,
        context: Mapping[str, Any],
        results_key: str,
        plans: Sequence[ShardPlan],
    ) -> EvalRun | None:
        with self.engine.begin() as conn:
            run = self._lock(conn, eval_run_id)
            if run is None or run.status != "queued":
                return run
            run = self._update(
                conn,
                run,
                status="running",
                started_at=sa.func.now(),
                context=dict(context),
                results_key=results_key,
            )
            for ordinal, plan in enumerate(plans):
                shard_id = uuid.uuid4()
                values: dict[str, Any] = {
                    "id": shard_id,
                    "workspace_id": run.workspace_id,
                    "eval_run_id": run.id,
                    "ordinal": ordinal,
                    "cache_key": plan.cache_key,
                    "item_ids": list(plan.item_ids),
                    "status": "cached" if plan.cached else "queued",
                    "cache_hit": plan.cached,
                }
                if plan.cached:
                    values.update(
                        started_at=sa.func.now(), finished_at=sa.func.now(), duration_ms=0.0
                    )
                conn.execute(sa.insert(eval_shards).values(**values))
                if not plan.cached:
                    job_id = self._enqueue(conn, run, SHARD_JOB, {"shard_id": str(shard_id)})
                    conn.execute(
                        sa.update(eval_shards)
                        .where(eval_shards.c.id == shard_id)
                        .values(job_id=job_id)
                    )
            return self._after_shard(conn, run)

    def fail_run(self, eval_run_id: uuid.UUID, code: str) -> EvalRun | None:
        """An eval run that couldn't start, or couldn't finish."""
        with self.engine.begin() as conn:
            run = self._lock(conn, eval_run_id)
            if run is None or run.finished:
                return run
            run = self._update(
                conn, run, status="failed", error_code=code, finished_at=sa.func.now()
            )
            self._finished(conn, run)
            return self._settle(conn, run)

    # --- eval.shard ---------------------------------------------------------------------------

    def begin_shard(self, shard_id: uuid.UUID, attempts: int) -> tuple[EvalShard, EvalRun] | None:
        with self.engine.begin() as conn:
            shard = get_shard(conn, shard_id)
            if shard is None:
                return None
            run = self._lock(conn, shard.eval_run_id)
            if run is None or shard.status not in ACTIVE_SHARD_STATUSES:
                return None
            if run.status != "running":
                self._cancel_shard(conn, shard)
                self._after_shard(conn, run)
                return None
            row = conn.execute(
                sa.update(eval_shards)
                .where(eval_shards.c.id == shard.id)
                .values(
                    status="running",
                    attempts=attempts,
                    started_at=sa.func.coalesce(eval_shards.c.started_at, sa.func.now()),
                )
                .returning(*eval_shards.columns)
            ).one()
            return _shard(row), run

    def complete_shard(
        self,
        shard_id: uuid.UUID,
        *,
        record: ArtifactRecord,
        inputs: Iterable[str | ArtifactInput],
        metrics: Mapping[str, Any],
        duration_ms: float,
    ) -> EvalRun | None:
        with self.engine.begin() as conn:
            shard = get_shard(conn, shard_id)
            if shard is None:
                return None
            run = self._lock(conn, shard.eval_run_id)
            if run is None or shard.status not in ACTIVE_SHARD_STATUSES:
                return run
            record_artifact(conn, record, list(inputs))
            conn.execute(
                sa.update(eval_shards)
                .where(eval_shards.c.id == shard.id)
                .values(
                    status="succeeded",
                    metrics=dict(metrics),
                    finished_at=sa.func.now(),
                    duration_ms=round(duration_ms, 3),
                    error_code=None,
                )
            )
            return self._after_shard(conn, run)

    def fail_shard(
        self, shard_id: uuid.UUID, code: str, *, duration_ms: float | None = None
    ) -> tuple[EvalRun | None, list[uuid.UUID]]:
        """Fail a shard and its run; returns the run and the other shards' jobs to cancel."""
        with self.engine.begin() as conn:
            shard = get_shard(conn, shard_id)
            if shard is None:
                return None, []
            run = self._lock(conn, shard.eval_run_id)
            if run is None or shard.status not in ACTIVE_SHARD_STATUSES:
                return run, []
            conn.execute(
                sa.update(eval_shards)
                .where(eval_shards.c.id == shard.id)
                .values(
                    status="failed",
                    error_code=code,
                    finished_at=sa.func.now(),
                    duration_ms=round(duration_ms, 3) if duration_ms is not None else None,
                )
            )
            to_cancel = [
                other.job_id
                for other in shards_of(conn, run.id)
                if other.status == "queued" and other.job_id is not None
            ]
            for other in shards_of(conn, run.id):
                if other.status == "queued":
                    self._cancel_shard(conn, other)
            return self._after_shard(conn, run), to_cancel

    def _cancel_shard(self, conn: Connection, shard: EvalShard) -> None:
        conn.execute(
            sa.update(eval_shards)
            .where(eval_shards.c.id == shard.id, eval_shards.c.status.in_(ACTIVE_SHARD_STATUSES))
            .values(status="cancelled", finished_at=sa.func.now())
        )

    def cancel_shard(self, shard_id: uuid.UUID) -> EvalRun | None:
        with self.engine.begin() as conn:
            shard = get_shard(conn, shard_id)
            if shard is None:
                return None
            run = self._lock(conn, shard.eval_run_id)
            if run is None:
                return None
            self._cancel_shard(conn, shard)
            return self._after_shard(conn, run)

    # --- eval.finish --------------------------------------------------------------------------

    def begin_finish(self, eval_run_id: uuid.UUID) -> EvalRun | None:
        """The run, if it's ready to finish: running, with every shard done."""
        with self.engine.connect() as conn:
            run = get_eval_run(conn, None, eval_run_id)
            if run is None or run.status != "running":
                return None
            if any(shard.status not in DONE_SHARD_STATUSES for shard in shards_of(conn, run.id)):
                return None
            return run

    def finish(
        self,
        eval_run_id: uuid.UUID,
        *,
        results: Sequence[Mapping[str, Any]],
        metrics: Mapping[str, Any],
        manifest: Mapping[str, Any],
        manifest_hash: str,
        record: ArtifactRecord | None,
        inputs: Sequence[str],
        results_cached: bool,
    ) -> EvalRun | None:
        with self.engine.begin() as conn:
            run = self._lock(conn, eval_run_id)
            if run is None or run.status != "running":
                return run
            if record is not None:
                record_artifact(conn, record, list(inputs))
            conn.execute(
                sa.delete(eval_query_results).where(eval_query_results.c.eval_run_id == run.id)
            )
            if results:
                conn.execute(
                    sa.insert(eval_query_results),
                    [{"workspace_id": run.workspace_id, "eval_run_id": run.id, **row}
                     for row in results],
                )  # fmt: skip
            run = self._update(
                conn,
                run,
                status="succeeded",
                finished_at=sa.func.now(),
                metrics=dict(metrics),
                manifest=dict(manifest),
                manifest_hash=manifest_hash,
                results_cached=results_cached,
            )
            self._finished(conn, run)
            return self._settle(conn, run)

    # --- the client's side and the sweep ------------------------------------------------------

    def cancel(
        self, eval_run_id: uuid.UUID, workspace_id: uuid.UUID
    ) -> tuple[EvalRun, list[uuid.UUID]] | None:
        with self.engine.begin() as conn:
            run = self._lock(conn, eval_run_id)
            if run is None or run.workspace_id != workspace_id:
                return None
            if run.finished:
                return run, []
            shard_list = shards_of(conn, run.id)
            job_ids = [
                shard.job_id
                for shard in shard_list
                if shard.status in ACTIVE_SHARD_STATUSES and shard.job_id is not None
            ]
            if run.status == "queued":
                run = self._update(conn, run, status="cancelled", finished_at=sa.func.now())
                self._finished(conn, run)
                return self._settle(conn, run), job_ids
            run = self._update(conn, run, status="cancelling")
            for shard in shard_list:
                if shard.status == "queued":
                    self._cancel_shard(conn, shard)
            return self._after_shard(conn, run), job_ids

    def reconcile(self) -> int:
        """Settle what a job ended without recording: a shard whose job failed or was cancelled
        outside its handler, and a run whose start or finish job did. Returns how many."""
        with self.engine.connect() as conn:
            orphaned = conn.execute(
                sa.select(eval_shards.c.id, jobs.c.status, jobs.c.error_code)
                .join(jobs, jobs.c.id == eval_shards.c.job_id)
                .where(
                    eval_shards.c.status.in_(ACTIVE_SHARD_STATUSES),
                    jobs.c.status.in_(("failed", "cancelled")),
                )
            ).all()
            stuck = conn.execute(
                sa.select(eval_runs.c.id, jobs.c.error_code)
                .join(
                    jobs,
                    sa.and_(
                        jobs.c.kind.in_((START_JOB, FINISH_JOB)),
                        jobs.c.workspace_id == eval_runs.c.workspace_id,
                        jobs.c.payload["eval_run_id"].astext == sa.cast(eval_runs.c.id, sa.Text),
                    ),
                )
                .where(
                    eval_runs.c.status.in_(("queued", "running")),
                    jobs.c.status.in_(("failed", "cancelled")),
                )
            ).all()
        for row in orphaned:
            if row.status == "cancelled":
                self.cancel_shard(row.id)
            else:
                self.fail_shard(row.id, row.error_code or "job_failed")
        for row in stuck:
            self.fail_run(row.id, row.error_code or "job_failed")
        return len(orphaned) + len(stuck)
