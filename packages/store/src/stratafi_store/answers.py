"""Answer runs, their per-query scores, and calibration labels.

An answer run is created ``queued`` for a succeeded eval run, with its configuration: the
generator and the judge (a component and its parameters), the prompts they use, pinned, and how
many of each query's final passages the generator reads. The worker's ``answer_run.start`` job
answers, judges and scores every query, and keeps the answers and their passages in an
artifact; ``answer_results`` holds each query's numbers and flags and the hash of its answer.
A run with the same configuration, revisions and eval results as an earlier one reuses its
artifact (``results_key``), as eval runs do.

Labels are people's judgements of one answer (``calibration_labels``): faithful, correct,
abstained. A label attaches to the answer's hash, so it follows the answer into every run that
reuses it, and a person's later label replaces their earlier one. ``label_pairs`` pairs the
labels with what the judge or the abstention detector said, for
``stratafi_core.answers.agreement``.
"""

import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection

from stratafi_core.answers import LABEL_DIMENSIONS, automatic_label
from stratafi_store.tables import answer_results, answer_runs, calibration_labels

START_JOB = "answer_run.start"
FINISHED = ("succeeded", "failed", "cancelled")


@dataclass(frozen=True)
class AnswerRun:
    id: uuid.UUID
    workspace_id: uuid.UUID
    eval_run_id: uuid.UUID
    status: str
    config: dict[str, Any]
    results_key: str | None
    results_cached: bool
    metrics: dict[str, Any] | None
    usage: dict[str, Any]
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
class AnswerResult:
    item_id: str
    ordinal: int
    answer_key: str
    score: dict[str, Any]


@dataclass(frozen=True)
class Label:
    answer_key: str
    item_id: str
    dimension: str
    label: bool
    labeled_by: str
    labeled_at: datetime


def _run(row: Any) -> AnswerRun:
    return AnswerRun(**{column.name: getattr(row, column.name) for column in answer_runs.columns})


def create_answer_run(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    eval_run_id: uuid.UUID,
    config: dict[str, Any],
    created_by: str,
) -> AnswerRun:
    row = conn.execute(
        sa.insert(answer_runs)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            eval_run_id=eval_run_id,
            config=config,
            created_by=created_by,
        )
        .returning(*answer_runs.columns)
    ).one()
    return _run(row)


def get_answer_run(
    conn: Connection, workspace_id: uuid.UUID, answer_run_id: uuid.UUID
) -> AnswerRun | None:
    row = conn.execute(
        sa.select(answer_runs).where(
            answer_runs.c.workspace_id == workspace_id, answer_runs.c.id == answer_run_id
        )
    ).first()
    return _run(row) if row is not None else None


def list_answer_runs(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    eval_run_id: uuid.UUID | None = None,
) -> list[AnswerRun]:
    """Newest first; one eval run's, or every one."""
    query = sa.select(answer_runs).where(answer_runs.c.workspace_id == workspace_id)
    if eval_run_id is not None:
        query = query.where(answer_runs.c.eval_run_id == eval_run_id)
    rows = conn.execute(
        query.order_by(answer_runs.c.created_at.desc(), answer_runs.c.id.desc()).limit(limit)
    )
    return [_run(row) for row in rows]


def begin(conn: Connection, answer_run_id: uuid.UUID) -> AnswerRun | None:
    """Mark a queued (or a retried running) run running; None for a finished one."""
    row = conn.execute(
        sa.update(answer_runs)
        .where(answer_runs.c.id == answer_run_id, answer_runs.c.status.in_(("queued", "running")))
        .values(status="running", started_at=sa.func.now(), updated_at=sa.func.now())
        .returning(*answer_runs.columns)
    ).first()
    return _run(row) if row is not None else None


def finish(
    conn: Connection,
    run: AnswerRun,
    *,
    status: str,
    results: Sequence[AnswerResult] = (),
    **values: Any,
) -> AnswerRun:
    """End a run: its per-query results (replacing any from an earlier attempt), status and
    whatever else ``values`` sets (``metrics``, ``usage``, ``results_key``, ``error_code``)."""
    conn.execute(sa.delete(answer_results).where(answer_results.c.answer_run_id == run.id))
    if results:
        conn.execute(
            sa.insert(answer_results),
            [
                {
                    "answer_run_id": run.id,
                    "item_id": result.item_id,
                    "ordinal": result.ordinal,
                    "answer_key": result.answer_key,
                    "score": result.score,
                }
                for result in results
            ],
        )
    row = conn.execute(
        sa.update(answer_runs)
        .where(answer_runs.c.id == run.id)
        .values(status=status, finished_at=sa.func.now(), updated_at=sa.func.now(), **values)
        .returning(*answer_runs.columns)
    ).one()
    return _run(row)


def results_of(conn: Connection, answer_run_id: uuid.UUID) -> list[AnswerResult]:
    rows = conn.execute(
        sa.select(answer_results)
        .where(answer_results.c.answer_run_id == answer_run_id)
        .order_by(answer_results.c.ordinal)
    )
    return [AnswerResult(row.item_id, row.ordinal, row.answer_key, row.score) for row in rows]


# --- labels --------------------------------------------------------------------------------


def set_label(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    answer_key: str,
    item_id: str,
    dimension: str,
    label: bool,
    labeled_by: str,
) -> Label:
    """A person's label for one answer on one dimension, replacing their earlier one."""
    if dimension not in LABEL_DIMENSIONS:
        raise ValueError("unknown dimension")
    statement = postgresql.insert(calibration_labels).values(
        id=uuid.uuid4(),
        workspace_id=workspace_id,
        answer_key=answer_key,
        item_id=item_id,
        dimension=dimension,
        label=label,
        labeled_by=labeled_by,
    )
    row = conn.execute(
        statement.on_conflict_do_update(
            index_elements=["workspace_id", "answer_key", "dimension", "labeled_by"],
            set_={"label": statement.excluded.label, "labeled_at": sa.func.now()},
        ).returning(*calibration_labels.columns)
    ).one()
    return Label(
        row.answer_key, row.item_id, row.dimension, row.label, row.labeled_by, row.labeled_at
    )


def labels_for(
    conn: Connection, workspace_id: uuid.UUID, answer_keys: Iterable[str]
) -> dict[str, list[Label]]:
    """Every label on these answers, by answer."""
    keys = list(set(answer_keys))
    if not keys:
        return {}
    rows = conn.execute(
        sa.select(calibration_labels)
        .where(
            calibration_labels.c.workspace_id == workspace_id,
            calibration_labels.c.answer_key.in_(keys),
        )
        .order_by(calibration_labels.c.labeled_at, calibration_labels.c.id)
    )
    found: dict[str, list[Label]] = defaultdict(list)
    for row in rows:
        found[row.answer_key].append(
            Label(
                row.answer_key,
                row.item_id,
                row.dimension,
                row.label,
                row.labeled_by,
                row.labeled_at,
            )
        )
    return dict(found)


def label_pairs(
    results: Sequence[AnswerResult], labels: dict[str, list[Label]]
) -> dict[str, list[tuple[bool, bool]]]:
    """For each dimension, (automatic, human) pairs: one per label on an answer the judge or
    the abstention detector also labelled."""
    pairs: dict[str, list[tuple[bool, bool]]] = {dimension: [] for dimension in LABEL_DIMENSIONS}
    for result in results:
        for label in labels.get(result.answer_key, []):
            automatic = automatic_label(result.score, label.dimension)
            if automatic is not None:
                pairs[label.dimension].append((automatic, label.label))
    return pairs
