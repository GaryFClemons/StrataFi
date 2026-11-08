"""Drafted questions and their review queue.

A batch drafts questions from a sample of one pipeline run's passages, for one eval set: the
worker's ``questions.draft`` job asks a generator (``stratafi_core.answers.DRAFT_QUESTIONS``)
and keeps each question whose evidence really is a sentence of its passage, as a ``proposed``
candidate with that passage's document, version and section.

A person reviews each candidate (``decide``): accepts it, maybe with a better query, evidence,
gold document, split or tags, or rejects it; a candidate accepted without a gold document and
evidence is refused (``evidence_required``). ``add_accepted`` adds every accepted candidate to a
new version of the eval set, the latest version's items and then the new ones, and marks them
``added``. Nothing enters a version without a person's acceptance.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from stratafi_core.evalsets import EvalItem, EvidenceRef
from stratafi_store import evalsets
from stratafi_store.tables import question_batches, question_candidates

DRAFT_JOB = "questions.draft"
GENERATED_TAG = "generated"


class CandidateRefused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class QuestionBatch:
    id: uuid.UUID
    workspace_id: uuid.UUID
    eval_set_id: uuid.UUID
    pipeline_run_id: uuid.UUID
    status: str
    config: dict[str, Any]
    counts: dict[str, Any]
    error_code: str | None
    created_by: str
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True)
class Candidate:
    id: uuid.UUID
    workspace_id: uuid.UUID
    eval_set_id: uuid.UUID
    batch_id: uuid.UUID
    ordinal: int
    document_id: uuid.UUID | None
    chunk_id: str | None
    doc_key: str
    version: str | None
    section: str | None
    query: str
    evidence: str
    drafted_query: str
    split: str
    tags: list[str]
    status: str
    decided_by: str | None
    decided_at: datetime | None
    item_id: str | None
    added_version_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime

    @property
    def edited(self) -> bool:
        return self.query != self.drafted_query

    def item(self) -> EvalItem:
        """The eval set item it becomes, ``gen-`` and its ID, tagged ``generated``."""
        return EvalItem(
            id=f"gen-{self.id.hex[:12]}",
            query=self.query,
            expected=[
                EvidenceRef(
                    doc_key=self.doc_key,
                    version=self.version,
                    section=self.section or None,
                    evidence=self.evidence,
                )
            ],
            tags=list(dict.fromkeys([*self.tags, GENERATED_TAG])),
            split=self.split,  # type: ignore[arg-type]
        )


def _batch(row: Any) -> QuestionBatch:
    return QuestionBatch(
        **{column.name: getattr(row, column.name) for column in question_batches.columns}
    )


def _candidate(row: Any) -> Candidate:
    return Candidate(
        **{column.name: getattr(row, column.name) for column in question_candidates.columns}
    )


def create_batch(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    eval_set_id: uuid.UUID,
    pipeline_run_id: uuid.UUID,
    config: dict[str, Any],
    created_by: str,
) -> QuestionBatch:
    row = conn.execute(
        sa.insert(question_batches)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            eval_set_id=eval_set_id,
            pipeline_run_id=pipeline_run_id,
            config=config,
            created_by=created_by,
        )
        .returning(*question_batches.columns)
    ).one()
    return _batch(row)


def get_batch(
    conn: Connection, workspace_id: uuid.UUID, batch_id: uuid.UUID
) -> QuestionBatch | None:
    row = conn.execute(
        sa.select(question_batches).where(
            question_batches.c.workspace_id == workspace_id, question_batches.c.id == batch_id
        )
    ).first()
    return _batch(row) if row is not None else None


def list_batches(
    conn: Connection, workspace_id: uuid.UUID, eval_set_id: uuid.UUID, *, limit: int
) -> list[QuestionBatch]:
    rows = conn.execute(
        sa.select(question_batches)
        .where(
            question_batches.c.workspace_id == workspace_id,
            question_batches.c.eval_set_id == eval_set_id,
        )
        .order_by(question_batches.c.created_at.desc(), question_batches.c.id.desc())
        .limit(limit)
    )
    return [_batch(row) for row in rows]


def begin_batch(conn: Connection, batch_id: uuid.UUID) -> QuestionBatch | None:
    """Mark a queued (or a retried running) batch running; None for a finished one."""
    row = conn.execute(
        sa.update(question_batches)
        .where(
            question_batches.c.id == batch_id,
            question_batches.c.status.in_(("queued", "running")),
        )
        .values(status="running", updated_at=sa.func.now())
        .returning(*question_batches.columns)
    ).first()
    return _batch(row) if row is not None else None


def finish_batch(
    conn: Connection,
    batch: QuestionBatch,
    *,
    status: str,
    drafts: Sequence[Mapping[str, Any]] = (),
    counts: Mapping[str, Any] | None = None,
    error_code: str | None = None,
) -> QuestionBatch:
    """End a batch with its drafted candidates (replacing any from an earlier attempt)."""
    conn.execute(sa.delete(question_candidates).where(question_candidates.c.batch_id == batch.id))
    if drafts:
        conn.execute(
            sa.insert(question_candidates),
            [
                {
                    "id": uuid.uuid4(),
                    "workspace_id": batch.workspace_id,
                    "eval_set_id": batch.eval_set_id,
                    "batch_id": batch.id,
                    "ordinal": ordinal,
                    "drafted_query": draft["query"],
                    **draft,
                }
                for ordinal, draft in enumerate(drafts)
            ],
        )
    row = conn.execute(
        sa.update(question_batches)
        .where(question_batches.c.id == batch.id)
        .values(
            status=status,
            counts=dict(counts or {}),
            error_code=error_code,
            finished_at=sa.func.now(),
            updated_at=sa.func.now(),
        )
        .returning(*question_batches.columns)
    ).one()
    return _batch(row)


def list_candidates(
    conn: Connection,
    workspace_id: uuid.UUID,
    eval_set_id: uuid.UUID,
    *,
    status: str | None = None,
    limit: int = 200,
) -> list[Candidate]:
    """Oldest batch first, in each batch's order."""
    query = sa.select(question_candidates).where(
        question_candidates.c.workspace_id == workspace_id,
        question_candidates.c.eval_set_id == eval_set_id,
    )
    if status is not None:
        query = query.where(question_candidates.c.status == status)
    rows = conn.execute(
        query.order_by(question_candidates.c.created_at, question_candidates.c.ordinal).limit(limit)
    )
    return [_candidate(row) for row in rows]


def get_candidate(
    conn: Connection, workspace_id: uuid.UUID, candidate_id: uuid.UUID
) -> Candidate | None:
    row = conn.execute(
        sa.select(question_candidates).where(
            question_candidates.c.workspace_id == workspace_id,
            question_candidates.c.id == candidate_id,
        )
    ).first()
    return _candidate(row) if row is not None else None


def decide(
    conn: Connection,
    candidate: Candidate,
    *,
    decision: str,
    decided_by: str,
    query: str | None = None,
    evidence: str | None = None,
    split: str | None = None,
    tags: Sequence[str] | None = None,
    doc_key: str | None = None,
    version: str | None = None,
    section: str | None = None,
) -> Candidate:
    """Accept (with any edits) or reject a candidate. One already added to a version is
    settled (``already_added``); one accepted without a gold document and evidence is refused
    (``evidence_required``)."""
    if candidate.status == "added":
        raise CandidateRefused("already_added")
    if decision == "accepted" and not (
        (evidence if evidence is not None else candidate.evidence).strip()
        and (doc_key if doc_key is not None else candidate.doc_key).strip()
    ):
        raise CandidateRefused("evidence_required")
    values: dict[str, Any] = {
        "status": decision,
        "decided_by": decided_by,
        "decided_at": sa.func.now(),
        "updated_at": sa.func.now(),
    }
    if decision == "accepted":
        values.update(
            {
                key: value
                for key, value in (
                    ("query", query),
                    ("evidence", evidence),
                    ("split", split),
                    ("tags", list(tags) if tags is not None else None),
                    ("doc_key", doc_key),
                    ("version", version),
                    ("section", section),
                )
                if value is not None
            }
        )
    row = conn.execute(
        sa.update(question_candidates)
        .where(question_candidates.c.id == candidate.id)
        .values(**values)
        .returning(*question_candidates.columns)
    ).one()
    return _candidate(row)


def add_accepted(
    conn: Connection, *, workspace_id: uuid.UUID, eval_set_id: uuid.UUID, created_by: str
) -> tuple[evalsets.EvalSetVersion, list[Candidate]]:
    """A new version of the eval set: the latest version's items, then every accepted
    candidate's. Refuses with ``nothing_accepted`` or ``no_version``."""
    conn.execute(
        sa.text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"stratafi:question_candidates:{workspace_id}:{eval_set_id}"},
    )
    accepted = list_candidates(conn, workspace_id, eval_set_id, status="accepted", limit=1000)
    if not accepted:
        raise CandidateRefused("nothing_accepted")
    latest = evalsets.latest_version(conn, workspace_id, eval_set_id)
    if latest is None:
        raise CandidateRefused("no_version")
    current = evalsets.stored_eval_set(conn, latest)
    known = {item.id for item in current.items}
    new_items = [item for candidate in accepted if (item := candidate.item()).id not in known]
    eval_set = current.model_copy(update={"items": [*current.items, *new_items]})
    version, _ = evalsets.save_version(
        conn,
        workspace_id=workspace_id,
        eval_set_id=eval_set_id,
        eval_set=eval_set,
        created_by=created_by,
        split_params=latest.split_params,
        source={"format": "generated", "from_version": latest.version, "added": len(new_items)},
    )
    added = []
    for candidate in accepted:
        row = conn.execute(
            sa.update(question_candidates)
            .where(question_candidates.c.id == candidate.id)
            .values(
                status="added",
                item_id=candidate.item().id,
                added_version_id=version.id,
                updated_at=sa.func.now(),
            )
            .returning(*question_candidates.columns)
        ).one()
        added.append(_candidate(row))
    return version, added
