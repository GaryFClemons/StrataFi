"""Drafted questions and their review queue.

``POST /eval-sets/{id}/question-batches`` queues a batch: a generator drafts questions from a
seeded sample of a pipeline run's passages, each with the sentence that answers it, copied from
its passage. The drafts wait as ``proposed`` candidates. A person accepts each one, maybe with a
better query, evidence, split or tags, or rejects it; ``POST .../candidates/add`` adds the
accepted ones to a new version of the eval set. Nothing enters a version unreviewed.

Audit events carry IDs, decisions and counts, never a question's text.
"""

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_api.routes.answers import GeneratorIn, _generator
from stratafi_api.routes.evalsets import EvalSetVersionOut
from stratafi_core.answers import DRAFT_QUESTIONS
from stratafi_core.logs import current_log_context
from stratafi_store import evalsets, runs
from stratafi_store import questions as store
from stratafi_store.audit import record_audit_event
from stratafi_store.jobs import JobQueue

router = APIRouter(tags=["eval sets"])

Tag = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._:-]{0,63}$")]
BatchStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
CandidateStatus = Literal["proposed", "accepted", "rejected", "added"]


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID = Field(description="A succeeded pipeline run: its passages")
    generator: GeneratorIn
    count: int = Field(default=20, ge=1, le=100, description="Questions to draft")
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Seeds the passages' sample")


class BatchOut(BaseModel):
    id: uuid.UUID
    eval_set_id: uuid.UUID
    run_id: uuid.UUID = Field(description="The pipeline run drafted from")
    status: BatchStatus
    config: dict[str, Any]
    counts: dict[str, Any] = Field(description="Passages asked, questions drafted and dropped")
    error_code: str | None
    created_by: str
    created_at: datetime
    finished_at: datetime | None

    @classmethod
    def of(cls, batch: store.QuestionBatch) -> "BatchOut":
        return cls(
            id=batch.id,
            eval_set_id=batch.eval_set_id,
            run_id=batch.pipeline_run_id,
            status=batch.status,  # type: ignore[arg-type]
            config=batch.config,
            counts=batch.counts,
            error_code=batch.error_code,
            created_by=batch.created_by,
            created_at=batch.created_at,
            finished_at=batch.finished_at,
        )


class BatchesOut(BaseModel):
    items: list[BatchOut]


class CandidateOut(BaseModel):
    id: uuid.UUID
    batch_id: uuid.UUID
    status: CandidateStatus
    query: str
    evidence: str = Field(description="The sentence that answers it, from its passage")
    drafted_query: str = Field(description="The query as the generator drafted it")
    edited: bool
    doc_key: str
    version: str | None
    section: str | None
    chunk_id: str | None
    split: Literal["dev", "test"]
    tags: list[str]
    decided_by: str | None
    decided_at: datetime | None
    item_id: str | None = Field(description="Its item, once added to a version")
    added_version_id: uuid.UUID | None

    @classmethod
    def of(cls, candidate: store.Candidate) -> "CandidateOut":
        return cls(
            id=candidate.id,
            batch_id=candidate.batch_id,
            status=candidate.status,  # type: ignore[arg-type]
            query=candidate.query,
            evidence=candidate.evidence,
            drafted_query=candidate.drafted_query,
            edited=candidate.edited,
            doc_key=candidate.doc_key,
            version=candidate.version,
            section=candidate.section,
            chunk_id=candidate.chunk_id,
            split=candidate.split,  # type: ignore[arg-type]
            tags=candidate.tags,
            decided_by=candidate.decided_by,
            decided_at=candidate.decided_at,
            item_id=candidate.item_id,
            added_version_id=candidate.added_version_id,
        )


class CandidatesOut(BaseModel):
    items: list[CandidateOut]


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accepted", "rejected"]
    query: str | None = Field(default=None, min_length=8, max_length=4000)
    evidence: str | None = Field(default=None, min_length=8, max_length=2000)
    split: Literal["dev", "test"] | None = None
    tags: list[Tag] | None = Field(default=None, max_length=32)
    doc_key: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$",
        description="The gold document, when the drafted one is wrong",
    )
    version: str | None = Field(default=None, min_length=1, max_length=64)
    section: str | None = Field(default=None, min_length=1, max_length=500)


class AddedOut(BaseModel):
    version: EvalSetVersionOut
    added: list[CandidateOut]


def _audit(
    conn: Connection, principal: Principal, action: str, resource_id: uuid.UUID, **details: Any
) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type="eval_set",
        resource_id=str(resource_id),
        request_id=current_log_context().get("request_id"),
        details=details,
    )


def _set_or_404(conn: Connection, principal: Principal, eval_set_id: uuid.UUID) -> None:
    if evalsets.get_set(conn, principal.workspace.id, eval_set_id) is None:
        raise Problem("not_found", code="eval_set_not_found")


def _queue(request: Request) -> JobQueue:
    queue = request.app.state.jobs
    if queue is None:
        raise Problem("unavailable", code="database_not_configured")
    return queue


@router.post(
    "/eval-sets/{eval_set_id}/question-batches",
    status_code=202,
    response_model=BatchOut,
    summary="Draft questions from a pipeline run's passages, for review",
    description="202 with the queued batch. A worker drafts a question from each passage in a "
    "seeded sample, with the sentence that answers it; each waits in the review queue.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def start_batch(
    request: Request,
    eval_set_id: uuid.UUID,
    body: BatchIn,
    principal: Principal = Depends(require_role("editor")),
) -> JSONResponse:
    generator = _generator(request, body.generator, "generator")
    with engine_of(request).begin() as conn:
        _set_or_404(conn, principal, eval_set_id)
        pipeline = runs.get_run(conn, principal.workspace.id, body.run_id)
        if pipeline is None:
            raise Problem("not_found", code="run_not_found", errors=["run_id"])
        if pipeline.status != "succeeded":
            raise Problem("conflict", "The run hasn't succeeded.", code="run_not_done")
        batch = store.create_batch(
            conn,
            workspace_id=principal.workspace.id,
            eval_set_id=eval_set_id,
            pipeline_run_id=pipeline.id,
            config={
                "generator": generator,
                "count": body.count,
                "seed": body.seed,
                "prompt": DRAFT_QUESTIONS.pinned(),
            },
            created_by=principal.subject,
        )
        _queue(request).enqueue_in(
            conn,
            workspace_id=principal.workspace.id,
            kind=store.DRAFT_JOB,
            payload={"batch_id": str(batch.id)},
            resource_class="io",
            priority=5,
            max_attempts=request.app.state.settings.worker.max_attempts,
        )
        _audit(conn, principal, "eval_set.questions_drafted", eval_set_id, batch=str(batch.id))
    return JSONResponse(BatchOut.of(batch).model_dump(mode="json"), status_code=202)


@router.get(
    "/eval-sets/{eval_set_id}/question-batches",
    response_model=BatchesOut,
    summary="The eval set's question batches, newest first",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_batches(
    request: Request,
    eval_set_id: uuid.UUID,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    principal: Principal = Depends(require_role("viewer")),
) -> BatchesOut:
    with engine_of(request).connect() as conn:
        _set_or_404(conn, principal, eval_set_id)
        found = store.list_batches(conn, principal.workspace.id, eval_set_id, limit=limit)
    return BatchesOut(items=[BatchOut.of(batch) for batch in found])


@router.get(
    "/eval-sets/{eval_set_id}/question-batches/{batch_id}",
    response_model=BatchOut,
    summary="One question batch: its status and counts",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_batch(
    request: Request,
    eval_set_id: uuid.UUID,
    batch_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> BatchOut:
    with engine_of(request).connect() as conn:
        batch = store.get_batch(conn, principal.workspace.id, batch_id)
    if batch is None or batch.eval_set_id != eval_set_id:
        raise Problem("not_found", code="batch_not_found")
    return BatchOut.of(batch)


@router.get(
    "/eval-sets/{eval_set_id}/candidates",
    response_model=CandidatesOut,
    summary="The review queue: drafted questions, one status's with status",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_candidates(
    request: Request,
    eval_set_id: uuid.UUID,
    status: Annotated[CandidateStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    principal: Principal = Depends(require_role("viewer")),
) -> CandidatesOut:
    with engine_of(request).connect() as conn:
        _set_or_404(conn, principal, eval_set_id)
        found = store.list_candidates(
            conn, principal.workspace.id, eval_set_id, status=status, limit=limit
        )
    return CandidatesOut(items=[CandidateOut.of(candidate) for candidate in found])


@router.post(
    "/eval-sets/{eval_set_id}/candidates/{candidate_id}/decision",
    response_model=CandidateOut,
    summary="Accept a drafted question, maybe edited, or reject it",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def decide_candidate(
    request: Request,
    eval_set_id: uuid.UUID,
    candidate_id: uuid.UUID,
    body: DecisionIn,
    principal: Principal = Depends(require_role("editor")),
) -> CandidateOut:
    with engine_of(request).begin() as conn:
        candidate = store.get_candidate(conn, principal.workspace.id, candidate_id)
        if candidate is None or candidate.eval_set_id != eval_set_id:
            raise Problem("not_found", code="candidate_not_found")
        try:
            decided = store.decide(
                conn,
                candidate,
                decision=body.decision,
                decided_by=principal.subject,
                query=body.query,
                evidence=body.evidence,
                split=body.split,
                tags=body.tags,
                doc_key=body.doc_key,
                version=body.version,
                section=body.section,
            )
        except store.CandidateRefused as refused:
            raise Problem("conflict", code=refused.code) from None
        _audit(
            conn,
            principal,
            "eval_set.question_reviewed",
            eval_set_id,
            candidate=str(candidate.id),
            decision=body.decision,
            edited=decided.edited,
        )
    return CandidateOut.of(decided)


@router.post(
    "/eval-sets/{eval_set_id}/candidates/add",
    response_model=AddedOut,
    status_code=201,
    summary="Add every accepted question to a new version of the eval set",
    description="The new version holds the latest version's items, then the accepted "
    "questions, each with its passage's document, version, section and evidence.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def add_accepted(
    request: Request,
    eval_set_id: uuid.UUID,
    principal: Principal = Depends(require_role("editor")),
) -> AddedOut:
    with engine_of(request).begin() as conn:
        _set_or_404(conn, principal, eval_set_id)
        try:
            version, added = store.add_accepted(
                conn,
                workspace_id=principal.workspace.id,
                eval_set_id=eval_set_id,
                created_by=principal.subject,
            )
        except store.CandidateRefused as refused:
            raise Problem("conflict", code=refused.code) from None
        _audit(
            conn,
            principal,
            "eval_set.questions_added",
            eval_set_id,
            version=version.version,
            added=len(added),
        )
    return AddedOut(
        version=EvalSetVersionOut.of(version),
        added=[CandidateOut.of(candidate) for candidate in added],
    )
