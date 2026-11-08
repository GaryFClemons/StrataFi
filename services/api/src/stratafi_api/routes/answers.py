"""Answer runs: answers to an eval run's queries, judged, scored and labelled.

``POST /answer-runs`` queues a run for a succeeded eval run: a generator (a ``generate/``
component and its parameters) answers each query from its final passages with the pinned
``grounded-qa`` prompt, and a judge, if given, scores faithfulness and correctness. The run's
metrics are means with 95% intervals; ``GET /answer-runs/{id}/queries`` lists each query's
scores, and ``.../queries/{item_id}`` shows its answer and passages.

Calibration: a person labels an answer (``PUT .../labels``: faithful, correct, abstained), and
``GET .../agreement`` compares the labels with what the judge and the abstention detector said.
Audit events carry IDs, never text.
"""

import json
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of, store_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_core.answers import LABEL_DIMENSIONS, PROMPTS, agreement
from stratafi_core.components import UnknownComponent
from stratafi_core.logs import current_log_context
from stratafi_store import answers as store
from stratafi_store import evalruns
from stratafi_store.artifacts import ArtifactNotFound, artifact_key
from stratafi_store.audit import record_audit_event
from stratafi_store.jobs import JobQueue

router = APIRouter(tags=["answers"])

AnswerStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
Dimension = Literal["faithful", "correct", "abstained"]


class GeneratorIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str = Field(description="A generate/ component, such as generate/gateway")
    params: dict[str, Any] = Field(default_factory=dict)


class AnswerRunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_run_id: uuid.UUID = Field(description="A succeeded eval run: its queries and passages")
    generator: GeneratorIn
    judge: GeneratorIn | None = Field(
        default=None, description="Scores faithfulness and correctness; none: only the graders"
    )
    passages: int = Field(default=8, ge=1, le=50, description="Final passages the answer reads")


class AnswerRunOut(BaseModel):
    id: uuid.UUID
    eval_run_id: uuid.UUID
    status: AnswerStatus
    config: dict[str, Any] = Field(description="Generator, judge, pinned prompts and passages")
    results_cached: bool = Field(description="An identical earlier run's answers were reused")
    metrics: dict[str, Any] | None = Field(description="Each metric's mean with its 95% CI")
    usage: dict[str, Any] = Field(description="Calls, tokens, model revisions and egress")
    error_code: str | None
    created_by: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def of(cls, run: store.AnswerRun) -> "AnswerRunOut":
        return cls(**{key: getattr(run, key) for key in cls.model_fields})


class AnswerRunsOut(BaseModel):
    items: list[AnswerRunOut]


class LabelOut(BaseModel):
    dimension: Dimension
    label: bool
    labeled_by: str
    labeled_at: datetime


class AnswerResultOut(BaseModel):
    item_id: str
    ordinal: int
    answer_key: str = Field(description="The hash of the item and its answer: labels attach here")
    score: dict[str, Any] = Field(description="Abstention, citations, reference F1, judge scores")
    labels: list[LabelOut]


class AnswerResultsOut(BaseModel):
    items: list[AnswerResultOut]


class AnswerDetailOut(BaseModel):
    item_id: str
    answer: str
    passages: list[dict[str, Any]] = Field(description="Source ID, document, section, text")
    score: dict[str, Any]
    labels: list[LabelOut]


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, max_length=128)
    dimension: Dimension
    label: bool


class AgreementOut(BaseModel):
    dimensions: dict[str, dict[str, Any]] = Field(
        description="Per dimension: labels, agreement with its 95% CI, Cohen's kappa, rates"
    )


def _queue(request: Request) -> JobQueue:
    queue = request.app.state.jobs
    if queue is None:
        raise Problem("unavailable", code="database_not_configured")
    return queue


def _audit(
    conn: Connection, principal: Principal, action: str, resource_id: str, **details: Any
) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type="answer_run",
        resource_id=resource_id,
        request_id=current_log_context().get("request_id"),
        details=details,
    )


def _generator(request: Request, ref: GeneratorIn, field: str) -> dict[str, Any]:
    try:
        component = request.app.state.registry.get(ref.component)
    except UnknownComponent:
        raise Problem("invalid_request", code="unknown_component", errors=[field]) from None
    if component.info.stage != "generate":
        raise Problem("invalid_request", code="not_a_generator", errors=[field])
    try:
        params = component.resolve_params(ref.params)
    except ValueError:
        raise Problem(
            "invalid_request", code="invalid_params", errors=[f"{field}.params"]
        ) from None
    return {"component": ref.component, "params": component.dump_params(params)}


def _run_or_404(conn: Connection, principal: Principal, run_id: uuid.UUID) -> store.AnswerRun:
    run = store.get_answer_run(conn, principal.workspace.id, run_id)
    if run is None:
        raise Problem("not_found", code="answer_run_not_found")
    return run


def _labels(found: list[store.Label]) -> list[LabelOut]:
    return [
        LabelOut(
            dimension=label.dimension,  # type: ignore[arg-type]
            label=label.label,
            labeled_by=label.labeled_by,
            labeled_at=label.labeled_at,
        )
        for label in found
    ]


@router.post(
    "/answer-runs",
    status_code=202,
    response_model=AnswerRunOut,
    summary="Answer an eval run's queries from their passages, and grade the answers",
    description="202 with the queued run. A worker answers every query with the pinned "
    "grounded-qa prompt and the judge, if any, scores the answers; an identical earlier run's "
    "answers are reused. generate/gateway answers through the model gateway's chat models; "
    "generate/fake is the lab's stand-in.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def start_answer_run(
    request: Request, body: AnswerRunIn, principal: Principal = Depends(require_role("editor"))
) -> JSONResponse:
    generator = _generator(request, body.generator, "generator")
    judge = _generator(request, body.judge, "judge") if body.judge else None
    config = {
        "generator": generator,
        "judge": judge,
        "passages": body.passages,
        "prompts": {name: prompt.pinned() for name, prompt in PROMPTS.items()},
    }
    with engine_of(request).begin() as conn:
        evaluation = evalruns.get_eval_run(conn, principal.workspace.id, body.eval_run_id)
        if evaluation is None:
            raise Problem("not_found", code="eval_run_not_found", errors=["eval_run_id"])
        if evaluation.status != "succeeded":
            raise Problem("conflict", "The eval run hasn't succeeded.", code="eval_run_not_done")
        run = store.create_answer_run(
            conn,
            workspace_id=principal.workspace.id,
            eval_run_id=evaluation.id,
            config=config,
            created_by=principal.subject,
        )
        _queue(request).enqueue_in(
            conn,
            workspace_id=principal.workspace.id,
            kind=store.START_JOB,
            payload={"answer_run_id": str(run.id)},
            resource_class="io",
            priority=5,
            max_attempts=request.app.state.settings.worker.max_attempts,
        )
        _audit(
            conn,
            principal,
            "answer_run.start",
            str(run.id),
            eval_run=str(evaluation.id),
            generator=generator["component"],
            judge=judge["component"] if judge else None,
        )
    return JSONResponse(AnswerRunOut.of(run).model_dump(mode="json"), status_code=202)


@router.get(
    "/answer-runs",
    response_model=AnswerRunsOut,
    summary="Answer runs, newest first; one eval run's with eval_run_id",
    responses=problem_responses(400, 401, 403, 503),
)
def list_answer_runs(
    request: Request,
    eval_run_id: Annotated[uuid.UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    principal: Principal = Depends(require_role("viewer")),
) -> AnswerRunsOut:
    with engine_of(request).connect() as conn:
        found = store.list_answer_runs(
            conn, principal.workspace.id, limit=limit, eval_run_id=eval_run_id
        )
    return AnswerRunsOut(items=[AnswerRunOut.of(run) for run in found])


@router.get(
    "/answer-runs/{answer_run_id}",
    response_model=AnswerRunOut,
    summary="One answer run: its configuration, metrics and usage",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_answer_run(
    request: Request,
    answer_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> AnswerRunOut:
    with engine_of(request).connect() as conn:
        return AnswerRunOut.of(_run_or_404(conn, principal, answer_run_id))


@router.get(
    "/answer-runs/{answer_run_id}/queries",
    response_model=AnswerResultsOut,
    summary="Each query's scores and labels, in the eval set's order",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_answer_results(
    request: Request,
    answer_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> AnswerResultsOut:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, answer_run_id)
        results = store.results_of(conn, run.id)
        labels = store.labels_for(
            conn, principal.workspace.id, [result.answer_key for result in results]
        )
    return AnswerResultsOut(
        items=[
            AnswerResultOut(
                item_id=result.item_id,
                ordinal=result.ordinal,
                answer_key=result.answer_key,
                score=result.score,
                labels=_labels(labels.get(result.answer_key, [])),
            )
            for result in results
        ]
    )


@router.get(
    "/answer-runs/{answer_run_id}/queries/{item_id}",
    response_model=AnswerDetailOut,
    summary="One query's answer, with the passages it was given",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def get_answer_detail(
    request: Request,
    answer_run_id: uuid.UUID,
    item_id: str,
    principal: Principal = Depends(require_role("viewer")),
) -> AnswerDetailOut:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, answer_run_id)
        if run.status != "succeeded" or not run.results_key:
            raise Problem("conflict", code="answer_run_not_done")
        result = next((r for r in store.results_of(conn, run.id) if r.item_id == item_id), None)
        if result is None:
            raise Problem("not_found", code="query_not_found")
        labels = store.labels_for(conn, principal.workspace.id, [result.answer_key])
    try:
        data = store_of(request).get(artifact_key(run.workspace_id, run.results_key))
    except ArtifactNotFound:
        raise Problem("conflict", code="artifact_missing") from None
    row = next(row for row in json.loads(data)["items"] if row["item_id"] == item_id)
    return AnswerDetailOut(
        item_id=item_id,
        answer=row["answer"],
        passages=row["passages"],
        score=result.score,
        labels=_labels(labels.get(result.answer_key, [])),
    )


@router.put(
    "/answer-runs/{answer_run_id}/labels",
    response_model=LabelOut,
    summary="Label one answer: faithful, correct or abstained (calibration)",
    description="The label attaches to the answer itself, so it counts wherever the answer is "
    "reused; a person's later label on the same dimension replaces their earlier one.",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def label_answer(
    request: Request,
    answer_run_id: uuid.UUID,
    body: LabelIn,
    principal: Principal = Depends(require_role("editor")),
) -> LabelOut:
    with engine_of(request).begin() as conn:
        run = _run_or_404(conn, principal, answer_run_id)
        result = next(
            (r for r in store.results_of(conn, run.id) if r.item_id == body.item_id), None
        )
        if result is None:
            raise Problem("not_found", code="query_not_found", errors=["item_id"])
        label = store.set_label(
            conn,
            workspace_id=principal.workspace.id,
            answer_key=result.answer_key,
            item_id=body.item_id,
            dimension=body.dimension,
            label=body.label,
            labeled_by=principal.subject,
        )
        _audit(conn, principal, "answer.label", str(run.id), dimension=body.dimension)
    return _labels([label])[0]


@router.get(
    "/answer-runs/{answer_run_id}/agreement",
    response_model=AgreementOut,
    summary="How often the judge and the abstention detector agree with people's labels",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_agreement(
    request: Request,
    answer_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> AgreementOut:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, answer_run_id)
        results = store.results_of(conn, run.id)
        labels = store.labels_for(
            conn, principal.workspace.id, [result.answer_key for result in results]
        )
    pairs = store.label_pairs(results, labels)
    return AgreementOut(
        dimensions={
            dimension: agreement(pairs[dimension]).as_dict() for dimension in LABEL_DIMENSIONS
        }
    )
