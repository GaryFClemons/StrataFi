"""Traces and latency profiles of eval runs.

- ``GET /eval-runs/{id}/queries/{item_id}/trace``: one query's retrieval as spans, built from
  its stored score (step latencies, relevant chunk IDs) and its shard's ranked lists.
- ``GET /eval-runs/{id}/latency``: per step, the percentiles and histogram of a run's queries,
  and its slowest queries, from the stored scores alone.

IDs and numbers only; a query's text and its passages stay with the query detail.
"""

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_api.routes.evalruns import _ranked_item, _run_or_404
from stratafi_api.trace_models import TraceBody
from stratafi_core.traces import build_trace, latency_profile
from stratafi_store import evalruns as store

router = APIRouter(tags=["eval-runs"])

Rule = Literal["knowledge-hit-v1", "evidence-coverage-v1", "doc-v1"]


class TraceOut(TraceBody):
    eval_run_id: uuid.UUID
    item_id: str
    rule: str = Field(description="The match rule that marks relevant chunks")
    has_evidence: bool
    hit: bool | None
    lost_at: str | None = Field(description="Where the eval run says a miss was lost")
    embedding_revision: str | None
    rerank_revision: str | None
    visible_chunks: int = Field(description="Chunks the asker could see on the as-of date")


class HistogramOut(BaseModel):
    edges: list[float]
    counts: list[int]


class StepProfileOut(BaseModel):
    n: int
    mean: float
    min: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    max: float
    histogram: HistogramOut


class SlowQueryOut(BaseModel):
    item_id: str
    total_ms: float
    steps: dict[str, float]


class LatencyProfileOut(BaseModel):
    eval_run_id: uuid.UUID
    queries: int
    steps: dict[str, StepProfileOut] = Field(description="By step: embed_ms, ..., total_ms")
    slowest: list[SlowQueryOut]


@router.get(
    "/eval-runs/{eval_run_id}/queries/{item_id}/trace",
    response_model=TraceOut,
    summary="One query's trace: retrieval's steps on a timeline, each with its lists",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def get_trace(
    request: Request,
    eval_run_id: uuid.UUID,
    item_id: str,
    rule: Annotated[Rule, Query(description="Which rule marks chunks relevant")] = (
        "knowledge-hit-v1"
    ),
    principal: Principal = Depends(require_role("viewer")),
) -> TraceOut:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, eval_run_id)
        row = store.query_result(conn, run.id, item_id)
        if row is None:
            raise Problem("not_found", code="query_not_found")
        shard = next((s for s in store.shards_of(conn, run.id) if s.ordinal == row.shard), None)
    if shard is None:
        raise Problem("not_found", code="query_not_found")
    ranked = _ranked_item(request, run, shard.cache_key, item_id)
    relevant = (row.score.get("relevant") or {}).get(rule, []) if row.has_evidence else None
    trace = build_trace(
        latency_ms=ranked.latency_ms,
        checkpoints=ranked.checkpoints,
        relevant=relevant,
        config=(run.context or {}).get("retrieval"),
        models=(run.context or {}).get("models"),
    )
    return TraceOut(
        eval_run_id=run.id,
        item_id=item_id,
        rule=rule,
        has_evidence=row.has_evidence,
        hit=row.hit,
        lost_at=row.lost_at,
        embedding_revision=ranked.embedding_revision,
        rerank_revision=ranked.rerank_revision,
        visible_chunks=ranked.visible_chunks,
        **trace,
    )


@router.get(
    "/eval-runs/{eval_run_id}/latency",
    response_model=LatencyProfileOut,
    summary="An eval run's latency per step: percentiles, histograms and its slowest queries",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def get_latency_profile(
    request: Request,
    eval_run_id: uuid.UUID,
    bins: Annotated[int, Query(ge=5, le=60)] = 20,
    principal: Principal = Depends(require_role("viewer")),
) -> LatencyProfileOut:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, eval_run_id)
        if run.status != "succeeded":
            raise Problem("conflict", "The eval run hasn't finished.", code="eval_run_not_finished")
        latencies = {
            row.item_id: row.score.get("latency_ms") or {}
            for row in store.query_results(conn, run.id)
        }
    profile = latency_profile(latencies, bins=bins)
    return LatencyProfileOut(eval_run_id=run.id, **profile)
