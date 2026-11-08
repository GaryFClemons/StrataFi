"""Comparing up to six eval runs: RAG pipelines side by side.

``GET /eval-comparisons/many`` pairs two to six succeeded eval runs of one eval set version on
the queries they all scored, against a baseline: each metric's interval per run, deltas with
Holm's adjustment, groups against the best run; each run's curves, checkpoint funnel, latency,
where lost, violations, findability and slices; the parameters that differ; and the queries the
runs disagree on. Numbers, IDs and the runs' labels only.
"""

import re
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_api.tracked import label_of, tracked_run
from stratafi_core.comparison import MAX_RUNS, MIN_RUNS, RunScores, compare_many
from stratafi_core.tracking import differing_params, headline_names
from stratafi_store import evalruns
from stratafi_store import tracking as store

router = APIRouter(tags=["eval-runs"])

Rule = Literal["knowledge-hit-v1", "evidence-coverage-v1", "doc-v1"]
Checkpoint = Literal["dense", "keyword", "fused", "candidates", "reranked", "final"]
METRIC_RE = re.compile(r"(hit|recall|precision|mrr|ndcg)@(\d{1,3}|all)")


class ComparedRunOut(BaseModel):
    id: uuid.UUID
    label: str = Field(description="Its title, or its config and pipeline")
    title: str | None
    tags: dict[str, str]
    split: str
    created_at: datetime
    run_id: uuid.UUID = Field(description="The pipeline run it searched")
    retrieval_config: dict[str, Any] = Field(description="id, name and version")
    pipeline_spec: dict[str, Any] = Field(description="id, name and version")
    corpus: dict[str, Any] = Field(description="Name and version")
    k_eval: int | None
    models: dict[str, str | None] = Field(
        description="Role → the model it used: an alias, a catalog model or a trained model"
    )


class CellOut(BaseModel):
    mean: float | None = None
    low: float | None = None
    high: float | None = None
    n: int = 0
    delta: float | None = Field(default=None, description="This run minus the baseline")
    delta_low: float | None = None
    delta_high: float | None = None
    p_value: float | None = Field(default=None, description="Paired randomization test")
    p_holm: float | None = Field(default=None, description="Holm-adjusted across the runs")
    wins: int | None = None
    losses: int | None = None
    ties: int | None = None
    group: Literal["leader", "tied", "worse"] | None = Field(
        default=None, description="Against the best run, after Holm's adjustment"
    )
    behind_best: float | None = None


class MetricRowOut(BaseModel):
    metric: str
    n: int = Field(description="Queries paired for this metric")
    best: int | None = Field(description="The best run's index")
    cells: list[CellOut] = Field(description="One per run, in the runs' order")


class CurveOut(BaseModel):
    metric: str
    ks: list[str]
    series: list[list[float | None]] = Field(description="One per run: the mean at each k")


class FunnelOut(BaseModel):
    checkpoints: list[str]
    hit: list[list[float | None]] = Field(
        description="Per run and checkpoint: the share of queries with a relevant chunk anywhere"
    )
    recall: list[list[float | None]]


class LatencyOut(BaseModel):
    steps: list[str]
    p50: list[list[float | None]]
    p95: list[list[float | None]]
    mean: list[list[float | None]]


class WhereLostOut(BaseModel):
    buckets: list[str]
    counts: list[list[int | None]]


class SliceOut(BaseModel):
    kind: str = Field(description="file_type or tag")
    label: str
    values: list[float | None]


class ParamDiffOut(BaseModel):
    key: str
    values: list[Any]


class QueryRowOut(BaseModel):
    item_id: str
    values: list[float | None]
    spread: float


class CompareManyOut(BaseModel):
    runs: list[ComparedRunOut]
    baseline: int = Field(description="The baseline's index")
    rule: str
    checkpoint: str
    metrics: list[str]
    paired: int = Field(description="Queries with evidence every run scored")
    resamples: int
    seed: int
    table: list[MetricRowOut]
    curves: list[CurveOut]
    funnel: FunnelOut
    latency: LatencyOut
    where_lost: WhereLostOut
    violations: list[dict[str, int | None]]
    findability: list[float | None]
    slices: list[SliceOut]
    params: list[ParamDiffOut] = Field(description="The parameters that differ, per run")
    queries: list[QueryRowOut] = Field(
        description="The first metric per query and run; the widest spread first"
    )
    warnings: list[str] = Field(
        description="few_queries: fewer than 50 paired; different_splits: the runs read "
        "different splits, and only shared items pair"
    )


def model_names(context: dict[str, Any]) -> dict[str, str | None]:
    """Role → the model name an eval run recorded."""
    found: dict[str, str | None] = {}
    for role, record in (context.get("models") or {}).items():
        if not isinstance(record, dict):
            continue
        params = record.get("params") or {}
        found[role] = (
            record.get("alias")
            or record.get("model_id")
            or params.get("alias")
            or params.get("model")
            or record.get("component")
        )
    return found


def compared_run(row: store.SearchRow) -> ComparedRunOut:
    return ComparedRunOut(
        id=row.id,
        label=label_of(row),
        title=row.title,
        tags=dict(row.tags or {}),
        split=row.split,
        created_at=row.created_at,
        run_id=row.pipeline_run_id,
        retrieval_config={
            "id": str(row.retrieval_config_id),
            "name": row.retrieval_name,
            "version": row.retrieval_version,
        },
        pipeline_spec={
            "id": str(row.pipeline_spec_id) if row.pipeline_spec_id else None,
            "name": row.pipeline_name,
            "version": row.pipeline_version,
        },
        corpus={"name": row.corpus_name, "version": row.corpus_version},
        k_eval=(row.context or {}).get("k_eval"),
        models=model_names(row.context or {}),
    )


@router.get(
    "/eval-comparisons/many",
    response_model=CompareManyOut,
    summary="Compare two to six eval runs of one eval set version, against a baseline",
    description="Each metric's 95% interval per run on the queries they all scored; each run "
    "against the baseline (paired interval, randomization test, Holm across the runs); groups "
    "against the best run; curves, the checkpoint funnel, latency, where lost and slices; the "
    "parameters that differ; and the queries the runs disagree on.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def compare_many_eval_runs(
    request: Request,
    ids: Annotated[
        list[uuid.UUID], Query(description="Two to six eval runs, in the order to show them")
    ],
    baseline: Annotated[uuid.UUID | None, Query(description="One of them; the first")] = None,
    rule: Annotated[Rule, Query()] = "knowledge-hit-v1",
    checkpoint: Annotated[Checkpoint, Query()] = "final",
    metric: Annotated[
        list[str] | None,
        Query(description="Metrics such as recall@5; the eval set's headline ones by default"),
    ] = None,
    resamples: Annotated[int, Query(ge=100, le=20_000)] = 2000,
    seed: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
    principal: Principal = Depends(require_role("viewer")),
) -> CompareManyOut:
    if len(ids) < MIN_RUNS:
        raise Problem("invalid_request", code="too_few_runs", errors=["ids"])
    if len(ids) > MAX_RUNS:
        raise Problem("invalid_request", code="too_many_runs", errors=["ids"])
    if len(set(ids)) != len(ids):
        raise Problem("invalid_request", code="repeated_run", errors=["ids"])
    if baseline is not None and baseline not in ids:
        raise Problem("invalid_request", code="unknown_baseline", errors=["baseline"])
    if metric and (len(metric) > 8 or any(not METRIC_RE.fullmatch(name) for name in metric)):
        raise Problem("invalid_request", code="unknown_metric", errors=["metric"])
    with engine_of(request).connect() as conn:
        found = {
            row.id: row
            for row in store.search_rows(
                conn, principal.workspace.id, include_archived=True, ids=ids
            )
        }
        if len(found) != len(ids):
            raise Problem("not_found", code="eval_run_not_found", errors=["ids"])
        rows = [found[identifier] for identifier in ids]
        if any(row.status != "succeeded" for row in rows):
            raise Problem("conflict", "Every eval run must have succeeded.", code="not_finished")
        if len({row.eval_set_version_id for row in rows}) > 1:
            raise Problem(
                "conflict",
                "Only eval runs of the same eval set version pair query by query.",
                code="not_paired",
            )
        scores = [
            RunScores(
                id=str(row.id),
                scores={
                    result.item_id: result.score for result in evalruns.query_results(conn, row.id)
                },
                summary=row.metrics,
            )
            for row in rows
        ]
    k_eval = int((rows[0].context or {}).get("k_eval") or 5)
    names = list(metric) if metric else headline_names(k_eval)
    index = ids.index(baseline) if baseline is not None else 0
    result = compare_many(
        scores,
        rule=rule,
        checkpoint=checkpoint,
        metrics=names,
        baseline=index,
        resamples=resamples,
        seed=seed,
    )
    tracked = [tracked_run(row) for row in rows]
    warnings = list(result["warnings"])
    if len({row.split for row in rows}) > 1:
        warnings.append("different_splits")
    return CompareManyOut(
        runs=[compared_run(row) for row in rows],
        baseline=index,
        rule=rule,
        checkpoint=checkpoint,
        metrics=names,
        paired=result["paired"],
        resamples=resamples,
        seed=seed,
        table=[MetricRowOut(**row) for row in result["table"]],
        curves=[CurveOut(**row) for row in result["curves"]],
        funnel=FunnelOut(**result["funnel"]),
        latency=LatencyOut(**result["latency"]),
        where_lost=WhereLostOut(**result["where_lost"]),
        violations=result["violations"],
        findability=result["findability"],
        slices=[SliceOut(**row) for row in result["slices"]],
        params=[
            ParamDiffOut(key=key, values=[run.params.get(key) for run in tracked])
            for key in differing_params(tracked)
        ],
        queries=[QueryRowOut(**row) for row in result["queries"]],
        warnings=warnings,
    )
