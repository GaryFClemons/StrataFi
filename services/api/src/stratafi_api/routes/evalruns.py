"""Eval runs: start, list, read, per-query results and detail, cancel.

``POST /eval-runs`` records a queued eval run and its ``eval.start`` job and answers 202; a
worker plans and runs it (``stratafi_worker.evals``). Progress arrives as ``eval_run`` events on
``GET /events?run_id=``.

The test split is locked: an eval run that reads it must say ``confirm_test_read``, and
each such read is counted on the eval set version and audited. A finished run's results are its
aggregates with confidence intervals (``metrics``), one row per query, and each query's detail:
its ranked lists at every checkpoint with the evidence marked, read back from the run's
artifacts. ``GET /eval-comparisons`` compares two runs of one eval set version query by
query. Audit events carry IDs and counts, never a query.
"""

import re
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of, store_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, problem_responses
from stratafi_api.routes.runs import RunStatus
from stratafi_api.routes.search import IndexCache
from stratafi_components.retrieval import EntryMeta, section_of
from stratafi_core.compare import compare_runs
from stratafi_core.evalruns import EvalOptions, EvalRunManifest, RankedItem
from stratafi_core.evalsets import EvalItem
from stratafi_core.logs import current_log_context
from stratafi_core.relevance import RULES, Matcher, Passage, highlights, same_version
from stratafi_core.specs import (
    SpecInvalid,
    diff_documents,
    dump_document,
    resolve_retrieval,
    spec_hash,
)
from stratafi_store import corpora, evalsets, runs, specs
from stratafi_store import evalruns as store
from stratafi_store.artifacts import ArtifactNotFound, artifact_key
from stratafi_store.audit import record_audit_event
from stratafi_store.jobs import JobQueue

router = APIRouter(tags=["eval-runs"])

Split = Literal["dev", "test", "all"]
Outcome = Literal["hit", "miss", "abstain"]
LostAt = Literal["parse", "retrieval", "rerank"]
DETAIL_DEPTH = 20


class EvalRunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID = Field(description="A succeeded pipeline run with its indexes")
    retrieval_config_id: uuid.UUID = Field(description="A saved retrieval config version")
    eval_set_version_id: uuid.UUID = Field(description="A frozen eval set version")
    split: Split = Field(default="dev", description="dev for tuning; test only to confirm")
    confirm_test_read: bool = Field(
        default=False,
        description="Reading the locked test split is a confirmation, counted and audited",
    )
    options: EvalOptions = Field(default_factory=EvalOptions)


class EvalRunOut(BaseModel):
    id: uuid.UUID
    status: RunStatus
    run_id: uuid.UUID = Field(description="The pipeline run searched")
    retrieval_config_id: uuid.UUID
    eval_set_version_id: uuid.UUID
    split: Split
    options: dict[str, Any]
    retrieval_hash: str
    test_read: bool
    results_cached: bool = Field(description="An identical earlier run's results were reused")
    retrieval_ref: dict[str, Any] | None = Field(description="The config's name and version")
    eval_set: dict[str, Any] | None = Field(description="The eval set's name, version, items")
    k_eval: int | None = Field(description="The cutoff the eval set's own rule uses")
    counts: dict[str, Any]
    metrics: dict[str, Any] | None = Field(
        description="Aggregates with 95% intervals, slices, latency, findability and cost"
    )
    manifest_hash: str | None
    error_code: str | None
    created_by: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def of(cls, run: store.EvalRun) -> "EvalRunOut":
        context = run.context or {}
        return cls(
            id=run.id,
            status=run.status,  # type: ignore[arg-type]
            run_id=run.pipeline_run_id,
            retrieval_config_id=run.retrieval_config_id,
            eval_set_version_id=run.eval_set_version_id,
            split=run.split,  # type: ignore[arg-type]
            options=run.options,
            retrieval_hash=run.retrieval_hash,
            test_read=run.test_read,
            results_cached=run.results_cached,
            retrieval_ref=context.get("retrieval_ref"),
            eval_set=context.get("eval_set"),
            k_eval=context.get("k_eval"),
            counts=run.counts or {},
            metrics=run.metrics,
            manifest_hash=run.manifest_hash,
            error_code=run.error_code,
            created_by=run.created_by,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )


class QueryResultOut(BaseModel):
    item_id: str
    ordinal: int
    split: str
    tags: list[str]
    file_types: list[str]
    has_evidence: bool
    hit: bool | None = Field(description="knowledge-hit-v1 at the eval set's k, in the final list")
    first_hit_rank: int | None
    lost_at: LostAt | None = Field(description="Where a miss lost its evidence")
    avoid_violation: bool | None
    must_not_return_violation: bool
    score: dict[str, Any] = Field(description="Every metric at every checkpoint, and diagnoses")

    @classmethod
    def of(cls, row: store.QueryResult) -> "QueryResultOut":
        return cls(
            item_id=row.item_id,
            ordinal=row.ordinal,
            split=row.split,
            tags=row.tags,
            file_types=row.file_types,
            has_evidence=row.has_evidence,
            hit=row.hit,
            first_hit_rank=row.first_hit_rank,
            lost_at=row.lost_at,  # type: ignore[arg-type]
            avoid_violation=row.avoid_violation,
            must_not_return_violation=row.must_not_return_violation,
            score=row.score,
        )


class PassageOut(BaseModel):
    rank: int
    chunk_id: str
    score: float
    document_id: str
    entry: str | None
    doc_key: str | None
    version: str | None
    section: str | None
    heading_path: list[str]
    text: str
    relevant: dict[str, bool] = Field(description="By match rule")
    grades: list[float] = Field(description="evidence-coverage-v1, per evidence item")
    highlights: list[list[int]] = Field(
        description="Evidence spans in text: [start, end) character offsets"
    )


class QueryDetailOut(BaseModel):
    result: QueryResultOut
    item: dict[str, Any] = Field(description="The eval item: query, asker, evidence, tags")
    checkpoints: dict[str, list[PassageOut]] = Field(
        description="Each checkpoint's list, its first 20 passages"
    )
    lengths: dict[str, int] = Field(description="Each checkpoint's full length")


def _queue(request: Request) -> JobQueue:
    queue = request.app.state.jobs
    if queue is None:
        raise Problem("unavailable", code="database_not_configured")
    return queue


def _orchestrator(request: Request) -> store.EvalOrchestrator:
    orchestrator = request.app.state.evals
    if orchestrator is None:
        raise Problem("unavailable", code="database_not_configured")
    return orchestrator


def _audit(
    conn: Connection, principal: Principal, action: str, resource: str, resource_id: str, **details
) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type=resource,
        resource_id=resource_id,
        request_id=current_log_context().get("request_id"),
        details=details,
    )


def _run_or_404(conn: Connection, principal: Principal, eval_run_id: uuid.UUID) -> store.EvalRun:
    run = store.get_eval_run(conn, principal.workspace.id, eval_run_id)
    if run is None:
        raise Problem("not_found", code="eval_run_not_found")
    return run


@router.post(
    "/eval-runs",
    response_model=EvalRunOut,
    status_code=202,
    summary="Evaluate a retrieval config on a pipeline run, over an eval set's split",
    description="202 with the queued eval run. A worker searches every query in shards and "
    "scores them; an identical earlier run's work is reused. Reading the test split needs "
    "`confirm_test_read`, and is counted. Follow it with `GET /events?run_id=`.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def start_eval_run(
    request: Request, body: EvalRunIn, principal: Principal = Depends(require_role("editor"))
) -> JSONResponse:
    workspace_id = principal.workspace.id
    registry = request.app.state.registry
    with engine_of(request).begin() as conn:
        pipeline = runs.get_run(conn, workspace_id, body.run_id)
        if pipeline is None:
            raise Problem("not_found", code="run_not_found", errors=["run_id"])
        if pipeline.status != "succeeded" or not any(
            task.status in runs.DONE_TASK_STATUSES
            for task in runs.tasks_of(conn, pipeline.id, stage="index")
        ):
            raise Problem("conflict", "The run hasn't built its indexes.", code="run_not_indexed")
        stored = specs.get_by_id(conn, "retrieval", workspace_id, body.retrieval_config_id)
        if stored is None:
            raise Problem(
                "not_found", code="retrieval_config_not_found", errors=["retrieval_config_id"]
            )
        try:
            config = resolve_retrieval(stored.document, registry)
        except SpecInvalid:
            raise Problem("conflict", code="retrieval_config_invalid") from None
        version = evalsets.get_version(conn, workspace_id, version_id=body.eval_set_version_id)
        if version is None:
            raise Problem(
                "not_found", code="eval_set_version_not_found", errors=["eval_set_version_id"]
            )
        if not version.frozen:
            raise Problem(
                "conflict", "Freeze the eval set version first.", code="eval_set_not_frozen"
            )
        reads_test = body.split in ("test", "all") and version.split_counts.get("test", 0) > 0
        if reads_test and version.test_locked and not body.confirm_test_read:
            raise Problem(
                "conflict",
                "The test split is locked: evaluate it only to confirm a candidate chosen on "
                "dev, with confirm_test_read.",
                code="test_split_locked",
            )
        run = store.create_eval_run(
            conn,
            workspace_id=workspace_id,
            pipeline_run_id=pipeline.id,
            retrieval_config_id=stored.id,
            eval_set_version_id=version.id,
            split=body.split,
            options=body.options.dump(),
            retrieval_hash=spec_hash(config),
            test_read=reads_test,
            created_by=principal.subject,
        )
        if reads_test:
            counted = evalsets.record_test_read(conn, version.id)
            _audit(
                conn,
                principal,
                "eval_set.test_read",
                "eval_set",
                str(version.eval_set_id),
                version=version.version,
                reads=counted.test_reads,
                eval_run=str(run.id),
            )
        _queue(request).enqueue_in(
            conn,
            workspace_id=workspace_id,
            kind=store.START_JOB,
            payload={"eval_run_id": str(run.id)},
            resource_class="io",
            priority=5,
            max_attempts=request.app.state.settings.worker.max_attempts,
        )
        _audit(conn, principal, "eval_run.start", "eval_run", str(run.id), split=body.split)
    return JSONResponse(EvalRunOut.of(run).model_dump(mode="json"), status_code=202)


@router.get(
    "/eval-runs",
    response_model=Page[EvalRunOut],
    summary="The workspace's eval runs, newest first",
    responses=problem_responses(400, 401, 403, 503),
)
def list_eval_runs(
    request: Request,
    page: PageParams = Depends(page_params),
    run_id: Annotated[uuid.UUID | None, Query(description="Only this pipeline run's")] = None,
    eval_set_version_id: Annotated[uuid.UUID | None, Query()] = None,
    status: Annotated[RunStatus | None, Query()] = None,
    include_archived: Annotated[bool, Query(description="Include archived ones")] = False,
    principal: Principal = Depends(require_role("viewer")),
) -> Page[EvalRunOut]:
    after = None
    if page.cursor is not None:
        created_at, run_ref = decode_cursor(page.cursor, size=2)
        try:
            after = (datetime.fromisoformat(created_at), uuid.UUID(run_ref))
        except (TypeError, ValueError, AttributeError):
            raise Problem(
                "invalid_request", "The cursor isn't valid.", code="invalid_cursor"
            ) from None
    with engine_of(request).connect() as conn:
        found = store.list_eval_runs(
            conn,
            principal.workspace.id,
            limit=page.limit + 1,
            after=after,
            pipeline_run_id=run_id,
            eval_set_version_id=eval_set_version_id,
            status=status,
            include_archived=include_archived,
        )
    items = [EvalRunOut.of(run) for run in found[: page.limit]]
    cursor = None
    if len(found) > page.limit:
        last = found[page.limit - 1]
        cursor = encode_cursor([last.created_at.isoformat(), str(last.id)])
    return Page[EvalRunOut](items=items, next_cursor=cursor)


@router.get(
    "/eval-runs/{eval_run_id}",
    response_model=EvalRunOut,
    summary="One eval run: its status, counts and, once finished, its aggregates",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_eval_run(
    request: Request,
    eval_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> EvalRunOut:
    with engine_of(request).connect() as conn:
        return EvalRunOut.of(_run_or_404(conn, principal, eval_run_id))


@router.get(
    "/eval-runs/{eval_run_id}/queries",
    response_model=Page[QueryResultOut],
    summary="An eval run's per-query results, in the eval set's order",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_query_results(
    request: Request,
    eval_run_id: uuid.UUID,
    page: PageParams = Depends(page_params),
    outcome: Annotated[Outcome | None, Query(description="hit, miss or abstain")] = None,
    lost_at: Annotated[LostAt | None, Query(description="Only misses lost there")] = None,
    tag: Annotated[str | None, Query(max_length=64)] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> Page[QueryResultOut]:
    after = None
    if page.cursor is not None:
        (after,) = decode_cursor(page.cursor, size=1)
        if not isinstance(after, int):
            raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, eval_run_id)
        found = store.query_results(
            conn,
            run.id,
            limit=page.limit + 1,
            after_ordinal=after,
            outcome=outcome,
            lost_at=lost_at,
            tag=tag,
        )
    items = [QueryResultOut.of(row) for row in found[: page.limit]]
    cursor = encode_cursor([found[page.limit - 1].ordinal]) if len(found) > page.limit else None
    return Page[QueryResultOut](items=items, next_cursor=cursor)


def _ranked_item(request: Request, run: store.EvalRun, cache_key: str, item_id: str) -> RankedItem:
    try:
        data = store_of(request).get(artifact_key(run.workspace_id, cache_key))
    except ArtifactNotFound:
        raise Problem(
            "conflict", "The run's ranked lists are missing.", code="artifact_missing"
        ) from None
    for line in data.splitlines():
        if line.strip():
            ranked = RankedItem.model_validate_json(line)
            if ranked.item_id == item_id:
                return ranked
    raise Problem("not_found", code="query_not_found")


@router.get(
    "/eval-runs/{eval_run_id}/queries/{item_id}",
    response_model=QueryDetailOut,
    summary="One query's detail: its item, and every checkpoint's passages with evidence marked",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def get_query_detail(
    request: Request,
    eval_run_id: uuid.UUID,
    item_id: str,
    principal: Principal = Depends(require_role("viewer")),
) -> QueryDetailOut:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, eval_run_id)
        row = store.query_result(conn, run.id, item_id)
        if row is None:
            raise Problem("not_found", code="query_not_found")
        shard = next((s for s in store.shards_of(conn, run.id) if s.ordinal == row.shard), None)
        stored = next(
            (
                item
                for item in evalsets.list_items(conn, run.eval_set_version_id)
                if item.item_id == item_id
            ),
            None,
        )
        corpus_version = uuid.UUID(run.context["pipeline_run"]["corpus_version_id"])
        entries = corpora.version_entries(conn, run.workspace_id, corpus_version)
    if shard is None or stored is None:
        raise Problem("not_found", code="query_not_found")
    item: EvalItem = stored.model()
    ranked = _ranked_item(request, run, shard.cache_key, item_id)
    cache: IndexCache = request.app.state.indexes
    chunks = {}
    for kind, key in run.context["indexes"].items():
        chunks.update(cache.get(store_of(request), run.workspace_id, key, kind).chunks)
    metas: dict[str, EntryMeta] = {}
    for entry in entries:
        metas.setdefault(str(entry.document_id), EntryMeta.from_entry(entry))
    threshold = float(run.options.get("coverage_threshold", 0.7))
    matcher = Matcher(item.expected, threshold)
    checkpoints: dict[str, list[PassageOut]] = {}
    for name, ranking in ranked.checkpoints.items():
        passages = []
        for rank, (chunk_id, score) in enumerate(ranking[:DETAIL_DEPTH], start=1):
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            meta = metas.get(chunk.document_id)
            passage = Passage(
                chunk_id,
                meta.doc_key if meta else None,
                meta.version if meta else None,
                section_of(chunk),
                chunk.text,
            )
            judgement = matcher.judge(passage)
            spans: list[tuple[int, int]] = []
            for ref in item.expected:
                if ref.evidence and same_version(passage, ref):
                    spans += highlights(ref.evidence, chunk.text)
            passages.append(
                PassageOut(
                    rank=rank,
                    chunk_id=chunk_id,
                    score=score,
                    document_id=chunk.document_id,
                    entry=meta.name if meta else None,
                    doc_key=passage.doc_key,
                    version=passage.version,
                    section=passage.section,
                    heading_path=list(chunk.heading_path),
                    text=chunk.text,
                    relevant={rule: judgement.relevant(rule) for rule in RULES},
                    grades=[round(grade, 6) for grade in judgement.grades],
                    highlights=[list(span) for span in sorted(set(spans))],
                )
            )
        checkpoints[name] = passages
    return QueryDetailOut(
        result=QueryResultOut.of(row),
        item=item.model_dump(mode="json", exclude_none=True),
        checkpoints=checkpoints,
        lengths={name: len(ranking) for name, ranking in ranked.checkpoints.items()},
    )


@router.get(
    "/eval-runs/{eval_run_id}/manifest",
    response_model=EvalRunManifest,
    summary="An eval run's record (stratafi.eval-run.v1), once it has finished",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def get_eval_run_manifest(
    request: Request,
    eval_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> JSONResponse:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, eval_run_id)
    if run.manifest is None:
        raise Problem("conflict", "The eval run hasn't finished.", code="eval_run_not_finished")
    return JSONResponse(
        EvalRunManifest.model_validate(run.manifest).model_dump(mode="json", by_alias=True)
    )


@router.post(
    "/eval-runs/{eval_run_id}/cancel",
    response_model=EvalRunOut,
    summary="Cancel an eval run: waiting shards at once, running ones at their next query",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def cancel_eval_run(
    request: Request,
    eval_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("editor")),
) -> EvalRunOut:
    cancelled = _orchestrator(request).cancel(eval_run_id, principal.workspace.id)
    if cancelled is None:
        raise Problem("not_found", code="eval_run_not_found")
    run, job_ids = cancelled
    queue = _queue(request)
    for job_id in job_ids:
        queue.cancel(job_id, principal.workspace.id)
    with engine_of(request).begin() as conn:
        _audit(conn, principal, "eval_run.cancel", "eval_run", str(run.id), status=run.status)
        run = store.get_eval_run(conn, principal.workspace.id, run.id) or run
    return EvalRunOut.of(cast(store.EvalRun, run))


# --- comparing two eval runs ---------------------------------------------------------


class RunRef(BaseModel):
    id: uuid.UUID
    split: Split
    retrieval_ref: dict[str, Any] | None
    eval_set: dict[str, Any] | None
    pipeline_run_id: uuid.UUID
    created_at: datetime


class MetricDelta(BaseModel):
    metric: str
    n: int = Field(description="Queries paired for this metric")
    mean_a: float | None = None
    mean_b: float | None = None
    delta: float | None = Field(default=None, description="B minus A")
    low: float | None = Field(default=None, description="The 95% interval of the delta")
    high: float | None = None
    p_value: float | None = Field(default=None, description="Paired randomization test")
    p_holm: float | None = Field(default=None, description="Holm-adjusted across the metrics")
    wins: int | None = Field(default=None, description="Queries where B scored higher")
    losses: int | None = None
    ties: int | None = None


class QueryDelta(BaseModel):
    item_id: str
    a: float
    b: float
    delta: float


class ChangeOut(BaseModel):
    path: str
    op: Literal["added", "removed", "changed"]
    before: Any = None
    after: Any = None


class CompareOut(BaseModel):
    a: RunRef
    b: RunRef
    rule: str
    checkpoint: str
    paired: int = Field(description="Queries with evidence that both runs scored")
    resamples: int
    seed: int
    metrics: list[MetricDelta]
    queries: list[QueryDelta] = Field(description="The first metric per query, worst for B first")
    retrieval_diff: list[ChangeOut] = Field(description="How B's retrieval config differs")
    pipeline_diff: list[ChangeOut] = Field(description="How B's pipeline spec differs")
    warnings: list[str] = Field(
        description="few_queries: fewer than 50 paired, so the intervals are wide; "
        "different_splits: the runs read different splits, and only shared items pair"
    )


def _ref(run: store.EvalRun) -> RunRef:
    context = run.context or {}
    return RunRef(
        id=run.id,
        split=run.split,  # type: ignore[arg-type]
        retrieval_ref=context.get("retrieval_ref"),
        eval_set=context.get("eval_set"),
        pipeline_run_id=run.pipeline_run_id,
        created_at=run.created_at,
    )


def _changes(before: Any, after: Any) -> list[ChangeOut]:
    return [
        ChangeOut(path=change.path, op=change.op, before=change.before, after=change.after)
        for change in diff_documents(before, after)
    ]


@router.get(
    "/eval-comparisons",
    response_model=CompareOut,
    summary="Compare two eval runs of one eval set version, query by query",
    description="B against A on the queries both scored: each metric's means, B minus A with "
    "a 95% bootstrap interval, a paired randomization test's p-value (and Holm's across the "
    "metrics), and wins, losses and ties; per query; and how the configurations differ.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def compare_eval_runs(
    request: Request,
    a: Annotated[uuid.UUID, Query(description="The baseline eval run")],
    b: Annotated[uuid.UUID, Query(description="The eval run compared with it")],
    rule: Annotated[
        Literal["knowledge-hit-v1", "evidence-coverage-v1", "doc-v1"], Query()
    ] = "knowledge-hit-v1",
    checkpoint: Annotated[
        Literal["dense", "keyword", "fused", "candidates", "reranked", "final"], Query()
    ] = "final",
    metric: Annotated[
        list[str] | None,
        Query(description="Metrics such as recall@5; the eval set's headline ones by default"),
    ] = None,
    resamples: Annotated[int, Query(ge=100, le=20_000)] = 2000,
    seed: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
    principal: Principal = Depends(require_role("viewer")),
) -> CompareOut:
    with engine_of(request).connect() as conn:
        run_a = _run_or_404(conn, principal, a)
        run_b = _run_or_404(conn, principal, b)
        if run_a.status != "succeeded" or run_b.status != "succeeded":
            raise Problem("conflict", "Both eval runs must have succeeded.", code="not_finished")
        if run_a.eval_set_version_id != run_b.eval_set_version_id:
            raise Problem(
                "conflict",
                "Only eval runs of the same eval set version pair query by query.",
                code="not_paired",
            )
        rows_a = {row.item_id: row.score for row in store.query_results(conn, run_a.id)}
        rows_b = {row.item_id: row.score for row in store.query_results(conn, run_b.id)}
        config_a = specs.get_by_id(
            conn, "retrieval", principal.workspace.id, run_a.retrieval_config_id
        )
        config_b = specs.get_by_id(
            conn, "retrieval", principal.workspace.id, run_b.retrieval_config_id
        )
        pipelines = []
        for run in (run_a, run_b):
            pipeline = runs.get_run(conn, principal.workspace.id, run.pipeline_run_id)
            pipelines.append(dump_document(runs.run_spec(conn, pipeline)) if pipeline else {})
    k = int((run_a.context or {}).get("k_eval", 5))
    names = metric or [f"hit@{k}", f"recall@{k}", "mrr@10", "ndcg@10"]
    if any(
        not re.fullmatch(r"(hit|recall|precision|mrr|ndcg)@(\d{1,3}|all)", name) for name in names
    ):
        raise Problem("invalid_request", code="unknown_metric", errors=["metric"])
    result = compare_runs(
        rows_a,
        rows_b,
        rule=rule,
        checkpoint=checkpoint,
        metrics=names,
        resamples=resamples,
        seed=seed,
    )
    warnings = list(result["warnings"])
    if run_a.split != run_b.split:
        warnings.append("different_splits")
    return CompareOut(
        a=_ref(run_a),
        b=_ref(run_b),
        rule=rule,
        checkpoint=checkpoint,
        paired=result["paired"],
        resamples=resamples,
        seed=seed,
        metrics=[MetricDelta(**row) for row in result["metrics"]],
        queries=[QueryDelta(**row) for row in result["queries"]],
        retrieval_diff=_changes(
            {**(config_a.document if config_a else {}), "name": None},
            {**(config_b.document if config_b else {}), "name": None},
        ),
        pipeline_diff=_changes({**pipelines[0], "name": None}, {**pipelines[1], "name": None}),
        warnings=warnings,
    )
