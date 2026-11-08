"""Tracking: annotations and the search language.

- ``GET`` and ``PATCH /runs|eval-runs|experiments/{id}/annotation``: a run's title, tags,
  Markdown note and archive. Editors change them; viewers read them. They're people's text: no
  log, audit event or error message holds a title, a tag or a note, and an
  ``<kind>.annotate`` audit event says only which fields changed and how many tags there are.
- ``POST /eval-runs/search``: MLflow's ``search_runs`` over the workspace's eval runs. The
  filter is parsed and evaluated here, over the newest 5,000 (archived ones only when asked), so
  it never reaches SQL as text; a syntax error is a 400 ``invalid_filter`` with its code and
  character position, never the filter. Each item carries the requested columns' values, and
  the response lists the keys the matches have, for column pickers and charts.
"""

import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import decode_cursor, encode_cursor
from stratafi_api.problems import Problem, problem_responses
from stratafi_api.tracked import label_of, tracked_run
from stratafi_core.logs import current_log_context
from stratafi_core.tracking import (
    AnnotationError,
    SearchError,
    TrackedRun,
    available_keys,
    check_note,
    check_tag_key,
    check_title,
    headline_names,
    merge_tags,
    parse_filter,
    parse_order,
    sort_runs,
    split_column,
)
from stratafi_core.tracking import matches as filter_matches
from stratafi_store import tracking as store
from stratafi_store.audit import record_audit_event

router = APIRouter(tags=["tracking"])

Kind = Literal["run", "eval_run", "experiment"]
MAX_COLUMNS = 40
MAX_PAGE = 500


class AnnotationIn(BaseModel):
    """What to change; fields left out stay as they are."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(
        default=None, max_length=1000, description="Up to 120 characters; empty or null clears it"
    )
    note: str | None = Field(
        default=None,
        max_length=25_000,
        description="Markdown, up to 20,000 characters; empty or null clears it",
    )
    set_tags: dict[str, str] = Field(
        default_factory=dict,
        description="Tags to add or replace: keys of letters, digits and _ . - / :, not "
        "starting with stratafi.; values up to 500 characters",
    )
    delete_tags: list[str] = Field(default_factory=list, max_length=50)
    archived: bool | None = Field(
        default=None, description="true hides it from lists and searches; false restores it"
    )


class AnnotationOut(BaseModel):
    title: str | None
    tags: dict[str, str]
    note: str | None
    archived: bool
    archived_at: datetime | None
    updated_at: datetime | None = Field(description="Null until the first change")

    @classmethod
    def of(cls, annotation: store.Annotation) -> "AnnotationOut":
        return cls(
            title=annotation.title,
            tags=annotation.tags,
            note=annotation.note,
            archived=annotation.archived_at is not None,
            archived_at=annotation.archived_at,
            updated_at=annotation.updated_at,
        )


NOT_FOUND = {"run": "run_not_found", "eval_run": "eval_run_not_found",
             "experiment": "experiment_not_found"}  # fmt: skip


def _read(request: Request, kind: Kind, resource_id: uuid.UUID, principal: Principal):
    with engine_of(request).connect() as conn:
        found = store.get_annotation(conn, kind, principal.workspace.id, resource_id)
    if found is None:
        raise Problem("not_found", code=NOT_FOUND[kind])
    return AnnotationOut.of(found)


def _annotate_and_audit(
    request: Request, kind: Kind, resource_id: uuid.UUID, body: AnnotationIn, principal: Principal
) -> AnnotationOut:
    given = body.model_fields_set
    try:
        title = check_title(body.title) if "title" in given else store.UNSET
        note = check_note(body.note) if "note" in given else store.UNSET
        for key in body.delete_tags:
            check_tag_key(key)
    except AnnotationError as error:
        raise Problem("invalid_request", code=error.code, errors=[error.field]) from None
    changed = [name for name in ("title", "note", "archived") if name in given]
    workspace_id = principal.workspace.id
    with engine_of(request).begin() as conn:
        tags = None
        if body.set_tags or body.delete_tags:
            current = store.current_tags(conn, kind, workspace_id, resource_id)
            if current is None:
                raise Problem("not_found", code=NOT_FOUND[kind])
            try:
                tags = merge_tags(current, set_tags=body.set_tags, delete=body.delete_tags)
            except AnnotationError as error:
                raise Problem("invalid_request", code=error.code, errors=[error.field]) from None
            changed.append("tags")
        annotation = store.annotate(
            conn,
            kind,
            workspace_id,
            resource_id,
            by=principal.subject,
            title=title,
            note=note,
            tags=tags,
            archived=body.archived,
        )
        if annotation is None:
            raise Problem("not_found", code=NOT_FOUND[kind])
        _audit(conn, principal, kind, resource_id, changed, len(annotation.tags))
    return AnnotationOut.of(annotation)


def _audit(
    conn: Connection,
    principal: Principal,
    kind: str,
    resource_id: uuid.UUID,
    changed: list[str],
    tag_count: int,
) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=f"{kind}.annotate",
        resource_type=kind,
        resource_id=str(resource_id),
        request_id=current_log_context().get("request_id"),
        details={"changed": sorted(set(changed)), "tag_count": tag_count},
    )


READ = {
    "response_model": AnnotationOut,
    "responses": problem_responses(400, 401, 403, 404, 503),
}
CHANGE = {
    "response_model": AnnotationOut,
    "description": "Fields left out stay as they are. `set_tags` adds or replaces tags and "
    "`delete_tags` removes them. Archiving hides it from lists and searches; nothing is deleted.",
    "responses": problem_responses(400, 401, 403, 404, 503),
}


@router.get("/runs/{run_id}/annotation", summary="A pipeline run's title, tags, note", **READ)
def read_run_annotation(
    request: Request, run_id: uuid.UUID, principal: Principal = Depends(require_role("viewer"))
) -> AnnotationOut:
    return _read(request, "run", run_id, principal)


@router.patch("/runs/{run_id}/annotation", summary="Annotate or archive a pipeline run", **CHANGE)
def annotate_run(
    request: Request,
    run_id: uuid.UUID,
    body: AnnotationIn,
    principal: Principal = Depends(require_role("editor")),
) -> AnnotationOut:
    return _annotate_and_audit(request, "run", run_id, body, principal)


@router.get(
    "/eval-runs/{eval_run_id}/annotation", summary="An eval run's title, tags, note", **READ
)
def read_eval_run_annotation(
    request: Request,
    eval_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> AnnotationOut:
    return _read(request, "eval_run", eval_run_id, principal)


@router.patch(
    "/eval-runs/{eval_run_id}/annotation", summary="Annotate or archive an eval run", **CHANGE
)
def annotate_eval_run(
    request: Request,
    eval_run_id: uuid.UUID,
    body: AnnotationIn,
    principal: Principal = Depends(require_role("editor")),
) -> AnnotationOut:
    return _annotate_and_audit(request, "eval_run", eval_run_id, body, principal)


@router.get(
    "/experiments/{experiment_id}/annotation",
    summary="An experiment's title, tags, note",
    **READ,
)
def read_experiment_annotation(
    request: Request,
    experiment_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> AnnotationOut:
    return _read(request, "experiment", experiment_id, principal)


@router.patch(
    "/experiments/{experiment_id}/annotation",
    summary="Annotate or archive an experiment",
    **CHANGE,
)
def annotate_experiment(
    request: Request,
    experiment_id: uuid.UUID,
    body: AnnotationIn,
    principal: Principal = Depends(require_role("editor")),
) -> AnnotationOut:
    return _annotate_and_audit(request, "experiment", experiment_id, body, principal)


# --- searching ----------------------------------------------------------------------------


class EvalRunSearchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filter: str = Field(
        default="",
        max_length=4000,
        description="Clauses joined by `and`: `metrics.recall@5 > 0.8 and "
        "params.rerank.component = 'rerank/none' and tags.owner = 'ana'`",
    )
    order_by: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Keys with ASC or DESC, such as `metrics.recall@5 DESC`; newest first by "
        "default",
    )
    columns: list[str] = Field(
        default_factory=list,
        max_length=MAX_COLUMNS,
        description="Keys whose values each item carries, such as `params.chunk.component`",
    )
    include_archived: bool = False
    limit: int = Field(default=50, ge=1, le=MAX_PAGE)
    cursor: str | None = Field(default=None, max_length=512)


class TrackedRunOut(BaseModel):
    id: uuid.UUID
    label: str = Field(description="Its title, or its config and pipeline")
    status: str
    split: str
    title: str | None
    tags: dict[str, str]
    archived: bool
    created_by: str
    created_at: datetime
    finished_at: datetime | None
    run_id: uuid.UUID = Field(description="The pipeline run searched")
    retrieval_config_id: uuid.UUID
    pipeline_spec_id: uuid.UUID | None
    eval_set_version_id: uuid.UUID
    attributes: dict[str, Any]
    headline: dict[str, float | None] = Field(
        description="hit and recall at the eval set's k, MRR and nDCG at 10, final list"
    )
    latency_p95_ms: float | None
    values: dict[str, Any] = Field(description="The requested columns' values")


class EvalRunSearchKeys(BaseModel):
    metrics: list[str]
    params: list[str]
    tags: list[str]


class EvalRunSearchOut(BaseModel):
    items: list[TrackedRunOut]
    next_cursor: str | None
    total: int = Field(description="Runs the filter matched")
    scanned: int = Field(description="Runs searched: the newest 5,000 at most")
    keys: EvalRunSearchKeys = Field(
        description="Every metric, parameter and tag key the matches have"
    )


def tracked_out(row: store.SearchRow, run: TrackedRun, columns: list[str]) -> TrackedRunOut:
    k_eval = int(run.attributes.get("k_eval") or 5)
    return TrackedRunOut(
        id=row.id,
        label=label_of(row),
        status=row.status,
        split=row.split,
        title=row.title,
        tags=run.tags,
        archived=row.archived_at is not None,
        created_by=row.created_by,
        created_at=row.created_at,
        finished_at=row.finished_at,
        run_id=row.pipeline_run_id,
        retrieval_config_id=row.retrieval_config_id,
        pipeline_spec_id=row.pipeline_spec_id,
        eval_set_version_id=row.eval_set_version_id,
        attributes=run.attributes,
        headline={name: run.metrics.get(name) for name in headline_names(k_eval)},
        latency_p95_ms=run.metrics.get("latency.total_ms.p95"),
        values={column: run.value(column) for column in columns},
    )


def _invalid(error: SearchError, field: str) -> Problem:
    return Problem(
        "invalid_request",
        f"The {field} doesn't parse ({error.code} at character {error.position}).",
        code=f"invalid_{field}",
        errors=[field],
    )


def run_search(
    rows: list[store.SearchRow], *, filter: str, order_by: list[str]
) -> tuple[list[tuple[store.SearchRow, TrackedRun]], list[TrackedRun]]:
    """The rows a filter matches, in order, each with its tracked run; and every row's."""
    try:
        clauses = parse_filter(filter)
    except SearchError as error:
        raise _invalid(error, "filter") from None
    try:
        orders = parse_order(order_by)
    except SearchError as error:
        raise _invalid(error, "order") from None
    tracked = [tracked_run(row) for row in rows]
    by_id = {run.id: row for run, row in zip(tracked, rows, strict=True)}
    matched = sort_runs([run for run in tracked if filter_matches(run, clauses)], orders)
    return [(by_id[run.id], run) for run in matched], tracked


@router.post(
    "/eval-runs/search",
    response_model=EvalRunSearchOut,
    summary="Search the workspace's eval runs with the search language",
    description="MLflow's `search_runs`: a filter over `metrics.`, `params.`, `tags.` and "
    "`attributes.` keys, an ordering and the columns to return. Evaluated over the "
    "newest 5,000 eval runs.",
    responses=problem_responses(400, 401, 403, 503),
)
def search_eval_runs(
    request: Request, body: EvalRunSearchIn, principal: Principal = Depends(require_role("viewer"))
) -> EvalRunSearchOut:
    for column in body.columns:
        try:
            split_column(column)
        except SearchError:
            raise Problem("invalid_request", code="unknown_column", errors=["columns"]) from None
    with engine_of(request).connect() as conn:
        rows = store.search_rows(
            conn, principal.workspace.id, include_archived=body.include_archived, slices=False
        )
    matched, _ = run_search(rows, filter=body.filter, order_by=body.order_by)
    start = 0
    if body.cursor is not None:
        (after,) = decode_cursor(body.cursor, size=1)
        positions = {str(row.id): index for index, (row, _) in enumerate(matched)}
        if not isinstance(after, str) or after not in positions:
            raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
        start = positions[after] + 1
    page = matched[start : start + body.limit]
    items = [tracked_out(row, run, body.columns) for row, run in page]
    more = start + body.limit < len(matched)
    keys = available_keys(run for _, run in matched)
    return EvalRunSearchOut(
        items=items,
        next_cursor=encode_cursor([str(page[-1][0].id)]) if more and page else None,
        total=len(matched),
        scanned=len(rows),
        keys=EvalRunSearchKeys(**keys),
    )
