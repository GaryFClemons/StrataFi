"""Eval sets: import, list, read, split and freeze; and items authored from a passage on the
Parse screen.

``POST /eval-sets/import`` takes a YAML (``stratafi.eval-set.v1``, or the retrieval YAML format
unchanged), JSONL or CSV document, resolves it (askers, tags, a seeded and stratified split)
and stores it as the eval set's next version: frozen at once by default, or a draft that can be
split again and frozen later. Importing the same set twice returns the version that holds it.
Its ``askers`` names who each asker is when the document has no table of its own (a CSV).

``POST /eval-sets/author`` adds one item, "a question about this passage", as the set's next
version: a draft, so more items can follow before it is frozen (a draft's items can be split
again). Its evidence is the passage itself.

A version's test split is locked: evaluations read it only as counted confirmations;
``test_reads`` says how often that has happened. Audit events carry counts, never text.
"""

import uuid
from datetime import date, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, problem_responses
from stratafi_api.routes.specs import invalid
from stratafi_core.evalsets import (
    DEFAULT_TEST_FRACTION,
    MAX_TEST_FRACTION,
    Asker,
    AskerName,
    EvalItem,
    EvalSet,
    EvidenceRef,
    ResolveReport,
    load_eval_set,
    resolve_eval_set,
)
from stratafi_core.logs import current_log_context
from stratafi_core.specs import NAME_PATTERN, SpecInvalid
from stratafi_store import evalsets as store
from stratafi_store.audit import record_audit_event

router = APIRouter(tags=["eval-sets"])

MAX_CONTENT_CHARS = 8_000_000
Split = Literal["dev", "test"]


class EvalSetImport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME_PATTERN, description="The eval set: created if it's new")
    description: str | None = Field(default=None, max_length=2000)
    format: Literal["yaml", "jsonl", "csv"] = Field(
        default="yaml",
        description="yaml: stratafi.eval-set.v1, or the retrieval YAML format unchanged; "
        "jsonl: one item per line; csv: one row per evidence item",
    )
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    as_of: date | None = Field(default=None, description='Overrides the document\'s "today"')
    k: int | None = Field(default=None, ge=1, le=100, description="Overrides defaults.k")
    askers: dict[AskerName, Asker] | None = Field(
        default=None,
        max_length=256,
        description="Overrides the askers table: who each asker name in the items is (a CSV "
        "has no table of its own)",
    )
    test_fraction: float = Field(
        default=DEFAULT_TEST_FRACTION,
        ge=0,
        le=MAX_TEST_FRACTION,
        description="The share of items to put in the locked test split",
    )
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Seeds the stratified split")
    resplit: bool = Field(default=False, description="Split items that name a split too")
    freeze: bool = Field(default=True, description="Freeze the version at once")


class SplitIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_fraction: float = Field(default=DEFAULT_TEST_FRACTION, ge=0, le=MAX_TEST_FRACTION)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class EvalSetVersionOut(BaseModel):
    id: uuid.UUID
    eval_set_id: uuid.UUID
    version: int
    document: dict[str, Any] = Field(description="The set's own fields: as_of, defaults, askers")
    document_hash: str
    item_count: int
    split_counts: dict[str, int]
    split_params: dict[str, Any] | None = Field(description="How the split was made: seed, share")
    source: dict[str, Any] = Field(description="Where the items came from: format and counts")
    frozen: bool
    frozen_at: datetime | None
    test_locked: bool = Field(description="Evaluating the test split is a counted confirmation")
    test_reads: int = Field(description="How often the test split has been evaluated")
    last_test_read_at: datetime | None
    created_by: str
    created_at: datetime

    @classmethod
    def of(cls, version: store.EvalSetVersion) -> "EvalSetVersionOut":
        return cls(**{key: getattr(version, key) for key in cls.model_fields})


class EvalSetOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    latest_version: EvalSetVersionOut | None

    @classmethod
    def of(cls, row: store.EvalSetRow, latest: store.EvalSetVersion | None) -> "EvalSetOut":
        return cls(
            id=row.id,
            name=row.name,
            description=row.description,
            created_by=row.created_by,
            created_at=row.created_at,
            updated_at=row.updated_at,
            latest_version=EvalSetVersionOut.of(latest) if latest is not None else None,
        )


class ImportReport(BaseModel):
    items: int
    derived_tags: int = Field(description="Items whose tags were derived")
    assigned_splits: int = Field(description="Items this import split")
    splits: dict[str, int]

    @classmethod
    def of(cls, report: ResolveReport) -> "ImportReport":
        return cls(
            items=report.items,
            derived_tags=report.derived_tags,
            assigned_splits=report.assigned_splits,
            splits=report.splits,
        )


class ImportOut(BaseModel):
    eval_set: EvalSetOut
    version: EvalSetVersionOut
    created: bool = Field(description="False: the same set is already this version")
    report: ImportReport


class EvalItemOut(BaseModel):
    id: str
    ordinal: int
    query: str
    asker: dict[str, Any]
    expected: list[dict[str, Any]]
    avoid: list[dict[str, Any]]
    must_not_return: list[dict[str, Any]]
    near_misses: list[dict[str, Any]]
    reference_answer: str | None
    tags: list[str]
    split: Split

    @classmethod
    def of(cls, item: store.StoredItem) -> "EvalItemOut":
        return cls(
            id=item.item_id,
            ordinal=item.ordinal,
            query=item.query,
            asker=item.asker,
            expected=item.expected,
            avoid=item.avoid,
            must_not_return=item.must_not_return,
            near_misses=item.near_misses,
            reference_answer=item.reference_answer,
            tags=item.tags,
            split=item.split,  # type: ignore[arg-type]
        )


def _audit(conn: Connection, principal: Principal, action: str, resource_id: uuid.UUID, **details):
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


def _set_or_404(conn: Connection, principal: Principal, eval_set_id: uuid.UUID) -> store.EvalSetRow:
    found = store.get_set(conn, principal.workspace.id, eval_set_id)
    if found is None:
        raise Problem("not_found", code="eval_set_not_found")
    return found


def _version_or_404(
    conn: Connection, principal: Principal, eval_set_id: uuid.UUID, version: int
) -> store.EvalSetVersion:
    _set_or_404(conn, principal, eval_set_id)
    found = store.get_version(
        conn, principal.workspace.id, eval_set_id=eval_set_id, version=version
    )
    if found is None:
        raise Problem("not_found", code="eval_set_version_not_found")
    return found


def _split_params(test_fraction: float, seed: int) -> dict[str, Any]:
    return {"test_fraction": test_fraction, "seed": seed, "stratified_by": ["tags"]}


@router.post(
    "/eval-sets/import",
    response_model=ImportOut,
    status_code=201,
    summary="Import an eval set as its next version: YAML, JSONL or CSV",
    description="201 with a new version; 200 with the version that already holds the same set. "
    "Askers, tags and splits are resolved first, so the stored version is complete.",
    responses={
        200: {"model": ImportOut, "description": "Unchanged: the existing version"},
        **problem_responses(400, 401, 403, 503),
    },
)
def import_eval_set(
    request: Request, body: EvalSetImport, principal: Principal = Depends(require_role("editor"))
) -> JSONResponse:
    overrides: dict[str, Any] = {
        "description": body.description,
        "as_of": body.as_of,
        "askers": (
            {name: asker.model_dump(exclude_none=True) for name, asker in body.askers.items()}
            if body.askers is not None
            else None
        ),
    }
    try:
        parsed = load_eval_set(body.content, body.format, name=body.name, overrides=overrides)
        if body.k is not None:
            parsed = parsed.model_copy(
                update={"defaults": parsed.defaults.model_copy(update={"k": body.k})}
            )
        resolved, report = resolve_eval_set(
            parsed, test_fraction=body.test_fraction, seed=body.seed, resplit=body.resplit
        )
    except SpecInvalid as error:
        raise invalid(error) from None
    with engine_of(request).begin() as conn:
        row = store.get_or_create_set(
            conn, principal.workspace.id, body.name, principal.subject, resolved.description
        )
        version, created = store.save_version(
            conn,
            workspace_id=principal.workspace.id,
            eval_set_id=row.id,
            eval_set=resolved,
            created_by=principal.subject,
            split_params=_split_params(body.test_fraction, body.seed),
            source={"format": body.format, "items": report.items},
            freeze=body.freeze,
        )
        _audit(
            conn,
            principal,
            "eval_set.import",
            row.id,
            version=version.version,
            created=created,
            items=report.items,
            frozen=version.frozen,
        )
        row = store.get_set(conn, principal.workspace.id, row.id) or row
    out = ImportOut(
        eval_set=EvalSetOut.of(row, version),
        version=EvalSetVersionOut.of(version),
        created=created,
        report=ImportReport.of(report),
    )
    return JSONResponse(out.model_dump(mode="json"), status_code=201 if created else 200)


def _name_cursor(page: PageParams) -> str | None:
    if page.cursor is None:
        return None
    (name,) = decode_cursor(page.cursor, size=1)
    if not isinstance(name, str):
        raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
    return name


@router.get(
    "/eval-sets",
    response_model=Page[EvalSetOut],
    summary="The workspace's eval sets, by name, each with its latest version",
    responses=problem_responses(400, 401, 403, 503),
)
def list_eval_sets(
    request: Request,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("viewer")),
) -> Page[EvalSetOut]:
    with engine_of(request).connect() as conn:
        found = store.list_sets(
            conn, principal.workspace.id, limit=page.limit + 1, after_name=_name_cursor(page)
        )
        latest = store.latest_versions(conn, principal.workspace.id, [row.id for row in found])
    items = [EvalSetOut.of(row, latest.get(row.id)) for row in found[: page.limit]]
    cursor = encode_cursor([found[page.limit - 1].name]) if len(found) > page.limit else None
    return Page[EvalSetOut](items=items, next_cursor=cursor)


@router.get(
    "/eval-sets/{eval_set_id}",
    response_model=EvalSetOut,
    summary="One eval set, with its latest version",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_eval_set(
    request: Request,
    eval_set_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> EvalSetOut:
    with engine_of(request).connect() as conn:
        row = _set_or_404(conn, principal, eval_set_id)
        latest = store.latest_version(conn, principal.workspace.id, row.id)
    return EvalSetOut.of(row, latest)


@router.get(
    "/eval-sets/{eval_set_id}/versions",
    response_model=Page[EvalSetVersionOut],
    summary="An eval set's versions, newest first",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_eval_set_versions(
    request: Request,
    eval_set_id: uuid.UUID,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("viewer")),
) -> Page[EvalSetVersionOut]:
    before = None
    if page.cursor is not None:
        (before,) = decode_cursor(page.cursor, size=1)
        if not isinstance(before, int):
            raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
    with engine_of(request).connect() as conn:
        _set_or_404(conn, principal, eval_set_id)
        found = store.list_versions(
            conn, principal.workspace.id, eval_set_id, limit=page.limit + 1, before_version=before
        )
    items = [EvalSetVersionOut.of(version) for version in found[: page.limit]]
    cursor = encode_cursor([found[page.limit - 1].version]) if len(found) > page.limit else None
    return Page[EvalSetVersionOut](items=items, next_cursor=cursor)


@router.get(
    "/eval-sets/{eval_set_id}/versions/{version}",
    response_model=EvalSetVersionOut,
    summary="One version of an eval set",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_eval_set_version(
    request: Request,
    eval_set_id: uuid.UUID,
    version: int,
    principal: Principal = Depends(require_role("viewer")),
) -> EvalSetVersionOut:
    with engine_of(request).connect() as conn:
        return EvalSetVersionOut.of(_version_or_404(conn, principal, eval_set_id, version))


@router.get(
    "/eval-sets/{eval_set_id}/versions/{version}/items",
    response_model=Page[EvalItemOut],
    summary="A version's items, in order: queries, askers, evidence, tags and splits",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_eval_items(
    request: Request,
    eval_set_id: uuid.UUID,
    version: int,
    page: PageParams = Depends(page_params),
    split: Annotated[Split | None, Query(description="Only this split's items")] = None,
    tag: Annotated[str | None, Query(max_length=64, description="Only items with this tag")] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> Page[EvalItemOut]:
    after = None
    if page.cursor is not None:
        (after,) = decode_cursor(page.cursor, size=1)
        if not isinstance(after, int):
            raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
    with engine_of(request).connect() as conn:
        found_version = _version_or_404(conn, principal, eval_set_id, version)
        found = store.list_items(
            conn,
            found_version.id,
            split=split,
            tag=tag,
            limit=page.limit + 1,
            after_ordinal=after,
        )
    items = [EvalItemOut.of(item) for item in found[: page.limit]]
    cursor = encode_cursor([found[page.limit - 1].ordinal]) if len(found) > page.limit else None
    return Page[EvalItemOut](items=items, next_cursor=cursor)


@router.post(
    "/eval-sets/{eval_set_id}/versions/{version}/split",
    response_model=EvalSetVersionOut,
    summary="Split a draft version again: seeded and stratified by tags",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def split_eval_set_version(
    request: Request,
    eval_set_id: uuid.UUID,
    version: int,
    body: SplitIn,
    principal: Principal = Depends(require_role("editor")),
) -> EvalSetVersionOut:
    with engine_of(request).begin() as conn:
        found = _version_or_404(conn, principal, eval_set_id, version)
        if found.frozen:
            raise Problem("conflict", "A frozen version can't change.", code="version_frozen")
        current: EvalSet = store.stored_eval_set(conn, found)
        resplit, _ = resolve_eval_set(
            current, test_fraction=body.test_fraction, seed=body.seed, resplit=True
        )
        updated = store.replace_items(
            conn, found, resplit, split_params=_split_params(body.test_fraction, body.seed)
        )
        _audit(
            conn, principal, "eval_set.split", eval_set_id, version=version, **updated.split_counts
        )
    return EvalSetVersionOut.of(updated)


@router.post(
    "/eval-sets/{eval_set_id}/versions/{version}/freeze",
    response_model=EvalSetVersionOut,
    summary="Freeze a version: it never changes again, and evaluations can use it",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def freeze_eval_set_version(
    request: Request,
    eval_set_id: uuid.UUID,
    version: int,
    principal: Principal = Depends(require_role("editor")),
) -> EvalSetVersionOut:
    with engine_of(request).begin() as conn:
        found = _version_or_404(conn, principal, eval_set_id, version)
        if not found.frozen:
            found = store.freeze_version(conn, found)
            _audit(conn, principal, "eval_set.freeze", eval_set_id, version=version)
    return EvalSetVersionOut.of(found)


class AuthorIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_set: str = Field(pattern=NAME_PATTERN, description="The set: created if it's new")
    query: str = Field(min_length=3, max_length=4000, description="The question")
    evidence: EvidenceRef = Field(description="The passage the question is about")
    asker: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    tags: list[str] = Field(default_factory=list, max_length=8)


class AuthorOut(BaseModel):
    eval_set: EvalSetOut
    version: EvalSetVersionOut
    item_id: str


@router.post(
    "/eval-sets/author",
    response_model=AuthorOut,
    status_code=201,
    summary="Add a question about a passage to an eval set, as its next (draft) version",
    responses=problem_responses(400, 401, 403, 409, 503),
)
def author_item(
    request: Request, body: AuthorIn, principal: Principal = Depends(require_role("editor"))
) -> AuthorOut:
    workspace = principal.workspace.id
    with engine_of(request).begin() as conn:
        row = store.get_or_create_set(conn, workspace, body.eval_set, principal.subject, "")
        latest = store.latest_version(conn, workspace, row.id)
        existing = store.stored_eval_set(conn, latest) if latest is not None else None
        items = list(existing.items) if existing is not None else []
        taken = {item.id for item in items}
        number = len(items) + 1
        while f"authored-{number}" in taken:
            number += 1
        item_id = f"authored-{number}"
        try:
            item = EvalItem(
                id=item_id,
                query=body.query,
                asker=body.asker,
                expected=[body.evidence],
                tags=["authored", *body.tags],
            )
            base = existing.model_copy(update={"items": [*items, item]}) if existing else None
            eval_set = base or EvalSet(name=body.eval_set, items=[item])
            resolved, report = resolve_eval_set(eval_set, test_fraction=0.0, seed=0)
        except (SpecInvalid, ValueError) as error:
            if isinstance(error, SpecInvalid):
                raise invalid(error) from None
            raise Problem("invalid_request", code="invalid_item") from None
        version, _ = store.save_version(
            conn,
            workspace_id=workspace,
            eval_set_id=row.id,
            eval_set=resolved,
            created_by=principal.subject,
            split_params=_split_params(0.0, 0),
            source={"format": "authored", "items": report.items},
            freeze=False,
        )
        _audit(conn, principal, "eval_set.author", row.id, version=version.version, items=1)
        row = store.get_set(conn, workspace, row.id) or row
    return AuthorOut(
        eval_set=EvalSetOut.of(row, version),
        version=EvalSetVersionOut.of(version),
        item_id=item_id,
    )
