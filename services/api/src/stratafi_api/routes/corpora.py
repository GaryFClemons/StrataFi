"""Corpora: create, edit the working set, freeze versions, and read them.

A corpus's working set can change at any time; ``POST /corpora/{id}/versions`` freezes it as the
next version, with a manifest hash. A frozen version never changes: there is no route that
edits one, and the database refuses it too.
"""

import uuid
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from stratafi_api.automation import fire
from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, problem_responses
from stratafi_components.detect import duplicate_groups
from stratafi_components.intake import InvalidName, entry_name
from stratafi_core.logs import current_log_context
from stratafi_store import corpora as store
from stratafi_store.audit import record_audit_event
from stratafi_store.documents import get_documents

router = APIRouter(tags=["corpora"])

NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"


class CorpusIn(BaseModel):
    name: str = Field(pattern=NAME_PATTERN)
    description: str = Field(default="", max_length=2000)


class VersionOut(BaseModel):
    id: uuid.UUID
    corpus_id: uuid.UUID
    version: int
    manifest_hash: str
    document_count: int
    total_bytes: int
    created_by: str
    frozen_at: datetime

    @classmethod
    def of(cls, version: store.CorpusVersion) -> "VersionOut":
        return cls(
            id=version.id,
            corpus_id=version.corpus_id,
            version=version.version,
            manifest_hash=version.manifest_hash,
            document_count=version.document_count,
            total_bytes=version.total_bytes,
            created_by=version.created_by,
            frozen_at=version.frozen_at,
        )


class CorpusOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    created_by: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, corpus: store.Corpus) -> "CorpusOut":
        return cls(
            id=corpus.id,
            name=corpus.name,
            description=corpus.description,
            created_by=corpus.created_by,
            created_at=corpus.created_at,
            updated_at=corpus.updated_at,
        )


class CorpusDetail(CorpusOut):
    entries: int = Field(description="Entries in the working set")
    total_bytes: int
    latest_version: VersionOut | None
    duplicate_groups: list[list[str]] = Field(
        description="Working-set names that share content; scan/dedupe keeps each group's first"
    )


class EntryOut(BaseModel):
    name: str
    document_id: uuid.UUID
    sha256: str
    size_bytes: int
    media_type: str
    traits: dict[str, Any]
    status: str
    reason_code: str | None
    doc_key: str | None = None
    doc_version: str | None = None
    collection: str | None = None
    acl: dict[str, Any] | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def of(cls, entry: store.Entry) -> "EntryOut":
        return cls(**entry.__dict__)


class EntryIn(BaseModel):
    document_id: uuid.UUID
    name: str | None = Field(
        default=None, max_length=512, description="Defaults to the document's name"
    )


class EntriesIn(BaseModel):
    entries: list[EntryIn] = Field(min_length=1, max_length=10_000)


class NamesIn(BaseModel):
    names: list[str] = Field(min_length=1, max_length=10_000)


class Count(BaseModel):
    count: int


def _corpus_or_404(conn: Any, principal: Principal, corpus_id: uuid.UUID) -> store.Corpus:
    corpus = store.get_corpus(conn, principal.workspace.id, corpus_id)
    if corpus is None:
        raise Problem("not_found", code="corpus_not_found")
    return corpus


def _audit(conn: Any, principal: Principal, action: str, resource_id: str, **details: Any) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type="corpus",
        resource_id=resource_id,
        request_id=current_log_context().get("request_id"),
        details=details,
    )


def _name_cursor(page: PageParams) -> str | None:
    if page.cursor is None:
        return None
    (name,) = decode_cursor(page.cursor, size=1)
    if not isinstance(name, str):
        raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
    return name


@router.post(
    "/corpora",
    response_model=CorpusOut,
    status_code=201,
    summary="Create a corpus",
    responses=problem_responses(400, 401, 403, 409, 503),
)
def create_corpus(
    request: Request, body: CorpusIn, principal: Principal = Depends(require_role("editor"))
) -> CorpusOut:
    with engine_of(request).begin() as conn:
        corpus = store.create_corpus(
            conn, principal.workspace.id, body.name, principal.subject, body.description
        )
        if corpus is None:
            raise Problem("conflict", "A corpus with this name exists.", code="name_taken")
        _audit(conn, principal, "corpus.create", str(corpus.id))
    return CorpusOut.of(corpus)


@router.get(
    "/corpora",
    response_model=Page[CorpusOut],
    summary="The workspace's corpora, by name",
    responses=problem_responses(400, 401, 403, 503),
)
def list_corpora(
    request: Request,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("viewer")),
) -> Page[CorpusOut]:
    with engine_of(request).connect() as conn:
        found = store.list_corpora(
            conn, principal.workspace.id, limit=page.limit + 1, after_name=_name_cursor(page)
        )
    items = [CorpusOut.of(corpus) for corpus in found[: page.limit]]
    cursor = encode_cursor([found[page.limit - 1].name]) if len(found) > page.limit else None
    return Page[CorpusOut](items=items, next_cursor=cursor)


@router.get(
    "/corpora/{corpus_id}",
    response_model=CorpusDetail,
    summary="A corpus: its working set's size, duplicates and latest version",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_corpus(
    request: Request, corpus_id: uuid.UUID, principal: Principal = Depends(require_role("viewer"))
) -> CorpusDetail:
    with engine_of(request).connect() as conn:
        corpus = _corpus_or_404(conn, principal, corpus_id)
        entries = store.list_entries(conn, principal.workspace.id, corpus_id)
        latest = store.latest_version(conn, principal.workspace.id, corpus_id)
    return CorpusDetail(
        **CorpusOut.of(corpus).model_dump(),
        entries=len(entries),
        total_bytes=sum(entry.size_bytes for entry in entries),
        latest_version=VersionOut.of(latest) if latest is not None else None,
        duplicate_groups=duplicate_groups([(entry.name, entry.sha256) for entry in entries]),
    )


@router.get(
    "/corpora/{corpus_id}/documents",
    response_model=Page[EntryOut],
    summary="The working set, by name",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_entries(
    request: Request,
    corpus_id: uuid.UUID,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("viewer")),
) -> Page[EntryOut]:
    with engine_of(request).connect() as conn:
        _corpus_or_404(conn, principal, corpus_id)
        found = store.list_entries(
            conn,
            principal.workspace.id,
            corpus_id,
            limit=page.limit + 1,
            after_name=_name_cursor(page),
        )
    items = [EntryOut.of(entry) for entry in found[: page.limit]]
    cursor = encode_cursor([found[page.limit - 1].name]) if len(found) > page.limit else None
    return Page[EntryOut](items=items, next_cursor=cursor)


@router.post(
    "/corpora/{corpus_id}/documents",
    response_model=Count,
    summary="Add documents to the working set; an entry with the same name is replaced",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def add_entries(
    request: Request,
    corpus_id: uuid.UUID,
    body: EntriesIn,
    principal: Principal = Depends(require_role("editor")),
) -> Count:
    with engine_of(request).begin() as conn:
        _corpus_or_404(conn, principal, corpus_id)
        found = get_documents(
            conn, principal.workspace.id, [entry.document_id for entry in body.entries]
        )
        new: list[store.NewEntry] = []
        for index, entry in enumerate(body.entries):
            document = found.get(entry.document_id)
            if document is None:
                raise Problem(
                    "not_found",
                    code="document_not_found",
                    errors=[f"entries.{index}.document_id"],
                )
            try:
                name = entry_name(entry.name if entry.name is not None else document.name)
            except InvalidName:
                raise Problem(
                    "invalid_request", code="invalid_name", errors=[f"entries.{index}.name"]
                ) from None
            new.append(store.NewEntry(name, document.id))
        count = store.add_entries(conn, principal.workspace.id, corpus_id, new, principal.subject)
        _audit(conn, principal, "corpus.add_documents", str(corpus_id), entries=count)
    return Count(count=count)


@router.post(
    "/corpora/{corpus_id}/documents/remove",
    response_model=Count,
    summary="Remove entries from the working set, by name",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def remove_entries(
    request: Request,
    corpus_id: uuid.UUID,
    body: NamesIn,
    principal: Principal = Depends(require_role("editor")),
) -> Count:
    with engine_of(request).begin() as conn:
        _corpus_or_404(conn, principal, corpus_id)
        count = store.remove_entries(conn, principal.workspace.id, corpus_id, body.names)
        _audit(conn, principal, "corpus.remove_documents", str(corpus_id), entries=count)
    return Count(count=count)


@router.post(
    "/corpora/{corpus_id}/versions",
    response_model=VersionOut,
    status_code=201,
    summary="Freeze the working set as the next version",
    description="201 with a new version; 200 with the latest version when the working set "
    "hasn't changed since.",
    responses={
        200: {"model": VersionOut, "description": "Unchanged: the latest version"},
        **problem_responses(400, 401, 403, 404, 503),
    },
)
def freeze(
    request: Request, corpus_id: uuid.UUID, principal: Principal = Depends(require_role("editor"))
) -> JSONResponse:
    with engine_of(request).begin() as conn:
        corpus = _corpus_or_404(conn, principal, corpus_id)
        version, created = store.freeze(conn, principal.workspace.id, corpus_id, principal.subject)
        if created:
            _audit(
                conn,
                principal,
                "corpus.freeze",
                str(corpus_id),
                version=version.version,
                documents=version.document_count,
            )
            fire(request, conn, principal, "corpus_version_frozen", corpus=corpus.name)
    return JSONResponse(
        VersionOut.of(version).model_dump(mode="json"), status_code=201 if created else 200
    )


@router.get(
    "/corpora/{corpus_id}/versions",
    response_model=Page[VersionOut],
    summary="The corpus's frozen versions, newest first",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def list_versions(
    request: Request,
    corpus_id: uuid.UUID,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("viewer")),
) -> Page[VersionOut]:
    before = None
    if page.cursor is not None:
        (before,) = decode_cursor(page.cursor, size=1)
        if not isinstance(before, int):
            raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
    with engine_of(request).connect() as conn:
        _corpus_or_404(conn, principal, corpus_id)
        found = store.list_versions(
            conn, principal.workspace.id, corpus_id, limit=page.limit + 1, before_version=before
        )
    items = [VersionOut.of(version) for version in found[: page.limit]]
    cursor = encode_cursor([found[page.limit - 1].version]) if len(found) > page.limit else None
    return Page[VersionOut](items=items, next_cursor=cursor)


def _version_or_404(conn: Any, principal: Principal, corpus_id: uuid.UUID, version: int):
    found = store.get_version(conn, principal.workspace.id, corpus_id=corpus_id, version=version)
    if found is None:
        raise Problem("not_found", code="corpus_version_not_found")
    return found


@router.get(
    "/corpora/{corpus_id}/versions/{version}",
    response_model=VersionOut,
    summary="One frozen version",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_version(
    request: Request,
    corpus_id: uuid.UUID,
    version: int,
    principal: Principal = Depends(require_role("viewer")),
) -> VersionOut:
    with engine_of(request).connect() as conn:
        return VersionOut.of(_version_or_404(conn, principal, corpus_id, version))


@router.get(
    "/corpora/{corpus_id}/versions/{version}/documents",
    response_model=Page[EntryOut],
    summary="A frozen version's entries, by name, as they were when it was frozen",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def version_entries(
    request: Request,
    corpus_id: uuid.UUID,
    version: int,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("viewer")),
) -> Page[EntryOut]:
    with engine_of(request).connect() as conn:
        found_version = _version_or_404(conn, principal, corpus_id, version)
        found = store.version_entries(
            conn,
            principal.workspace.id,
            found_version.id,
            limit=page.limit + 1,
            after_name=_name_cursor(page),
        )
    items = [EntryOut.of(entry) for entry in found[: page.limit]]
    cursor = encode_cursor([found[page.limit - 1].name]) if len(found) > page.limit else None
    return Page[EntryOut](items=items, next_cursor=cursor)
