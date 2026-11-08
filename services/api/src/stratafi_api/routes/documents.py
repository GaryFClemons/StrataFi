"""Documents: upload, import from a server folder, list, and read with their provenance.

``POST /documents`` streams a ``multipart/form-data`` body with one or more ``file`` parts into
quarantine (``stratafi_api.intake``); ``?corpus_id=`` also adds them to a corpus's working set.
A ``manifest`` part (``extract/manifest``) gives the entries their metadata, as
``POST /documents/import-manifest`` does for a server folder. ``POST /documents/import-s3``
imports a bucket prefix the configuration names (``extract/s3``), and
``POST /documents/import-url`` fetches one URL through the SSRF guard (``extract/url``).
``GET /file-types`` describes the file-type taxonomy and the traits detection sets.
"""

import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from stratafi_api.deps import engine_of, store_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.intake import (
    IngestedFile,
    IngestReport,
    MultipartFiles,
    UploadError,
    boundary_of,
    folder_files,
    folder_under,
    ingest,
    manifest_entry_fields,
    manifest_files,
    read_manifest_text,
    s3_files,
    url_file,
)
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, ProblemIssue, problem_responses
from stratafi_components.manifest import CorpusManifest, load_corpus_manifest
from stratafi_components.taxonomy import FILE_TYPES, TRAITS
from stratafi_core.logs import current_log_context
from stratafi_core.settings import Settings
from stratafi_core.specs import SpecInvalid
from stratafi_store import purge
from stratafi_store.audit import record_audit_event
from stratafi_store.corpora import get_corpus
from stratafi_store.documents import DocumentRow, get_document, list_documents, origins_of

router = APIRouter(tags=["documents"])


class DocumentOut(BaseModel):
    id: uuid.UUID
    name: str = Field(description="The name it first arrived with")
    sha256: str
    size_bytes: int
    extension: str
    media_type: str = Field(description="What its bytes are")
    kind: str
    mismatch: bool = Field(description="Its bytes disagree with its extension")
    known: bool = Field(description="Its extension is one Stratafi lists")
    traits: dict[str, Any]
    status: str = Field(description="stored, or refused (an archive that broke a limit)")
    reason_code: str | None
    created_at: datetime

    @classmethod
    def of(cls, row: DocumentRow) -> "DocumentOut":
        return cls(
            id=row.id,
            name=row.name,
            sha256=row.sha256,
            size_bytes=row.size_bytes,
            extension=row.extension,
            media_type=row.media_type,
            kind=row.kind,
            mismatch=row.mismatch,
            known=row.known,
            traits=row.traits,
            status=row.status,
            reason_code=row.reason_code,
            created_at=row.created_at,
        )


class OriginOut(BaseModel):
    source_id: uuid.UUID
    source_kind: str
    name: str
    parent_document_id: uuid.UUID | None
    member_path: str | None
    created_at: datetime


class DocumentDetail(DocumentOut):
    origins: list[OriginOut]


class IngestedOut(BaseModel):
    name: str
    document: DocumentOut
    deduplicated: bool = Field(description="These bytes were already in the workspace")
    refused: dict[str, str] | None = Field(
        default=None, description="A refused archive: its outcome and reason"
    )
    members: list["IngestedOut"] = Field(default_factory=list)

    @classmethod
    def of(cls, item: IngestedFile) -> "IngestedOut":
        return cls(
            name=item.name,
            document=DocumentOut.of(item.document),
            deduplicated=item.deduplicated,
            refused={"outcome": item.refusal[0], "reason": item.refusal[1]}
            if item.refusal
            else None,
            members=[cls.of(member) for member in item.members],
        )


class IngestOut(BaseModel):
    source_id: uuid.UUID
    files: list[IngestedOut]
    added_to_corpus: int
    counts: dict[str, int]

    @classmethod
    def of(cls, report: IngestReport) -> "IngestOut":
        return cls(
            source_id=report.source_id,
            files=[IngestedOut.of(item) for item in report.files],
            added_to_corpus=report.added_to_corpus,
            counts=report.counts(),
        )


class FolderImport(BaseModel):
    root: str = Field(min_length=1, max_length=64, description="A name from intake.folder_roots")
    path: str = Field(default="", max_length=512, description="A folder below the root")
    corpus_id: uuid.UUID | None = None


class ManifestImport(BaseModel):
    root: str = Field(min_length=1, max_length=64, description="A name from intake.folder_roots")
    path: str = Field(
        default="", max_length=512, description="A folder below the root that holds manifest.yaml"
    )
    corpus_id: uuid.UUID = Field(description="The corpus whose working set the entries join")


class S3Import(BaseModel):
    source: str = Field(min_length=1, max_length=64, description="A name from intake.s3_sources")
    prefix: str = Field(default="", max_length=512, description="A prefix below the source's")
    corpus_id: uuid.UUID | None = None


class UrlImport(BaseModel):
    url: str = Field(min_length=8, max_length=2048, description="An http or https URL")
    corpus_id: uuid.UUID | None = None


class FileTypeOut(BaseModel):
    media_type: str
    short: str = Field(description="The name parse routes use, e.g. pdf")
    family: str
    title: str
    extensions: list[str]


class TraitOut(BaseModel):
    name: str
    kind: str = Field(description="flag (true or false; can route a parser) or count")
    applies_to: list[str] = Field(description="Short type names")
    description: str


class FileTypesOut(BaseModel):
    types: list[FileTypeOut]
    traits: list[TraitOut]


class IntakeSourcesOut(BaseModel):
    folder_roots: list[str] = Field(description="Names of server folders (extract/folder)")
    s3_sources: list[str] = Field(description="Names of buckets (extract/s3)")
    url_import: bool = Field(description="Whether URLs can be fetched (extract/url)")
    unpack_archives: bool
    unpack_email: bool


def _spool(settings: Settings) -> Path:
    spool = settings.intake.spool_dir
    if spool is None:
        return Path(tempfile.gettempdir())
    spool.mkdir(parents=True, exist_ok=True)
    return spool


def _check_corpus(request: Request, principal: Principal, corpus_id: uuid.UUID | None) -> None:
    if corpus_id is None:
        return
    with engine_of(request).connect() as conn:
        if get_corpus(conn, principal.workspace.id, corpus_id) is None:
            raise Problem("not_found", code="corpus_not_found")


def _problem(error: UploadError) -> Problem:
    return Problem(error.kind, code=error.code)


def _manifest(text: str) -> CorpusManifest:
    try:
        return load_corpus_manifest(text)
    except SpecInvalid as error:
        raise Problem(
            "invalid_request",
            "The manifest is invalid.",
            code="invalid_manifest",
            errors=error.paths,
            issues=[ProblemIssue(path=issue.path, code=issue.code) for issue in error.issues],
        ) from None


def _ingest(request: Request, principal: Principal, **kwargs: Any) -> IngestReport:
    return ingest(
        engine=engine_of(request),
        store=store_of(request),
        settings=request.app.state.settings,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        subject=principal.subject,
        request_id=current_log_context().get("request_id"),
        **kwargs,
    )


@router.post(
    "/documents",
    response_model=IngestOut,
    status_code=201,
    summary="Upload documents: a multipart body with one or more `file` parts",
    description="Each file streams into quarantine, with `limits.upload_bytes` per file. ZIP "
    "archives are unpacked within the archive limits; their members become documents. Bytes "
    "already in the workspace are the same document (`deduplicated`).",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            },
                            "manifest": {
                                "type": "string",
                                "format": "binary",
                                "description": "A manifest.yaml naming the files by their paths, "
                                "with each one's metadata (needs corpus_id)",
                            },
                        },
                        "required": ["file"],
                    }
                }
            },
        }
    },
    responses=problem_responses(400, 401, 403, 404, 413, 415, 503),
)
async def upload_documents(
    request: Request,
    corpus_id: Annotated[
        uuid.UUID | None, Query(description="Also add them to this corpus's working set")
    ] = None,
    principal: Principal = Depends(require_role("editor")),
) -> JSONResponse:
    settings: Settings = request.app.state.settings
    engine_of(request)
    store_of(request)
    await anyio.to_thread.run_sync(_check_corpus, request, principal, corpus_id)
    try:
        boundary = boundary_of(request.headers.get("content-type", ""))
    except UploadError as error:
        raise _problem(error) from None
    spool = settings.intake.spool_dir
    if spool is not None:
        spool.mkdir(parents=True, exist_ok=True)
    reader = MultipartFiles(
        boundary,
        spool or Path(tempfile.gettempdir()),
        max_bytes=settings.limits.upload_bytes,
        max_files=settings.limits.upload_files,
    )
    try:
        async for chunk in request.stream():
            reader.feed(chunk)
        received = reader.finish()
        fields = None
        if reader.manifest is not None:
            if corpus_id is None:
                raise UploadError("invalid_request", "corpus_required")
            try:
                text = reader.manifest.decode("utf-8")
            except UnicodeDecodeError:
                raise UploadError("invalid_request", "manifest_unreadable") from None
            fields = manifest_entry_fields(_manifest(text))
            if set(fields) - {item.name for item in received}:
                raise UploadError("invalid_request", "manifest_file_missing")
        report = await anyio.to_thread.run_sync(
            lambda: _ingest(
                request,
                principal,
                received=received,
                source_kind="manifest" if fields is not None else "upload",
                corpus_id=corpus_id,
                entry_fields=fields,
            )
        )
    except UploadError as error:
        raise _problem(error) from None
    finally:
        reader.discard()
    return JSONResponse(IngestOut.of(report).model_dump(mode="json"), status_code=201)


@router.post(
    "/documents/import-folder",
    response_model=IngestOut,
    status_code=201,
    summary="Import every file under a server folder the configuration names (extract/folder)",
    responses=problem_responses(400, 401, 403, 404, 413, 503),
)
def import_folder(
    request: Request, body: FolderImport, principal: Principal = Depends(require_role("editor"))
) -> IngestOut:
    settings: Settings = request.app.state.settings
    root = settings.intake.folder_roots.get(body.root)
    if root is None:
        raise Problem("not_found", code="unknown_root")
    _check_corpus(request, principal, body.corpus_id)
    try:
        received = folder_files(
            root,
            body.path,
            max_files=settings.limits.upload_files,
            max_bytes=settings.limits.upload_bytes,
        )
        report = _ingest(
            request,
            principal,
            received=received,
            source_kind="folder",
            source_details={"root": body.root},
            corpus_id=body.corpus_id,
        )
    except UploadError as error:
        raise _problem(error) from None
    return IngestOut.of(report)


@router.post(
    "/documents/import-manifest",
    response_model=IngestOut,
    status_code=201,
    summary="Import a server folder its manifest.yaml describes (extract/manifest)",
    description="Each file the manifest lists joins the corpus's working set, named by its path, "
    "with its doc_key, version, collection, ACL, effective dates, title and status, as "
    "fixtures/demo-corpus/corpus/manifest.yaml describes the Halcyon demo corpus.",
    responses=problem_responses(400, 401, 403, 404, 413, 503),
)
def import_manifest(
    request: Request, body: ManifestImport, principal: Principal = Depends(require_role("editor"))
) -> IngestOut:
    settings: Settings = request.app.state.settings
    root = settings.intake.folder_roots.get(body.root)
    if root is None:
        raise Problem("not_found", code="unknown_root")
    _check_corpus(request, principal, body.corpus_id)
    try:
        folder = folder_under(root, body.path)
        manifest = _manifest(read_manifest_text(folder))
        fields = manifest_entry_fields(manifest)
        received = manifest_files(
            folder,
            manifest,
            max_files=settings.limits.upload_files,
            max_bytes=settings.limits.upload_bytes,
        )
        report = _ingest(
            request,
            principal,
            received=received,
            source_kind="manifest",
            source_details={"root": body.root},
            corpus_id=body.corpus_id,
            entry_fields=fields,
        )
    except UploadError as error:
        raise _problem(error) from None
    return IngestOut.of(report)


@router.post(
    "/documents/import-s3",
    response_model=IngestOut,
    status_code=201,
    summary="Import every object under a bucket prefix the configuration names (extract/s3)",
    responses=problem_responses(400, 401, 403, 404, 413, 503),
)
def import_s3(
    request: Request, body: S3Import, principal: Principal = Depends(require_role("editor"))
) -> IngestOut:
    settings: Settings = request.app.state.settings
    source = settings.intake.s3_sources.get(body.source)
    if source is None:
        raise Problem("not_found", code="unknown_source")
    _check_corpus(request, principal, body.corpus_id)
    received = []
    try:
        received = s3_files(
            source,
            body.prefix,
            _spool(settings),
            max_files=settings.limits.upload_files,
            max_bytes=settings.limits.upload_bytes,
        )
        report = _ingest(
            request,
            principal,
            received=received,
            source_kind="s3",
            source_details={"source": body.source},
            corpus_id=body.corpus_id,
        )
    except UploadError as error:
        raise _problem(error) from None
    finally:
        for item in received:
            item.path.unlink(missing_ok=True)
    return IngestOut.of(report)


@router.post(
    "/documents/import-url",
    response_model=IngestOut,
    status_code=201,
    summary="Fetch one document from a URL through the SSRF guard (extract/url)",
    description="Public addresses only, checked after DNS resolution and again on every "
    "redirect; `limits.upload_bytes` at most; a content type Stratafi can process. The URL "
    "isn't stored or logged; the document is named from the URL's path.",
    responses=problem_responses(400, 401, 403, 404, 413, 503),
)
def import_url(
    request: Request, body: UrlImport, principal: Principal = Depends(require_role("editor"))
) -> IngestOut:
    settings: Settings = request.app.state.settings
    if not settings.intake.url_import:
        raise Problem("forbidden", code="url_import_disabled")
    _check_corpus(request, principal, body.corpus_id)
    received = None
    try:
        received = url_file(body.url, _spool(settings), settings)
        report = _ingest(
            request,
            principal,
            received=[received],
            source_kind="url",
            corpus_id=body.corpus_id,
        )
    except UploadError as error:
        # The code is one of the guard's own, never the URL, so the detail can name it.
        detail = f"The URL was refused: {error.code.replace('_', ' ')}."
        raise Problem(error.kind, detail, code=error.code) from None
    finally:
        if received is not None:
            received.path.unlink(missing_ok=True)
    return IngestOut.of(report)


@router.get(
    "/intake/sources",
    response_model=IntakeSourcesOut,
    summary="Where documents can come from: configured folders, buckets, and URL import",
    responses=problem_responses(401, 403),
)
def intake_sources(
    request: Request, _: Principal = Depends(require_role("viewer"))
) -> IntakeSourcesOut:
    intake = request.app.state.settings.intake
    return IntakeSourcesOut(
        folder_roots=sorted(intake.folder_roots),
        s3_sources=sorted(intake.s3_sources),
        url_import=intake.url_import,
        unpack_archives=intake.unpack_archives,
        unpack_email=intake.unpack_email,
    )


@router.get(
    "/file-types",
    response_model=FileTypesOut,
    summary="The file-type taxonomy: families, short names for parse routes, and traits",
    responses=problem_responses(401, 403),
)
def file_types(_: Principal = Depends(require_role("viewer"))) -> FileTypesOut:
    return FileTypesOut(
        types=[
            FileTypeOut(
                media_type=item.media_type,
                short=item.short,
                family=item.family,
                title=item.title,
                extensions=list(item.extensions),
            )
            for item in FILE_TYPES
        ],
        traits=[
            TraitOut(
                name=trait.name,
                kind=trait.kind,
                applies_to=list(trait.applies_to),
                description=trait.description,
            )
            for trait in TRAITS
        ],
    )


@router.get(
    "/documents",
    response_model=Page[DocumentOut],
    summary="The workspace's documents, newest first",
    responses=problem_responses(400, 401, 403, 503),
)
def documents(
    request: Request,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("viewer")),
) -> Page[DocumentOut]:
    after = None
    if page.cursor is not None:
        created_at, document_id = decode_cursor(page.cursor, size=2)
        try:
            after = (datetime.fromisoformat(created_at), uuid.UUID(document_id))
        except (TypeError, ValueError, AttributeError):
            raise Problem(
                "invalid_request", "The cursor isn't valid.", code="invalid_cursor"
            ) from None
    with engine_of(request).connect() as conn:
        found = list_documents(conn, principal.workspace.id, limit=page.limit + 1, after=after)
    items = [DocumentOut.of(row) for row in found[: page.limit]]
    cursor = None
    if len(found) > page.limit:
        last = found[page.limit - 1]
        cursor = encode_cursor([last.created_at.isoformat(), str(last.id)])
    return Page[DocumentOut](items=items, next_cursor=cursor)


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetail,
    summary="One document, with every arrival of it (its provenance)",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def document(
    request: Request, document_id: uuid.UUID, principal: Principal = Depends(require_role("viewer"))
) -> DocumentDetail:
    with engine_of(request).connect() as conn:
        row = get_document(conn, principal.workspace.id, document_id)
        if row is None:
            raise Problem("not_found", code="document_not_found")
        origins = origins_of(conn, principal.workspace.id, document_id)
    return DocumentDetail(
        **DocumentOut.of(row).model_dump(),
        origins=[
            OriginOut(
                source_id=origin.source_id,
                source_kind=origin.source_kind,
                name=origin.name,
                parent_document_id=origin.parent_document_id,
                member_path=origin.member_path,
                created_at=origin.created_at,
            )
            for origin in origins
        ],
    )


class PurgeOut(BaseModel):
    document_id: uuid.UUID
    status: Literal["withdrawn"]
    artifacts: int = Field(description="Artifacts derived from it that went, with their parts")
    tables: int = Field(description="Postgres index tables built with it that went")
    working_sets: int = Field(description="Corpora whose working set lost it")
    candidates: int = Field(description="Question candidates drafted from it that went")
    models: int = Field(
        default=0, description="Trained models withdrawn: their training data quoted it"
    )
    objects: int = Field(description="Stored objects deleted")


@router.delete(
    "/documents/{document_id}",
    response_model=PurgeOut,
    summary="Purge a document by lineage: its bytes and everything derived from it",
    description="Deletes the document's bytes, every artifact lineage derives from it (an "
    "index built with it included, and training sets and models built from those), their "
    "stored parts and index tables, and the question candidates drafted from it that weren't "
    "accepted; trained models that lose their artifact are withdrawn. Its row stays as a tombstone "
    "(`withdrawn`, reason `deleted`), since frozen corpus versions and run records name it. "
    "Owners only; it can't be undone. Purging a withdrawn document again clears what's left.",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def purge_document(
    request: Request, document_id: uuid.UUID, principal: Principal = Depends(require_role("owner"))
) -> PurgeOut:
    engine, store = engine_of(request), store_of(request)
    with engine.begin() as conn:
        plan = purge.plan_purge(conn, store, principal.workspace.id, document_id, reason="deleted")
        if plan is None:
            raise Problem("not_found", code="document_not_found")
        record_audit_event(
            conn,
            workspace_id=principal.workspace.id,
            actor_kind=principal.actor_kind,
            actor_subject=principal.subject,
            action="document.purge",
            resource_type="document",
            resource_id=str(document_id),
            request_id=current_log_context().get("request_id"),
            details={"reason": "deleted", **plan.counts()},
        )
    objects = purge.remove(engine, store, plan)
    # A loaded index holds its chunks' text.
    request.app.state.indexes.forget(principal.workspace.id, plan.artifact_keys)
    counts = plan.counts()
    return PurgeOut(
        document_id=document_id,
        status="withdrawn",
        artifacts=counts["artifacts"],
        tables=counts["tables"],
        working_sets=counts["working_sets"],
        candidates=counts["candidates"],
        models=counts["models"],
        objects=objects,
    )
