"""Artifacts: their records, their content, chunks a page at a time (for the inspectors), and a
cleaned document's edits as a diff.

An artifact is addressed by its cache key within the workspace. Content comes back as stored:
JSON for raw documents, scan reports, parsed and cleaned documents; NDJSON for chunks; binary
vectors and indexes as downloads. Document text is in some of it: it goes to the caller, never
into logs.

The diff rebuilds each cleaning rule's before and after text from the parsed document and the
edit log (``stratafi_components.clean.diff_hunks``): every hunk is an edit with context. A
redacted value is masked in every string it returns, context included, so the diff never shows
what redaction removed.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of, store_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, problem_responses
from stratafi_components.clean import diff_hunks
from stratafi_core.artifacts import Chunk, ParsedDocument
from stratafi_store.artifacts import ArtifactNotFound, artifact_key
from stratafi_store.lineage import (
    ArtifactRecord,
    get_artifact,
    inputs_of,
    redacted_downstream,
)

router = APIRouter(tags=["artifacts"])

CACHE_KEY_PATTERN = r"^[0-9a-f]{64}$"
TEXT_MEDIA_TYPES = ("application/json", "application/x-ndjson")
CHUNK_KINDS = ("chunks", "enriched")
STREAM_BYTES = 1024 * 1024


class ArtifactInputOut(BaseModel):
    key: str
    role: str
    ordinal: int


class ArtifactOut(BaseModel):
    cache_key: str
    kind: str
    size_bytes: int
    content_sha256: str
    media_type: str
    summary: dict[str, Any]
    created_at: datetime | None
    inputs: list[ArtifactInputOut] = Field(description="What it was made from, in order")


class ChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    ordinal: int
    text: str
    index_text: str
    char_start: int = Field(description="Offset into the cleaned document's text")
    char_end: int
    page_start: int | None
    page_end: int | None
    heading_path: list[str]
    tokens: dict[str, int]
    element_ids: list[str]
    context: str | None = Field(default=None, description="The enricher's context line")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="The chunker's notes: section, table and header_repeated, a parent section",
    )

    @classmethod
    def of(cls, chunk: Chunk) -> "ChunkOut":
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            ordinal=chunk.ordinal,
            text=chunk.text,
            index_text=chunk.index_text,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            heading_path=list(chunk.heading_path),
            tokens=dict(chunk.tokens),
            element_ids=list(chunk.element_ids),
            context=chunk.context,
            metadata=dict(chunk.metadata),
        )


class DiffHunkOut(BaseModel):
    rule: str
    element_id: str
    start: int = Field(description="Offset into the rule's input text")
    end: int
    kind: str = Field(description="The edit's replacement kind")
    before: str = Field(description="The edited span with context; redacted values masked")
    after: str
    redaction: bool


class RuleCountOut(BaseModel):
    rule: str
    edits: int


class DiffOut(BaseModel):
    cache_key: str
    parsed_key: str = Field(description="The parsed document the edits apply to")
    rules: list[RuleCountOut] = Field(description="Edits by rule, in the order they ran")
    hunks: list[DiffHunkOut]
    total_edits: int
    truncated: bool = Field(description="More edits than hunks shown")
    before_chars: int
    after_chars: int
    language: str | None
    warnings: list[str]


def artifact_or_404(conn: Connection, principal: Principal, cache_key: str) -> ArtifactRecord:
    if len(cache_key) != 64 or any(c not in "0123456789abcdef" for c in cache_key):
        raise Problem("not_found", code="artifact_not_found")
    found = get_artifact(conn, principal.workspace.id, cache_key)
    if found is None:
        raise Problem("not_found", code="artifact_not_found")
    return found


def read_chunks(data: bytes) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in data.splitlines() if line.strip()]


@router.get(
    "/artifacts/{cache_key}",
    response_model=ArtifactOut,
    summary="An artifact's record: kind, size, summary and inputs",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_artifact_record(
    request: Request, cache_key: str, principal: Principal = Depends(require_role("viewer"))
) -> ArtifactOut:
    with engine_of(request).connect() as conn:
        found = artifact_or_404(conn, principal, cache_key)
        inputs = inputs_of(conn, principal.workspace.id, cache_key)
    return ArtifactOut(
        cache_key=found.cache_key,
        kind=found.kind,
        size_bytes=found.size_bytes,
        content_sha256=found.content_sha256,
        media_type=found.media_type,
        summary=dict(found.summary),
        created_at=found.created_at,
        inputs=[ArtifactInputOut(key=i.key, role=i.role, ordinal=i.ordinal) for i in inputs],
    )


@router.get(
    "/artifacts/{cache_key}/content",
    summary="An artifact's content, as stored",
    description="JSON or NDJSON inline; vectors, indexes and documents as a download.",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"application/octet-stream": {"schema": {"type": "string"}}}},
        **problem_responses(400, 401, 403, 404, 503),
    },
)
def get_artifact_content(
    request: Request, cache_key: str, principal: Principal = Depends(require_role("viewer"))
) -> StreamingResponse:
    with engine_of(request).connect() as conn:
        found = artifact_or_404(conn, principal, cache_key)
        # A parsed document whose Clean redacted personal data holds the values it hid: only
        # owners read it. The cleaned artifact, the diff and the chunks mask them.
        pre_redaction = found.kind == "parsed" and redacted_downstream(
            conn, principal.workspace.id, cache_key
        )
    if pre_redaction and not principal.has_role("owner"):
        raise Problem(
            "forbidden",
            "This parsed document holds personal data its Clean step redacted: owners only.",
            code="pre_redaction_owner_only",
        )
    store = store_of(request)
    # A document's record points at its quarantined bytes; every other artifact is stored
    # under its cache key.
    if found.kind == "document":
        raise Problem("not_found", code="artifact_content_not_served")
    try:
        stream = store.open(artifact_key(principal.workspace.id, cache_key))
    except ArtifactNotFound:
        raise Problem("not_found", code="artifact_content_missing") from None

    def body():
        with stream:
            while chunk := stream.read(STREAM_BYTES):
                yield chunk

    headers = {"Cache-Control": "private, max-age=31536000, immutable"}
    if found.media_type not in TEXT_MEDIA_TYPES:
        headers["Content-Disposition"] = f'attachment; filename="{cache_key}"'
    return StreamingResponse(body(), media_type=found.media_type, headers=headers)


def _offset(page: PageParams) -> int:
    if page.cursor is None:
        return 0
    (offset,) = decode_cursor(page.cursor, size=1)
    if not isinstance(offset, int) or offset < 0:
        raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
    return offset


@router.get(
    "/artifacts/{cache_key}/chunks",
    response_model=Page[ChunkOut],
    summary="A chunks artifact's chunks, in order, a page at a time",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def list_chunks(
    request: Request,
    cache_key: str,
    page: PageParams = Depends(page_params),
    principal: Principal = Depends(require_role("viewer")),
) -> Page[ChunkOut]:
    offset = _offset(page)
    with engine_of(request).connect() as conn:
        found = artifact_or_404(conn, principal, cache_key)
    if found.kind not in CHUNK_KINDS:
        raise Problem("conflict", "This artifact isn't chunks.", code="not_chunks")
    try:
        data = store_of(request).get(artifact_key(principal.workspace.id, cache_key))
    except ArtifactNotFound:
        raise Problem("not_found", code="artifact_content_missing") from None
    chunks = read_chunks(data)[offset : offset + page.limit + 1]
    items = [ChunkOut.of(chunk) for chunk in chunks[: page.limit]]
    cursor = encode_cursor([offset + page.limit]) if len(chunks) > page.limit else None
    return Page[ChunkOut](items=items, next_cursor=cursor)


@router.get(
    "/artifacts/{cache_key}/diff",
    response_model=DiffOut,
    summary="A cleaned document's edits as before/after hunks, rule by rule",
    description="Rebuilt from the parsed document and the edit log. Redacted values are "
    "masked in every string, the context included.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def get_diff(
    request: Request,
    cache_key: str,
    limit: int = 500,
    principal: Principal = Depends(require_role("viewer")),
) -> DiffOut:
    limit = max(1, min(limit, 2000))
    workspace_id = principal.workspace.id
    with engine_of(request).connect() as conn:
        found = artifact_or_404(conn, principal, cache_key)
        inputs = inputs_of(conn, workspace_id, cache_key)
    if found.kind != "cleaned":
        raise Problem("conflict", "This artifact isn't a cleaned document.", code="not_cleaned")
    parsed_key = min(inputs, key=lambda item: item.ordinal).key if inputs else None
    if parsed_key is None:
        raise Problem("conflict", "The cleaned document has no input.", code="input_missing")
    store = store_of(request)
    try:
        cleaned = ParsedDocument.model_validate_json(
            store.get(artifact_key(workspace_id, cache_key))
        )
        parsed = ParsedDocument.model_validate_json(
            store.get(artifact_key(workspace_id, parsed_key))
        )
    except ArtifactNotFound:
        raise Problem("not_found", code="artifact_content_missing") from None
    edits = list(cleaned.edits or [])
    hunks, total = diff_hunks(parsed, edits, limit=limit)
    counts: dict[str, int] = {}
    for edit in edits:
        counts[edit.rule] = counts.get(edit.rule, 0) + 1
    return DiffOut(
        cache_key=cache_key,
        parsed_key=parsed_key,
        rules=[RuleCountOut(rule=rule, edits=count) for rule, count in counts.items()],
        hunks=[DiffHunkOut(**hunk.__dict__) for hunk in hunks],
        total_edits=total,
        truncated=total > len(hunks),
        before_chars=len(parsed.text),
        after_chars=len(cleaned.text),
        language=cleaned.language,
        warnings=list(cleaned.warnings),
    )
