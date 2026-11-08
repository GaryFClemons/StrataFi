"""Documents, their sources and their provenance.

A document is a file's bytes, unique per workspace by SHA-256: ``record_document`` returns the
existing document when the same bytes arrive again, so duplicates are found as they arrive.
Each arrival is an origin: the name it came with, its source (an upload, a folder import, an
archive) and, for an archive member, the archive and the member's path.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from stratafi_core.logs import check_metadata
from stratafi_store.tables import document_origins, documents, sources


@dataclass(frozen=True)
class DocumentRow:
    id: uuid.UUID
    workspace_id: uuid.UUID
    sha256: str
    size_bytes: int
    name: str
    extension: str
    media_type: str
    kind: str
    mismatch: bool
    known: bool
    traits: dict[str, Any]
    status: str
    reason_code: str | None
    storage_key: str
    created_by: str
    created_at: datetime


@dataclass(frozen=True)
class Origin:
    document_id: uuid.UUID
    source_id: uuid.UUID
    source_kind: str
    name: str
    parent_document_id: uuid.UUID | None
    member_path: str | None
    created_at: datetime


def _document(row: Any) -> DocumentRow:
    return DocumentRow(**{column.name: getattr(row, column.name) for column in documents.columns})


def create_source(
    conn: Connection,
    workspace_id: uuid.UUID,
    kind: str,
    created_by: str,
    details: Mapping[str, Any] | None = None,
) -> uuid.UUID:
    details = dict(details or {})
    check_metadata(details, where="source details")
    source_id = uuid.uuid4()
    conn.execute(
        sa.insert(sources).values(
            id=source_id,
            workspace_id=workspace_id,
            kind=kind,
            details=details,
            created_by=created_by,
        )
    )
    return source_id


def record_document(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    sha256: str,
    size_bytes: int,
    name: str,
    detection: Mapping[str, Any],
    storage_key: str,
    created_by: str,
    status: str = "stored",
    reason_code: str | None = None,
) -> tuple[DocumentRow, bool]:
    """The document for these bytes: new, or the one already in the workspace. Returns it and
    whether it's new."""
    traits = dict(detection.get("traits", {}))
    check_metadata(traits, where="document traits")
    row = conn.execute(
        insert(documents)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            sha256=sha256,
            size_bytes=size_bytes,
            name=name,
            extension=detection["extension"],
            media_type=detection["media_type"],
            kind=detection["kind"],
            mismatch=detection["mismatch"],
            known=detection["known"],
            traits=traits,
            status=status,
            reason_code=reason_code,
            storage_key=storage_key,
            created_by=created_by,
        )
        .on_conflict_do_nothing(index_elements=["workspace_id", "sha256"])
        .returning(*documents.columns)
    ).first()
    if row is not None:
        return _document(row), True
    existing = conn.execute(
        sa.select(documents).where(
            documents.c.workspace_id == workspace_id, documents.c.sha256 == sha256
        )
    ).one()
    return _document(existing), False


def record_origin(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    document_id: uuid.UUID,
    source_id: uuid.UUID,
    name: str,
    parent_document_id: uuid.UUID | None = None,
    member_path: str | None = None,
) -> None:
    conn.execute(
        sa.insert(document_origins).values(
            workspace_id=workspace_id,
            document_id=document_id,
            source_id=source_id,
            name=name,
            parent_document_id=parent_document_id,
            member_path=member_path,
        )
    )


def get_document(
    conn: Connection, workspace_id: uuid.UUID, document_id: uuid.UUID
) -> DocumentRow | None:
    row = conn.execute(
        sa.select(documents).where(
            documents.c.workspace_id == workspace_id, documents.c.id == document_id
        )
    ).first()
    return _document(row) if row is not None else None


def get_documents(
    conn: Connection, workspace_id: uuid.UUID, document_ids: list[uuid.UUID]
) -> dict[uuid.UUID, DocumentRow]:
    if not document_ids:
        return {}
    rows = conn.execute(
        sa.select(documents).where(
            documents.c.workspace_id == workspace_id, documents.c.id.in_(document_ids)
        )
    )
    return {row.id: _document(row) for row in rows}


def list_documents(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    after: tuple[datetime, uuid.UUID] | None = None,
) -> list[DocumentRow]:
    """Newest first. ``after`` is the (created_at, id) of the previous page's last document."""
    query = sa.select(documents).where(documents.c.workspace_id == workspace_id)
    if after is not None:
        query = query.where(sa.tuple_(documents.c.created_at, documents.c.id) < after)
    rows = conn.execute(
        query.order_by(documents.c.created_at.desc(), documents.c.id.desc()).limit(limit)
    )
    return [_document(row) for row in rows]


def origins_of(conn: Connection, workspace_id: uuid.UUID, document_id: uuid.UUID) -> list[Origin]:
    rows = conn.execute(
        sa.select(document_origins, sources.c.kind.label("source_kind"))
        .join(
            sources,
            sa.and_(
                sources.c.workspace_id == document_origins.c.workspace_id,
                sources.c.id == document_origins.c.source_id,
            ),
        )
        .where(
            document_origins.c.workspace_id == workspace_id,
            document_origins.c.document_id == document_id,
        )
        .order_by(document_origins.c.created_at, document_origins.c.id)
    )
    return [
        Origin(
            document_id=row.document_id,
            source_id=row.source_id,
            source_kind=row.source_kind,
            name=row.name,
            parent_document_id=row.parent_document_id,
            member_path=row.member_path,
            created_at=row.created_at,
        )
        for row in rows
    ]
