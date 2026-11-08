"""Versioned pipeline specs and retrieval configs.

Each name in a workspace is a chain of immutable versions. ``save_version`` appends the resolved
document as the next version, unless it matches the latest one exactly (same ``document_hash``),
in which case it returns that version: saving twice changes nothing, and old runs keep the
version they used. Saves of one name are serialized with a transaction-scoped advisory lock, so
two editors saving at once get versions n and n+1, never a conflict.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.engine import Connection

from stratafi_store.tables import pipeline_specs, retrieval_configs, test_suites

# Test suites are versioned the same way, in their own table.
DocumentKind = Literal["pipeline", "retrieval", "suite"]
TABLES: dict[str, sa.Table] = {
    "pipeline": pipeline_specs,
    "retrieval": retrieval_configs,
    "suite": test_suites,
}


@dataclass(frozen=True)
class StoredDocument:
    id: uuid.UUID
    workspace_id: uuid.UUID
    kind: DocumentKind
    name: str
    version: int
    document: dict[str, Any]
    spec_hash: str
    document_hash: str
    created_by: str
    created_at: datetime


def _stored(kind: DocumentKind, row: Any) -> StoredDocument:
    return StoredDocument(
        id=row.id,
        workspace_id=row.workspace_id,
        kind=kind,
        name=row.name,
        version=row.version,
        document=row.document,
        spec_hash=row.spec_hash,
        document_hash=row.document_hash,
        created_by=row.created_by,
        created_at=row.created_at,
    )


def save_version(
    conn: Connection,
    kind: DocumentKind,
    *,
    workspace_id: uuid.UUID,
    document: dict[str, Any],
    spec_hash: str,
    document_hash: str,
    created_by: str,
) -> tuple[StoredDocument, bool]:
    """Append ``document`` (already resolved) to its name's chain. Returns the version and
    whether it's new."""
    table = TABLES[kind]
    name = document["name"]
    conn.execute(
        sa.text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"stratafi:{table.name}:{workspace_id}:{name}"},
    )
    latest = conn.execute(
        sa.select(table)
        .where(table.c.workspace_id == workspace_id, table.c.name == name)
        .order_by(table.c.version.desc())
        .limit(1)
    ).first()
    if latest is not None and latest.document_hash == document_hash:
        return _stored(kind, latest), False
    row = conn.execute(
        sa.insert(table)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            name=name,
            version=(latest.version + 1) if latest is not None else 1,
            document=document,
            spec_hash=spec_hash,
            document_hash=document_hash,
            created_by=created_by,
        )
        .returning(*table.columns)
    ).one()
    return _stored(kind, row), True


def get_version(
    conn: Connection,
    kind: DocumentKind,
    workspace_id: uuid.UUID,
    name: str,
    version: int | None = None,
) -> StoredDocument | None:
    """A version of ``name``; the latest when ``version`` is None."""
    table = TABLES[kind]
    query = sa.select(table).where(table.c.workspace_id == workspace_id, table.c.name == name)
    if version is not None:
        query = query.where(table.c.version == version)
    row = conn.execute(query.order_by(table.c.version.desc()).limit(1)).first()
    return _stored(kind, row) if row is not None else None


def get_by_id(
    conn: Connection, kind: DocumentKind, workspace_id: uuid.UUID, document_id: uuid.UUID
) -> StoredDocument | None:
    table = TABLES[kind]
    row = conn.execute(
        sa.select(table).where(table.c.workspace_id == workspace_id, table.c.id == document_id)
    ).first()
    return _stored(kind, row) if row is not None else None


def list_latest(
    conn: Connection,
    kind: DocumentKind,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    after_name: str | None = None,
) -> list[StoredDocument]:
    """The latest version of each name, by name."""
    table = TABLES[kind]
    query = (
        sa.select(table)
        .where(table.c.workspace_id == workspace_id)
        .ext(distinct_on(table.c.name))
        .order_by(table.c.name, table.c.version.desc())
    )
    if after_name is not None:
        query = query.where(table.c.name > after_name)
    return [_stored(kind, row) for row in conn.execute(query.limit(limit))]


def list_versions(
    conn: Connection,
    kind: DocumentKind,
    workspace_id: uuid.UUID,
    name: str,
    *,
    limit: int,
    before_version: int | None = None,
) -> list[StoredDocument]:
    """A name's versions, newest first."""
    table = TABLES[kind]
    query = sa.select(table).where(table.c.workspace_id == workspace_id, table.c.name == name)
    if before_version is not None:
        query = query.where(table.c.version < before_version)
    rows = conn.execute(query.order_by(table.c.version.desc()).limit(limit))
    return [_stored(kind, row) for row in rows]
