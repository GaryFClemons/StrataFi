"""Corpora, their working sets, and frozen versions.

A corpus's working set maps entry names to documents and can change at any time. ``freeze``
snapshots it as the next corpus version: the entries with what each document was at that moment,
and a manifest hash (``stratafi_core.corpus``). Freezing an unchanged working set returns the
latest version instead of a copy. A version is sealed in the transaction that writes it, and the
database refuses any later change to it (a trigger in the baseline migration).
"""

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import PurePosixPath
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from stratafi_core.corpus import ManifestEntry, manifest_hash
from stratafi_core.planner import EntryInput
from stratafi_store.tables import (
    corpora,
    corpus_documents,
    corpus_version_documents,
    corpus_versions,
    documents,
)

ENTRY_FIELDS = ("doc_key", "doc_version", "collection", "acl", "effective_from", "effective_to")


@dataclass(frozen=True)
class Corpus:
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str
    created_by: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Entry:
    """A corpus entry: in the working set, or in a frozen version."""

    name: str
    document_id: uuid.UUID
    sha256: str
    size_bytes: int
    media_type: str
    traits: dict[str, Any]
    status: str = "stored"
    reason_code: str | None = None
    doc_key: str | None = None
    doc_version: str | None = None
    collection: str | None = None
    acl: dict[str, Any] | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest_entry(self) -> ManifestEntry:
        return ManifestEntry(
            name=self.name,
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            media_type=self.media_type,
            traits=self.traits,
            doc_key=self.doc_key,
            doc_version=self.doc_version,
            collection=self.collection,
            acl=self.acl,
            effective_from=self.effective_from,
            effective_to=self.effective_to,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class CorpusVersion:
    id: uuid.UUID
    workspace_id: uuid.UUID
    corpus_id: uuid.UUID
    version: int
    manifest_hash: str
    document_count: int
    total_bytes: int
    created_by: str
    frozen_at: datetime


def _corpus(row: Any) -> Corpus:
    return Corpus(**{column.name: getattr(row, column.name) for column in corpora.columns})


def _version(row: Any) -> CorpusVersion:
    return CorpusVersion(
        id=row.id,
        workspace_id=row.workspace_id,
        corpus_id=row.corpus_id,
        version=row.version,
        manifest_hash=row.manifest_hash,
        document_count=row.document_count,
        total_bytes=row.total_bytes,
        created_by=row.created_by,
        frozen_at=row.frozen_at,
    )


# --- corpora ----------------------------------------------------------------------------------


def create_corpus(
    conn: Connection, workspace_id: uuid.UUID, name: str, created_by: str, description: str = ""
) -> Corpus | None:
    """A new corpus, or None if the name is taken in the workspace."""
    row = conn.execute(
        insert(corpora)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            name=name,
            description=description,
            created_by=created_by,
        )
        .on_conflict_do_nothing(index_elements=["workspace_id", "name"])
        .returning(*corpora.columns)
    ).first()
    return _corpus(row) if row is not None else None


def get_corpus(conn: Connection, workspace_id: uuid.UUID, corpus_id: uuid.UUID) -> Corpus | None:
    row = conn.execute(
        sa.select(corpora).where(corpora.c.workspace_id == workspace_id, corpora.c.id == corpus_id)
    ).first()
    return _corpus(row) if row is not None else None


def get_corpus_by_name(conn: Connection, workspace_id: uuid.UUID, name: str) -> Corpus | None:
    row = conn.execute(
        sa.select(corpora).where(corpora.c.workspace_id == workspace_id, corpora.c.name == name)
    ).first()
    return _corpus(row) if row is not None else None


def list_corpora(
    conn: Connection, workspace_id: uuid.UUID, *, limit: int, after_name: str | None = None
) -> list[Corpus]:
    query = sa.select(corpora).where(corpora.c.workspace_id == workspace_id)
    if after_name is not None:
        query = query.where(corpora.c.name > after_name)
    return [_corpus(row) for row in conn.execute(query.order_by(corpora.c.name).limit(limit))]


# --- the working set --------------------------------------------------------------------------


@dataclass(frozen=True)
class NewEntry:
    name: str
    document_id: uuid.UUID
    fields: Mapping[str, Any] = field(default_factory=dict)


def add_entries(
    conn: Connection,
    workspace_id: uuid.UUID,
    corpus_id: uuid.UUID,
    entries: Iterable[NewEntry],
    added_by: str,
) -> int:
    """Add entries, replacing any with the same name. Returns how many were written."""
    rows = [
        {
            "workspace_id": workspace_id,
            "corpus_id": corpus_id,
            "name": entry.name,
            "document_id": entry.document_id,
            "added_by": added_by,
            **{key: entry.fields.get(key) for key in ENTRY_FIELDS},
            "metadata": dict(entry.fields.get("metadata") or {}),
        }
        for entry in entries
    ]
    if not rows:
        return 0
    statement = insert(corpus_documents).values(rows)
    conn.execute(
        statement.on_conflict_do_update(
            index_elements=["corpus_id", "name"],
            set_={
                "document_id": statement.excluded.document_id,
                "added_by": statement.excluded.added_by,
                "added_at": sa.func.now(),
                **{key: statement.excluded[key] for key in (*ENTRY_FIELDS, "metadata")},
            },
        )
    )
    conn.execute(
        sa.update(corpora)
        .where(corpora.c.workspace_id == workspace_id, corpora.c.id == corpus_id)
        .values(updated_at=sa.func.now())
    )
    return len(rows)


def remove_entries(
    conn: Connection, workspace_id: uuid.UUID, corpus_id: uuid.UUID, names: Iterable[str]
) -> int:
    names = list(names)
    if not names:
        return 0
    result = conn.execute(
        sa.delete(corpus_documents).where(
            corpus_documents.c.workspace_id == workspace_id,
            corpus_documents.c.corpus_id == corpus_id,
            corpus_documents.c.name.in_(names),
        )
    )
    return result.rowcount or 0


def _entry_columns(table: sa.Table) -> list[Any]:
    return [table.c[key] for key in (*ENTRY_FIELDS, "metadata")]


def list_entries(
    conn: Connection,
    workspace_id: uuid.UUID,
    corpus_id: uuid.UUID,
    *,
    limit: int | None = None,
    after_name: str | None = None,
) -> list[Entry]:
    """The working set by name, with each document's current record."""
    query = (
        sa.select(
            corpus_documents.c.name,
            corpus_documents.c.document_id,
            documents.c.sha256,
            documents.c.size_bytes,
            documents.c.media_type,
            documents.c.traits,
            documents.c.status,
            documents.c.reason_code,
            *_entry_columns(corpus_documents),
        )
        .join(
            documents,
            sa.and_(
                documents.c.workspace_id == corpus_documents.c.workspace_id,
                documents.c.id == corpus_documents.c.document_id,
            ),
        )
        .where(
            corpus_documents.c.workspace_id == workspace_id,
            corpus_documents.c.corpus_id == corpus_id,
        )
        .order_by(corpus_documents.c.name)
    )
    if after_name is not None:
        query = query.where(corpus_documents.c.name > after_name)
    if limit is not None:
        query = query.limit(limit)
    return [Entry(**row._asdict()) for row in conn.execute(query)]


# --- versions ---------------------------------------------------------------------------------


def latest_version(
    conn: Connection, workspace_id: uuid.UUID, corpus_id: uuid.UUID
) -> CorpusVersion | None:
    row = conn.execute(
        sa.select(corpus_versions)
        .where(
            corpus_versions.c.workspace_id == workspace_id,
            corpus_versions.c.corpus_id == corpus_id,
        )
        .order_by(corpus_versions.c.version.desc())
        .limit(1)
    ).first()
    return _version(row) if row is not None else None


def freeze(
    conn: Connection, workspace_id: uuid.UUID, corpus_id: uuid.UUID, created_by: str
) -> tuple[CorpusVersion, bool]:
    """Freeze the working set as the next version. Returns it and whether it's new; an
    unchanged working set returns the latest version."""
    conn.execute(
        sa.select(corpora.c.id)
        .where(corpora.c.workspace_id == workspace_id, corpora.c.id == corpus_id)
        .with_for_update()
    ).one()
    entries = list_entries(conn, workspace_id, corpus_id)
    digest = manifest_hash([entry.manifest_entry() for entry in entries])
    latest = latest_version(conn, workspace_id, corpus_id)
    if latest is not None and latest.manifest_hash == digest:
        return latest, False
    version_id = uuid.uuid4()
    row = conn.execute(
        sa.insert(corpus_versions)
        .values(
            id=version_id,
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            version=(latest.version + 1) if latest is not None else 1,
            manifest_hash=digest,
            document_count=len(entries),
            total_bytes=sum(entry.size_bytes for entry in entries),
            sealed=False,
            created_by=created_by,
        )
        .returning(*corpus_versions.columns)
    ).one()
    if entries:
        conn.execute(
            sa.insert(corpus_version_documents),
            [
                {
                    "workspace_id": workspace_id,
                    "corpus_version_id": version_id,
                    "name": entry.name,
                    "document_id": entry.document_id,
                    "sha256": entry.sha256,
                    "size_bytes": entry.size_bytes,
                    "media_type": entry.media_type,
                    "traits": entry.traits,
                    **{key: getattr(entry, key) for key in ENTRY_FIELDS},
                    "metadata": entry.metadata,
                }
                for entry in entries
            ],
        )
    conn.execute(
        sa.update(corpus_versions).where(corpus_versions.c.id == version_id).values(sealed=True)
    )
    return _version(row), True


def get_version(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    version_id: uuid.UUID | None = None,
    corpus_id: uuid.UUID | None = None,
    version: int | None = None,
) -> CorpusVersion | None:
    """A version by its ID, or by its corpus and number."""
    query = sa.select(corpus_versions).where(corpus_versions.c.workspace_id == workspace_id)
    if version_id is not None:
        query = query.where(corpus_versions.c.id == version_id)
    else:
        query = query.where(
            corpus_versions.c.corpus_id == corpus_id, corpus_versions.c.version == version
        )
    row = conn.execute(query).first()
    return _version(row) if row is not None else None


def list_versions(
    conn: Connection,
    workspace_id: uuid.UUID,
    corpus_id: uuid.UUID,
    *,
    limit: int,
    before_version: int | None = None,
) -> list[CorpusVersion]:
    query = sa.select(corpus_versions).where(
        corpus_versions.c.workspace_id == workspace_id, corpus_versions.c.corpus_id == corpus_id
    )
    if before_version is not None:
        query = query.where(corpus_versions.c.version < before_version)
    rows = conn.execute(query.order_by(corpus_versions.c.version.desc()).limit(limit))
    return [_version(row) for row in rows]


def version_entries(
    conn: Connection,
    workspace_id: uuid.UUID,
    version_id: uuid.UUID,
    *,
    limit: int | None = None,
    after_name: str | None = None,
) -> list[Entry]:
    """A frozen version's entries by name, as they were when it was frozen."""
    table = corpus_version_documents
    query = (
        sa.select(
            table.c.name,
            table.c.document_id,
            table.c.sha256,
            table.c.size_bytes,
            table.c.media_type,
            table.c.traits,
            documents.c.status,
            documents.c.reason_code,
            *_entry_columns(table),
        )
        .join(
            documents,
            sa.and_(
                documents.c.workspace_id == table.c.workspace_id,
                documents.c.id == table.c.document_id,
            ),
        )
        .where(table.c.workspace_id == workspace_id, table.c.corpus_version_id == version_id)
        .order_by(table.c.name)
    )
    if after_name is not None:
        query = query.where(table.c.name > after_name)
    if limit is not None:
        query = query.limit(limit)
    return [Entry(**row._asdict()) for row in conn.execute(query)]


def planner_entries(entries: Sequence[Entry]) -> list[EntryInput]:
    """A version's entries as the run planner sees them, in name order: a duplicate's bytes
    match an earlier entry's, and a document without a title is labelled by its key, else by
    its file's stem. Runs and experiment plans both read entries this way, so their keys agree.
    The name order is Python's, not the database's: a collation depends on how the database was
    installed, and which of two entries is the duplicate must not."""
    seen: set[str] = set()
    found = []
    for entry in sorted(entries, key=lambda entry: entry.name):
        path = PurePosixPath(entry.name)
        found.append(
            EntryInput(
                name=entry.name,
                document_id=str(entry.document_id),
                sha256=entry.sha256,
                extension=path.suffix.lower(),
                duplicate=entry.sha256 in seen,
                label=entry.doc_key or path.stem,
            )
        )
        seen.add(entry.sha256)
    return found
