"""Artifact records and lineage in Postgres (the ``artifacts`` and ``artifact_inputs`` tables).

``record_artifact`` writes an artifact and the inputs it was made from; artifacts are immutable,
so recording the same cache key again changes nothing. ``descendants`` and ``ancestors`` walk the
lineage with recursive queries. They return the same sets as ``stratafi_core.lineage``'s
in-memory graph, which the integration tests check.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from stratafi_core.logs import check_metadata
from stratafi_store.tables import artifact_inputs, artifacts, stage_tasks


@dataclass(frozen=True)
class ArtifactInput:
    key: str
    role: str = "input"
    ordinal: int = 0


@dataclass(frozen=True)
class ArtifactRecord:
    workspace_id: uuid.UUID
    cache_key: str
    kind: str
    storage_uri: str
    size_bytes: int
    content_sha256: str
    media_type: str = "application/octet-stream"
    summary: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


def _row_to_record(row: Any) -> ArtifactRecord:
    return ArtifactRecord(
        workspace_id=row.workspace_id,
        cache_key=row.cache_key,
        kind=row.kind,
        storage_uri=row.storage_uri,
        size_bytes=row.size_bytes,
        content_sha256=row.content_sha256,
        media_type=row.media_type,
        summary=row.summary,
        created_at=row.created_at,
    )


def record_artifact(
    conn: Connection,
    record: ArtifactRecord,
    inputs: Sequence[str | ArtifactInput] = (),
) -> bool:
    """Record ``record`` and its inputs, which must be recorded already. Returns False if the
    artifact was recorded before; the first record wins, since artifacts are immutable."""
    check_metadata(record.summary, where="artifact summary")
    values = {
        "workspace_id": record.workspace_id,
        "cache_key": record.cache_key,
        "kind": record.kind,
        "storage_uri": record.storage_uri,
        "size_bytes": record.size_bytes,
        "content_sha256": record.content_sha256,
        "media_type": record.media_type,
        "summary": dict(record.summary),
    }
    inserted = conn.execute(
        insert(artifacts)
        .values(**values)
        .on_conflict_do_nothing(index_elements=["workspace_id", "cache_key"])
        .returning(artifacts.c.cache_key)
    ).first()
    if inserted is None:
        return False
    edges = [
        item if isinstance(item, ArtifactInput) else ArtifactInput(item, ordinal=index)
        for index, item in enumerate(inputs)
    ]
    if edges:
        conn.execute(
            insert(artifact_inputs),
            [
                {
                    "workspace_id": record.workspace_id,
                    "artifact_key": record.cache_key,
                    "input_key": edge.key,
                    "role": edge.role,
                    "ordinal": edge.ordinal,
                }
                for edge in edges
            ],
        )
    return True


def get_artifact(
    conn: Connection, workspace_id: uuid.UUID, cache_key: str
) -> ArtifactRecord | None:
    row = conn.execute(
        sa.select(artifacts).where(
            artifacts.c.workspace_id == workspace_id, artifacts.c.cache_key == cache_key
        )
    ).first()
    return _row_to_record(row) if row is not None else None


def inputs_of(conn: Connection, workspace_id: uuid.UUID, cache_key: str) -> list[ArtifactInput]:
    rows = conn.execute(
        sa.select(artifact_inputs.c.input_key, artifact_inputs.c.role, artifact_inputs.c.ordinal)
        .where(
            artifact_inputs.c.workspace_id == workspace_id,
            artifact_inputs.c.artifact_key == cache_key,
        )
        .order_by(artifact_inputs.c.ordinal, artifact_inputs.c.input_key)
    )
    return [ArtifactInput(row.input_key, row.role, row.ordinal) for row in rows]


def _walk(conn: Connection, workspace_id: uuid.UUID, start: str, *, down: bool) -> set[str]:
    # UNION (not UNION ALL) drops a key already found, so each key is visited once, diamonds
    # don't multiply paths, and even a malformed cycle ends.
    near, far = ("input_key", "artifact_key") if down else ("artifact_key", "input_key")
    query = sa.text(
        f"""
        WITH RECURSIVE walk(key) AS (
            SELECT {far} FROM stratafi.artifact_inputs
             WHERE workspace_id = :workspace_id AND {near} = :start
            UNION
            SELECT edge.{far} FROM stratafi.artifact_inputs AS edge
              JOIN walk ON edge.{near} = walk.key
             WHERE edge.workspace_id = :workspace_id
        )
        SELECT key FROM walk WHERE key <> :start
        """  # noqa: S608 - column names are fixed above, values are bound
    )
    rows = conn.execute(query, {"workspace_id": workspace_id, "start": start})
    return {row.key for row in rows}


def descendants(conn: Connection, workspace_id: uuid.UUID, cache_key: str) -> set[str]:
    """Every artifact derived from ``cache_key``, directly or not."""
    return _walk(conn, workspace_id, cache_key, down=True)


def ancestors(conn: Connection, workspace_id: uuid.UUID, cache_key: str) -> set[str]:
    """Every artifact ``cache_key`` was derived from, directly or not."""
    return _walk(conn, workspace_id, cache_key, down=False)


def lineage_edges(conn: Connection, workspace_id: uuid.UUID) -> list[tuple[str, str]]:
    """Every ``(artifact, input)`` pair in a workspace, for ``LineageGraph.from_edges``."""
    rows = conn.execute(
        sa.select(artifact_inputs.c.artifact_key, artifact_inputs.c.input_key).where(
            artifact_inputs.c.workspace_id == workspace_id
        )
    )
    return [(row.artifact_key, row.input_key) for row in rows]


def unreferenced_artifacts(
    conn: Connection, workspace_id: uuid.UUID, *, created_before: datetime
) -> list[ArtifactRecord]:
    """A stub for garbage collection: artifacts that no other artifact uses as an input and no
    run used, recorded before ``created_before``.

    A run's stage tasks name every artifact its manifest lists, so a run keeps its artifacts.
    Nothing deletes by it yet: purging (``stratafi_store.purge``) follows a document's lineage.
    """
    used = sa.exists().where(
        artifact_inputs.c.workspace_id == artifacts.c.workspace_id,
        artifact_inputs.c.input_key == artifacts.c.cache_key,
    )
    in_a_run = sa.exists().where(
        stage_tasks.c.workspace_id == artifacts.c.workspace_id,
        stage_tasks.c.cache_key == artifacts.c.cache_key,
    )
    rows = conn.execute(
        sa.select(artifacts)
        .where(
            artifacts.c.workspace_id == workspace_id,
            artifacts.c.created_at < created_before,
            ~used,
            ~in_a_run,
        )
        .order_by(artifacts.c.created_at, artifacts.c.cache_key)
    )
    return [_row_to_record(row) for row in rows]


def redacted_downstream(conn: Connection, workspace_id: uuid.UUID, cache_key: str) -> bool:
    """Whether a cleaned artifact made directly from ``cache_key`` (a parsed document) redacted
    something: then ``cache_key`` holds what the redaction hid."""
    return (
        conn.execute(
            sa.select(sa.literal(1))
            .select_from(artifact_inputs)
            .join(
                artifacts,
                sa.and_(
                    artifacts.c.workspace_id == artifact_inputs.c.workspace_id,
                    artifacts.c.cache_key == artifact_inputs.c.artifact_key,
                ),
            )
            .where(
                artifact_inputs.c.workspace_id == workspace_id,
                artifact_inputs.c.input_key == cache_key,
                artifacts.c.kind == "cleaned",
                sa.func.coalesce(artifacts.c.summary["redactions"].as_integer(), 0) > 0,
            )
            .limit(1)
        ).first()
        is not None
    )
