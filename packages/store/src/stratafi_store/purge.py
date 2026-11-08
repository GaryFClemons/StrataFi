"""Purging by lineage.

**A document.** When a person deletes it or retention expires it, the document goes: its bytes
in quarantine, every artifact lineage derives from it (a document's SHA-256 is its root
artifact's key, so ``lineage.descendants`` finds them; an index or a ranked list built with it
included, and the training sets and models built from those, whose registry rows are
withdrawn), their stored parts, the Postgres tables of indexes built with it, and the question
candidates drafted from it that no one accepted (their text quotes it). The document's row stays
as a tombstone (``withdrawn``, with the reason), because frozen corpus versions and run records
name it; its working-set entries go. What stays holds no document text: run, task and eval
records keep IDs, keys, counts and scores.

**A workspace.** ``plan_workspace_purge`` deletes every row of the workspace, in every table,
and ``remove`` then deletes every object under its prefix and its indexes' tables.

``plan_*`` changes the database inside the caller's transaction and returns what's left to
remove; ``remove`` deletes the objects and tables after the commit, so a failed transaction
leaves nothing half-done in the database (a crash between the two leaves objects whose rows
are gone, which ``orphans`` finds and the next ``remove`` of the same plan clears). Nothing here
logs a name or text: IDs and counts.
"""

import json
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from stratafi_store.artifacts import (
    ArtifactNotFound,
    ArtifactStore,
    artifact_key,
    artifact_part_key,
    workspace_prefix,
)
from stratafi_store.lineage import descendants
from stratafi_store.scope import system_scope
from stratafi_store.tables import (
    answer_results,
    answer_runs,
    artifacts,
    corpus_documents,
    documents,
    metadata,
    question_candidates,
    workspaces,
)
from stratafi_store.training import withdraw_purged

INDEX_PART = "index"
INDEX_SCHEMA = "stratafi_index"
# pg_advisory_lock key a backup holds while it runs: "STRB" (stratafi_store.backup).
# Deleting objects takes it shared, so no object goes between a backup's dump and its copy.
BACKUP_LOCK = 0x5354_5242


@dataclass
class PurgePlan:
    workspace_id: uuid.UUID
    document_id: uuid.UUID | None
    artifact_keys: list[str] = field(default_factory=list)
    quarantine_key: str | None = None
    tables: list[str] = field(default_factory=list)
    working_sets: int = 0
    candidates: int = 0
    models: int = 0
    # A whole workspace: every object under its prefix goes.
    whole_workspace: bool = False
    rows: int = 0

    def counts(self) -> dict[str, int]:
        return {
            "artifacts": len(self.artifact_keys),
            "tables": len(self.tables),
            "working_sets": self.working_sets,
            "candidates": self.candidates,
            "models": self.models,
            "rows": self.rows,
        }


def _index_table(store: ArtifactStore, workspace_id: uuid.UUID, cache_key: str) -> str | None:
    """The Postgres table of an index artifact with a Postgres backend, from its manifest part."""
    try:
        manifest = json.loads(store.get(artifact_part_key(workspace_id, cache_key, INDEX_PART)))
    except (ArtifactNotFound, ValueError, OSError):
        return None
    table = manifest.get("table") if isinstance(manifest, dict) else None
    if manifest.get("backend") != "postgres" or not isinstance(table, str):
        return None
    if not table.replace("_", "").isalnum() or len(table) > 63:
        return None
    return table


def _index_tables(
    store: ArtifactStore, workspace_id: uuid.UUID, rows: Iterable[sa.Row]
) -> list[str]:
    return [
        table
        for row in rows
        if row.kind == "index" and (table := _index_table(store, workspace_id, row.cache_key))
    ]


def plan_artifacts_purge(
    conn: Connection, store: ArtifactStore, workspace_id: uuid.UUID, roots: Iterable[str]
) -> PurgePlan:
    """Delete the rows of ``roots`` and every artifact derived from them (retention)."""
    plan = PurgePlan(workspace_id, None)
    keys: set[str] = set()
    for root in roots:
        keys |= descendants(conn, workspace_id, root) | {root}
    _drop_artifact_rows(conn, store, plan, keys)
    return plan


def _drop_artifact_rows(
    conn: Connection, store: ArtifactStore, plan: PurgePlan, keys: set[str]
) -> None:
    if not keys:
        return
    found = conn.execute(
        sa.select(artifacts.c.cache_key, artifacts.c.kind).where(
            artifacts.c.workspace_id == plan.workspace_id,
            artifacts.c.cache_key.in_(sorted(keys)),
        )
    ).all()
    plan.artifact_keys = sorted({*plan.artifact_keys, *(row.cache_key for row in found)})
    plan.tables += _index_tables(store, plan.workspace_id, found)
    if found:
        conn.execute(
            sa.delete(artifacts).where(
                artifacts.c.workspace_id == plan.workspace_id,
                artifacts.c.cache_key.in_([row.cache_key for row in found]),
            )
        )
        # A model trained on the purged text goes with it: its registry row stays,
        # withdrawn, so its card still says what it was.
        plan.models += withdraw_purged(conn, plan.workspace_id, {row.cache_key for row in found})


def plan_purge(
    conn: Connection,
    store: ArtifactStore,
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    reason: str,
) -> PurgePlan | None:
    """Tombstone the document and drop its derived artifacts' rows and drafted candidates;
    None if it's not there."""
    document = conn.execute(
        sa.select(documents).where(
            documents.c.workspace_id == workspace_id, documents.c.id == document_id
        )
    ).first()
    if document is None:
        return None
    plan = PurgePlan(workspace_id, document_id, quarantine_key=document.storage_key)
    keys = descendants(conn, workspace_id, document.sha256) | {document.sha256}
    _drop_artifact_rows(conn, store, plan, keys)
    plan.working_sets = (
        conn.execute(
            sa.delete(corpus_documents).where(
                corpus_documents.c.workspace_id == workspace_id,
                corpus_documents.c.document_id == document_id,
            )
        ).rowcount
        or 0
    )
    # Drafted from its text and not (yet) a reviewed eval item: they quote it.
    plan.candidates = (
        conn.execute(
            sa.delete(question_candidates).where(
                question_candidates.c.workspace_id == workspace_id,
                question_candidates.c.document_id == document_id,
                question_candidates.c.status.in_(("proposed", "rejected")),
            )
        ).rowcount
        or 0
    )
    conn.execute(
        sa.update(documents)
        .where(documents.c.workspace_id == workspace_id, documents.c.id == document_id)
        .values(status="withdrawn", reason_code=reason)
    )
    return plan


def plan_workspace_purge(
    conn: Connection, store: ArtifactStore, workspace_id: uuid.UUID
) -> PurgePlan | None:
    """Delete every row the workspace has, in every table; None if it doesn't exist."""
    exists = conn.execute(sa.select(workspaces.c.id).where(workspaces.c.id == workspace_id)).first()
    if exists is None:
        return None
    plan = PurgePlan(workspace_id, None, whole_workspace=True)
    indexes = conn.execute(
        sa.select(artifacts.c.cache_key, artifacts.c.kind).where(
            artifacts.c.workspace_id == workspace_id, artifacts.c.kind == "index"
        )
    ).all()
    plan.tables = _index_tables(store, workspace_id, indexes)
    tables = [table for table in metadata.sorted_tables if "workspace_id" in table.c]
    for table in tables:
        plan.rows += conn.execute(
            sa.select(sa.func.count())
            .select_from(table)
            .where(table.c.workspace_id == workspace_id)
        ).scalar_one()
    plan.rows += (
        conn.execute(
            sa.delete(answer_results).where(
                answer_results.c.answer_run_id.in_(
                    sa.select(answer_runs.c.id).where(answer_runs.c.workspace_id == workspace_id)
                )
            )
        ).rowcount
        or 0
    )
    # The workspace's row first: its cascades reach frozen versions, which a direct delete may
    # not (their triggers allow only a cascade). Then whatever no cascade reached, children
    # first; lineage's deferred keys are checked at the commit.
    conn.execute(sa.delete(workspaces).where(workspaces.c.id == workspace_id))
    for table in reversed(tables):
        conn.execute(sa.delete(table).where(table.c.workspace_id == workspace_id))
    return plan


@contextmanager
def deletions_allowed(engine: Engine) -> Iterator[None]:
    """Wait until no backup runs, then keep one from starting while objects are deleted."""
    with system_scope(), engine.connect() as conn:
        conn.execute(sa.text("SELECT pg_advisory_lock_shared(:key)"), {"key": BACKUP_LOCK})
        conn.commit()
        try:
            yield
        finally:
            conn.execute(sa.text("SELECT pg_advisory_unlock_shared(:key)"), {"key": BACKUP_LOCK})
            conn.commit()


def remove(engine: Engine, store: ArtifactStore, plan: PurgePlan) -> int:
    """Delete the plan's objects and drop its index tables, after the database committed, once
    no backup is running. Returns how many objects went."""
    with deletions_allowed(engine):
        return _remove(engine, store, plan)


def _remove(engine: Engine, store: ArtifactStore, plan: PurgePlan) -> int:
    removed = 0
    if plan.whole_workspace:
        for key in list(store.keys(workspace_prefix(plan.workspace_id))):
            store.delete(key)
            removed += 1
    else:
        for cache_key in plan.artifact_keys:
            for key in list(store.keys(artifact_key(plan.workspace_id, cache_key))):
                store.delete(key)
                removed += 1
        if plan.quarantine_key and store.exists(plan.quarantine_key):
            store.delete(plan.quarantine_key)
            removed += 1
    if plan.tables:
        with engine.begin() as conn:
            for table in plan.tables:
                conn.execute(sa.text(f"DROP TABLE IF EXISTS {INDEX_SCHEMA}.{table}"))
    return removed


def orphans(conn: Connection, store: ArtifactStore, workspace_id: uuid.UUID) -> list[str]:
    """Objects under the workspace's artifacts prefix with no artifact row: what a purge that
    stopped between its commit and ``remove`` left behind."""
    known = set(
        conn.execute(
            sa.select(artifacts.c.cache_key).where(artifacts.c.workspace_id == workspace_id)
        ).scalars()
    )
    prefix = f"{workspace_prefix(workspace_id)}artifacts/"
    left = []
    for key in store.keys(prefix):
        cache_key = key.removeprefix(prefix).split(".", 1)[0]
        if cache_key not in known:
            left.append(key)
    return left
