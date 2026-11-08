"""Retention: what expires, and the sweep that purges it.

``retention`` in the settings names how many days each kind of record lives; unset keeps it:

- ``document_days``: a document imported that many days ago is purged by lineage
  (``purge.plan_purge``, reason ``retention``): its bytes and everything derived from it, leaving
  a tombstone
- ``artifact_days``: a derived artifact made that many days ago goes, with every artifact derived
  from it, so lineage stays whole; the documents' own bytes stay, and a run that needs it again
  makes it again. A cache hit doesn't renew an artifact
- ``audit_days``: audit events older than that are deleted

The worker's heartbeat calls ``RetentionSweeper.tick``; one worker sweeps at a time (a session
advisory lock), every ``sweep_minutes``, at most ``batch`` documents and artifact roots per
sweep. Each purge is audited as ``document.purge`` or ``artifact.expire`` by
``system:retention``. Nothing here logs a name or text.
"""

import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from stratafi_core.settings import RetentionSettings
from stratafi_store import purge
from stratafi_store.artifacts import ArtifactStore
from stratafi_store.audit import record_audit_event
from stratafi_store.scope import system_scope, workspace_scope
from stratafi_store.tables import artifacts, audit_events, documents

logger = logging.getLogger("stratafi.store.retention")

ACTOR = "system:retention"
# pg_try_advisory_lock key for the sweep: "STRR".
SWEEP_LOCK = 0x5354_5252


@dataclass
class RetentionReport:
    documents: int = 0
    artifacts: int = 0
    audit_events: int = 0
    objects: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _cutoff(now: datetime, days: int | None) -> datetime | None:
    return now - timedelta(days=days) if days else None


def _purge_document(
    engine: Engine, store: ArtifactStore, workspace_id: uuid.UUID, document_id: uuid.UUID
) -> int:
    with workspace_scope(workspace_id):
        with engine.begin() as conn:
            plan = purge.plan_purge(conn, store, workspace_id, document_id, reason="retention")
            if plan is None:
                return 0
            record_audit_event(
                conn,
                workspace_id=workspace_id,
                actor_kind="system",
                actor_subject=ACTOR,
                action="document.purge",
                resource_type="document",
                resource_id=str(document_id),
                details={"reason": "retention", **plan.counts()},
            )
        return purge.remove(engine, store, plan)


def _expire_artifacts(
    engine: Engine, store: ArtifactStore, workspace_id: uuid.UUID, roots: list[str]
) -> tuple[int, int]:
    with workspace_scope(workspace_id):
        with engine.begin() as conn:
            plan = purge.plan_artifacts_purge(conn, store, workspace_id, roots)
            if not plan.artifact_keys:
                return 0, 0
            record_audit_event(
                conn,
                workspace_id=workspace_id,
                actor_kind="system",
                actor_subject=ACTOR,
                action="artifact.expire",
                resource_type="workspace",
                resource_id=str(workspace_id),
                details={"reason": "retention", **plan.counts()},
            )
        return len(plan.artifact_keys), purge.remove(engine, store, plan)


def sweep(
    engine: Engine,
    store: ArtifactStore,
    settings: RetentionSettings,
    *,
    now: datetime | None = None,
) -> RetentionReport:
    """Purge what has expired, up to ``settings.batch`` documents and artifact roots."""
    now = now or datetime.now(UTC)
    report = RetentionReport()
    if cutoff := _cutoff(now, settings.document_days):
        with system_scope(), engine.connect() as conn:
            expired = conn.execute(
                sa.select(documents.c.workspace_id, documents.c.id)
                .where(documents.c.status != "withdrawn", documents.c.created_at < cutoff)
                .order_by(documents.c.created_at)
                .limit(settings.batch)
            ).all()
        for row in expired:
            report.objects += _purge_document(engine, store, row.workspace_id, row.id)
            report.documents += 1
    if cutoff := _cutoff(now, settings.artifact_days):
        with system_scope(), engine.connect() as conn:
            roots = conn.execute(
                sa.select(artifacts.c.workspace_id, artifacts.c.cache_key)
                .where(
                    artifacts.c.created_at < cutoff,
                    # A document's bytes are its root artifact: document_days governs those.
                    ~sa.exists().where(
                        documents.c.workspace_id == artifacts.c.workspace_id,
                        documents.c.sha256 == artifacts.c.cache_key,
                    ),
                )
                .order_by(artifacts.c.created_at)
                .limit(settings.batch)
            ).all()
        by_workspace: dict[uuid.UUID, list[str]] = defaultdict(list)
        for row in roots:
            by_workspace[row.workspace_id].append(row.cache_key)
        for workspace_id, keys in by_workspace.items():
            expired_artifacts, objects = _expire_artifacts(engine, store, workspace_id, keys)
            report.artifacts += expired_artifacts
            report.objects += objects
    if cutoff := _cutoff(now, settings.audit_days):
        with system_scope(), engine.begin() as conn:
            report.audit_events = (
                conn.execute(
                    sa.delete(audit_events).where(audit_events.c.occurred_at < cutoff)
                ).rowcount
                or 0
            )
    if any(asdict(report).values()):
        logger.info("retention sweep", extra=report.as_dict())
    return report


class RetentionSweeper:
    """Runs ``sweep`` from the worker's heartbeat, every ``sweep_minutes``, on one worker."""

    def __init__(
        self,
        engine: Engine,
        store: ArtifactStore,
        settings: RetentionSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.engine = engine
        self.store = store
        self.settings = settings
        self.clock = clock
        self._next = 0.0

    @property
    def enabled(self) -> bool:
        s = self.settings
        return any((s.document_days, s.artifact_days, s.audit_days))

    def tick(self) -> RetentionReport | None:
        if not self.enabled or self.clock() < self._next:
            return None
        self._next = self.clock() + self.settings.sweep_minutes * 60
        with system_scope(), self.engine.connect() as conn:
            if not conn.execute(
                sa.text("SELECT pg_try_advisory_lock(:key)"), {"key": SWEEP_LOCK}
            ).scalar_one():
                return None
            conn.commit()
            try:
                return sweep(self.engine, self.store, self.settings)
            finally:
                conn.execute(sa.text("SELECT pg_advisory_unlock(:key)"), {"key": SWEEP_LOCK})
                conn.commit()
