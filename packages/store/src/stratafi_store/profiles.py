"""Knowledge profiles: their versions and their review.

Each name in a workspace is a chain of versions. ``save_profile`` appends a profile as the next
version, unless the latest one is the same profile (its hash, which the review doesn't change),
in which case it returns that one. Saves of one name are serialized with a transaction-scoped
advisory lock.

A version starts ``draft``; an owner approves or rejects it (``decide``), each decision a
record, and the latest decision is its status. Its author may decide it too, and the record
marks that a self-review.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from stratafi_core.profiles import Approval, KnowledgeProfile, Review, profile_hash
from stratafi_store.tables import knowledge_profiles, profile_approvals

Source = Literal["trial", "import"]
# The table's target and level columns are unused: nothing reads them, and every version
# records these placeholders.
LEGACY_TARGET = "none"
LEGACY_LEVEL = "C"


class ReviewRefused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class StoredProfile:
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    version: int
    document: dict[str, Any]
    document_hash: str
    target: str
    level: str
    status: str
    source: str
    experiment_id: uuid.UUID | None
    trial_id: uuid.UUID | None
    confirmation_id: uuid.UUID | None
    created_by: str
    created_at: datetime
    updated_at: datetime

    def profile(self) -> KnowledgeProfile:
        return KnowledgeProfile.model_validate(self.document)


@dataclass(frozen=True)
class StoredApproval:
    id: uuid.UUID
    decision: str
    decided_by: str
    decided_at: datetime
    self_review: bool = False

    def approval(self) -> Approval:
        return Approval(
            by=self.decided_by,
            at=self.decided_at,
            decision=self.decision,  # type: ignore[arg-type]
            self_review=self.self_review,
        )


def _stored(row: Any) -> StoredProfile:
    return StoredProfile(
        **{column.name: getattr(row, column.name) for column in knowledge_profiles.columns}
    )


def save_profile(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    profile: KnowledgeProfile,
    source: Source,
    created_by: str,
    experiment_id: uuid.UUID | None = None,
    trial_id: uuid.UUID | None = None,
    confirmation_id: uuid.UUID | None = None,
) -> tuple[StoredProfile, bool]:
    """Append ``profile`` to its name's versions; returns the version and whether it's new."""
    conn.execute(
        sa.text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"stratafi:knowledge_profiles:{workspace_id}:{profile.name}"},
    )
    digest = profile_hash(profile)
    latest = conn.execute(
        sa.select(knowledge_profiles)
        .where(
            knowledge_profiles.c.workspace_id == workspace_id,
            knowledge_profiles.c.name == profile.name,
        )
        .order_by(knowledge_profiles.c.version.desc())
        .limit(1)
    ).first()
    if latest is not None and latest.document_hash == digest:
        return _stored(latest), False
    document = profile.model_dump(mode="json", by_alias=True, exclude={"review"})
    row = conn.execute(
        sa.insert(knowledge_profiles)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            name=profile.name,
            version=(latest.version + 1) if latest is not None else 1,
            document=document,
            document_hash=digest,
            target=LEGACY_TARGET,
            level=LEGACY_LEVEL,
            source=source,
            experiment_id=experiment_id,
            trial_id=trial_id,
            confirmation_id=confirmation_id,
            created_by=created_by,
        )
        .returning(*knowledge_profiles.columns)
    ).one()
    return _stored(row), True


def get_profile(
    conn: Connection, workspace_id: uuid.UUID, profile_id: uuid.UUID
) -> StoredProfile | None:
    row = conn.execute(
        sa.select(knowledge_profiles).where(
            knowledge_profiles.c.workspace_id == workspace_id,
            knowledge_profiles.c.id == profile_id,
        )
    ).first()
    return _stored(row) if row is not None else None


def list_profiles(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    after: tuple[datetime, uuid.UUID] | None = None,
    name: str | None = None,
) -> list[StoredProfile]:
    """Every version, newest first; or one name's."""
    query = sa.select(knowledge_profiles).where(knowledge_profiles.c.workspace_id == workspace_id)
    if name is not None:
        query = query.where(knowledge_profiles.c.name == name)
    if after is not None:
        query = query.where(
            sa.tuple_(knowledge_profiles.c.created_at, knowledge_profiles.c.id) < after
        )
    rows = conn.execute(
        query.order_by(
            knowledge_profiles.c.created_at.desc(), knowledge_profiles.c.id.desc()
        ).limit(limit)
    )
    return [_stored(row) for row in rows]


def previous_version(conn: Connection, profile: StoredProfile) -> StoredProfile | None:
    row = conn.execute(
        sa.select(knowledge_profiles)
        .where(
            knowledge_profiles.c.workspace_id == profile.workspace_id,
            knowledge_profiles.c.name == profile.name,
            knowledge_profiles.c.version < profile.version,
        )
        .order_by(knowledge_profiles.c.version.desc())
        .limit(1)
    ).first()
    return _stored(row) if row is not None else None


def approvals_of(conn: Connection, profile_id: uuid.UUID) -> list[StoredApproval]:
    rows = conn.execute(
        sa.select(profile_approvals)
        .where(profile_approvals.c.profile_id == profile_id)
        .order_by(profile_approvals.c.decided_at, profile_approvals.c.id)
    )
    return [
        StoredApproval(row.id, row.decision, row.decided_by, row.decided_at, row.self_review)
        for row in rows
    ]


def review_of(profile: StoredProfile, approvals: list[StoredApproval]) -> Review:
    return Review(
        status=profile.status,  # type: ignore[arg-type]
        approvals=[approval.approval() for approval in approvals],
    )


def decide(
    conn: Connection,
    profile: StoredProfile,
    *,
    decision: Literal["approved", "rejected"],
    decided_by: str,
) -> StoredProfile:
    """Record a review decision. Its author may decide it too: the record says it's a
    self-review."""
    conn.execute(
        sa.insert(profile_approvals).values(
            id=uuid.uuid4(),
            workspace_id=profile.workspace_id,
            profile_id=profile.id,
            decision=decision,
            decided_by=decided_by,
            self_review=decided_by == profile.created_by,
        )
    )
    row = conn.execute(
        sa.update(knowledge_profiles)
        .where(knowledge_profiles.c.id == profile.id)
        .values(status=decision, updated_at=sa.func.now())
        .returning(*knowledge_profiles.columns)
    ).one()
    return _stored(row)
