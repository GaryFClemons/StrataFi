"""Workspace features: what a workspace's owner turned on.

``custom_checks`` lets the workspace's test suites run custom checks in the sandbox. Turning it
on needs the deployment's sandbox to be configured and its security review signed off
(``Settings.suites.custom_checks_blocked``); the row keeps the review's reference. A missing
row is off.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from stratafi_store.tables import WORKSPACE_FEATURES, workspace_features

CUSTOM_CHECKS = "custom_checks"


@dataclass(frozen=True)
class Feature:
    name: str
    enabled: bool
    review: str | None = None
    changed_by: str | None = None
    changed_at: datetime | None = None


def features_of(conn: Connection, workspace_id: uuid.UUID) -> dict[str, Feature]:
    """Every feature, on or off."""
    rows = conn.execute(
        sa.select(workspace_features).where(workspace_features.c.workspace_id == workspace_id)
    ).all()
    found = {
        row.feature: Feature(row.feature, row.enabled, row.review, row.changed_by, row.changed_at)
        for row in rows
    }
    return {name: found.get(name, Feature(name, False)) for name in WORKSPACE_FEATURES}


def enabled(conn: Connection, workspace_id: uuid.UUID, feature: str) -> bool:
    return bool(
        conn.execute(
            sa.select(workspace_features.c.enabled).where(
                workspace_features.c.workspace_id == workspace_id,
                workspace_features.c.feature == feature,
            )
        ).scalar_one_or_none()
    )


def set_feature(
    conn: Connection,
    workspace_id: uuid.UUID,
    feature: str,
    *,
    on: bool,
    changed_by: str,
    review: str | None = None,
) -> Feature:
    if feature not in WORKSPACE_FEATURES:
        raise ValueError("unknown feature")
    statement = insert(workspace_features).values(
        workspace_id=workspace_id,
        feature=feature,
        enabled=on,
        review=review,
        changed_by=changed_by,
    )
    row = conn.execute(
        statement.on_conflict_do_update(
            index_elements=["workspace_id", "feature"],
            set_={
                "enabled": statement.excluded.enabled,
                "review": statement.excluded.review,
                "changed_by": statement.excluded.changed_by,
                "changed_at": sa.func.now(),
            },
        ).returning(workspace_features)
    ).one()
    return Feature(row.feature, row.enabled, row.review, row.changed_by, row.changed_at)
