"""Eval sets, their versions and items.

An eval set is a name; its versions hold the questions. ``save_version`` stores a resolved
``stratafi.eval-set.v1`` (``stratafi_core.evalsets``) as the next version, unless it matches the
latest version exactly, in which case it returns that one. A version starts as a draft, which
can be split again; ``freeze`` seals it, and the database refuses any later change to it, except
for its test split's lock and read counter.

The test split is locked: an evaluation reads it only as a confirmation, which
``record_test_read`` counts and the caller audits. Saves of one name are serialized with a
transaction-scoped advisory lock, as specs are.
"""

import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from stratafi_core.evalsets import SPLITS, EvalItem, EvalSet, dump_eval_set, eval_set_hash
from stratafi_store.tables import eval_items, eval_set_versions, eval_sets

# The set's own fields, stored on each version; the items have their own table.
SET_FIELDS = ("schema", "name", "description", "tenant", "as_of", "defaults", "askers")


class VersionFrozen(Exception):
    """A frozen version can't change."""


@dataclass(frozen=True)
class EvalSetRow:
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str
    created_by: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class EvalSetVersion:
    id: uuid.UUID
    workspace_id: uuid.UUID
    eval_set_id: uuid.UUID
    version: int
    document: dict[str, Any]
    document_hash: str
    item_count: int
    split_counts: dict[str, int]
    split_params: dict[str, Any] | None
    source: dict[str, Any]
    frozen: bool
    frozen_at: datetime | None
    test_locked: bool
    test_reads: int
    last_test_read_at: datetime | None
    created_by: str
    created_at: datetime


@dataclass(frozen=True)
class StoredItem:
    item_id: str
    ordinal: int
    query: str
    asker: dict[str, Any]
    expected: list[dict[str, Any]]
    avoid: list[dict[str, Any]]
    must_not_return: list[dict[str, Any]]
    near_misses: list[dict[str, Any]]
    reference_answer: str | None
    tags: list[str]
    split: str

    def model(self) -> EvalItem:
        return EvalItem.model_validate(
            {
                "id": self.item_id,
                "query": self.query,
                "asker": self.asker,
                "expected": self.expected,
                "avoid": self.avoid,
                "must_not_return": self.must_not_return,
                "near_misses": self.near_misses,
                "reference_answer": self.reference_answer,
                "tags": self.tags,
                "split": self.split,
            }
        )


def _set(row: Any) -> EvalSetRow:
    return EvalSetRow(**{column.name: getattr(row, column.name) for column in eval_sets.columns})


def _version(row: Any) -> EvalSetVersion:
    return EvalSetVersion(
        **{column.name: getattr(row, column.name) for column in eval_set_versions.columns}
    )


def _item(row: Any) -> StoredItem:
    return StoredItem(
        **{
            column.name: getattr(row, column.name)
            for column in eval_items.columns
            if column.name not in ("workspace_id", "eval_set_version_id")
        }
    )


def _item_row(workspace_id: uuid.UUID, version_id: uuid.UUID, ordinal: int, item: EvalItem):
    data = item.model_dump(mode="json", exclude_none=True)
    return {
        "workspace_id": workspace_id,
        "eval_set_version_id": version_id,
        "item_id": item.id,
        "ordinal": ordinal,
        "query": item.query,
        "asker": data.get("asker", {}),
        "expected": data.get("expected", []),
        "avoid": data.get("avoid", []),
        "must_not_return": data.get("must_not_return", []),
        "near_misses": data.get("near_misses", []),
        "reference_answer": item.reference_answer,
        "tags": list(item.tags),
        "split": item.split,
    }


def split_counts(items: Sequence[EvalItem]) -> dict[str, int]:
    counts = Counter(item.split for item in items)
    return {split: counts.get(split, 0) for split in SPLITS}


# --- eval sets --------------------------------------------------------------------------------


def get_or_create_set(
    conn: Connection, workspace_id: uuid.UUID, name: str, created_by: str, description: str = ""
) -> EvalSetRow:
    conn.execute(
        insert(eval_sets)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            name=name,
            description=description,
            created_by=created_by,
        )
        .on_conflict_do_nothing(index_elements=["workspace_id", "name"])
    )
    row = conn.execute(
        sa.select(eval_sets).where(
            eval_sets.c.workspace_id == workspace_id, eval_sets.c.name == name
        )
    ).one()
    return _set(row)


def get_set(conn: Connection, workspace_id: uuid.UUID, eval_set_id: uuid.UUID) -> EvalSetRow | None:
    row = conn.execute(
        sa.select(eval_sets).where(
            eval_sets.c.workspace_id == workspace_id, eval_sets.c.id == eval_set_id
        )
    ).first()
    return _set(row) if row is not None else None


def get_set_by_name(conn: Connection, workspace_id: uuid.UUID, name: str) -> EvalSetRow | None:
    row = conn.execute(
        sa.select(eval_sets).where(
            eval_sets.c.workspace_id == workspace_id, eval_sets.c.name == name
        )
    ).first()
    return _set(row) if row is not None else None


def list_sets(
    conn: Connection, workspace_id: uuid.UUID, *, limit: int, after_name: str | None = None
) -> list[EvalSetRow]:
    query = sa.select(eval_sets).where(eval_sets.c.workspace_id == workspace_id)
    if after_name is not None:
        query = query.where(eval_sets.c.name > after_name)
    return [_set(row) for row in conn.execute(query.order_by(eval_sets.c.name).limit(limit))]


# --- versions ---------------------------------------------------------------------------------


def latest_version(
    conn: Connection, workspace_id: uuid.UUID, eval_set_id: uuid.UUID
) -> EvalSetVersion | None:
    row = conn.execute(
        sa.select(eval_set_versions)
        .where(
            eval_set_versions.c.workspace_id == workspace_id,
            eval_set_versions.c.eval_set_id == eval_set_id,
        )
        .order_by(eval_set_versions.c.version.desc())
        .limit(1)
    ).first()
    return _version(row) if row is not None else None


def latest_versions(
    conn: Connection, workspace_id: uuid.UUID, eval_set_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, EvalSetVersion]:
    if not eval_set_ids:
        return {}
    rows = conn.execute(
        sa.select(eval_set_versions)
        .where(
            eval_set_versions.c.workspace_id == workspace_id,
            eval_set_versions.c.eval_set_id.in_(list(eval_set_ids)),
        )
        .order_by(eval_set_versions.c.eval_set_id, eval_set_versions.c.version.desc())
    )
    latest: dict[uuid.UUID, EvalSetVersion] = {}
    for row in rows:
        latest.setdefault(row.eval_set_id, _version(row))
    return latest


def get_version(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    version_id: uuid.UUID | None = None,
    eval_set_id: uuid.UUID | None = None,
    version: int | None = None,
) -> EvalSetVersion | None:
    """A version by its ID, or by its eval set and number."""
    query = sa.select(eval_set_versions).where(eval_set_versions.c.workspace_id == workspace_id)
    if version_id is not None:
        query = query.where(eval_set_versions.c.id == version_id)
    else:
        query = query.where(
            eval_set_versions.c.eval_set_id == eval_set_id, eval_set_versions.c.version == version
        )
    row = conn.execute(query).first()
    return _version(row) if row is not None else None


def list_versions(
    conn: Connection,
    workspace_id: uuid.UUID,
    eval_set_id: uuid.UUID,
    *,
    limit: int,
    before_version: int | None = None,
) -> list[EvalSetVersion]:
    """Newest first."""
    query = sa.select(eval_set_versions).where(
        eval_set_versions.c.workspace_id == workspace_id,
        eval_set_versions.c.eval_set_id == eval_set_id,
    )
    if before_version is not None:
        query = query.where(eval_set_versions.c.version < before_version)
    rows = conn.execute(query.order_by(eval_set_versions.c.version.desc()).limit(limit))
    return [_version(row) for row in rows]


def _set_document(eval_set: EvalSet) -> dict[str, Any]:
    dumped = dump_eval_set(eval_set)
    return {key: dumped[key] for key in SET_FIELDS if key in dumped}


def save_version(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    eval_set_id: uuid.UUID,
    eval_set: EvalSet,
    created_by: str,
    split_params: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    freeze: bool = False,
) -> tuple[EvalSetVersion, bool]:
    """Store a resolved eval set as the next version. Returns it and whether it's new; a set
    identical to the latest version returns that version (frozen now if ``freeze``)."""
    conn.execute(
        sa.text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"stratafi:eval_set_versions:{workspace_id}:{eval_set_id}"},
    )
    digest = eval_set_hash(eval_set)
    latest = latest_version(conn, workspace_id, eval_set_id)
    if latest is not None and latest.document_hash == digest:
        if freeze and not latest.frozen:
            latest = freeze_version(conn, latest)
        return latest, False
    version_id = uuid.uuid4()
    row = conn.execute(
        sa.insert(eval_set_versions)
        .values(
            id=version_id,
            workspace_id=workspace_id,
            eval_set_id=eval_set_id,
            version=(latest.version + 1) if latest is not None else 1,
            document=_set_document(eval_set),
            document_hash=digest,
            item_count=len(eval_set.items),
            split_counts=split_counts(eval_set.items),
            split_params=split_params,
            source=dict(source or {}),
            created_by=created_by,
        )
        .returning(*eval_set_versions.columns)
    ).one()
    conn.execute(
        sa.insert(eval_items),
        [
            _item_row(workspace_id, version_id, ordinal, item)
            for ordinal, item in enumerate(eval_set.items)
        ],
    )
    conn.execute(
        sa.update(eval_sets)
        .where(eval_sets.c.id == eval_set_id)
        .values(updated_at=sa.func.now(), description=eval_set.description)
    )
    version = _version(row)
    if freeze:
        version = freeze_version(conn, version)
    return version, True


def freeze_version(conn: Connection, version: EvalSetVersion) -> EvalSetVersion:
    """Seal a version; freezing a frozen one changes nothing."""
    if version.frozen:
        return version
    row = conn.execute(
        sa.update(eval_set_versions)
        .where(eval_set_versions.c.id == version.id)
        .values(frozen=True, frozen_at=sa.func.now())
        .returning(*eval_set_versions.columns)
    ).one()
    return _version(row)


def replace_items(
    conn: Connection,
    version: EvalSetVersion,
    eval_set: EvalSet,
    *,
    split_params: dict[str, Any] | None,
) -> EvalSetVersion:
    """A draft's items, split again. Raises ``VersionFrozen`` for a frozen version."""
    if version.frozen:
        raise VersionFrozen()
    conn.execute(sa.delete(eval_items).where(eval_items.c.eval_set_version_id == version.id))
    conn.execute(
        sa.insert(eval_items),
        [
            _item_row(version.workspace_id, version.id, ordinal, item)
            for ordinal, item in enumerate(eval_set.items)
        ],
    )
    row = conn.execute(
        sa.update(eval_set_versions)
        .where(eval_set_versions.c.id == version.id)
        .values(
            document=_set_document(eval_set),
            document_hash=eval_set_hash(eval_set),
            item_count=len(eval_set.items),
            split_counts=split_counts(eval_set.items),
            split_params=split_params,
        )
        .returning(*eval_set_versions.columns)
    ).one()
    return _version(row)


def record_test_read(conn: Connection, version_id: uuid.UUID) -> EvalSetVersion:
    """Count one confirmation read of a version's test split."""
    row = conn.execute(
        sa.update(eval_set_versions)
        .where(eval_set_versions.c.id == version_id)
        .values(test_reads=eval_set_versions.c.test_reads + 1, last_test_read_at=sa.func.now())
        .returning(*eval_set_versions.columns)
    ).one()
    return _version(row)


# --- items ------------------------------------------------------------------------------------


def list_items(
    conn: Connection,
    version_id: uuid.UUID,
    *,
    split: str | None = None,
    tag: str | None = None,
    limit: int | None = None,
    after_ordinal: int | None = None,
) -> list[StoredItem]:
    """A version's items in order, optionally one split's or one tag's."""
    query = sa.select(eval_items).where(eval_items.c.eval_set_version_id == version_id)
    if split is not None:
        query = query.where(eval_items.c.split == split)
    if tag is not None:
        query = query.where(eval_items.c.tags.contains([tag]))
    if after_ordinal is not None:
        query = query.where(eval_items.c.ordinal > after_ordinal)
    query = query.order_by(eval_items.c.ordinal)
    if limit is not None:
        query = query.limit(limit)
    return [_item(row) for row in conn.execute(query)]


def stored_eval_set(conn: Connection, version: EvalSetVersion) -> EvalSet:
    """A stored version as the resolved eval set it was saved from."""
    items = [item.model() for item in list_items(conn, version.id)]
    return EvalSet.model_validate({**version.document, "items": items})
