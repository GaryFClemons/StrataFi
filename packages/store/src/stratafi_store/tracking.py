"""Tracking records.

- **Annotations:** the title, tags, note and archive of a pipeline run, an eval run or an
  experiment: one ``annotations`` row per resource, made on its first change. ``current_tags``
  locks the resource's row (``FOR UPDATE``) so a read-modify-write of its tags is serialized,
  and ``annotate`` upserts.
- **Search rows:** the workspace's eval runs with what flattening needs: their resolved
  retrieval config and pipeline stages, their models, metrics and tags, and the names of their
  eval set, configuration, pipeline, corpus and experiment. Newest first, at most ``limit``.
- **Aliases:** one name per version of a trained model or a knowledge profile, per alias.

Nothing here logs a title, a tag or a note.
"""

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection

from stratafi_store.tables import (
    annotations,
    corpora,
    corpus_versions,
    eval_runs,
    eval_set_versions,
    eval_sets,
    experiments,
    pipeline_specs,
    registry_aliases,
    retrieval_configs,
    runs,
    trials,
)

AnnotatedKind = Literal["run", "eval_run", "experiment"]
TABLES: dict[str, sa.Table] = {"run": runs, "eval_run": eval_runs, "experiment": experiments}
COLUMNS: dict[str, str] = {
    "run": "run_id",
    "eval_run": "eval_run_id",
    "experiment": "experiment_id",
}
SEARCH_LIMIT = 5000


class _Unset:
    pass


UNSET: Any = _Unset()


@dataclass(frozen=True)
class Annotation:
    title: str | None
    tags: dict[str, str]
    note: str | None
    archived_at: datetime | None
    updated_at: datetime | None  # None until the first change


EMPTY = Annotation(None, {}, None, None, None)


def _annotation(row: Any) -> Annotation:
    return Annotation(row.title, dict(row.tags or {}), row.note, row.archived_at, row.updated_at)


def _exists(
    conn: Connection,
    kind: AnnotatedKind,
    workspace_id: uuid.UUID,
    resource_id: uuid.UUID,
    *,
    lock: bool = False,
) -> bool:
    table = TABLES[kind]
    query = sa.select(table.c.id).where(
        table.c.workspace_id == workspace_id, table.c.id == resource_id
    )
    return conn.execute(query.with_for_update() if lock else query).first() is not None


def _target(kind: AnnotatedKind) -> sa.Column:
    return annotations.c[COLUMNS[kind]]


def get_annotation(
    conn: Connection, kind: AnnotatedKind, workspace_id: uuid.UUID, resource_id: uuid.UUID
) -> Annotation | None:
    """The resource's annotation (empty if it has none), or None if there's no such resource."""
    if not _exists(conn, kind, workspace_id, resource_id):
        return None
    row = conn.execute(
        sa.select(annotations).where(
            annotations.c.workspace_id == workspace_id, _target(kind) == resource_id
        )
    ).first()
    return _annotation(row) if row is not None else EMPTY


def current_tags(
    conn: Connection, kind: AnnotatedKind, workspace_id: uuid.UUID, resource_id: uuid.UUID
) -> dict[str, str] | None:
    """The tags, with the resource's row locked for a read-modify-write; None if there's no
    such resource."""
    if not _exists(conn, kind, workspace_id, resource_id, lock=True):
        return None
    row = conn.execute(
        sa.select(annotations.c.tags).where(
            annotations.c.workspace_id == workspace_id, _target(kind) == resource_id
        )
    ).first()
    return dict(row.tags or {}) if row is not None else {}


def annotate(
    conn: Connection,
    kind: AnnotatedKind,
    workspace_id: uuid.UUID,
    resource_id: uuid.UUID,
    *,
    by: str,
    title: str | None = UNSET,
    note: str | None = UNSET,
    tags: Mapping[str, str] | None = None,
    archived: bool | None = None,
) -> Annotation | None:
    """Change what's given; ``tags`` replaces the tags (merge them with ``current_tags``
    first). Archiving an archived resource keeps its time. None if there's no such resource."""
    if not _exists(conn, kind, workspace_id, resource_id, lock=True):
        return None
    changes: dict[str, Any] = {}
    if title is not UNSET:
        changes["title"] = title
    if note is not UNSET:
        changes["note"] = note
    if tags is not None:
        changes["tags"] = dict(tags)
    first = {**changes}
    if archived is True:
        first["archived_at"] = sa.func.now()
        changes["archived_at"] = sa.func.coalesce(annotations.c.archived_at, sa.func.now())
    elif archived is False:
        first["archived_at"] = None
        changes["archived_at"] = None
    statement = postgresql.insert(annotations).values(
        id=uuid.uuid4(),
        workspace_id=workspace_id,
        updated_by=by,
        **{COLUMNS[kind]: resource_id},
        **first,
    )
    row = conn.execute(
        statement.on_conflict_do_update(
            index_elements=[_target(kind)],
            set_={**changes, "updated_by": by, "updated_at": sa.func.now()},
        ).returning(*annotations.columns)
    ).one()
    return _annotation(row)


# --- searching eval runs --------------------------------------------------------------------


@dataclass(frozen=True)
class SearchRow:
    """An eval run with what flattening and display need."""

    id: uuid.UUID
    status: str
    split: str
    created_by: str
    created_at: datetime
    finished_at: datetime | None
    test_read: bool
    results_cached: bool
    title: str | None
    tags: dict[str, str]
    archived_at: datetime | None
    context: dict[str, Any]
    metrics: dict[str, Any] | None
    eval_set_version_id: uuid.UUID
    pipeline_run_id: uuid.UUID
    retrieval_config_id: uuid.UUID
    retrieval: dict[str, Any] | None
    retrieval_name: str | None
    retrieval_version: int | None
    stages: dict[str, Any] | None
    pipeline_spec_id: uuid.UUID | None
    pipeline_name: str | None
    pipeline_version: int | None
    corpus_version_id: uuid.UUID | None
    corpus_name: str | None
    corpus_version: int | None
    eval_set_name: str | None
    eval_set_version: int | None
    experiment_id: uuid.UUID | None
    experiment_name: str | None
    trial: int | None


def search_rows(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    include_archived: bool = False,
    limit: int = SEARCH_LIMIT,
    eval_set_version_id: uuid.UUID | None = None,
    ids: Sequence[uuid.UUID] | None = None,
    slices: bool = True,
) -> list[SearchRow]:
    """The workspace's eval runs, newest first (the module docstring says what each holds).
    Without ``slices``, their summaries leave out the slices, which flattening never reads, so a
    search over thousands of runs moves less."""
    er = eval_runs
    trial = (
        sa.select(trials.c.ordinal, trials.c.experiment_id)
        .where(trials.c.eval_run_id == er.c.id)
        .order_by(trials.c.ordinal)
        .limit(1)
        .lateral("trial")
    )
    query = (
        sa.select(
            er.c.id,
            er.c.status,
            er.c.split,
            er.c.created_by,
            er.c.created_at,
            er.c.finished_at,
            er.c.test_read,
            er.c.results_cached,
            annotations.c.title,
            sa.func.coalesce(annotations.c.tags, sa.text("'{}'::jsonb")).label("tags"),
            annotations.c.archived_at,
            er.c.context,
            (
                er.c.metrics
                if slices
                else er.c.metrics.op("-")(sa.literal("slices", type_=sa.Text))
            ).label("metrics"),
            er.c.eval_set_version_id,
            er.c.pipeline_run_id,
            er.c.retrieval_config_id,
            sa.func.coalesce(er.c.context["retrieval"], retrieval_configs.c.document).label(
                "retrieval"
            ),
            retrieval_configs.c.name.label("retrieval_name"),
            retrieval_configs.c.version.label("retrieval_version"),
            pipeline_specs.c.document["stages"].label("stages"),
            runs.c.pipeline_spec_id,
            pipeline_specs.c.name.label("pipeline_name"),
            pipeline_specs.c.version.label("pipeline_version"),
            runs.c.corpus_version_id,
            corpora.c.name.label("corpus_name"),
            corpus_versions.c.version.label("corpus_version"),
            eval_sets.c.name.label("eval_set_name"),
            eval_set_versions.c.version.label("eval_set_version"),
            experiments.c.id.label("experiment_id"),
            experiments.c.name.label("experiment_name"),
            trial.c.ordinal.label("trial"),
        )
        .select_from(er)
        .join(
            runs,
            sa.and_(runs.c.workspace_id == er.c.workspace_id, runs.c.id == er.c.pipeline_run_id),
        )
        .outerjoin(annotations, annotations.c.eval_run_id == er.c.id)
        .outerjoin(pipeline_specs, pipeline_specs.c.id == runs.c.pipeline_spec_id)
        .outerjoin(retrieval_configs, retrieval_configs.c.id == er.c.retrieval_config_id)
        .outerjoin(corpus_versions, corpus_versions.c.id == runs.c.corpus_version_id)
        .outerjoin(corpora, corpora.c.id == corpus_versions.c.corpus_id)
        .outerjoin(eval_set_versions, eval_set_versions.c.id == er.c.eval_set_version_id)
        .outerjoin(eval_sets, eval_sets.c.id == eval_set_versions.c.eval_set_id)
        .outerjoin(trial, sa.true())
        .outerjoin(experiments, experiments.c.id == trial.c.experiment_id)
        .where(er.c.workspace_id == workspace_id)
    )
    if not include_archived:
        query = query.where(annotations.c.archived_at.is_(None))
    if eval_set_version_id is not None:
        query = query.where(er.c.eval_set_version_id == eval_set_version_id)
    if ids is not None:
        query = query.where(er.c.id.in_(list(ids)))
    query = query.order_by(er.c.created_at.desc(), er.c.id.desc()).limit(limit)
    return [
        SearchRow(
            **{**row._asdict(), "tags": dict(row.tags or {}), "context": dict(row.context or {})}
        )
        for row in conn.execute(query)
    ]


# --- registry aliases -----------------------------------------------------------------

AliasKind = Literal["trained_model", "knowledge_profile"]


@dataclass(frozen=True)
class Alias:
    kind: str
    name: str
    alias: str
    target_id: uuid.UUID
    set_by: str
    updated_at: datetime


def _alias(row: Any) -> Alias:
    return Alias(row.kind, row.name, row.alias, row.target_id, row.set_by, row.updated_at)


def set_alias(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    kind: AliasKind,
    name: str,
    alias: str,
    target_id: uuid.UUID,
    set_by: str,
) -> tuple[Alias, uuid.UUID | None]:
    """Point ``name@alias`` at ``target_id``; the alias and the version it moved from, if any."""
    table = registry_aliases
    previous = conn.execute(
        sa.select(table.c.target_id)
        .where(
            table.c.workspace_id == workspace_id,
            table.c.kind == kind,
            table.c.name == name,
            table.c.alias == alias,
        )
        .with_for_update()
    ).scalar_one_or_none()
    statement = postgresql.insert(table).values(
        workspace_id=workspace_id,
        kind=kind,
        name=name,
        alias=alias,
        target_id=target_id,
        set_by=set_by,
    )
    row = conn.execute(
        statement.on_conflict_do_update(
            index_elements=[table.c.workspace_id, table.c.kind, table.c.name, table.c.alias],
            set_={"target_id": target_id, "set_by": set_by, "updated_at": datetime.now(UTC)},
        ).returning(*table.columns)
    ).one()
    return _alias(row), previous if previous != target_id else None


def delete_alias(
    conn: Connection, workspace_id: uuid.UUID, *, kind: AliasKind, name: str, alias: str
) -> uuid.UUID | None:
    """Remove ``name@alias``; the version it named, or None if there was none."""
    table = registry_aliases
    return conn.execute(
        sa.delete(table)
        .where(
            table.c.workspace_id == workspace_id,
            table.c.kind == kind,
            table.c.name == name,
            table.c.alias == alias,
        )
        .returning(table.c.target_id)
    ).scalar_one_or_none()


def resolve_alias(
    conn: Connection, workspace_id: uuid.UUID, *, kind: AliasKind, name: str, alias: str
) -> uuid.UUID | None:
    table = registry_aliases
    return conn.execute(
        sa.select(table.c.target_id).where(
            table.c.workspace_id == workspace_id,
            table.c.kind == kind,
            table.c.name == name,
            table.c.alias == alias,
        )
    ).scalar_one_or_none()


def aliases_of(
    conn: Connection, workspace_id: uuid.UUID, kind: AliasKind, target_ids: Iterable[uuid.UUID]
) -> dict[uuid.UUID, list[str]]:
    """Each target's aliases, sorted."""
    ids = list(target_ids)
    if not ids:
        return {}
    table = registry_aliases
    found: dict[uuid.UUID, list[str]] = {}
    for row in conn.execute(
        sa.select(table.c.target_id, table.c.alias)
        .where(
            table.c.workspace_id == workspace_id,
            table.c.kind == kind,
            table.c.target_id.in_(ids),
        )
        .order_by(table.c.alias)
    ):
        found.setdefault(row.target_id, []).append(row.alias)
    return found


def list_aliases(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    kind: AliasKind | None = None,
    name: str | None = None,
) -> list[Alias]:
    table = registry_aliases
    query = sa.select(table).where(table.c.workspace_id == workspace_id)
    if kind is not None:
        query = query.where(table.c.kind == kind)
    if name is not None:
        query = query.where(table.c.name == name)
    rows = conn.execute(query.order_by(table.c.kind, table.c.name, table.c.alias))
    return [_alias(row) for row in rows]
