"""The ``components`` table: a snapshot of the component registry.

``sync_components`` runs when the API starts. It records every registered component version
(its info and parameter schema), marks versions the code no longer has as unavailable, and keeps
their rows, so a run that used one can still say what it was.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from stratafi_core.components import ComponentDescription
from stratafi_store.tables import components


@dataclass(frozen=True)
class ComponentRow:
    component_id: str
    version: str
    stage: str
    info: dict[str, Any]
    params_schema: dict[str, Any]
    available: bool
    first_seen_at: datetime
    last_seen_at: datetime


def sync_components(conn: Connection, descriptions: Iterable[ComponentDescription]) -> int:
    """Record the registry. Returns how many component versions it holds."""
    rows = [
        {
            "component_id": description.info.id,
            "version": description.info.version,
            "stage": description.info.stage,
            "info": description.info.model_dump(mode="json"),
            "params_schema": description.params_schema,
            "available": True,
        }
        for description in descriptions
    ]
    # One process at a time, so the API and a worker starting together don't interleave.
    conn.execute(
        sa.text("SELECT pg_advisory_xact_lock(hashtextextended('stratafi:components', 0))")
    )
    if rows:
        statement = insert(components).values(rows)
        conn.execute(
            statement.on_conflict_do_update(
                index_elements=["component_id", "version"],
                set_={
                    "stage": statement.excluded.stage,
                    "info": statement.excluded.info,
                    "params_schema": statement.excluded.params_schema,
                    "available": True,
                    "last_seen_at": sa.func.now(),
                },
            )
        )
    current = {(row["component_id"], row["version"]) for row in rows}
    stale = [
        (row.component_id, row.version)
        for row in conn.execute(
            sa.select(components.c.component_id, components.c.version).where(components.c.available)
        )
        if (row.component_id, row.version) not in current
    ]
    for component_id, version in stale:
        conn.execute(
            sa.update(components)
            .where(components.c.component_id == component_id, components.c.version == version)
            .values(available=False)
        )
    return len(rows)


def list_component_rows(conn: Connection, *, available_only: bool = False) -> list[ComponentRow]:
    query = sa.select(components).order_by(components.c.component_id, components.c.version)
    if available_only:
        query = query.where(components.c.available)
    return [
        ComponentRow(**{column.name: getattr(row, column.name) for column in components.columns})
        for row in conn.execute(query)
    ]
