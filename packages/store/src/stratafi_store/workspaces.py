"""Workspaces and memberships: the tenancy unit.

A workspace maps one-to-one to a tenant (an identity token's ``tid``). Identity resolves the
workspace from the token; with auto-provisioning (development) the first token of a new tenant
creates its workspace. Memberships grant workspace roles to a subject locally, on top of the
roles the token carries; no user table is the source of truth.
"""

import uuid
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from stratafi_store.tables import memberships, workspaces


@dataclass(frozen=True)
class Workspace:
    id: uuid.UUID
    slug: str
    name: str
    external_tenant_id: str | None = None


def _workspace(row: sa.Row) -> Workspace:
    return Workspace(row.id, row.slug, row.name, row.external_tenant_id)


def workspace_for_tenant(conn: Connection, tenant: str) -> Workspace | None:
    row = conn.execute(
        sa.select(workspaces).where(workspaces.c.external_tenant_id == tenant)
    ).first()
    return _workspace(row) if row is not None else None


def provision_workspace(conn: Connection, tenant: str) -> tuple[Workspace, bool]:
    """The tenant's workspace, created if missing. Returns it and whether it was created.

    The slug and name start as the tenant ID, which already has a slug's shape.
    """
    created = conn.execute(
        insert(workspaces)
        .values(id=uuid.uuid4(), slug=tenant, name=tenant, external_tenant_id=tenant)
        .on_conflict_do_nothing()
        .returning(workspaces)
    ).first()
    if created is not None:
        return _workspace(created), True
    existing = workspace_for_tenant(conn, tenant)
    if existing is None:
        # The slug is taken by a workspace of another tenant: a configuration error.
        raise LookupError("a workspace already uses this tenant's slug")
    return existing, False


def membership_role(conn: Connection, workspace_id: uuid.UUID, subject: str) -> str | None:
    return conn.execute(
        sa.select(memberships.c.role).where(
            memberships.c.workspace_id == workspace_id, memberships.c.subject == subject
        )
    ).scalar_one_or_none()


def grant_role(conn: Connection, workspace_id: uuid.UUID, subject: str, role: str) -> None:
    statement = insert(memberships).values(workspace_id=workspace_id, subject=subject, role=role)
    conn.execute(
        statement.on_conflict_do_update(
            index_elements=["workspace_id", "subject"],
            set_={"role": statement.excluded.role, "updated_at": sa.func.now()},
        )
    )
