"""The caller: who they are, their workspace, and their roles in it."""

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from stratafi_api.identity import Principal, get_principal
from stratafi_api.problems import problem_responses

router = APIRouter(tags=["identity"])


class WorkspaceOut(BaseModel):
    id: uuid.UUID
    slug: str
    name: str


class Me(BaseModel):
    subject: str
    tenant: str
    roles: list[str]
    scopes: list[str] | None = Field(
        default=None, description="An API token's scopes; none for people, whose roles decide"
    )
    workspace: WorkspaceOut


@router.get(
    "/me",
    response_model=Me,
    summary="The caller and their workspace roles",
    responses=problem_responses(401, 403),
)
def me(principal: Principal = Depends(get_principal)) -> Me:
    workspace = principal.workspace
    return Me(
        subject=principal.subject,
        tenant=principal.tenant,
        roles=sorted(principal.roles),
        scopes=sorted(principal.scopes) if principal.scopes is not None else None,
        workspace=WorkspaceOut(id=workspace.id, slug=workspace.slug, name=workspace.name),
    )
