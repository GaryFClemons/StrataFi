"""The component registry: what this installation can run, with each component's parameter
JSON Schema, which the UI turns into forms."""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_core.components import (
    ComponentDescription,
    ComponentRegistry,
    ComponentStage,
    UnknownComponent,
)

router = APIRouter(tags=["components"])


class ComponentOut(BaseModel):
    id: str
    stage: ComponentStage
    version: str
    title: str
    description: str
    accepts: list[str]
    resource_class: Literal["cpu", "gpu", "io"]
    deterministic: bool
    license: str
    platforms: list[str]
    egress: str
    services: list[str] = Field(
        default_factory=list,
        description="Stores and services it uses besides the artifact store and the database ",
    )
    lab_only: bool
    baseline: bool
    params_schema: dict[str, Any]
    available: bool = Field(description="This installation can run it")
    unavailable_reason: str | None = Field(
        description="Why it can't, such as a missing optional extra"
    )

    @classmethod
    def of(cls, description: ComponentDescription) -> "ComponentOut":
        info = description.info
        return cls(
            **info.model_dump(mode="json", exclude={"accepts", "platforms", "services"}),
            accepts=list(info.accepts),
            platforms=list(info.platforms),
            services=list(info.services),
            params_schema=description.params_schema,
            available=description.unavailable is None,
            unavailable_reason=description.unavailable,
        )


class ComponentList(BaseModel):
    items: list[ComponentOut]


def registry(request: Request) -> ComponentRegistry:
    return request.app.state.registry


@router.get(
    "/components",
    response_model=ComponentList,
    summary="Every component this installation can run, with its parameter schema",
    responses=problem_responses(400, 401, 403),
)
def list_components(
    request: Request,
    stage: Annotated[ComponentStage | None, Query(description="Only this stage's")] = None,
    _: Principal = Depends(require_role("viewer")),
) -> ComponentList:
    descriptions = registry(request).descriptions()
    return ComponentList(
        items=[
            ComponentOut.of(description)
            for description in descriptions
            if stage is None or description.info.stage == stage
        ]
    )


@router.get(
    "/components/{kind}/{name}",
    response_model=ComponentOut,
    summary="One component, by its ID (kind/name)",
    responses=problem_responses(400, 401, 403, 404),
)
def get_component(
    request: Request, kind: str, name: str, _: Principal = Depends(require_role("viewer"))
) -> ComponentOut:
    try:
        return ComponentOut.of(registry(request).describe(f"{kind}/{name}"))
    except UnknownComponent:
        raise Problem("not_found", code="component_not_found") from None
