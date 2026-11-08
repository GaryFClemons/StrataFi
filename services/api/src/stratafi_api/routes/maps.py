"""Pipeline maps: each RAG pipeline with its datastores.

- ``GET /pipeline-maps``: the map of a saved pipeline spec version, a saved retrieval config
  version, or both.
- ``GET /runs/{id}/map``: a pipeline run's spec, with each stage's tasks, cache hits, time and
  bytes written.
- ``GET /eval-runs/{id}/map``: the pipeline and the retrieval config an eval run measured, the
  pipeline run's numbers, and each query step's latency and the share of queries whose evidence
  reaches it.
- ``GET /rag-pipelines``: every pair of a pipeline spec version and a retrieval config version
  the workspace has evaluated, with its eval runs and its best headline result.

Every map comes with its Mermaid text. Maps name datastores and services and say whether this
deployment configures them; they never hold a host, a URL or a credential.
"""

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Any, cast

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_components.index import IndexComponent
from stratafi_core.components import ComponentRegistry, UnknownComponent
from stratafi_core.maps import (
    Deployment,
    attach_overlay,
    build_map,
    eval_overlay,
    overlay_stages,
    to_mermaid,
)
from stratafi_core.settings import Settings
from stratafi_core.tracking import headline_names
from stratafi_store import evalruns, runs, specs
from stratafi_store import tracking as store
from stratafi_store.artifacts import LocalArtifactStore
from stratafi_store.tables import artifacts

router = APIRouter(tags=["maps"])


class MapNodeOut(BaseModel):
    id: str
    kind: str = Field(description="input, stage, query, output, datastore or service")
    lane: str = Field(description="services, ingest, query or stores")
    label: str
    column: int
    row: int
    stage: str | None
    component: str | None
    version: str | None
    params: dict[str, Any]
    params_hash: str | None
    description: str | None
    egress: str | None
    lab_only: bool
    noop: bool = Field(description="It passes its input through: enrich/none, rerank/none")
    configured: bool | None = Field(description="This deployment configures the store or service")
    backend: str | None = Field(description="A datastore's kind: postgres, s3, filesystem, files")
    details: dict[str, Any]
    overlay: dict[str, Any] | None = Field(
        default=None, description="A run's numbers for this node, on a run's or eval run's map"
    )


class MapEdgeOut(BaseModel):
    id: str
    source: str
    target: str
    kind: str = Field(description="flow, read, write or call")
    label: str
    sends_text: bool = Field(description="It sends document or query text out of the lab")


class MapOut(BaseModel):
    schema_: str = Field(alias="schema", serialization_alias="schema")
    pipeline: dict[str, Any] | None = Field(description="The pipeline spec: id, name, version")
    retrieval: dict[str, Any] | None = Field(description="The retrieval config: id, name, version")
    nodes: list[MapNodeOut]
    edges: list[MapEdgeOut]
    summary: dict[str, Any] = Field(
        description="Stages, datastores, services, where text goes, and lab-only components"
    )
    mermaid: str = Field(description="The map as a Mermaid flowchart")
    run_id: uuid.UUID | None = None
    eval_run_id: uuid.UUID | None = None

    model_config = {"populate_by_name": True}


def deployment_of(request: Request) -> Deployment:
    settings = cast(Settings, request.app.state.settings)
    store_kind = request.app.state.store
    providers = set(settings.providers.allowed)
    if settings.services.tika_url:
        providers.add("tika")
    return Deployment(
        database=request.app.state.engine is not None,
        objects="filesystem" if isinstance(store_kind, LocalArtifactStore) else "s3",
        gateway=bool(settings.models.gateway_url),
        clamd=bool(settings.services.clamd),
        providers=frozenset(providers),
        in_process=settings.models.in_process,
    )


def _ref(stored: specs.StoredDocument | None) -> dict[str, Any] | None:
    if stored is None:
        return None
    return {"id": str(stored.id), "name": stored.name, "version": stored.version}


def _out(
    found: dict[str, Any],
    *,
    pipeline: specs.StoredDocument | None,
    retrieval: specs.StoredDocument | None,
    run_id: uuid.UUID | None = None,
    eval_run_id: uuid.UUID | None = None,
) -> MapOut:
    return MapOut(
        schema=found["schema"],
        pipeline=_ref(pipeline),
        retrieval=_ref(retrieval),
        nodes=[MapNodeOut(**node) for node in found["nodes"]],
        edges=[MapEdgeOut(**edge) for edge in found["edges"]],
        summary=found["summary"],
        mermaid=to_mermaid(found),
        run_id=run_id,
        eval_run_id=eval_run_id,
    )


def _document(
    conn: Connection, kind: str, principal: Principal, document_id: uuid.UUID | None, code: str
) -> specs.StoredDocument | None:
    if document_id is None:
        return None
    stored = specs.get_by_id(conn, kind, principal.workspace.id, document_id)  # type: ignore[arg-type]
    if stored is None:
        raise Problem("not_found", code=code)
    return stored


@router.get(
    "/pipeline-maps",
    response_model=MapOut,
    response_model_by_alias=True,
    summary="The map of a saved pipeline spec, retrieval config, or both",
    description="Each stage (one per parse route and clean rule), query time's steps, and the "
    "datastores and services each touches; calls that send text out of the lab are marked "
    ".",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_pipeline_map(
    request: Request,
    pipeline_spec_id: Annotated[uuid.UUID | None, Query()] = None,
    retrieval_config_id: Annotated[uuid.UUID | None, Query()] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> MapOut:
    if pipeline_spec_id is None and retrieval_config_id is None:
        raise Problem(
            "invalid_request",
            "Name a pipeline spec, a retrieval config, or both.",
            code="nothing_to_map",
            errors=["pipeline_spec_id", "retrieval_config_id"],
        )
    with engine_of(request).connect() as conn:
        pipeline = _document(conn, "pipeline", principal, pipeline_spec_id, "spec_not_found")
        retrieval = _document(
            conn, "retrieval", principal, retrieval_config_id, "retrieval_config_not_found"
        )
    found = build_map(
        pipeline.document if pipeline else None,
        retrieval.document if retrieval else None,
        request.app.state.registry,
        deployment_of(request),
    )
    return _out(found, pipeline=pipeline, retrieval=retrieval)


def _index_kind(registry: ComponentRegistry, component_id: str) -> str | None:
    try:
        component = registry.get(component_id)
    except UnknownComponent:
        return None
    return getattr(cast(type[IndexComponent], component), "kind", None)


def stage_overlay(
    conn: Connection, registry: ComponentRegistry, run: runs.Run
) -> dict[str, dict[str, Any]]:
    tasks = runs.tasks_of(conn, run.id)
    sizes = dict(
        conn.execute(
            sa.select(artifacts.c.cache_key, artifacts.c.size_bytes).where(
                artifacts.c.workspace_id == run.workspace_id,
                artifacts.c.cache_key.in_({task.cache_key for task in tasks}),
            )
        ).all()
    )
    return overlay_stages(
        [
            {
                "stage": task.stage,
                "route": task.route,
                "index_kind": _index_kind(registry, task.component_id)
                if task.stage == "index"
                else None,
                "status": task.status,
                "cache_hit": task.cache_hit,
                "duration_ms": task.duration_ms,
                "size_bytes": sizes.get(task.cache_key),
            }
            for task in tasks
        ]
    )


@router.get(
    "/runs/{run_id}/map",
    response_model=MapOut,
    response_model_by_alias=True,
    summary="A pipeline run's map, with each stage's tasks, cache hits, time and bytes",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_run_map(
    request: Request, run_id: uuid.UUID, principal: Principal = Depends(require_role("viewer"))
) -> MapOut:
    registry = request.app.state.registry
    with engine_of(request).connect() as conn:
        run = runs.get_run(conn, principal.workspace.id, run_id)
        if run is None:
            raise Problem("not_found", code="run_not_found")
        pipeline = specs.get_by_id(conn, "pipeline", principal.workspace.id, run.pipeline_spec_id)
        overlay = stage_overlay(conn, registry, run)
    if pipeline is None:  # pragma: no cover - a run keeps its spec
        raise Problem("not_found", code="spec_not_found")
    found = build_map(pipeline.document, None, registry, deployment_of(request))
    attach_overlay(found, stages=overlay)
    return _out(found, pipeline=pipeline, retrieval=None, run_id=run.id)


@router.get(
    "/eval-runs/{eval_run_id}/map",
    response_model=MapOut,
    response_model_by_alias=True,
    summary="An eval run's RAG pipeline: its pipeline run's stages and its query steps",
    description="The pipeline run's numbers on the ingest stages; each query step's latency "
    "and the share of queries whose evidence reaches its list.",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_eval_run_map(
    request: Request,
    eval_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> MapOut:
    registry = request.app.state.registry
    with engine_of(request).connect() as conn:
        run = evalruns.get_eval_run(conn, principal.workspace.id, eval_run_id)
        if run is None:
            raise Problem("not_found", code="eval_run_not_found")
        pipeline_run = runs.get_run(conn, principal.workspace.id, run.pipeline_run_id)
        pipeline = (
            specs.get_by_id(conn, "pipeline", principal.workspace.id, pipeline_run.pipeline_spec_id)
            if pipeline_run is not None
            else None
        )
        retrieval = specs.get_by_id(
            conn, "retrieval", principal.workspace.id, run.retrieval_config_id
        )
        overlay = stage_overlay(conn, registry, pipeline_run) if pipeline_run else {}
    resolved = (run.context or {}).get("retrieval") or (retrieval.document if retrieval else None)
    found = build_map(
        pipeline.document if pipeline else None,
        resolved,
        registry,
        deployment_of(request),
        evaluation=True,
    )
    attach_overlay(found, stages=overlay, queries=eval_overlay(run.metrics))
    return _out(
        found,
        pipeline=pipeline,
        retrieval=retrieval,
        run_id=run.pipeline_run_id,
        eval_run_id=run.id,
    )


# --- every RAG pipeline -------------------------------------------------------------------


class RagPipelineOut(BaseModel):
    pipeline_spec: dict[str, Any] = Field(description="id, name and version")
    retrieval_config: dict[str, Any] = Field(description="id, name and version")
    eval_runs: int
    eval_sets: list[str] = Field(description="The eval sets it was measured on: name@version")
    latest_eval_run_id: uuid.UUID
    latest_at: datetime
    best: dict[str, Any] | None = Field(
        description="Its best eval run on the headline recall: id, metric, mean, eval set"
    )
    components: list[str] = Field(description="Every component it uses, ingest and query")
    services: list[str] = Field(description="The services it calls, from its map")
    datastores: list[str]
    sends_text_to: list[str]


class RagPipelinesOut(BaseModel):
    items: list[RagPipelineOut]


def _components(found: dict[str, Any]) -> list[str]:
    return sorted({node["component"] for node in found["nodes"] if node.get("component")})


def _best(rows: Sequence[store.SearchRow]) -> dict[str, Any] | None:
    best = None
    for row in rows:
        k_eval = int((row.context or {}).get("k_eval") or 5)
        name = headline_names(k_eval)[1]
        estimate = (
            ((row.metrics or {}).get("metrics") or {}).get("knowledge-hit-v1", {}).get("final", {})
        ).get(name)
        mean = estimate.get("mean") if isinstance(estimate, dict) else None
        if isinstance(mean, int | float) and (best is None or mean > best["mean"]):
            best = {
                "eval_run_id": str(row.id),
                "metric": name,
                "mean": mean,
                "eval_set": f"{row.eval_set_name}@{row.eval_set_version}",
                "split": row.split,
            }
    return best


@router.get(
    "/rag-pipelines",
    response_model=RagPipelinesOut,
    summary="Every RAG pipeline the workspace has evaluated: spec and config, with its map's "
    "services and stores",
    responses=problem_responses(400, 401, 403, 503),
)
def list_rag_pipelines(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    principal: Principal = Depends(require_role("viewer")),
) -> RagPipelinesOut:
    registry = request.app.state.registry
    deployment = deployment_of(request)
    with engine_of(request).connect() as conn:
        rows = store.search_rows(conn, principal.workspace.id, limit=2000, slices=False)
        groups: dict[tuple[str, str], list[store.SearchRow]] = {}
        for row in rows:
            if row.pipeline_spec_id is None:
                continue
            groups.setdefault((str(row.pipeline_spec_id), str(row.retrieval_config_id)), []).append(
                row
            )
        documents: dict[str, dict[str, Any]] = {}
        for pipeline_id, retrieval_id in groups:
            for kind, identifier in (("pipeline", pipeline_id), ("retrieval", retrieval_id)):
                if identifier not in documents:
                    stored = specs.get_by_id(
                        conn,
                        kind,  # type: ignore[arg-type]
                        principal.workspace.id,
                        uuid.UUID(identifier),
                    )
                    documents[identifier] = stored.document if stored else {}
    items = []
    for (pipeline_id, retrieval_id), members in list(groups.items())[:limit]:
        latest = members[0]
        found = build_map(documents[pipeline_id], documents[retrieval_id], registry, deployment)
        items.append(
            RagPipelineOut(
                pipeline_spec={
                    "id": pipeline_id,
                    "name": latest.pipeline_name,
                    "version": latest.pipeline_version,
                },
                retrieval_config={
                    "id": retrieval_id,
                    "name": latest.retrieval_name,
                    "version": latest.retrieval_version,
                },
                eval_runs=len(members),
                eval_sets=sorted(
                    {f"{row.eval_set_name}@{row.eval_set_version}" for row in members}
                ),
                latest_eval_run_id=latest.id,
                latest_at=latest.created_at,
                best=_best(members),
                components=_components(found),
                services=found["summary"]["services"],
                datastores=found["summary"]["datastores"],
                sends_text_to=found["summary"]["sends_text_to"],
            )
        )
    return RagPipelinesOut(items=items)
