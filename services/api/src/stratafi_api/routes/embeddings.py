"""The Embed screen's API: the model catalog, a 2-D projection of a run's
vectors, and nearest neighbours.

- ``GET /models/catalog`` lists the embedding models this installation can use and where each
  runs: the gateway's aliases (from its ``/v1/models``), the in-process catalog (lab only,
  when ``models.in_process`` is on and the extra is installed), and the allow-listed
  OpenAI-compatible endpoints; with dimensions, input limits, licenses and prefixes.
- ``GET /runs/{id}/projection`` projects the run's dense index to two dimensions: PCA (numpy's
  SVD, no extra), or UMAP with the ``umap`` extra. At most ``limit`` points, chosen with a
  fixed seed, each with its document, entry name and type, for colouring.
- ``GET /runs/{id}/neighbours`` gives a chunk's nearest chunks in the same index, by cosine.

Texts come back only where the screen shows them (neighbours); nothing here logs them.
"""

import random
import uuid
from typing import Annotated, Literal, cast

import numpy as np
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from stratafi_api.deps import engine_of, store_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_components.embedders import CATALOG, SentenceTransformersEmbed
from stratafi_components.index import DenseIndex, IndexComponent
from stratafi_components.models import GatewayProvider
from stratafi_core.components import ComponentError
from stratafi_store import corpora, runs

router = APIRouter(tags=["embeddings"])

MAX_POINTS = 20_000


class CatalogModelOut(BaseModel):
    where: Literal["gateway", "in_process", "hosted"]
    component: str
    model: str = Field(description="An alias, a Hugging Face ID, or provider:model")
    title: str
    dims: int | None
    max_tokens: int | None
    license: str
    query_prefix: str = ""
    document_prefix: str = ""
    matryoshka: list[int] = Field(default_factory=list)
    available: bool
    reason: str | None = None


class CatalogOut(BaseModel):
    models: list[CatalogModelOut]


@router.get(
    "/models/catalog",
    response_model=CatalogOut,
    summary="Embedding models this installation can use, and where each runs",
    responses=problem_responses(401, 403),
)
def model_catalog(request: Request, _: Principal = Depends(require_role("viewer"))) -> CatalogOut:
    settings = request.app.state.settings
    models: list[CatalogModelOut] = []
    if settings.models.gateway_url:
        provider = GatewayProvider(settings.models.gateway_url, max_attempts=1)
        try:
            for info in provider.models().values():
                if info.purpose in (None, "embed"):
                    models.append(
                        CatalogModelOut(
                            where="gateway",
                            component="embed/gateway",
                            model=info.alias,
                            title=info.alias,
                            dims=info.dimensions,
                            max_tokens=None,
                            license="per alias",
                            available=True,
                        )
                    )
        except ComponentError:
            pass
        finally:
            provider.close()
    missing = SentenceTransformersEmbed.unavailable()
    reason = "models.in_process is off" if not settings.models.in_process else missing
    for entry in CATALOG:
        models.append(
            CatalogModelOut(
                where="in_process",
                component="embed/sentence-transformers",
                model=entry.model,
                title=entry.title,
                dims=entry.dims,
                max_tokens=entry.max_tokens,
                license=entry.license,
                query_prefix=entry.query_prefix,
                document_prefix=entry.document_prefix,
                matryoshka=list(entry.matryoshka),
                available=reason is None,
                reason=reason,
            )
        )
    for name in settings.providers.allowed:
        if name in settings.providers.endpoints:
            models.append(
                CatalogModelOut(
                    where="hosted",
                    component="embed/openai-compatible",
                    model=f"{name}:*",
                    title=name,
                    dims=None,
                    max_tokens=None,
                    license="per endpoint",
                    available=True,
                )
            )
    return CatalogOut(models=models)


class ProjectionPoint(BaseModel):
    chunk_id: str
    x: float
    y: float
    document_id: str
    entry: str | None
    media_type: str | None


class ProjectionOut(BaseModel):
    method: Literal["pca", "umap"]
    points: list[ProjectionPoint]
    total: int = Field(description="Vectors in the index; points is a seeded sample")
    explained_variance: list[float] = Field(default_factory=list, description="PCA only")


def _dense(request: Request, principal: Principal, run_id: uuid.UUID):
    """A run's dense index, loaded, with entry names and types by document."""
    workspace_id = principal.workspace.id
    with engine_of(request).connect() as conn:
        run = runs.get_run(conn, workspace_id, run_id)
        if run is None:
            raise Problem("not_found", code="run_not_found")
        tasks = [
            task
            for task in runs.tasks_of(conn, run.id, stage="index")
            if task.status in runs.DONE_TASK_STATUSES
        ]
        entries = corpora.version_entries(conn, workspace_id, run.corpus_version_id)
    registry = request.app.state.registry
    dense = [
        task
        for task in tasks
        if cast(type[IndexComponent], registry.get(task.component_id)).kind == "dense"
    ]
    if not dense:
        raise Problem("conflict", "The run has no dense index.", code="index_missing")
    loaded = request.app.state.indexes.get(
        store_of(request), workspace_id, dense[0].cache_key, "dense", engine_of(request)
    )
    if not isinstance(loaded.index, DenseIndex):
        raise Problem("conflict", "Projections read in-memory indexes.", code="index_not_in_memory")
    names: dict[str, tuple[str, str | None]] = {}
    for entry in entries:
        names.setdefault(str(entry.document_id), (entry.name, entry.media_type))
    return loaded, names


@router.get(
    "/runs/{run_id}/projection",
    response_model=ProjectionOut,
    summary="A run's vectors projected to two dimensions (PCA, or UMAP with its extra)",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def projection(
    request: Request,
    run_id: uuid.UUID,
    method: Literal["pca", "umap"] = "pca",
    limit: Annotated[int, Query(ge=10, le=MAX_POINTS)] = 5000,
    principal: Principal = Depends(require_role("viewer")),
) -> ProjectionOut:
    loaded, names = _dense(request, principal, run_id)
    index = cast(DenseIndex, loaded.index)
    matrix = index.matrix.astype(np.float32)
    total = len(matrix)
    rows = list(range(total))
    if total > limit:
        rows = sorted(random.Random(0).sample(rows, limit))  # noqa: S311 - a fixed sample
    sample = matrix[rows]
    variance: list[float] = []
    if method == "umap":
        try:
            import umap
        except ImportError:
            raise Problem("conflict", "UMAP needs the umap extra.", code="extra_missing") from None
        coordinates = umap.UMAP(n_components=2, random_state=0).fit_transform(sample)
    else:
        centered = sample - sample.mean(axis=0, keepdims=True)
        if len(sample) < 2:
            coordinates = np.zeros((len(sample), 2), dtype=np.float32)
        else:
            _, singular, vt = np.linalg.svd(centered, full_matrices=False)
            coordinates = centered @ vt[:2].T
            energy = singular**2
            variance = [round(float(v), 6) for v in (energy[:2] / (energy.sum() or 1.0))]
    points = []
    for position, row in enumerate(rows):
        document_id = index.manifest.document_ids[row]
        entry, media_type = names.get(document_id, (None, None))
        points.append(
            ProjectionPoint(
                chunk_id=index.manifest.chunk_ids[row],
                x=round(float(coordinates[position][0]), 6),
                y=round(float(coordinates[position][1]) if coordinates.shape[1] > 1 else 0.0, 6),
                document_id=document_id,
                entry=entry,
                media_type=media_type,
            )
        )
    return ProjectionOut(method=method, points=points, total=total, explained_variance=variance)


class NeighbourOut(BaseModel):
    chunk_id: str
    score: float
    document_id: str
    entry: str | None
    text: str


class NeighboursOut(BaseModel):
    chunk_id: str
    neighbours: list[NeighbourOut]


@router.get(
    "/runs/{run_id}/neighbours",
    response_model=NeighboursOut,
    summary="A chunk's nearest chunks in the run's dense index, by cosine",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def neighbours(
    request: Request,
    run_id: uuid.UUID,
    chunk_id: Annotated[str, Query(pattern=r"^[0-9a-f]{32}$")],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    principal: Principal = Depends(require_role("viewer")),
) -> NeighboursOut:
    loaded, names = _dense(request, principal, run_id)
    index = cast(DenseIndex, loaded.index)
    try:
        row = index.manifest.chunk_ids.index(chunk_id)
    except ValueError:
        raise Problem("not_found", code="chunk_not_found") from None
    matrix = index.matrix
    vector = matrix[row] / (float(np.linalg.norm(matrix[row])) or 1.0)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    scores = (matrix @ vector) / norms
    order = [int(i) for i in np.argsort(-scores, kind="stable") if int(i) != row][:k]
    found: list[NeighbourOut] = []
    for other in order:
        chunk = loaded.chunks.get(index.manifest.chunk_ids[other])
        if chunk is None:
            continue
        found.append(
            NeighbourOut(
                chunk_id=chunk.chunk_id,
                score=round(float(scores[other]), 6),
                document_id=chunk.document_id,
                entry=names.get(chunk.document_id, (None, None))[0],
                text=chunk.text,
            )
        )
    return NeighboursOut(chunk_id=chunk_id, neighbours=found)
