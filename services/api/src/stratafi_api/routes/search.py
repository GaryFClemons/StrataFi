"""Search a run's indexes through the retrieval engine.

By ``mode``: dense, BM25, or both fused with reciprocal rank fusion (k = 60) over the best 50 of
each; the best ``top_k`` come back. A saved retrieval config (``retrieval_config_id``) replaces
the mode with its retrievers, fusion, filters, reranker and ``top_k``. The engine
(``stratafi_components.retrieval``) is the one evaluation uses, so what the lab shows is
what it measures.

An ``asker`` (a user, roles and groups) filters by the entries' ACLs, and ``as_of`` by their
effective dates; without them the lab sees its whole run. A dense query is embedded
with the run's embedder through the model gateway, carrying the caller's identity, and must come
from the revision the index was built with. Each checkpoint's list comes back too, so the UI can
show where a passage ranked before fusion.

Indexes and their chunks are immutable, so loaded ones are kept in a small cache; query vectors
and rerank scores are cached for the process. The query is never logged.

The search playground sends an unsaved ``retrieval_config`` instead, validated like a
saved one, to compare two configurations side by side; and an eval item's ``evidence``, so each
hit says where the evidence sits in its text (``highlights``, the relevance module's).
"""

import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterable
from datetime import date
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from stratafi_api.deps import engine_of, store_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_api.routes.specs import invalid
from stratafi_api.trace_models import TraceBody
from stratafi_components.embed import EmbedComponent
from stratafi_components.index import DenseIndex, IndexComponent
from stratafi_components.models import GatewayProvider
from stratafi_components.parsers.hosted import hosted_services
from stratafi_components.rerank import RerankComponent
from stratafi_components.retrieval import (
    EntryMeta,
    LoadedIndex,
    QueryEmbedder,
    Reranker,
    RetrievalEngine,
    load_index,
    with_adapter,
)
from stratafi_core.artifacts import Chunk
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.evalsets import Asker
from stratafi_core.identity import HEADER as IDENTITY_HEADER
from stratafi_core.relevance import highlights
from stratafi_core.specs import RetrievalConfig, SpecInvalid, dump_document, resolve_retrieval
from stratafi_core.traces import build_trace
from stratafi_store import corpora, runs, specs
from stratafi_store.artifacts import (
    ArtifactNotFound,
    ArtifactStore,
    artifact_key,
    artifact_part_key,
)
from stratafi_store.training import WorkspaceModels

router = APIRouter(tags=["search"])

CANDIDATES = 50
CACHED_INDEXES = 8
UNAVAILABLE_CODES = ("models_not_configured", "gateway_timeout", "gateway_unreachable")
Mode = Literal["dense", "keyword", "hybrid"]


class AskerIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_-]{0,63}$",
        description="A label for the asker, such as an eval set's asker name; it grants nothing",
    )
    user: str | None = Field(
        default=None, max_length=128, description="Their user ID, for ACLs that list users"
    )
    roles: list[str] = Field(default_factory=list, max_length=64)
    groups: list[str] = Field(
        default_factory=list, max_length=256, description="The ACL groups they belong to"
    )


class SearchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID = Field(description="A run whose indexes to search")
    query: str = Field(min_length=1, max_length=2000)
    mode: Mode = "hybrid"
    top_k: int = Field(default=10, ge=1, le=50)
    retrieval_config_id: uuid.UUID | None = Field(
        default=None, description="A saved retrieval config: replaces mode and top_k"
    )
    retrieval_config: dict[str, Any] | None = Field(
        default=None,
        description="An unsaved stratafi.retrieval.v1 document, validated like a saved one",
    )
    evidence: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="An eval item's evidence phrases: each hit says where they sit in its text",
    )
    asker: AskerIn | None = Field(
        default=None, description="Filter by ACL as this asker; without one, nothing is hidden"
    )
    as_of: date | None = Field(default=None, description="Filter by effective dates as of this day")


class Hit(BaseModel):
    rank: int
    chunk_id: str
    score: float
    document_id: str
    entry: str | None = Field(description="The document's name in the run's corpus version")
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    text: str
    highlights: list[tuple[int, int]] = Field(
        default_factory=list, description="Where the asked evidence sits in the text"
    )


class SearchOut(BaseModel):
    mode: Mode
    revision: str | None = Field(description="The embedding revision a dense query used")
    results: list[Hit]
    checkpoints: dict[str, list[Hit]] = Field(
        description="Each list before the cut: dense, keyword, and fused for hybrid; with a "
        "retrieval config, also candidates and reranked"
    )
    trace: TraceBody | None = Field(
        default=None,
        description="The search's steps on a timeline, each with its lists; the "
        "playground's query has no labels, so nothing is marked relevant",
    )


class IndexCache:
    """The last few loaded indexes, by workspace and cache key."""

    def __init__(self, size: int = CACHED_INDEXES) -> None:
        self.size = size
        self._items: OrderedDict[tuple[uuid.UUID, str], LoadedIndex] = OrderedDict()
        self._lock = threading.Lock()

    def get(
        self,
        store: ArtifactStore,
        workspace_id: uuid.UUID,
        cache_key: str,
        kind: str,
        engine: Any = None,
    ) -> LoadedIndex:
        key = (workspace_id, cache_key)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]

        def read(artifact: str, part: str | None) -> bytes:
            if part is None:
                return store.get(artifact_key(workspace_id, artifact))
            return store.get(artifact_part_key(workspace_id, artifact, part))

        try:
            loaded = load_index(read, cache_key, kind, engine=engine)
        except ArtifactNotFound:
            raise Problem("conflict", "The run's index is missing.", code="index_missing") from None
        with self._lock:
            self._items[key] = loaded
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return loaded

    def forget(self, workspace_id: uuid.UUID, cache_keys: Iterable[str]) -> None:
        """Drop purged indexes: a loaded index holds its chunks' text."""
        with self._lock:
            for cache_key in cache_keys:
                self._items.pop((workspace_id, cache_key), None)


def mode_config(mode: Mode, top_k: int) -> RetrievalConfig:
    """What a mode searches with: RRF (k = 60) for hybrid, no reranker."""
    retrievers: dict[str, Any] = {
        "dense": {"top_n": CANDIDATES} if mode != "keyword" else None,
        "keyword": {"top_n": CANDIDATES} if mode != "dense" else None,
    }
    return RetrievalConfig.model_validate(
        {
            "schema": "stratafi.retrieval.v1",
            "name": f"lab-{mode}",
            "retrievers": retrievers,
            "fusion": {"method": "rrf" if mode == "hybrid" else "none", "k": 60},
            "candidates": CANDIDATES,
            "filters": {"current_only": True},
            "rerank": "rerank/none",
            "top_k": top_k,
        }
    )


def asker_of(body: AskerIn | None) -> Asker | None:
    if body is None:
        return None
    return Asker(name=body.name, user=body.user, roles=body.roles, groups=body.groups)


def _hits(
    ranked: list[tuple[str, float]],
    chunks: dict[str, Chunk],
    names: dict[str, str],
    top_k: int,
    evidence: list[str] | None = None,
) -> list[Hit]:
    hits = []
    for chunk_id, score in ranked:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            continue
        spans: list[tuple[int, int]] = []
        for phrase in evidence or []:
            spans += [tuple(span) for span in highlights(phrase, chunk.text)]  # type: ignore[misc]
        hits.append(
            Hit(
                rank=len(hits) + 1,
                chunk_id=chunk_id,
                score=round(float(score), 6),
                document_id=chunk.document_id,
                entry=names.get(chunk.document_id),
                heading_path=list(chunk.heading_path),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                highlights=spans,
            )
        )
        if len(hits) == top_k:
            break
    return hits


def gateway_services(
    request: Request, tenant: str | None = None
) -> tuple[dict[str, Any], GatewayProvider | None]:
    """The model gateway, carrying the caller's identity token; a caller who sent none (an API
    token) calls as Stratafi's service identity in their tenant. And what gated rerankers need
    (allow-listed providers, licenses, the lab switches)."""
    settings = request.app.state.settings
    gated = hosted_services(settings)
    gateway_url = settings.models.gateway_url
    if not gateway_url:
        return gated, None
    token = request.headers.get(IDENTITY_HEADER)
    tokens = None if token else getattr(request.app.state, "service_tokens", None)

    def headers() -> dict[str, str]:
        if token:
            return {IDENTITY_HEADER: token}
        minted = tokens.token(tenant) if tokens is not None and tenant else None
        return {IDENTITY_HEADER: minted} if minted else {}

    provider = GatewayProvider(gateway_url, headers=headers)
    return {**gated, "models": provider}, provider


def _component_error(error: ComponentError) -> Problem:
    if error.code in UNAVAILABLE_CODES or error.code.startswith("gateway_"):
        return Problem("unavailable", code=error.code)
    if error.code == "revision_mismatch":
        return Problem(
            "conflict",
            "The embedding model changed since the index was built.",
            code="revision_changed",
        )
    return Problem("conflict", code=error.code)


@router.post(
    "/search",
    response_model=SearchOut,
    summary="Search a run's indexes: dense, BM25, hybrid with RRF, or a saved retrieval config",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def search(
    request: Request, body: SearchIn, principal: Principal = Depends(require_role("viewer"))
) -> SearchOut:
    workspace_id = principal.workspace.id
    with engine_of(request).connect() as conn:
        run = runs.get_run(conn, workspace_id, body.run_id)
        if run is None:
            raise Problem("not_found", code="run_not_found")
        tasks = [
            task
            for task in runs.tasks_of(conn, run.id, stage="index")
            if task.status in runs.DONE_TASK_STATUSES
        ]
        entries = corpora.version_entries(conn, workspace_id, run.corpus_version_id)
        spec = runs.run_spec(conn, run)
        stored = (
            specs.get_by_id(conn, "retrieval", workspace_id, body.retrieval_config_id)
            if body.retrieval_config_id is not None
            else None
        )
    registry = request.app.state.registry
    if body.retrieval_config_id is not None:
        if stored is None:
            raise Problem("not_found", code="retrieval_config_not_found")
        config = resolve_retrieval(stored.document, registry)
    elif body.retrieval_config is not None:
        try:
            config = resolve_retrieval(body.retrieval_config, registry)
        except SpecInvalid as error:
            raise invalid(error) from None
    else:
        config = mode_config(body.mode, body.top_k)
    by_kind = {
        cast(type[IndexComponent], registry.get(task.component_id)).kind: task.cache_key
        for task in tasks
    }
    wanted = [
        kind
        for kind, retriever in (
            ("keyword", config.retrievers.keyword),
            ("dense", config.retrievers.dense),
        )
        if retriever is not None
    ]
    if missing := [kind for kind in wanted if kind not in by_kind]:
        raise Problem("conflict", f"The run has no {missing[0]} index.", code="index_missing")
    cache: IndexCache = request.app.state.indexes
    store = store_of(request)
    database = engine_of(request)
    loaded = {
        kind: cache.get(store, workspace_id, by_kind[kind], kind, database) for kind in wanted
    }
    chunks: dict[str, Chunk] = {}
    for item in loaded.values():
        chunks.update(item.chunks)
    names: dict[str, str] = {}
    for entry in entries:
        names.setdefault(str(entry.document_id), entry.name)

    services, provider = gateway_services(request, principal.tenant)
    if request.app.state.model_files is not None:
        services["trained_models"] = WorkspaceModels(request.app.state.model_files, workspace_id)
    context = StageContext(str(workspace_id), services=services)
    try:
        embedder: QueryEmbedder | None = None
        if "dense" in loaded:
            component = cast(EmbedComponent, registry.get(spec.stages.embed.component)())
            embedder = QueryEmbedder(
                component, component.resolve_params(spec.stages.embed.params), context
            )
            adapter = config.retrievers.dense.adapter if config.retrievers.dense else None
            if adapter is not None:
                # A trained query adapter, checked against the dense index.
                embedder = with_adapter(
                    embedder, adapter, dense=cast(DenseIndex, loaded["dense"].index)
                )
        rerank = cast(RerankComponent, registry.get(config.rerank.component)())
        engine = RetrievalEngine(
            config,
            entries=[EntryMeta.from_entry(entry) for entry in entries],
            chunks=chunks,
            dense=loaded["dense"].index if "dense" in loaded else None,  # type: ignore[arg-type]
            keyword=loaded["keyword"].index if "keyword" in loaded else None,  # type: ignore[arg-type]
            embedder=embedder,
            reranker=Reranker.resolve(rerank, rerank.resolve_params(config.rerank.params), context),
            cache=request.app.state.retrieval_cache,
        )
        result = engine.search(body.query, asker=asker_of(body.asker), day=body.as_of)
    except ComponentError as error:
        raise _component_error(error) from None
    finally:
        if provider is not None:
            provider.close()
    if body.retrieval_config_id is None and body.retrieval_config is None:
        shown = {"keyword", "dense", "fused"} if body.mode == "hybrid" else {body.mode}
    else:
        shown = set(result.checkpoints) - {"final"}
    top_k = config.top_k
    checkpoints = {
        name: _hits(ranked, chunks, names, top_k, body.evidence)
        for name, ranked in result.checkpoints.items()
        if name in shown
    }
    models: dict[str, Any] = {
        "rerank": {
            "component": config.rerank.component,
            "params": dict(config.rerank.params),
            "revision": result.rerank_revision,
        }
    }
    if embedder is not None:
        models["embed"] = {
            "component": spec.stages.embed.component,
            "params": dict(spec.stages.embed.params),
            "revision": result.embedding_revision,
        }
    trace = build_trace(
        latency_ms=result.latency_ms,
        checkpoints=result.checkpoints,
        relevant=None,
        config=dump_document(config),
        models=models,
    )
    return SearchOut(
        mode=body.mode,
        revision=result.embedding_revision,
        results=_hits(result.checkpoints["final"], chunks, names, top_k, body.evidence),
        checkpoints=checkpoints,
        trace=TraceBody(**trace),
    )
