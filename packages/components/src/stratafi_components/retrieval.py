"""Retrieval for evaluation.

Given a run's indexes, the metadata of its corpus entries, a retrieval config
(``stratafi.retrieval.v1``) and a query with its asker and "as of" date,
``RetrievalEngine.search`` runs a hybrid search one step at a time and keeps every step's ranked
list:

1. **Embed the query** with the run's embedder and its query prefix (nomic and e5 need one),
   cached by the embedder, its parameters, the index's revision and the query.
2. **Retrieve**: exact dense search and BM25, ``top_n`` each, with the filter applied inside
   both, so ``top_n`` counts only what the asker may see.
3. **Fuse**: reciprocal rank fusion with ``k`` (``reciprocal_rank_fusion``), weighted fusion
   (each list's scores min-max normalized, then ``alpha`` * dense + (1 - ``alpha``) * keyword),
   or, with ``none``, the one list as it is.
4. **Candidates**: the best ``candidates`` fused results, checked against the filter once more
   (defense in depth), so a restricted chunk never reaches the reranker.
5. **Rerank** each candidate's title, section and text (``rerank.rerank_text``), each score
   cached by the reranker, its revision, the query and the text; ties keep the fused order.
6. **Final**: the best ``top_k``. Chunks from ``chunk/parent-child`` stand for their parent
   section: the final list holds each parent once, at its best child's rank and score.

The result holds each checkpoint's list with scores (``CHECKPOINTS``) and each step's latency,
so evaluation can say where a passage was lost.

**Visibility** (``Visibility.allows``). An entry is visible on day ``d`` when its status is
allowed (``published`` unless the config says otherwise), its collection is in the config's
(when it names any), the asker is in its ACL's groups or users, and, with ``current_only``,
``effective_from <= d`` and ``d`` is before its end: the earlier of its own ``effective_to`` and
the next published version's ``effective_from`` among the entries with its collection and key
(``visible_until``). Entries without that metadata (a corpus uploaded without a manifest) have
no ACL and no dates: every asker sees them on every day, unless the config names collections. A
search with no asker (the lab's own view of its run) skips the ACL, and one with no day skips
the dates.

Nothing here logs a query or a text; callers get them back in the result only as chunk IDs.
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Any

import numpy as np
from pydantic import BaseModel

from stratafi_components.chunk import label_title
from stratafi_components.embed import EmbedComponent, embed_query
from stratafi_components.index import MANIFEST_PART, DenseIndex, KeywordIndex
from stratafi_components.rerank import RerankComponent, rerank_text
from stratafi_core.artifacts import Chunk, IndexManifest
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.evalsets import Asker
from stratafi_core.specs import Fusion, RetrievalConfig
from stratafi_core.training import apply_adapter

logger = logging.getLogger("stratafi.retrieval")

CHECKPOINTS = ("dense", "keyword", "fused", "candidates", "reranked", "final")
PUBLISHED = "published"
Ranking = list[tuple[str, float]]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


# --- what filters see -------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryMeta:
    """One corpus entry: the metadata a manifest gave it, as filters and the reranker
    see it."""

    document_id: str
    name: str
    doc_key: str | None = None
    version: str | None = None
    collection: str | None = None
    acl_groups: frozenset[str] | None = None  # None: no ACL, so every asker may read it
    acl_users: frozenset[str] = frozenset()
    effective_from: date | None = None
    effective_to: date | None = None  # as the manifest asked; chains may end it sooner
    status: str = PUBLISHED
    title: str | None = None
    media_type: str | None = None

    @classmethod
    def of(
        cls,
        *,
        document_id: str,
        name: str,
        doc_key: str | None = None,
        doc_version: str | None = None,
        collection: str | None = None,
        acl: Mapping[str, Any] | None = None,
        effective_from: date | None = None,
        effective_to: date | None = None,
        metadata: Mapping[str, Any] | None = None,
        media_type: str | None = None,
    ) -> "EntryMeta":
        """From a corpus entry's fields (``stratafi_store.corpora.Entry``)."""
        metadata = metadata or {}
        return cls(
            document_id=document_id,
            name=name,
            doc_key=doc_key,
            version=doc_version,
            collection=collection,
            acl_groups=frozenset(acl.get("groups") or ()) if acl is not None else None,
            acl_users=frozenset(acl.get("users") or ()) if acl is not None else frozenset(),
            effective_from=effective_from,
            effective_to=effective_to,
            status=str(metadata.get("status") or PUBLISHED),
            title=metadata.get("title"),
            media_type=media_type,
        )

    @classmethod
    def from_entry(cls, entry: Any) -> "EntryMeta":
        """From a corpus version's entry (``stratafi_store.corpora.Entry``, read by name)."""
        return cls.of(
            document_id=str(entry.document_id),
            name=entry.name,
            doc_key=entry.doc_key,
            doc_version=entry.doc_version,
            collection=entry.collection,
            acl=entry.acl,
            effective_from=entry.effective_from,
            effective_to=entry.effective_to,
            metadata=entry.metadata,
            media_type=entry.media_type,
        )

    @property
    def label(self) -> str:
        """Its key, or its file's stem: what a chunker titles an untitled document by."""
        return self.doc_key or PurePosixPath(self.name).stem

    @property
    def display_title(self) -> str | None:
        return self.title or label_title(self.label)


def visible_until(entries: Iterable[EntryMeta]) -> dict[str, date | None]:
    """Each entry's end, by name: the first day it's no longer in effect, or None when it never
    ends. An entry ends at its own ``effective_to``, and a published entry with a key and a
    start date also ends where the next published version of its key (in its collection)
    begins: versions ordered by ``effective_from``, then name. So a new version replaces the
    one before it on its first day, a future-dated version leaves the current one in effect
    until then, and a draft never ends a published version."""
    entries = list(entries)
    ends: dict[str, date | None] = {entry.name: entry.effective_to for entry in entries}
    chains: dict[tuple[str | None, str], list[EntryMeta]] = defaultdict(list)
    for entry in entries:
        if (
            entry.status == PUBLISHED
            and entry.doc_key is not None
            and entry.effective_from is not None
        ):
            chains[(entry.collection, entry.doc_key)].append(entry)
    for chain in chains.values():
        chain.sort(key=lambda entry: (entry.effective_from, entry.name))
        for current, following in pairwise(chain):
            own, starts = ends[current.name], following.effective_from
            if starts is not None and (own is None or starts < own):
                ends[current.name] = starts
    return ends


@dataclass(frozen=True)
class Visibility:
    """What one search may see, for an asker on a day. Without an asker the ACL isn't checked;
    without a day, nor are effective dates."""

    day: date | None
    asker: bool
    user: str | None
    groups: frozenset[str]
    collections: frozenset[str] | None
    statuses: frozenset[str]
    current_only: bool

    @classmethod
    def for_asker(
        cls, config: RetrievalConfig, asker: Asker | None, day: date | None
    ) -> "Visibility":
        filters = config.filters
        return cls(
            day=day,
            asker=asker is not None,
            user=asker.user if asker is not None else None,
            groups=frozenset(asker.groups) if asker is not None else frozenset(),
            collections=frozenset(filters.collections) if filters.collections is not None else None,
            statuses=frozenset(filters.statuses or (PUBLISHED,)),
            current_only=filters.current_only and day is not None,
        )

    def allows(self, entry: EntryMeta, end: date | None) -> bool:
        if self.collections is not None and entry.collection not in self.collections:
            return False
        if entry.status not in self.statuses:
            return False
        if self.current_only and self.day is not None:
            if entry.effective_from is not None and self.day < entry.effective_from:
                return False
            if end is not None and self.day >= end:
                return False
        if not self.asker or (entry.acl_groups is None and not entry.acl_users):
            return True
        return (self.user is not None and self.user in entry.acl_users) or not (
            entry.acl_groups or frozenset()
        ).isdisjoint(self.groups)


# --- fusion -----------------------------------------------------------------------------------

RRF_K = 60


def reciprocal_rank_fusion(rankings: Iterable[Sequence[str]], *, k: int = RRF_K) -> Ranking:
    """Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009): an item scores the
    sum, over the lists that hold it, of ``1 / (k + rank)``, ranks counting from 1; an item a
    list repeats counts at its best rank there. Best first; ties go to the smaller ID."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item not in seen:
                seen.add(item)
                scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def _normalized(ranking: Ranking) -> dict[str, float]:
    if not ranking:
        return {}
    scores = [score for _, score in ranking]
    low, high = min(scores), max(scores)
    if high == low:
        return {chunk_id: 1.0 for chunk_id, _ in ranking}
    return {chunk_id: (score - low) / (high - low) for chunk_id, score in ranking}


def weighted_fusion(dense: Ranking | None, keyword: Ranking | None, *, alpha: float) -> Ranking:
    """Min-max normalized scores, ``alpha`` * dense + (1 - ``alpha``) * keyword; a chunk missing
    from a list scores 0 there. Ties go to the smaller ID."""
    dense_scores, keyword_scores = _normalized(dense or []), _normalized(keyword or [])
    fused = {
        chunk_id: alpha * dense_scores.get(chunk_id, 0.0)
        + (1 - alpha) * keyword_scores.get(chunk_id, 0.0)
        for chunk_id in set(dense_scores) | set(keyword_scores)
    }
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


def fuse(fusion: Fusion, *, dense: Ranking | None, keyword: Ranking | None) -> Ranking:
    """The fused list, best first: RRF over the lists' ranks, weighted fusion over their
    scores, or with ``none`` the one list there is."""
    if fusion.method == "rrf":
        lists = [ranking for ranking in (dense, keyword) if ranking is not None]
        return reciprocal_rank_fusion([[chunk_id for chunk_id, _ in r] for r in lists], k=fusion.k)
    if fusion.method == "weighted":
        return weighted_fusion(dense, keyword, alpha=fusion.alpha)
    if dense is not None and keyword is not None:
        raise ComponentError("fusion_required")
    return list(dense if dense is not None else keyword or [])


# --- loading a run's indexes ----------------------------------------------------------------

# read(cache_key, part) → the artifact's payload (part None) or one of its parts.
ArtifactReader = Callable[[str, str | None], bytes]


def read_chunks(data: bytes) -> list[Chunk]:
    """A chunks artifact: one ``stratafi.chunk.v1`` per line."""
    return [Chunk.model_validate_json(line) for line in data.splitlines() if line.strip()]


@dataclass(frozen=True)
class LoadedIndex:
    """An index with the chunks it was built from, by ID."""

    index: DenseIndex | KeywordIndex
    chunks: dict[str, Chunk]


def parent_chunk(child: Chunk) -> Chunk | None:
    """The parent section a ``chunk/parent-child`` child carries, as a chunk of its own."""
    parent = child.metadata.get("parent")
    if not isinstance(parent, dict):
        return None
    path = list(parent.get("heading_path") or [])
    return Chunk(
        chunk_id=str(parent["id"]),
        document_id=child.document_id,
        ordinal=child.ordinal,
        text=str(parent["text"]),
        index_text=str(parent["text"]),
        element_ids=list(parent.get("element_ids") or []),
        page_start=parent.get("page_start"),
        page_end=parent.get("page_end"),
        char_start=int(parent["char_start"]),
        char_end=int(parent["char_end"]),
        heading_path=path,
        metadata={"role": "parent", **({"section": path[-1]} if path else {})},
    )


def with_parents(chunks: dict[str, Chunk]) -> dict[str, Chunk]:
    """``chunks`` plus every parent their children carry, by ID."""
    parents = {}
    for chunk in chunks.values():
        parent = parent_chunk(chunk)
        if parent is not None and parent.chunk_id not in chunks:
            parents.setdefault(parent.chunk_id, parent)
    return {**chunks, **parents}


def collapse_to_parents(ranking: Ranking, chunks: Mapping[str, Chunk]) -> Ranking:
    """Each child replaced by its parent, kept once at its best rank; other chunks as they
    are."""
    seen: set[str] = set()
    collapsed: Ranking = []
    for chunk_id, score in ranking:
        chunk = chunks.get(chunk_id)
        parent = chunk.metadata.get("parent") if chunk is not None else None
        target = str(parent["id"]) if isinstance(parent, dict) else chunk_id
        if target not in seen:
            seen.add(target)
            collapsed.append((target, score))
    return collapsed


def load_index(
    read: ArtifactReader, cache_key: str, kind: str, *, engine: Any = None
) -> LoadedIndex:
    """A dense or keyword index artifact and its chunks, with the parents of parent-child
    chunks. An index that lives in Postgres (``backend: postgres``) searches through
    ``engine``. Raises what ``read`` raises for a missing artifact, and
    ``ComponentError("index_corrupt")`` for a broken one."""
    manifest = IndexManifest.model_validate_json(read(cache_key, MANIFEST_PART))
    payload = read(cache_key, None)
    chunks: dict[str, Chunk] = {}
    for chunks_key in manifest.chunks:
        for chunk in read_chunks(read(chunks_key, None)):
            chunks[chunk.chunk_id] = chunk
    chunks = with_parents(chunks)
    if manifest.backend == "postgres":
        from stratafi_components.pgindex import PgDenseIndex, PgKeywordIndex

        if engine is None:
            raise ComponentError("database_not_configured")
        searcher = PgDenseIndex if kind == "dense" else PgKeywordIndex
        return LoadedIndex(searcher(manifest, engine), chunks)  # type: ignore[arg-type]
    index = (
        DenseIndex.load(manifest, payload)
        if kind == "dense"
        else KeywordIndex.load(manifest, payload)
    )
    return LoadedIndex(index, chunks)


# --- caches -----------------------------------------------------------------------------------


class LruCache:
    """A thread-safe least-recently-used map."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Any:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                self.hits += 1
                return self._items[key]
            self.misses += 1
            return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)


@dataclass
class RetrievalCache:
    """Query vectors and rerank scores, shared by every engine of a process."""

    vectors: LruCache = field(default_factory=lambda: LruCache(4096))
    scores: LruCache = field(default_factory=lambda: LruCache(200_000))


# --- models -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryEmbedder:
    """The run's embedder, for queries: its component, resolved parameters and services."""

    component: EmbedComponent
    params: BaseModel
    context: StageContext

    def key(self, revision: str | None) -> tuple[str, ...]:
        info = self.component.info
        return (info.id, info.version, self.component.params_hash(self.params), revision or "")

    def __call__(self, query: str) -> tuple[np.ndarray, str]:
        vector, revision = embed_query(self.component, self.params, query, self.context)
        self.context.metric("query_embeddings", 1, add=True)
        if self.component.info.egress == "model_gateway":
            self.context.metric("gateway.calls", 1, add=True)
        return vector, revision


TRAINED_MODELS_SERVICE = "trained_models"


@dataclass(frozen=True)
class AdaptedQueryEmbedder(QueryEmbedder):
    """The run's embedder through a trained query adapter: each query's vector becomes
    ``W q``. Its key names the adapter's artifact, so cached query vectors never mix."""

    adapter_key: str = ""
    weights: np.ndarray | None = field(default=None, compare=False, hash=False, repr=False)

    def key(self, revision: str | None) -> tuple[str, ...]:
        return (*super().key(revision), "adapter", self.adapter_key)

    def __call__(self, query: str) -> tuple[np.ndarray, str]:
        vector, revision = super().__call__(query)
        if self.weights is None:
            raise ComponentError("adapter_missing")
        try:
            return apply_adapter(self.weights, vector), revision
        except ValueError:
            raise ComponentError("adapter_mismatch") from None


def with_adapter(
    embedder: QueryEmbedder, model_id: str, *, dense: DenseIndex
) -> AdaptedQueryEmbedder:
    """``embedder`` through the trained adapter ``model_id``. Lab only, and it must fit
    the dense index: the embedding model, revision and dimensions it was trained for."""
    if not embedder.context.services.get("in_process", False):
        raise ComponentError("in_process_disabled")
    model, weights = embedder.context.service(TRAINED_MODELS_SERVICE).adapter(model_id)
    base, manifest = model.base, dense.manifest
    if (
        base.get("model") != manifest.model
        or base.get("revision") != manifest.revision
        or int(weights.shape[0]) != manifest.dims
    ):
        raise ComponentError("adapter_mismatch")
    return AdaptedQueryEmbedder(
        embedder.component,
        embedder.params,
        embedder.context,
        adapter_key=model.model_key,
        weights=weights,
    )


@dataclass(frozen=True)
class Reranker:
    """A reranker with its resolved parameters, services and revision."""

    component: RerankComponent
    params: BaseModel
    context: StageContext
    revision: str | None = None

    @classmethod
    def resolve(
        cls, component: RerankComponent, params: BaseModel, context: StageContext
    ) -> "Reranker":
        return cls(component, params, context, component.revision_for(params, context))

    def key(self) -> tuple[str, ...]:
        info = self.component.info
        return (info.id, info.version, self.component.params_hash(self.params), self.revision or "")


# --- the engine -------------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalResult:
    checkpoints: dict[str, Ranking]
    latency_ms: dict[str, float]
    embedding_revision: str | None
    rerank_revision: str | None
    visible_chunks: int


class RetrievalEngine:
    """Search one run's indexes with one retrieval config (the module docstring says how)."""

    def __init__(
        self,
        config: RetrievalConfig,
        *,
        entries: Sequence[EntryMeta],
        chunks: Mapping[str, Chunk],
        dense: DenseIndex | None = None,
        keyword: KeywordIndex | None = None,
        embedder: QueryEmbedder | None = None,
        reranker: Reranker | None = None,
        cache: RetrievalCache | None = None,
    ) -> None:
        if config.retrievers.dense is not None and (dense is None or embedder is None):
            raise ComponentError("dense_index_missing")
        if config.retrievers.keyword is not None and keyword is None:
            raise ComponentError("keyword_index_missing")
        self.config = config
        self.chunks = chunks
        self.dense = dense if config.retrievers.dense is not None else None
        self.keyword = keyword if config.retrievers.keyword is not None else None
        self.embedder = embedder
        self.reranker = reranker
        self.cache = cache or RetrievalCache()
        # A document under two names (scan/dedupe kept the first) reads as its first entry.
        self.entries: dict[str, EntryMeta] = {}
        for entry in sorted(entries, key=lambda item: item.name):
            self.entries.setdefault(entry.document_id, entry)
        self.ends = visible_until(entries)
        self._masks: dict[Visibility, tuple[frozenset[str], dict[str, np.ndarray]]] = {}
        self._lock = threading.Lock()

    @property
    def rerank_revision(self) -> str | None:
        return self.reranker.revision if self.reranker is not None else None

    def entry_of(self, chunk_id: str) -> EntryMeta | None:
        chunk = self.chunks.get(chunk_id)
        return self.entries.get(chunk.document_id) if chunk is not None else None

    def rerank_text(self, chunk_id: str) -> str:
        chunk = self.chunks[chunk_id]
        entry = self.entries.get(chunk.document_id)
        return rerank_text(entry.display_title if entry else None, section_of(chunk), chunk.text)

    def _allowed(self, visibility: Visibility) -> tuple[frozenset[str], dict[str, np.ndarray]]:
        """The documents this visibility allows, and each index's rows it allows."""
        with self._lock:
            if visibility in self._masks:
                return self._masks[visibility]
        documents = frozenset(
            document_id
            for document_id, entry in self.entries.items()
            if visibility.allows(entry, self.ends.get(entry.name))
        )
        masks = {}
        for kind, index in (("dense", self.dense), ("keyword", self.keyword)):
            if index is not None:
                rows = index.manifest.document_ids
                masks[kind] = np.fromiter((d in documents for d in rows), bool, len(rows))
        with self._lock:
            if len(self._masks) >= 64:
                self._masks.clear()
            self._masks[visibility] = (documents, masks)
        return documents, masks

    def _embed(self, query: str) -> tuple[np.ndarray, str]:
        if self.embedder is None or self.dense is None:
            raise ComponentError("dense_index_missing")
        key = (*self.embedder.key(self.dense.manifest.revision), _sha(query))
        cached = self.cache.vectors.get(key)
        if cached is not None:
            return cached
        vector, revision = self.embedder(query)
        self.cache.vectors.put(key, (vector, revision))
        return vector, revision

    def _rerank(self, query: str, candidates: Ranking) -> Ranking:
        if self.reranker is None or not self.reranker.component.reorders or not candidates:
            return list(candidates)
        texts = [self.rerank_text(chunk_id) for chunk_id, _ in candidates]
        prefix = (*self.reranker.key(), _sha(query))
        if not self.reranker.component.pointwise:
            # A listwise score depends on the whole list: cache the list.
            key = (*prefix, _sha("\x1f".join(texts)))
            listed = self.cache.scores.get(key)
            if listed is None:
                listed = self.reranker.component.rerank(
                    query,
                    texts,
                    self.reranker.params,
                    self.reranker.context,
                    expected_revision=self.reranker.revision,
                )
                if listed is None:
                    return list(candidates)
                self.cache.scores.put(key, listed)
            order = sorted(range(len(candidates)), key=lambda i: (-float(listed[i]), i))
            return [(candidates[i][0], float(listed[i])) for i in order]
        keys = [(*prefix, _sha(text)) for text in texts]
        scores: list[float | None] = [self.cache.scores.get(key) for key in keys]
        missing = [index for index, score in enumerate(scores) if score is None]
        if missing:
            fresh = self.reranker.component.rerank(
                query,
                [texts[index] for index in missing],
                self.reranker.params,
                self.reranker.context,
                expected_revision=self.reranker.revision,
            )
            if fresh is None:
                return list(candidates)
            for index, score in zip(missing, fresh, strict=True):
                scores[index] = score
                self.cache.scores.put(keys[index], score)
        order = sorted(
            range(len(candidates)), key=lambda index: (-float(scores[index] or 0), index)
        )
        return [(candidates[index][0], float(scores[index] or 0)) for index in order]

    def search(self, query: str, *, asker: Asker | None, day: date | None) -> RetrievalResult:
        started = time.perf_counter()
        config = self.config
        visibility = Visibility.for_asker(config, asker, day)
        documents, masks = self._allowed(visibility)
        checkpoints: dict[str, Ranking] = {}
        latency: dict[str, float] = {}
        revision = None
        if self.keyword is not None and config.retrievers.keyword is not None:
            step = time.perf_counter()
            checkpoints["keyword"] = self.keyword.search(
                query, config.retrievers.keyword.top_n, allowed=masks["keyword"]
            )
            latency["keyword_ms"] = _ms(step)
        if self.dense is not None and config.retrievers.dense is not None:
            step = time.perf_counter()
            vector, revision = self._embed(query)
            latency["embed_ms"] = _ms(step)
            step = time.perf_counter()
            checkpoints["dense"] = self.dense.search(
                vector, config.retrievers.dense.top_n, revision=revision, allowed=masks["dense"]
            )
            latency["dense_ms"] = _ms(step)
        step = time.perf_counter()
        fused = fuse(
            config.fusion, dense=checkpoints.get("dense"), keyword=checkpoints.get("keyword")
        )
        checkpoints["fused"] = fused
        # Defense in depth: nothing outside the filter reaches the reranker or the caller.
        candidates = []
        for chunk_id, score in fused[: config.candidates]:
            chunk = self.chunks.get(chunk_id)
            if chunk is not None and chunk.document_id in documents:
                candidates.append((chunk_id, score))
        if dropped := min(len(fused), config.candidates) - len(candidates):
            logger.error("retrievers returned chunks outside the filter", extra={"count": dropped})
        checkpoints["candidates"] = candidates
        latency["fusion_ms"] = _ms(step)
        step = time.perf_counter()
        reranked = self._rerank(query, candidates)
        latency["rerank_ms"] = _ms(step)
        checkpoints["reranked"] = reranked
        checkpoints["final"] = collapse_to_parents(reranked, self.chunks)[: config.top_k]
        latency["total_ms"] = _ms(started)
        visible = int(next(iter(masks.values())).sum()) if masks else 0
        return RetrievalResult(
            checkpoints, latency, revision, self.rerank_revision, visible_chunks=visible
        )


def section_of(chunk: Chunk) -> str | None:
    """A chunk's section heading: what the chunker recorded, else its heading path's last."""
    section = chunk.metadata.get("section")
    if isinstance(section, str) and section:
        return section
    return chunk.heading_path[-1] if chunk.heading_path else None
