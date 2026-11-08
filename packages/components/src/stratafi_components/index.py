"""The index stage: corpus-level indexes over every embedded document of a run.

- ``index/exact`` stacks the run's vectors into one float32 matrix and searches it by brute-force
  cosine (or dot product) in numpy: no ANN error in model comparisons. It refuses vectors from
  more than one embedding revision, or of more than one dimension: such an index would compare
  vectors that don't share a space.
- ``index/bm25`` holds each chunk's term counts over its ``index_text`` (``keywords.terms``:
  Unicode words, case-folded, the Snowball English stop list and stemmer) and scores a query by
  Okapi BM25 (k1 = 1.2, b = 0.75 by default).

An index artifact is its payload (the matrix as ``.npy``, or the term statistics as JSON) with
an ``index`` part beside it: its ``stratafi.index.v1`` manifest. Searching returns
``(chunk_id, score)`` pairs, best first; ties go to the smaller chunk ID.
"""

import io
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.embed import Vectors, normalize_rows
from stratafi_components.keywords import ANALYZER, BM25_B, BM25_K1, bm25_idf, terms
from stratafi_components.vectors import binarize
from stratafi_core.artifacts import Chunk, IndexManifest
from stratafi_core.components import Component, ComponentError, ComponentInfo, StageContext

MANIFEST_PART = "index"


@dataclass(frozen=True)
class IndexInput:
    """One embedded document: its chunks artifact's key, its chunks, and its vectors."""

    chunks_key: str
    chunks: list[Chunk]
    vectors: Vectors | None = None


@dataclass(frozen=True)
class BuiltIndex:
    manifest: IndexManifest
    payload: bytes
    media_type: str


class IndexComponent(Component):
    kind: ClassVar[Literal["dense", "keyword"]]

    def build(
        self, inputs: list[IndexInput], params: BaseModel, context: StageContext
    ) -> BuiltIndex:
        raise NotImplementedError


def _manifest(
    component: IndexComponent, params: BaseModel, inputs: list[IndexInput], **extra: Any
) -> IndexManifest:
    return IndexManifest(
        kind=component.kind,
        component=component.info.id,
        version=component.info.version,
        params=component.dump_params(params),
        chunk_ids=[chunk.chunk_id for item in inputs for chunk in item.chunks],
        document_ids=[chunk.document_id for item in inputs for chunk in item.chunks],
        chunks=[item.chunks_key for item in inputs],
        **extra,
    )


def _top(
    chunk_ids: list[str], scores: np.ndarray, top_n: int, allowed: np.ndarray | None = None
) -> list[tuple[str, float]]:
    rows = range(len(chunk_ids)) if allowed is None else np.flatnonzero(allowed).tolist()
    order = sorted(rows, key=lambda i: (-float(scores[i]), chunk_ids[i]))
    return [(chunk_ids[i], float(scores[i])) for i in order[:top_n]]


# --- index/exact -------------------------------------------------------------------------------


class ExactParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: Literal["cosine", "dot"] = "cosine"
    rescore: bool = Field(
        default=True,
        description="Binary vectors: score with the float query (rescoring) or its signs only",
    )


class ExactIndex(IndexComponent):
    kind = "dense"
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="index/exact",
        stage="index",
        version="1.0.0",
        title="Exact vectors",
        description="Brute-force cosine or dot product in numpy over every vector: no ANN "
        "error in comparisons. Refuses vectors from mixed embedding revisions.",
        license="MIT (numpy: BSD-3-Clause)",
        baseline=True,
    )
    Params: ClassVar[type[BaseModel]] = ExactParams

    def build(
        self, inputs: list[IndexInput], params: BaseModel, context: StageContext
    ) -> BuiltIndex:
        params = cast(ExactParams, params)
        vectors = [item.vectors for item in inputs if item.vectors is not None]
        if len(vectors) != len(inputs) or not vectors:
            raise ComponentError("vectors_missing")
        identities = {(v.sidecar.model, v.sidecar.revision) for v in vectors}
        if len(identities) != 1:
            raise ComponentError("mixed_revisions")
        if len({v.sidecar.dims for v in vectors}) != 1:
            raise ComponentError("mixed_dimensions")
        for item, item_vectors in zip(inputs, vectors, strict=True):
            if item_vectors.sidecar.chunk_ids != [chunk.chunk_id for chunk in item.chunks]:
                raise ComponentError("vectors_out_of_step")
        matrix = np.vstack([v.matrix for v in vectors]).astype(np.float32)
        if params.metric == "cosine":
            matrix = normalize_rows(matrix)
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        first = vectors[0].sidecar
        context.metric("vectors", int(matrix.shape[0]))
        context.metric("index_bytes", buffer.tell())
        return BuiltIndex(
            _manifest(
                self,
                params,
                inputs,
                model=first.model,
                revision=first.revision,
                dims=first.dims,
                metric=params.metric,
                dtype=first.dtype,
            ),
            buffer.getvalue(),
            "application/x-npy",
        )


@dataclass
class DenseIndex:
    manifest: IndexManifest
    matrix: np.ndarray

    @classmethod
    def load(cls, manifest: IndexManifest, payload: bytes) -> "DenseIndex":
        matrix = np.load(io.BytesIO(payload), allow_pickle=False)
        if matrix.shape != (len(manifest.chunk_ids), manifest.dims):
            raise ComponentError("index_corrupt")
        return cls(manifest, matrix)

    def search(
        self,
        query: np.ndarray,
        top_n: int,
        *,
        revision: str,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        """The ``top_n`` nearest chunks to ``query`` (one vector). The query must come from the
        index's embedding revision. ``allowed`` (a boolean per row) filters inside the search,
        so ``top_n`` counts only allowed chunks."""
        if revision != self.manifest.revision:
            raise ComponentError("revision_mismatch")
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.manifest.dims:
            raise ComponentError("dimension_mismatch")
        if self.manifest.dtype == "binary" and not self.manifest.params.get("rescore", True):
            vector = binarize(vector)
        if self.manifest.metric == "cosine":
            norm = float(np.linalg.norm(vector)) or 1.0
            vector = vector / norm
        scores = self.matrix @ vector
        return _top(self.manifest.chunk_ids, scores, top_n, allowed)


# --- index/bm25 --------------------------------------------------------------------------------


class Bm25Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    k1: float = Field(default=BM25_K1, ge=0, le=10)
    b: float = Field(default=BM25_B, ge=0, le=1)


class Bm25Index(IndexComponent):
    kind = "keyword"
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="index/bm25",
        stage="index",
        version="2.0.0",
        title="BM25",
        description="In-process Okapi BM25 over each chunk's index text: Unicode words, "
        "case-folded, without Snowball's English stop words, Snowball-stemmed.",
        license="MIT (snowballstemmer: BSD-3-Clause)",
        baseline=True,
    )
    Params: ClassVar[type[BaseModel]] = Bm25Params

    def build(
        self, inputs: list[IndexInput], params: BaseModel, context: StageContext
    ) -> BuiltIndex:
        params = cast(Bm25Params, params)
        counts = []
        for item in inputs:
            context.checkpoint()
            counts += [dict(Counter(terms(chunk.index_text))) for chunk in item.chunks]
        payload = json.dumps(
            {"analyzer": ANALYZER, "k1": params.k1, "b": params.b, "counts": counts}
        ).encode()
        context.metric("chunks", len(counts))
        context.metric("index_bytes", len(payload))
        return BuiltIndex(_manifest(self, params, inputs), payload, "application/json")


@dataclass
class KeywordIndex:
    manifest: IndexManifest
    k1: float
    b: float
    counts: list[dict[str, int]]
    lengths: list[int] = field(default_factory=list)
    frequency: Counter[str] = field(default_factory=Counter)

    @classmethod
    def load(cls, manifest: IndexManifest, payload: bytes) -> "KeywordIndex":
        data = json.loads(payload)
        index = cls(manifest, float(data["k1"]), float(data["b"]), data["counts"])
        if len(index.counts) != len(manifest.chunk_ids):
            raise ComponentError("index_corrupt")
        index.lengths = [sum(counts.values()) for counts in index.counts]
        for counts in index.counts:
            index.frequency.update(counts.keys())
        return index

    def search(
        self, query: str, top_n: int, *, allowed: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        """The ``top_n`` chunks by BM25 score, best first, scoring each distinct query term once.
        ``allowed`` (a boolean per row) filters inside the search: the term statistics still
        cover every chunk, and only allowed chunks are scored. Chunks that share no term with
        the query aren't returned."""
        query_terms = list(dict.fromkeys(terms(query)))
        total = len(self.counts)
        if not total or not query_terms:
            return []
        average = sum(self.lengths) / total or 1.0
        idf = {term: bm25_idf(total, self.frequency[term]) for term in query_terms}
        scored = []
        for row, (chunk_id, counts, length) in enumerate(
            zip(self.manifest.chunk_ids, self.counts, self.lengths, strict=True)
        ):
            if allowed is not None and not allowed[row]:
                continue
            norm = self.k1 * (1 - self.b + self.b * length / average)
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency:
                    score += idf[term] * frequency * (self.k1 + 1) / (frequency + norm)
            if score > 0:
                scored.append((chunk_id, score))
        scored.sort(key=lambda hit: (-hit[1], hit[0]))
        return scored[:top_n]


COMPONENTS: list[type[Component]] = [ExactIndex, Bm25Index]
