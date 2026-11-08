"""The embed stage.

An embedder turns a document's chunks into a vectors artifact: a float32 ``.npy`` array
``[n, d]``, one row per chunk, and its sidecar (``stratafi.vectors.v1``) naming the chunks, the
model and its revision. What is embedded is each chunk's ``index_text``, with the embedder's
document prefix in front (nomic and e5 need one; many models need none).

- ``embed/gateway`` calls the model gateway by alias (``stratafi_components.models``). A run
  resolves the alias's revision when it starts (``revision_for``); the vectors must come back
  with that revision, or the task fails with ``revision_changed`` rather than mix them.
- ``embed/fake-hash`` makes deterministic vectors from hashed words, in process
  (``stratafi_core.fake_models.hashed_vector``, which the fake gateway uses too), so texts that
  share words score higher. For tests and offline development.

Every embedder takes the embed options (``stratafi_components.vectors``): Matryoshka
``dimensions``, ``normalize``, a stored ``dtype`` (float32, float16, int8, binary) and an input
limit with its ``truncation`` policy. ``embed/sentence-transformers`` and
``embed/openai-compatible`` live in ``stratafi_components.embedders``.
"""

import io
import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from stratafi_components.estimators import CHARS, EstimatorName, counter
from stratafi_components.models import GATEWAY, GatewayProvider
from stratafi_components.vectors import (
    Dtype,
    dequantize,
    limit_inputs,
    normalize_rows,  # noqa: F401 - index.py and callers import it from here
    quantize,
    shape,
)
from stratafi_core.artifacts import Chunk, VectorsSidecar
from stratafi_core.components import Component, ComponentError, ComponentInfo, StageContext
from stratafi_core.fake_models import EMBED_REVISION, hashed_vector

MODELS_SERVICE = "models"


@dataclass(frozen=True)
class Vectors:
    matrix: np.ndarray  # float32 [n, d]
    sidecar: VectorsSidecar


def vectors_to_bytes(vectors: Vectors) -> tuple[bytes, bytes]:
    """The ``.npy`` payload (in the sidecar's dtype) and the sidecar's JSON."""
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(vectors.matrix), allow_pickle=False)
    sidecar = json.dumps(vectors.sidecar.model_dump(mode="json", by_alias=True)).encode()
    return buffer.getvalue(), sidecar


def vectors_from_bytes(payload: bytes, sidecar: bytes) -> Vectors:
    """A vectors artifact, read back as float32 whatever it was stored as."""
    stored = np.load(io.BytesIO(payload), allow_pickle=False)
    meta = VectorsSidecar.model_validate_json(sidecar)
    if stored.ndim != 2 or stored.shape[0] != len(meta.chunk_ids):
        raise ComponentError("vectors_corrupt")
    matrix = dequantize(stored, meta.dtype, meta.dims)
    if matrix.shape != (len(meta.chunk_ids), meta.dims):
        raise ComponentError("vectors_corrupt")
    return Vectors(matrix, meta)


def stored_bytes(vectors: Vectors) -> int:
    return int(np.asarray(vectors.matrix).nbytes)


def hash_embedding(text: str, dimensions: int) -> list[float]:
    """``embed/fake-hash``'s vector for a text: a unit vector from hashed words, the same one
    the fake gateway returns (``stratafi_core.fake_models``)."""
    return hashed_vector(text, dimensions)


class EmbedComponent(Component):
    """An embedder: chunks → vectors."""

    def revision_for(self, params: BaseModel, context: StageContext) -> str:
        """The model revision the vectors will carry; part of the cache key."""
        raise NotImplementedError

    def model_name(self, params: BaseModel) -> str:
        raise NotImplementedError

    def embed(
        self,
        chunks: list[Chunk],
        params: BaseModel,
        *,
        expected_revision: str | None,
        context: StageContext,
    ) -> Vectors:
        raise NotImplementedError


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=32, ge=1, le=2048, description="Texts per request")
    normalize: bool = Field(default=False, description="Scale each vector to length 1")
    document_prefix: str = Field(
        default="", max_length=64, description="Put before each text (e.g. search_document: )"
    )
    query_prefix: str = Field(
        default="", max_length=64, description="Put before each query at search time"
    )
    dimensions: int | None = Field(
        default=None, ge=8, le=8192, description="Keep the first n values (Matryoshka)"
    )
    dtype: Dtype = Field(
        default="float32", description="How vectors are stored: float32, float16, int8, binary"
    )
    max_input_tokens: int | None = Field(
        default=None, ge=8, le=1_000_000, description="The model's input limit, for the check"
    )
    truncation: Literal["report", "truncate", "error"] = Field(
        default="report",
        description="Over the limit: report (the model cuts silently), truncate first, or fail",
    )
    token_estimator: EstimatorName = Field(
        default=CHARS, description="Counts tokens for the limit; hf:<model> is exact"
    )

    @model_validator(mode="after")
    def _int8_needs_unit_vectors(self) -> "_Params":
        if self.dtype == "int8" and not self.normalize:
            raise ValueError("int8 vectors need normalize: their scale assumes length 1")
        return self


def prepare_texts(
    chunks: list[Chunk], params: "_Params", context: StageContext, count=None
) -> list[str]:
    """Each chunk's index text with the document prefix, checked against the input limit."""
    texts = [params.document_prefix + chunk.index_text for chunk in chunks]
    texts, over = limit_inputs(
        texts,
        max_tokens=params.max_input_tokens,
        policy=params.truncation,
        count=count or counter(params.token_estimator),
    )
    context.metric("truncated_inputs", over)
    return texts


def finish_vectors(
    matrix: np.ndarray,
    chunks: list[Chunk],
    params: "_Params",
    *,
    component: str,
    model: str,
    revision: str,
    served_model: str | None = None,
    tokens: int = 0,
    truncated: int = 0,
) -> Vectors:
    """The model's float vectors shaped and stored as the options say."""
    full = int(matrix.shape[1])
    shaped = shape(matrix, dimensions=params.dimensions, normalize=params.normalize)
    stored, _ = quantize(shaped, params.dtype)
    return Vectors(
        stored,
        VectorsSidecar(
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            component=component,
            model=model,
            served_model=served_model,
            revision=revision,
            dims=int(shaped.shape[1]),
            full_dims=full,
            normalized=params.normalize,
            dtype=params.dtype,
            prefix=params.document_prefix,
            tokens=tokens,
            truncated_inputs=truncated,
        ),
    )


class GatewayEmbedParams(_Params):
    alias: str = Field(
        default="embed-default",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$",
        description="A model the gateway serves, by alias or name: embed-default is "
        "nomic-embed-text on the local gateway, a 384-dimension fake on the fake gateway",
    )


class GatewayEmbed(EmbedComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="embed/gateway",
        stage="embed",
        version="1.0.0",
        title="Model gateway",
        description="POST /v1/embeddings on the model gateway, by alias, in batches with "
        "retries and rate limits. Records X-Model-Revision.",
        resource_class="io",
        license="MIT (models: per alias)",
        egress="model_gateway",
        baseline=True,
    )
    Params: ClassVar[type[BaseModel]] = GatewayEmbedParams

    def model_name(self, params: BaseModel) -> str:
        return cast(GatewayEmbedParams, params).alias

    def revision_for(self, params: BaseModel, context: StageContext) -> str:
        provider: GatewayProvider = context.service(MODELS_SERVICE)
        return provider.revision(cast(GatewayEmbedParams, params).alias)

    def embed(
        self,
        chunks: list[Chunk],
        params: BaseModel,
        *,
        expected_revision: str | None,
        context: StageContext,
    ) -> Vectors:
        params = cast(GatewayEmbedParams, params)
        if not chunks:
            raise ComponentError("empty_input")
        provider: GatewayProvider = context.service(MODELS_SERVICE)
        texts = prepare_texts(chunks, params, context)
        result = provider.embed(
            texts,
            alias=params.alias,
            batch_size=params.batch_size,
            slot=lambda tokens: context.provider(GATEWAY, tokens=tokens),
            checkpoint=context.checkpoint,
        )
        if expected_revision is not None and result.revision != expected_revision:
            raise ComponentError("revision_changed")
        context.metric("chunks", len(chunks))
        context.metric("tokens", result.tokens)
        context.metric("gateway.calls", math.ceil(len(chunks) / params.batch_size))
        return finish_vectors(
            result.vectors,
            chunks,
            params,
            component=self.info.id,
            model=params.alias,
            revision=result.revision,
            served_model=result.served_model,
            tokens=result.tokens,
            truncated=int(context.metrics.get("truncated_inputs", 0)),
        )


class FakeHashParams(_Params):
    dimensions: int = Field(default=384, ge=8, le=4096, description="The hash's dimensions")
    truncate_to: int | None = Field(
        default=None, ge=8, le=4096, description="Keep the first n values (Matryoshka)"
    )
    normalize: bool = Field(default=True, description="Hash vectors are unit length already")


class FakeHashEmbed(EmbedComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="embed/fake-hash",
        stage="embed",
        version="2.0.0",
        title="Hashed words (fake)",
        description="Deterministic vectors from hashed words, in process: signed feature "
        "hashing, the fake gateway's algorithm too. For tests and offline development; texts "
        "that share words score higher.",
        license="MIT",
        lab_only=True,
    )
    Params: ClassVar[type[BaseModel]] = FakeHashParams

    def model_name(self, params: BaseModel) -> str:
        return f"fake-hash-{cast(FakeHashParams, params).dimensions}"

    def revision_for(self, params: BaseModel, context: StageContext) -> str:
        return f"{EMBED_REVISION}-{cast(FakeHashParams, params).dimensions}"

    def embed(
        self,
        chunks: list[Chunk],
        params: BaseModel,
        *,
        expected_revision: str | None,
        context: StageContext,
    ) -> Vectors:
        params = cast(FakeHashParams, params)
        if not chunks:
            raise ComponentError("empty_input")
        texts = prepare_texts(chunks, params, context)
        rows = [hash_embedding(text, params.dimensions) for text in texts]
        matrix = np.asarray(rows, dtype=np.float32)
        context.metric("chunks", len(chunks))
        return finish_vectors(
            matrix,
            chunks,
            params.model_copy(update={"dimensions": params.truncate_to}),
            component=self.info.id,
            model=self.model_name(params),
            revision=self.revision_for(params, context),
            truncated=int(context.metrics.get("truncated_inputs", 0)),
        )


def embed_query(
    component: EmbedComponent, params: BaseModel, query: str, context: StageContext
) -> tuple[np.ndarray, str]:
    """A query's vector with the embedder's query prefix, shaped as its documents were
    (truncation, normalization; never quantized), and its revision."""
    options = cast(_Params, params)
    prefix = options.query_prefix
    if isinstance(component, FakeHashEmbed):
        fake = cast(FakeHashParams, params)
        vector = np.asarray([hash_embedding(prefix + query, fake.dimensions)], dtype=np.float32)
        shaped = shape(vector, dimensions=fake.truncate_to, normalize=fake.normalize)
        return shaped, component.revision_for(params, context)
    query_vector = getattr(component, "embed_query_text", None)
    if query_vector is not None:
        vector, revision = query_vector(prefix + query, params, context)
    else:
        provider: GatewayProvider = context.service(MODELS_SERVICE)
        result = provider.embed([prefix + query], alias=component.model_name(params), batch_size=1)
        vector, revision = result.vectors, result.revision
    return shape(vector, dimensions=options.dimensions, normalize=options.normalize), revision


COMPONENTS: list[type[Component]] = [GatewayEmbed, FakeHashEmbed]


def describe(vectors: Vectors) -> dict[str, Any]:
    """A vectors artifact's summary: counts and model metadata, no text."""
    meta = vectors.sidecar
    matrix = dequantize(np.asarray(vectors.matrix), meta.dtype, meta.dims)
    norms = np.linalg.norm(matrix, axis=1) if len(matrix) else np.zeros(0)
    return {
        "chunks": len(meta.chunk_ids),
        "dims": meta.dims,
        "full_dims": meta.full_dims,
        "dtype": meta.dtype,
        "bytes": stored_bytes(vectors),
        "model": meta.model,
        "served_model": meta.served_model,
        "revision": meta.revision,
        "truncated_inputs": meta.truncated_inputs,
        "norm_min": round(float(norms.min()), 6) if len(norms) else None,
        "norm_max": round(float(norms.max()), 6) if len(norms) else None,
    }
