"""More embedders and their model catalog.

- ``embed/sentence-transformers`` runs a model in process with sentence-transformers (Apache-2.0,
  the ``sentence-transformers`` extra). **Lab only**: it refuses when ``models.in_process`` is
  off. The model must be in the catalog below, which records each one's license, dimensions,
  input limit, prefixes and Matryoshka support; a model whose license isn't permissive needs
  ``licenses.allowed``. Inputs are checked against the model's own tokenizer, so the
  truncation count is exact. ``trained/<model ID>`` names a model the lab fine-tuned: the
  registry is asked on every use, its base's catalog entry gives the prefixes, limits and
  license, and its artifact's key is its revision.
- ``embed/openai-compatible`` calls ``POST /v1/embeddings`` on an endpoint the configuration
  names (vLLM, Ollama, TEI or a hosted API) under ``providers.endpoints``, and refuses unless
  ``providers.allowed`` names it: it sends document text out.

Models download from Hugging Face at first use (into its cache); a lab without that network
runs the gateway or the fake embedder instead.
"""

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import httpx2
import numpy as np
from pydantic import BaseModel, Field

from stratafi_components.embed import (
    EmbedComponent,
    Vectors,
    _Params,
    finish_vectors,
    prepare_texts,
)
from stratafi_components.parsers.hosted import ALLOWED_SERVICE, ProviderAccess
from stratafi_core.artifacts import Chunk
from stratafi_core.components import ComponentError, ComponentInfo, StageContext

IN_PROCESS_SERVICE = "in_process"
TRAINED_MODELS_SERVICE = "trained_models"
# A model Stratafi trained: ``trained/<model ID>``, its base a catalog model.
TRAINED_PREFIX = "trained/"
PERMISSIVE = frozenset({"Apache-2.0", "MIT", "BSD-3-Clause"})
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class CatalogModel:
    model: str  # the Hugging Face ID
    title: str
    license: str
    dims: int
    max_tokens: int
    query_prefix: str = ""
    document_prefix: str = ""
    matryoshka: tuple[int, ...] = ()
    trust_remote_code: bool = False


CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel(
        "sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2", "Apache-2.0", 384, 256
    ),
    CatalogModel("BAAI/bge-small-en-v1.5", "BGE small v1.5", "MIT", 384, 512, QUERY_INSTRUCTION),
    CatalogModel("BAAI/bge-base-en-v1.5", "BGE base v1.5", "MIT", 768, 512, QUERY_INSTRUCTION),
    CatalogModel("BAAI/bge-large-en-v1.5", "BGE large v1.5", "MIT", 1024, 512, QUERY_INSTRUCTION),
    CatalogModel("intfloat/e5-small-v2", "E5 small v2", "MIT", 384, 512, "query: ", "passage: "),
    CatalogModel("intfloat/e5-base-v2", "E5 base v2", "MIT", 768, 512, "query: ", "passage: "),
    CatalogModel("thenlper/gte-small", "GTE small", "MIT", 384, 512),
    CatalogModel("thenlper/gte-base", "GTE base", "MIT", 768, 512),
    CatalogModel(
        "nomic-ai/nomic-embed-text-v1.5",
        "Nomic Embed v1.5",
        "Apache-2.0",
        768,
        8192,
        "search_query: ",
        "search_document: ",
        (512, 256, 128, 64),
        trust_remote_code=True,
    ),
    CatalogModel(
        "mixedbread-ai/mxbai-embed-large-v1",
        "mxbai embed large v1",
        "Apache-2.0",
        1024,
        512,
        QUERY_INSTRUCTION,
        matryoshka=(512, 256),
    ),
    CatalogModel(
        "Snowflake/snowflake-arctic-embed-m-v1.5",
        "Arctic Embed M v1.5",
        "Apache-2.0",
        768,
        512,
        QUERY_INSTRUCTION,
        matryoshka=(256,),
    ),
)
BY_MODEL = {entry.model: entry for entry in CATALOG}


def catalog_entry(model: str) -> CatalogModel:
    entry = BY_MODEL.get(model)
    if entry is None:
        raise ComponentError("model_not_cataloged")
    return entry


def is_trained(model: str) -> bool:
    return model.startswith(TRAINED_PREFIX)


def trained_model(model: str, kind: str, context: StageContext) -> Any:
    """The registry's row for ``trained/<model ID>``, ready to use; a
    ``ComponentError`` says why not."""
    models = context.service(TRAINED_MODELS_SERVICE)
    return models.ready(model.removeprefix(TRAINED_PREFIX), kind)


def load_trained(model: str, kind: str, context: StageContext, load: Any) -> Any:
    """Load a trained model with ``load(directory)`` from its files, written to a temporary
    directory that's deleted once it's loaded: no copy outlives a purge on disk."""
    models = context.service(TRAINED_MODELS_SERVICE)
    with tempfile.TemporaryDirectory(prefix="stratafi-model-", ignore_cleanup_errors=True) as tmp:
        models.materialize(model.removeprefix(TRAINED_PREFIX), kind, Path(tmp))
        try:
            return load(tmp)
        except Exception:
            raise ComponentError("model_unavailable") from None


# --- embed/sentence-transformers -------------------------------------------------------------


class SentenceTransformersParams(_Params):
    model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$",
        description="A model from the catalog (Hugging Face ID), or trained/<model ID>",
    )
    revision: str = Field(
        default="main", pattern=r"^[A-Za-z0-9._-]{1,64}$", description="A branch, tag or commit"
    )
    catalog_prefixes: bool = Field(
        default=True, description="Use the catalog's prefixes when none are given"
    )
    normalize: bool = Field(default=True, description="Scale each vector to length 1")


class SentenceTransformersEmbed(EmbedComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="embed/sentence-transformers",
        stage="embed",
        version="1.0.0",
        title="Sentence Transformers (in process)",
        description="A catalog model (BGE, E5, GTE, Nomic, mxbai, Arctic, MiniLM) in process, lab "
        "only: refuses when models.in_process is off. Exact truncation counts from the model's "
        "tokenizer.",
        resource_class="gpu",
        license="Apache-2.0 (models: per catalog entry)",
        lab_only=True,
        services=("model-files",),
    )
    Params: ClassVar[type[BaseModel]] = SentenceTransformersParams
    requires: ClassVar[tuple[str, ...]] = ("sentence_transformers",)
    extra: ClassVar[str | None] = "sentence-transformers"
    _models: ClassVar[dict[tuple[str, str], Any]] = {}

    def model_name(self, params: BaseModel) -> str:
        return cast(SentenceTransformersParams, params).model

    def revision_for(self, params: BaseModel, context: StageContext) -> str:
        params = cast(SentenceTransformersParams, params)
        if is_trained(params.model):
            # A trained model's files never change: its artifact's key is its revision.
            trained = trained_model(params.model, "embedder", context)
            return f"{params.model}@{trained.model_key[:16]}"
        return f"{params.model}@{params.revision}"[:128]

    def _check(self, params: SentenceTransformersParams, context: StageContext) -> CatalogModel:
        if not context.services.get(IN_PROCESS_SERVICE, False):
            raise ComponentError("in_process_disabled")
        if self.unavailable() is not None:
            raise ComponentError("extra_missing")
        if is_trained(params.model):
            trained = trained_model(params.model, "embedder", context)
            entry = catalog_entry(str(trained.base.get("model")))
        else:
            entry = catalog_entry(params.model)
        if entry.license not in PERMISSIVE and entry.license not in context.services.get(
            "licenses", ()
        ):
            raise ComponentError("license_not_allowed")
        return entry

    def _model(
        self, params: SentenceTransformersParams, entry: CatalogModel, context: StageContext
    ) -> Any:
        from sentence_transformers import SentenceTransformer

        if is_trained(params.model):
            trained = trained_model(params.model, "embedder", context)
            key = (params.model, trained.model_key)
            if key not in self._models:
                self._models[key] = load_trained(
                    params.model,
                    "embedder",
                    context,
                    lambda path: SentenceTransformer(
                        path, device="cpu", trust_remote_code=entry.trust_remote_code
                    ),
                )
            return self._models[key]
        key = (params.model, params.revision)
        if key not in self._models:
            try:
                self._models[key] = SentenceTransformer(
                    params.model,
                    revision=params.revision,
                    device="cpu",
                    trust_remote_code=entry.trust_remote_code,
                )
            except Exception:
                raise ComponentError("model_unavailable") from None
        return self._models[key]

    def _prefixed(self, params: SentenceTransformersParams, entry: CatalogModel) -> _Params:
        if not params.catalog_prefixes:
            return params
        return params.model_copy(
            update={
                "document_prefix": params.document_prefix or entry.document_prefix,
                "query_prefix": params.query_prefix or entry.query_prefix,
                "max_input_tokens": params.max_input_tokens or entry.max_tokens,
            }
        )

    def embed(
        self,
        chunks: list[Chunk],
        params: BaseModel,
        *,
        expected_revision: str | None,
        context: StageContext,
    ) -> Vectors:
        params = cast(SentenceTransformersParams, params)
        if not chunks:
            raise ComponentError("empty_input")
        entry = self._check(params, context)
        model = self._model(params, entry, context)
        options = self._prefixed(params, entry)
        tokenizer = getattr(model, "tokenizer", None)

        def count(text: str) -> int:
            if tokenizer is None:
                return math.ceil(len(text) / 4)
            return len(tokenizer(text, add_special_tokens=True)["input_ids"])

        texts = prepare_texts(chunks, options, context, count=count)
        matrix = np.asarray(
            model.encode(texts, batch_size=params.batch_size, convert_to_numpy=True),
            dtype=np.float32,
        )
        context.metric("chunks", len(chunks))
        return finish_vectors(
            matrix,
            chunks,
            options,
            component=self.info.id,
            model=params.model,
            revision=self.revision_for(params, context),
            truncated=int(context.metrics.get("truncated_inputs", 0)),
        )

    def embed_query_text(
        self, text: str, params: BaseModel, context: StageContext
    ) -> tuple[np.ndarray, str]:
        params = cast(SentenceTransformersParams, params)
        entry = self._check(params, context)
        options = self._prefixed(params, entry)
        query = options.query_prefix + text[len(params.query_prefix) :]
        vector = self._model(params, entry, context).encode([query], convert_to_numpy=True)
        return np.asarray(vector, dtype=np.float32), self.revision_for(params, context)


# --- embed/openai-compatible -----------------------------------------------------------------


class OpenAICompatibleParams(_Params):
    provider: str = Field(
        default="local-embeddings",
        pattern=r"^[a-z][a-z0-9-]{0,62}$",
        description="A name from providers.endpoints, allow-listed in providers.allowed",
    )
    model: str = Field(default="text-embedding", min_length=1, max_length=128)
    request_dimensions: int | None = Field(
        default=None, ge=8, le=8192, description="Ask the endpoint for fewer dimensions"
    )


class OpenAICompatibleEmbed(EmbedComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="embed/openai-compatible",
        stage="embed",
        version="1.0.0",
        title="OpenAI-compatible endpoint",
        description="POST /v1/embeddings on an endpoint the configuration names (vLLM, Ollama, "
        "TEI, hosted). Refuses unless the provider is allow-listed: it receives document text.",
        resource_class="io",
        license="per endpoint",
        egress="provider:openai-compatible",
    )
    Params: ClassVar[type[BaseModel]] = OpenAICompatibleParams

    def model_name(self, params: BaseModel) -> str:
        return cast(OpenAICompatibleParams, params).model

    def revision_for(self, params: BaseModel, context: StageContext) -> str:
        # Such endpoints don't report revisions; the provider, model and size stand in.
        params = cast(OpenAICompatibleParams, params)
        return f"{params.provider}:{params.model}:{params.request_dimensions or 'full'}"[:128]

    def _access(self, params: OpenAICompatibleParams, context: StageContext) -> ProviderAccess:
        if params.provider not in context.services.get(ALLOWED_SERVICE, ()):
            raise ComponentError("provider_not_allowed")
        access = context.services.get(f"provider:{params.provider}")
        if not isinstance(access, ProviderAccess) or not access.url:
            raise ComponentError("provider_not_configured")
        return access

    def _request(
        self, access: ProviderAccess, params: OpenAICompatibleParams, texts: list[str]
    ) -> tuple[np.ndarray, int]:
        body: dict[str, Any] = {"model": params.model, "input": texts, "encoding_format": "float"}
        if params.request_dimensions:
            body["dimensions"] = params.request_dimensions
        headers = {"Authorization": f"Bearer {access.key}"} if access.key else {}
        try:
            response = access.http().post(
                f"{access.url.rstrip('/')}/v1/embeddings", json=body, headers=headers
            )
        except httpx2.TimeoutException:
            raise ComponentError("provider_timeout", retryable=True) from None
        except httpx2.HTTPError:
            raise ComponentError("provider_unreachable", retryable=True) from None
        if response.status_code != 200:
            code = response.status_code
            raise ComponentError(f"provider_http_{code}", retryable=code == 429 or code >= 500)
        try:
            data = response.json()
            rows = sorted(data["data"], key=lambda item: item["index"])
            matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
            tokens = int((data.get("usage") or {}).get("total_tokens") or 0)
        except (ValueError, KeyError, TypeError):
            raise ComponentError("provider_bad_response") from None
        if matrix.shape[0] != len(texts) or matrix.ndim != 2:
            raise ComponentError("provider_bad_response")
        return matrix, tokens

    def embed(
        self,
        chunks: list[Chunk],
        params: BaseModel,
        *,
        expected_revision: str | None,
        context: StageContext,
    ) -> Vectors:
        params = cast(OpenAICompatibleParams, params)
        if not chunks:
            raise ComponentError("empty_input")
        access = self._access(params, context)
        texts = prepare_texts(chunks, params, context)
        parts, tokens = [], 0
        for start in range(0, len(texts), params.batch_size):
            context.checkpoint()
            with context.provider(params.provider):
                matrix, used = self._request(
                    access, params, texts[start : start + params.batch_size]
                )
            parts.append(matrix)
            tokens += used
        context.metric("chunks", len(chunks))
        context.metric("openai_compatible.calls", len(parts), add=True)
        context.metric("tokens", tokens, add=True)
        return finish_vectors(
            np.vstack(parts),
            chunks,
            params,
            component=self.info.id,
            model=params.model,
            revision=self.revision_for(params, context),
            tokens=tokens,
            truncated=int(context.metrics.get("truncated_inputs", 0)),
        )

    def embed_query_text(
        self, text: str, params: BaseModel, context: StageContext
    ) -> tuple[np.ndarray, str]:
        params = cast(OpenAICompatibleParams, params)
        matrix, _ = self._request(self._access(params, context), params, [text])
        return matrix, self.revision_for(params, context)


COMPONENTS = [SentenceTransformersEmbed, OpenAICompatibleEmbed]
