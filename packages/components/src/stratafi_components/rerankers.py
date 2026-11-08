"""More rerankers.

- ``rerank/cross-encoder`` scores each (query, passage) pair with a cross-encoder in process
  (sentence-transformers' ``CrossEncoder``, the ``sentence-transformers`` extra). **Lab only**:
  it refuses when ``models.in_process`` is off. Models come from its catalog: bge-reranker-v2-m3,
  bge-reranker-base, ms-marco-MiniLM-L-6-v2 and mxbai-rerank-base, each with its license; or
  ``trained/<model ID>``, one the lab fine-tuned from a catalog model.
- ``rerank/llm-listwise`` asks a gateway chat model to order the candidates (RankGPT's listwise
  prompt, versioned): a window of passages at a time, numbered; the answer's order becomes the
  scores. A passage's score depends on the others in its window, so the engine caches the whole
  list, not each pair. It refuses when ``models.llm_rerank_enabled`` is off.
- ``rerank/cohere``, ``rerank/jina``, ``rerank/voyage`` call those providers' rerank APIs (one
  shape: a query, documents, and a relevance score per index), and refuse unless the provider
  is allow-listed (``providers.allowed``): they receive passage text.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import httpx2
from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.embed import MODELS_SERVICE
from stratafi_components.embedders import (
    IN_PROCESS_SERVICE,
    PERMISSIVE,
    is_trained,
    load_trained,
    trained_model,
)
from stratafi_components.models import GATEWAY, GatewayProvider
from stratafi_components.parsers.hosted import ALLOWED_SERVICE, ProviderAccess
from stratafi_components.rerank import RerankComponent
from stratafi_core.components import ComponentError, ComponentInfo, StageContext

LLM_RERANK_SERVICE = "llm_rerank_enabled"
LISTWISE_PROMPT_VERSION = "listwise-v1"


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- rerank/cross-encoder ----------------------------------------------------------------------


@dataclass(frozen=True)
class RerankerModel:
    model: str
    title: str
    license: str
    max_tokens: int


RERANKERS: tuple[RerankerModel, ...] = (
    RerankerModel("BAAI/bge-reranker-v2-m3", "BGE reranker v2 m3", "Apache-2.0", 8192),
    RerankerModel("BAAI/bge-reranker-base", "BGE reranker base", "MIT", 512),
    RerankerModel("cross-encoder/ms-marco-MiniLM-L-6-v2", "ms-marco MiniLM L6", "Apache-2.0", 512),
    RerankerModel("mixedbread-ai/mxbai-rerank-base-v1", "mxbai rerank base v1", "Apache-2.0", 512),
)
RERANKER_MODELS = {entry.model: entry for entry in RERANKERS}


class CrossEncoderParams(_Params):
    model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$",
        description="A model from the reranker catalog, or trained/<model ID>",
    )
    revision: str = Field(default="main", pattern=r"^[A-Za-z0-9._-]{1,64}$")
    batch_size: int = Field(default=32, ge=1, le=512)
    max_length: int = Field(default=512, ge=16, le=8192, description="Tokens per pair")


class CrossEncoderRerank(RerankComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="rerank/cross-encoder",
        stage="rerank",
        version="1.0.0",
        title="Cross-encoder (in process)",
        description="A catalog cross-encoder (bge-reranker-v2-m3, bge-reranker-base, "
        "ms-marco-MiniLM, mxbai-rerank) in process, lab only: refuses when models.in_process "
        "is off.",
        resource_class="gpu",
        license="Apache-2.0 (models: per catalog entry)",
        lab_only=True,
        services=("model-files",),
    )
    Params: ClassVar[type[BaseModel]] = CrossEncoderParams
    requires: ClassVar[tuple[str, ...]] = ("sentence_transformers",)
    extra: ClassVar[str | None] = "sentence-transformers"
    _models: ClassVar[dict[tuple[str, str, int], Any]] = {}

    def revision_for(self, params: BaseModel, context: StageContext) -> str | None:
        params = cast(CrossEncoderParams, params)
        if is_trained(params.model):
            trained = trained_model(params.model, "reranker", context)
            return f"{params.model}@{trained.model_key[:16]}"
        return f"{params.model}@{params.revision}"[:128]

    def _model(self, params: CrossEncoderParams, context: StageContext) -> Any:
        if not context.services.get(IN_PROCESS_SERVICE, False):
            raise ComponentError("in_process_disabled")
        if self.unavailable() is not None:
            raise ComponentError("extra_missing")
        trained = (
            trained_model(params.model, "reranker", context) if is_trained(params.model) else None
        )
        entry = RERANKER_MODELS.get(str(trained.base.get("model")) if trained else params.model)
        if entry is None:
            raise ComponentError("model_not_cataloged")
        if entry.license not in PERMISSIVE and entry.license not in context.services.get(
            "licenses", ()
        ):
            raise ComponentError("license_not_allowed")
        from sentence_transformers import CrossEncoder

        if trained is not None:
            key = (params.model, trained.model_key, params.max_length)
            if key not in self._models:
                self._models[key] = load_trained(
                    params.model,
                    "reranker",
                    context,
                    lambda path: CrossEncoder(path, max_length=params.max_length),
                )
            return self._models[key]
        key = (params.model, params.revision, params.max_length)
        if key not in self._models:
            try:
                self._models[key] = CrossEncoder(
                    params.model, revision=params.revision, max_length=params.max_length
                )
            except Exception:
                raise ComponentError("model_unavailable") from None
        return self._models[key]

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        params: BaseModel,
        context: StageContext,
        *,
        expected_revision: str | None = None,
    ) -> list[float] | None:
        params = cast(CrossEncoderParams, params)
        if not texts:
            return []
        model = self._model(params, context)
        scores = model.predict([(query, text) for text in texts], batch_size=params.batch_size)
        context.metric("cross_encoder.pairs", len(texts), add=True)
        return [float(score) for score in scores]


# --- rerank/llm-listwise ---------------------------------------------------------------------

LISTWISE_SYSTEM = (
    "You rank passages by how well they answer a search query. Answer only with the passage "
    "numbers, best first, like [3] > [1] > [2]. Rank every passage once."
)


class ListwiseParams(_Params):
    alias: str = Field(default="chat-default", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
    window: int = Field(default=20, ge=2, le=50, description="Passages per prompt")
    passage_chars: int = Field(default=600, ge=100, le=4000, description="Each passage, at most")


def parse_ranking(answer: str, count: int) -> list[int]:
    """The passage order an answer gives (``[3] > [1]``), 0-based: each number once, in range;
    passages it leaves out follow in their original order."""
    seen: list[int] = []
    for match in re.finditer(r"\[(\d{1,3})\]", answer):
        number = int(match.group(1)) - 1
        if 0 <= number < count and number not in seen:
            seen.append(number)
    return seen + [index for index in range(count) if index not in seen]


class LlmListwiseRerank(RerankComponent):
    pointwise: ClassVar[bool] = False
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="rerank/llm-listwise",
        stage="rerank",
        version="1.0.0",
        title="LLM listwise",
        description="A gateway chat model orders a window of numbered passages (RankGPT's "
        "listwise prompt, versioned). Refuses when models.llm_rerank_enabled is off.",
        resource_class="io",
        deterministic=False,
        license="MIT (models: per alias)",
        egress="model_gateway",
    )
    Params: ClassVar[type[BaseModel]] = ListwiseParams

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        return {"prompt": LISTWISE_PROMPT_VERSION}

    def revision_for(self, params: BaseModel, context: StageContext) -> str | None:
        return f"{cast(ListwiseParams, params).alias}:{LISTWISE_PROMPT_VERSION}"

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        params: BaseModel,
        context: StageContext,
        *,
        expected_revision: str | None = None,
    ) -> list[float] | None:
        params = cast(ListwiseParams, params)
        if not context.services.get(LLM_RERANK_SERVICE):
            raise ComponentError("llm_rerank_not_enabled")
        if not texts:
            return []
        provider: GatewayProvider = context.service(MODELS_SERVICE)
        scores = [0.0] * len(texts)
        for start in range(0, len(texts), params.window):
            window = list(texts[start : start + params.window])
            numbered = "\n\n".join(
                f"[{i + 1}] {' '.join(text.split())[: params.passage_chars]}"
                for i, text in enumerate(window)
            )
            result = provider.chat(
                [
                    {"role": "system", "content": LISTWISE_SYSTEM},
                    {"role": "user", "content": f"Query: {query}\n\nPassages:\n{numbered}"},
                ],
                alias=params.alias,
                max_tokens=8 * len(window),
                slot=lambda tokens: context.provider(GATEWAY, tokens=tokens),
            )
            order = parse_ranking(result.text, len(window))
            # Earlier windows hold the better fused candidates: they keep higher scores.
            base = float(len(texts) - start)
            for rank, index in enumerate(order):
                scores[start + index] = base - rank / len(window)
            context.metric("gateway.calls", 1, add=True)
            context.metric("tokens", result.prompt_tokens + result.completion_tokens, add=True)
        return scores


# --- hosted rerankers ------------------------------------------------------------------------


class HostedRerankParams(_Params):
    model: str = Field(default="", max_length=128, description="The provider's model name")
    top_n: int | None = Field(default=None, ge=1, le=1000)


@dataclass(frozen=True)
class RerankApi:
    provider: str
    title: str
    path: str
    default_model: str
    results_key: str


class HostedRerank(RerankComponent):
    api: ClassVar[RerankApi]

    def revision_for(self, params: BaseModel, context: StageContext) -> str | None:
        model = cast(HostedRerankParams, params).model or self.api.default_model
        return f"{self.api.provider}:{model}"

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        params: BaseModel,
        context: StageContext,
        *,
        expected_revision: str | None = None,
    ) -> list[float] | None:
        params = cast(HostedRerankParams, params)
        if self.api.provider not in context.services.get(ALLOWED_SERVICE, ()):
            raise ComponentError("provider_not_allowed")
        access = context.services.get(f"provider:{self.api.provider}")
        if not isinstance(access, ProviderAccess) or not access.url or not access.key:
            raise ComponentError("provider_not_configured")
        if not texts:
            return []
        body: dict[str, Any] = {
            "model": params.model or self.api.default_model,
            "query": query,
            "documents": list(texts),
        }
        if params.top_n is not None:
            body["top_n"] = params.top_n
        try:
            with context.provider(self.api.provider):
                response = access.http().post(
                    f"{access.url.rstrip('/')}{self.api.path}",
                    json=body,
                    headers={"Authorization": f"Bearer {access.key}"},
                )
        except httpx2.TimeoutException:
            raise ComponentError("provider_timeout", retryable=True) from None
        except httpx2.HTTPError:
            raise ComponentError("provider_unreachable", retryable=True) from None
        if response.status_code != 200:
            code = response.status_code
            raise ComponentError(f"provider_http_{code}", retryable=code == 429 or code >= 500)
        try:
            results = response.json()[self.api.results_key]
            scored = {int(item["index"]): float(item["relevance_score"]) for item in results}
        except (ValueError, KeyError, TypeError):
            raise ComponentError("provider_bad_response") from None
        context.metric(f"{self.api.provider}.calls", 1, add=True)
        lowest = min(scored.values(), default=0.0) - 1.0
        return [scored.get(index, lowest) for index in range(len(texts))]


def _hosted(api: RerankApi) -> type[HostedRerank]:
    info = ComponentInfo(
        id=f"rerank/{api.provider}",
        stage="rerank",
        version="1.0.0",
        title=f"{api.title} rerank",
        description=f"{api.title}'s rerank API. Refuses unless {api.provider} is allow-listed "
        "(providers.allowed): it receives the candidates' text.",
        resource_class="io",
        license="commercial",
        egress=f"provider:{api.provider}",
    )
    name = f"{api.title.replace(' ', '')}Rerank"
    return type(name, (HostedRerank,), {"info": info, "api": api, "Params": HostedRerankParams})


CohereRerank = _hosted(RerankApi("cohere", "Cohere", "/v2/rerank", "rerank-v3.5", "results"))
JinaRerank = _hosted(
    RerankApi("jina", "Jina", "/v1/rerank", "jina-reranker-v2-base-multilingual", "results")
)
VoyageRerank = _hosted(RerankApi("voyage", "Voyage", "/v1/rerank", "rerank-2", "data"))

COMPONENTS = [CrossEncoderRerank, LlmListwiseRerank, CohereRerank, JinaRerank, VoyageRerank]
