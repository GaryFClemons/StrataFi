"""Rerankers: query-time components.

A reranker scores the fused candidates against the query. Each candidate is its document's
title, the chunk's section and the chunk's text, joined by newlines (``rerank_text``).
``rerank`` returns one score per text, in the texts' order, or None when it leaves the fused
order alone; the retrieval engine sorts by score, ties by fused rank, and caches each score by
the reranker, its revision, the query and the text.

- ``rerank/none`` turns reranking off: the fused list is the final one.
- ``rerank/fake-overlap`` scores how much of the query each text covers, in process
  (``stratafi_core.fake_models.overlap_score``, which the fake gateway uses too), for tests and
  offline work.
- ``rerank/gateway`` calls ``POST /v1/rerank`` on the model gateway by alias:
  ``rerank-default`` is a MiniLM cross-encoder on the local gateway (its ``rerank`` extra), and
  word overlap on the fake gateway.

The cross-encoder, LLM listwise and hosted rerankers are in ``stratafi_components.rerankers``.
"""

from collections.abc import Sequence
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.embed import MODELS_SERVICE
from stratafi_components.models import GATEWAY, GatewayProvider
from stratafi_core.components import Component, ComponentError, ComponentInfo, StageContext
from stratafi_core.fake_models import RERANK_REVISION, overlap_score


def rerank_text(title: str | None, section: str | None, text: str) -> str:
    """What a reranker reads: the title, the section and the text, one per line."""
    return "\n".join(part for part in (title, section, text) if part)


class RerankComponent(Component):
    """A reranker: (query, texts) → a score per text. A pointwise score depends on its query
    and text alone (a cross-encoder's), so the engine caches each one; a listwise reranker's
    depend on the whole list (``pointwise`` False), so the engine caches the list."""

    reorders: ClassVar[bool] = True
    pointwise: ClassVar[bool] = True

    def revision_for(self, params: BaseModel, context: StageContext) -> str | None:
        """The model revision its scores come from; part of each cached score's key."""
        return None

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        params: BaseModel,
        context: StageContext,
        *,
        expected_revision: str | None = None,
    ) -> list[float] | None:
        """A score per text. A remote model whose revision isn't ``expected_revision`` (a model
        swapped mid-run) fails with ``revision_changed``, as an embedder does."""
        raise NotImplementedError


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RerankNone(RerankComponent):
    reorders: ClassVar[bool] = False
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="rerank/none",
        stage="rerank",
        version="1.0.0",
        title="No reranking",
        description="Keeps the fused order: the final passages are the best fused candidates.",
        license="MIT",
    )

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        params: BaseModel,
        context: StageContext,
        *,
        expected_revision: str | None = None,
    ) -> list[float] | None:
        return None


class FakeOverlapRerank(RerankComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="rerank/fake-overlap",
        stage="rerank",
        version="2.0.0",
        title="Word overlap (fake)",
        description="Scores how much of the query each candidate covers, in process: the "
        "query's words found in it, longer words weighing more. For tests and offline "
        "development.",
        license="MIT",
        lab_only=True,
    )

    def revision_for(self, params: BaseModel, context: StageContext) -> str | None:
        return RERANK_REVISION

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        params: BaseModel,
        context: StageContext,
        *,
        expected_revision: str | None = None,
    ) -> list[float] | None:
        return [overlap_score(query, text) for text in texts]


class GatewayRerankParams(_Params):
    alias: str = Field(
        default="rerank-default",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$",
        description="The gateway's rerank alias: rerank-default is a MiniLM cross-encoder on the "
        "local gateway, word overlap on the fake gateway",
    )


class GatewayRerank(RerankComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="rerank/gateway",
        stage="rerank",
        version="1.0.0",
        title="Model gateway",
        description="POST /v1/rerank on the model gateway, by alias, scoring each candidate's "
        "title, section and text. Records X-Model-Revision.",
        resource_class="io",
        license="MIT (models: per alias)",
        egress="model_gateway",
        baseline=True,
    )
    Params: ClassVar[type[BaseModel]] = GatewayRerankParams

    def revision_for(self, params: BaseModel, context: StageContext) -> str | None:
        provider: GatewayProvider = context.service(MODELS_SERVICE)
        return provider.revision(cast(GatewayRerankParams, params).alias)

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        params: BaseModel,
        context: StageContext,
        *,
        expected_revision: str | None = None,
    ) -> list[float] | None:
        if not texts:
            return []
        provider: GatewayProvider = context.service(MODELS_SERVICE)
        result = provider.rerank(
            query,
            texts,
            alias=cast(GatewayRerankParams, params).alias,
            slot=lambda tokens: context.provider(GATEWAY, tokens=tokens),
        )
        if expected_revision is not None and result.revision != expected_revision:
            raise ComponentError("revision_changed")
        scores: dict[int, float] = dict(result.ranked)
        if len(scores) != len(texts):  # without top_n, the gateway scores every text
            raise ComponentError("gateway_bad_response")
        context.metric("gateway.calls", 1, add=True)
        context.metric("tokens", result.tokens, add=True)
        return [scores[index] for index in range(len(texts))]


COMPONENTS: list[type[Component]] = [RerankNone, FakeOverlapRerank, GatewayRerank]
