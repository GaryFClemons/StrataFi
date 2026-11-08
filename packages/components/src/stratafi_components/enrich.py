"""Context enrichment.

``enrich/llm-context`` is the "add context" step of contextual retrieval: for each chunk, a
chat model writes one line that situates it in its document, and the line goes into
``context``, which the index text can include (``{context}``). The prompt is versioned
(``PROMPT_VERSION``, part of the cache key), the model is a gateway alias whose revision is
recorded, and the tokens it used are recorded for the run's cost.

The configuration can turn it off (``models.enrichment_enabled``): then it refuses to run
with ``enrichment_not_enabled``. Answers vary between calls, so it is not deterministic; its
artifacts are cached like any other, so a cached run reproduces.
"""

from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.chunk import EnrichComponent
from stratafi_components.embed import MODELS_SERVICE
from stratafi_components.models import GATEWAY, GatewayProvider
from stratafi_core.artifacts import Chunk
from stratafi_core.components import ComponentError, ComponentInfo, StageContext

ENABLED_SERVICE = "enrichment_enabled"
PROMPT_VERSION = "context-v1"
SYSTEM_PROMPT = (
    "You situate a passage within its document for search. Answer with one short sentence "
    "naming the document and what the passage is about. Do not add facts."
)
USER_PROMPT = "Document: {title}\nSection: {section}\n\nPassage:\n{text}"


class LlmContextParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(
        default="chat-default",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$",
        description="The gateway's chat alias",
    )
    max_tokens: int = Field(default=64, ge=8, le=512, description="Tokens per context line")
    max_passage_chars: int = Field(default=4000, ge=200, le=32_000)


class LlmContext(EnrichComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="enrich/llm-context",
        stage="enrich",
        version="1.0.0",
        title="LLM context line",
        description="A one-line context per chunk from a gateway chat model, with a versioned "
        "prompt; tokens recorded for cost. Refuses when models.enrichment_enabled is off.",
        resource_class="io",
        deterministic=False,
        license="MIT (models: per alias)",
        egress="model_gateway",
    )
    Params: ClassVar[type[BaseModel]] = LlmContextParams

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        return {"prompt": PROMPT_VERSION}

    def enrich(
        self, chunks: list[Chunk], params: BaseModel, context: StageContext | None = None
    ) -> list[Chunk]:
        params = cast(LlmContextParams, params)
        if context is None or not context.services.get(ENABLED_SERVICE):
            raise ComponentError("enrichment_not_enabled")
        provider: GatewayProvider = context.service(MODELS_SERVICE)
        enriched: list[Chunk] = []
        prompt_tokens = completion_tokens = 0
        revision = None
        for chunk in chunks:
            context.checkpoint()
            user = USER_PROMPT.format(
                title=chunk.metadata.get("title") or chunk.index_text.split("\n", 1)[0],
                section=" > ".join(chunk.heading_path) or "-",
                text=chunk.text[: params.max_passage_chars],
            )
            result = provider.chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                alias=params.alias,
                max_tokens=params.max_tokens,
                slot=lambda tokens: context.provider(GATEWAY, tokens=tokens),
            )
            if revision is not None and result.revision != revision:
                raise ComponentError("revision_changed")
            revision = result.revision
            prompt_tokens += result.prompt_tokens
            completion_tokens += result.completion_tokens
            line = " ".join(result.text.split())[:500]
            enriched.append(
                chunk.model_copy(
                    update={
                        "context": line,
                        "index_text": f"{line}\n{chunk.index_text}" if line else chunk.index_text,
                        "metadata": {
                            **chunk.metadata,
                            "context_prompt": PROMPT_VERSION,
                            "context_model": params.alias,
                            "context_revision": result.revision,
                        },
                    }
                )
            )
        context.metric("gateway.calls", len(chunks), add=True)
        context.metric("tokens", prompt_tokens + completion_tokens, add=True)
        context.metric("prompt_tokens", prompt_tokens, add=True)
        context.metric("completion_tokens", completion_tokens, add=True)
        return enriched


COMPONENTS = [LlmContext]
