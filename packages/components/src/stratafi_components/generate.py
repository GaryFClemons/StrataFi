"""Generators: a prompt template and a model alias.

A generator answers a pinned prompt (``stratafi_core.answers``): grounded answers, a judge's
scores and drafted questions all come from one. ``generate`` returns the text and the tokens it
used; the caller parses it.

- ``generate/gateway`` calls ``POST /v1/chat/completions`` on the model gateway by alias.
- ``generate/fake`` answers in process, without a model, for tests and offline work: it answers
  with the source sentence that shares the most words with the question (or says the answer
  isn't in the sources), judges by word overlap, and drafts a question from each sentence it
  copies. Lab only.
"""

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.embed import MODELS_SERVICE
from stratafi_components.models import GATEWAY, GatewayProvider
from stratafi_core.answers import (
    NOT_IN_SOURCES,
    ContextPassage,
    GenerationRequest,
    abstains,
    citations_in,
    token_f1,
    words,
)
from stratafi_core.components import Component, ComponentError, ComponentInfo, StageContext

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CITED_RE = re.compile(r"\[[^\[\]\n]{1,200}\]")


@dataclass(frozen=True)
class Generation:
    text: str
    revision: str | None
    prompt_tokens: int
    completion_tokens: int


class GenerateComponent(Component):
    """A generator: a prompt with its inputs → text."""

    def revision_for(self, params: BaseModel, context: StageContext) -> str | None:
        """The model revision its text comes from; part of each cached answer's key."""
        return None

    def generate(
        self, request: GenerationRequest, params: BaseModel, context: StageContext
    ) -> Generation:
        raise NotImplementedError


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(" ".join(text.split())) if part.strip()]


def fake_answer(question: str, passages: Sequence[ContextPassage], *, min_overlap: float) -> str:
    """The sentence sharing the most of the question's words, cited; or "not in the sources"."""
    wanted = set(words(question))
    best: tuple[float, int, str, str] | None = None
    for rank, passage in enumerate(passages):
        for sentence in sentences(passage.text):
            share = len(wanted & set(words(sentence))) / len(wanted) if wanted else 0.0
            if best is None or (share, -rank) > (best[0], -best[1]):
                best = (share, rank, sentence, passage.source_id)
    if best is None or best[0] < min_overlap:
        return NOT_IN_SOURCES
    return f"{best[2]} [{best[3]}]"


def fake_faithfulness(answer: str, passages: Sequence[ContextPassage]) -> int:
    """5 when every claim's words are in the passages it cites; 1 when none are."""
    if abstains(answer):
        return 5
    by_id = {passage.source_id: passage for passage in passages}
    support = {
        word
        for source in citations_in(answer)
        if source in by_id
        for word in words(by_id[source].text)
    }
    claims = [found for claim in sentences(_CITED_RE.sub("", answer)) if (found := words(claim))]
    if not claims:
        return 1
    supported = sum(1 for claim in claims if sum(w in support for w in claim) >= 0.8 * len(claim))
    return 1 + round(4 * supported / len(claims))


def fake_correctness(answer: str, reference: str) -> int:
    f1 = token_f1(_CITED_RE.sub("", answer), reference)
    return next(score for score, floor in ((5, 0.6), (4, 0.45), (3, 0.3), (2, 0.15), (1, 0.0))
                if f1 >= floor)  # fmt: skip


def fake_questions(document: str, text: str, count: int) -> list[dict[str, str]]:
    """A question about each of the first ``count`` sentences of six to forty words, those with
    numbers first, quoting the sentence as its evidence."""
    candidates = [s for s in sentences(text) if 6 <= len(s.split()) <= 40]
    candidates.sort(key=lambda sentence: not any(char.isdigit() for char in sentence))
    drafted = []
    for sentence in candidates[:count]:
        topic = " ".join(words(sentence)[:4])
        drafted.append(
            {"question": f"What does {document} say about {topic}?", "evidence": sentence}
        )
    return drafted


class FakeParams(_Params):
    min_overlap: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="The share of the question's words a sentence needs, or it abstains",
    )


class FakeGenerate(GenerateComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="generate/fake",
        stage="generate",
        version="1.0.0",
        title="Extractive (fake)",
        description="Answers with the source sentence that shares the most words with the "
        "question, judges by word overlap and drafts a question per sentence, in process and "
        "without a model. For tests and offline development.",
        license="MIT",
        lab_only=True,
    )
    Params: ClassVar[type[BaseModel]] = FakeParams

    def revision_for(self, params: BaseModel, context: StageContext) -> str | None:
        return "fake-generate-v1"

    def generate(
        self, request: GenerationRequest, params: BaseModel, context: StageContext
    ) -> Generation:
        facts: dict[str, Any] = dict(request.facts)
        name = request.prompt.name
        if name == "grounded-qa":
            min_overlap = cast(FakeParams, params).min_overlap
            text = fake_answer(facts["question"], facts["passages"], min_overlap=min_overlap)
        elif name == "judge-faithfulness":
            text = json.dumps({"score": fake_faithfulness(facts["answer"], facts["passages"])})
        elif name == "judge-correctness":
            text = json.dumps({"score": fake_correctness(facts["answer"], facts["reference"])})
        elif name == "draft-questions":
            drafted = fake_questions(facts["document"], facts["text"], int(facts["count"]))
            text = json.dumps({"questions": drafted})
        else:
            raise ComponentError("unknown_prompt")
        prompt = sum(_tokens(message["content"]) for message in request.messages())
        return Generation(text, "fake-generate-v1", prompt, _tokens(text))


class GatewayParams(_Params):
    alias: str = Field(
        default="chat-default",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$",
        description="The gateway's chat alias",
    )
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=400, ge=16, le=4000)
    seed: int | None = Field(default=13, ge=0, le=2**31 - 1)


class GatewayGenerate(GenerateComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="generate/gateway",
        stage="generate",
        version="1.0.0",
        title="Model gateway",
        description="POST /v1/chat/completions on the model gateway, by alias.",
        resource_class="io",
        deterministic=False,
        license="MIT (models: per alias)",
        egress="model_gateway",
    )
    Params: ClassVar[type[BaseModel]] = GatewayParams

    def revision_for(self, params: BaseModel, context: StageContext) -> str | None:
        provider: GatewayProvider = context.service(MODELS_SERVICE)
        return provider.revision(cast(GatewayParams, params).alias)

    def generate(
        self, request: GenerationRequest, params: BaseModel, context: StageContext
    ) -> Generation:
        settings = cast(GatewayParams, params)
        provider: GatewayProvider = context.service(MODELS_SERVICE)
        result = provider.chat(
            request.messages(),
            alias=settings.alias,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            seed=settings.seed,
            slot=lambda tokens: context.provider(GATEWAY, tokens=tokens),
        )
        context.metric("gateway.calls", 1, add=True)
        context.metric("tokens", result.prompt_tokens + result.completion_tokens, add=True)
        return Generation(
            result.text, result.revision, result.prompt_tokens, result.completion_tokens
        )


COMPONENTS: list[type[Component]] = [FakeGenerate, GatewayGenerate]
