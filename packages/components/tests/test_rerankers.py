"""Rerankers: the cross-encoder behind its lab-only gate, LLM listwise behind its
switch with its ranking parsed and its list cached whole, and hosted rerankers behind the
allow-list."""

import json
import sys
import types
from typing import ClassVar

import httpx2
import numpy as np
import pytest

from stratafi_components.embed import MODELS_SERVICE
from stratafi_components.models import GatewayProvider
from stratafi_components.parsers.hosted import ProviderAccess
from stratafi_components.rerankers import (
    RERANKERS,
    CohereRerank,
    CrossEncoderRerank,
    JinaRerank,
    LlmListwiseRerank,
    VoyageRerank,
    parse_ranking,
)
from stratafi_components.retrieval import EntryMeta, Reranker, RetrievalCache, RetrievalEngine
from stratafi_core.artifacts import Chunk
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.specs import RetrievalConfig

TEXTS = ["wire callback rules", "branch hours", "callback to the phone on file for wires"]


class StandInCrossEncoder:
    loaded: ClassVar[list[tuple]] = []

    def __init__(self, model: str, revision: str, max_length: int) -> None:
        StandInCrossEncoder.loaded.append((model, revision, max_length))

    def predict(self, pairs, batch_size=32):
        return np.asarray([len(set(q.split()) & set(t.split())) for q, t in pairs], dtype=float)


@pytest.fixture
def cross_encoder(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = StandInCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(CrossEncoderRerank, "unavailable", classmethod(lambda cls: None))
    monkeypatch.setattr(CrossEncoderRerank, "_models", {})
    StandInCrossEncoder.loaded = []


def test_the_cross_encoder_is_lab_only_and_scores_pairs(cross_encoder) -> None:
    params = CrossEncoderRerank.resolve_params({})
    with pytest.raises(ComponentError) as error:
        CrossEncoderRerank().rerank("callback wires", TEXTS, params, StageContext())
    assert error.value.code == "in_process_disabled"
    context = StageContext(services={"in_process": True})
    scores = CrossEncoderRerank().rerank("callback for wires", TEXTS, params, context)
    assert scores == [1.0, 0.0, 3.0]
    assert StandInCrossEncoder.loaded == [("cross-encoder/ms-marco-MiniLM-L-6-v2", "main", 512)]
    assert CrossEncoderRerank().revision_for(params, context) == (
        "cross-encoder/ms-marco-MiniLM-L-6-v2@main"
    )
    unknown = CrossEncoderRerank.resolve_params({"model": "someone/unknown"})
    with pytest.raises(ComponentError) as error:
        CrossEncoderRerank().rerank("q", TEXTS, unknown, context)
    assert error.value.code == "model_not_cataloged"
    assert {entry.model for entry in RERANKERS} >= {
        "BAAI/bge-reranker-v2-m3",
        "BAAI/bge-reranker-base",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "mixedbread-ai/mxbai-rerank-base-v1",
    }


def test_rankings_parse_robustly() -> None:
    assert parse_ranking("[3] > [1] > [2]", 3) == [2, 0, 1]
    assert parse_ranking("I think [2] then [2] and [9] > [1]", 3) == [1, 0, 2]
    assert parse_ranking("no idea", 2) == [0, 1]


def chat(answers: list[str], seen: list) -> GatewayProvider:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={"choices": [{"message": {"content": answers[len(seen) - 1]}}], "usage": {}},
            headers={"X-Model-Revision": "chat-1"},
        )

    return GatewayProvider(
        "http://gw",
        client=httpx2.Client(base_url="http://gw", transport=httpx2.MockTransport(handler)),
    )


def test_llm_listwise_orders_windows_and_waits_for_its_switch() -> None:
    params = LlmListwiseRerank.resolve_params({"window": 2})
    with pytest.raises(ComponentError) as error:
        LlmListwiseRerank().rerank("q", TEXTS, params, StageContext())
    assert error.value.code == "llm_rerank_not_enabled"
    seen: list = []
    context = StageContext(
        services={MODELS_SERVICE: chat(["[2] > [1]", "[1]"], seen), "llm_rerank_enabled": True}
    )
    scores = LlmListwiseRerank().rerank("callback", TEXTS, params, context)
    assert len(seen) == 2  # windows of two
    assert "[1] wire callback rules" in seen[0]["messages"][1]["content"]
    assert scores[1] > scores[0] > scores[2]  # the first window ranks above the second
    assert LlmListwiseRerank.pointwise is False
    assert context.metrics["gateway.calls"] == 2


def test_the_engine_caches_a_listwise_list_whole() -> None:
    calls: list[list[str]] = []

    class Listwise(LlmListwiseRerank):
        def rerank(self, query, texts, params, context, *, expected_revision=None):
            calls.append(list(texts))
            return [float(len(texts) - i) for i in range(len(texts))][::-1]

    chunks = {
        f"{i:032x}": Chunk(
            chunk_id=f"{i:032x}",
            document_id="d",
            ordinal=i,
            text=text,
            index_text=text,
            char_start=0,
            char_end=1,
        )
        for i, text in enumerate(TEXTS)
    }
    from stratafi_components.index import Bm25Index, IndexInput, KeywordIndex

    built = Bm25Index().build(
        [IndexInput("c" * 64, list(chunks.values()))], Bm25Index.resolve_params({}), StageContext()
    )
    config = RetrievalConfig.model_validate(
        {
            "schema": "stratafi.retrieval.v1",
            "name": "lw",
            "retrievers": {"dense": None, "keyword": {"top_n": 10}},
            "fusion": {"method": "none"},
            "candidates": 10,
            "top_k": 3,
        }
    )
    component = Listwise()
    reranker = Reranker(component, component.resolve_params({}), StageContext(), "rev")
    engine = RetrievalEngine(
        config,
        entries=[EntryMeta(document_id="d", name="d.md")],
        chunks=chunks,
        keyword=KeywordIndex.load(built.manifest, built.payload),
        reranker=reranker,
        cache=RetrievalCache(),
    )
    first = engine.search("callback wires", asker=None, day=None)
    second = engine.search("callback wires", asker=None, day=None)
    assert len(calls) == 1
    assert first.checkpoints["final"] == second.checkpoints["final"]
    # The scores reversed the fused order.
    assert first.checkpoints["reranked"][0][0] == first.checkpoints["fused"][-1][0]


@pytest.mark.parametrize(
    ("component", "provider", "key"),
    [
        (CohereRerank, "cohere", "results"),
        (JinaRerank, "jina", "results"),
        (VoyageRerank, "voyage", "data"),
    ],
)
def test_hosted_rerankers_refuse_unless_allow_listed(component, provider: str, key: str) -> None:
    seen: list = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                key: [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}]
            },
        )

    access = ProviderAccess(
        "https://api.example",
        "secret",
        client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    params = component.resolve_params({})
    with pytest.raises(ComponentError) as error:
        component().rerank(
            "q", TEXTS, params, StageContext(services={f"provider:{provider}": access})
        )
    assert error.value.code == "provider_not_allowed"
    assert seen == []
    context = StageContext(
        services={"providers_allowed": frozenset({provider}), f"provider:{provider}": access}
    )
    scores = component().rerank("callback", TEXTS, params, context)
    assert scores[2] == 0.9
    assert scores[0] == 0.4
    assert scores[1] < 0.4  # unscored documents rank last
    assert seen[0]["documents"] == TEXTS
    assert component.info.egress == f"provider:{provider}"
