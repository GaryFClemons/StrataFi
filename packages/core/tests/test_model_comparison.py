"""Comparing models by controlled pairs (unit tier)."""

import pytest

from stratafi_core.model_comparison import (
    controlled,
    controlled_pairs,
    in_role,
    leaderboard,
    matrix,
    model_of,
    pair_result,
)
from stratafi_core.tracking import TrackedRun

ATTRIBUTES = {
    "split": "dev",
    "corpus": "halcyon",
    "corpus_version": 1,
    "eval_set": "halcyon",
    "eval_set_version": 1,
}


def tracked(
    identifier: str,
    *,
    rerank: str = "rerank/gateway",
    alias: str | None = "rerank-default",
    embed: str = "embed-local",
    top_k: int = 8,
    p95: float = 20.0,
    **attributes,
) -> TrackedRun:
    params = {
        "embed.component": "embed/gateway",
        "embed.params.alias": embed,
        "models.embed.component": "embed/gateway",
        "models.embed.alias": embed,
        "models.embed.revision": f"rev-{embed}",
        "rerank.component": rerank,
        "models.rerank.component": rerank,
        "top_k": top_k,
    }
    if alias:
        params["rerank.params.alias"] = alias
        params["models.rerank.params.alias"] = alias
    return TrackedRun(
        id=identifier,
        attributes={**ATTRIBUTES, **attributes},
        params=params,
        metrics={"latency.total_ms.p95": p95},
    )


GATEWAY = tracked("a")
NONE = tracked("b", rerank="rerank/none", alias=None, p95=5.0)
NONE_TOP5 = tracked("c", rerank="rerank/none", alias=None, top_k=5)
OTHER_EMBED = tracked("d", embed="embed-bge")


def test_a_model_is_its_component_name_and_revision() -> None:
    embed = model_of(GATEWAY, "embed")
    assert embed is not None
    assert (embed.component, embed.name, embed.revision) == (
        "embed/gateway",
        "embed-local",
        "rev-embed-local",
    )
    assert embed.label == "embed-local (rev-embed-lo)"
    rerank = model_of(NONE, "rerank")
    assert rerank is not None
    assert rerank.name == "rerank/none"
    assert rerank.revision is None
    assert model_of(GATEWAY, "adapter") is None
    assert in_role("rerank.params.alias", "rerank")
    assert in_role("retrievers.dense.adapter", "adapter")
    assert not in_role("top_k", "rerank")


def test_controlled_pairs_differ_only_in_the_role() -> None:
    assert controlled(GATEWAY, NONE, "rerank")
    # top_k differs too: not controlled.
    assert not controlled(GATEWAY, NONE_TOP5, "rerank")
    # The same reranker: nothing to compare.
    assert not controlled(NONE, NONE, "rerank")
    assert controlled(GATEWAY, OTHER_EMBED, "embed")
    assert not controlled(GATEWAY, OTHER_EMBED, "rerank")
    # Another split or corpus version never pairs.
    assert not controlled(
        GATEWAY, tracked("e", rerank="rerank/none", alias=None, split="test"), "rerank"
    )
    assert not controlled(
        GATEWAY, tracked("f", rerank="rerank/none", alias=None, corpus_version=2), "rerank"
    )
    assert controlled_pairs([GATEWAY, NONE, NONE_TOP5, OTHER_EMBED], "rerank") == [(0, 1)]
    assert controlled_pairs([GATEWAY, NONE, NONE_TOP5, OTHER_EMBED], "embed") == [(0, 3)]


def test_the_leaderboard_ranks_models_by_their_best_run() -> None:
    estimates = {
        "a": {"mean": 0.8, "low": 0.7, "high": 0.9, "n": 60},
        "b": {"mean": 0.7, "low": 0.6, "high": 0.8, "n": 60},
        "c": {"mean": 0.75, "low": 0.65, "high": 0.85, "n": 60},
        "d": None,
    }
    rows = leaderboard(
        [GATEWAY, NONE, NONE_TOP5, OTHER_EMBED],
        "rerank",
        "recall@5",
        estimates=estimates,
        egress=lambda run: (
            "model_gateway" if run.params["rerank.component"] == "rerank/gateway" else "none"
        ),
    )
    assert [row["name"] for row in rows] == ["rerank-default", "rerank/none"]
    gateway, none = rows
    assert gateway["runs"] == 2  # a and d share the gateway reranker; d has no estimate
    assert gateway["best_eval_run_id"] == "a"
    assert gateway["best"]["mean"] == 0.8
    assert none["runs"] == 2
    assert none["best_eval_run_id"] == "c"
    assert none["latency_p95_ms"] == 12.5  # the median of 5 and 20
    assert gateway["egress"] == "model_gateway"
    assert none["egress"] == "none"


def score(value: float, evidence: bool = True) -> dict:
    return {
        "has_evidence": evidence,
        "metrics": {"knowledge-hit-v1": {"final": {"recall@5": value}}},
    }


def test_a_pair_is_compared_query_by_query_and_fills_the_matrix() -> None:
    a = {f"q{i}": score(1.0 if i % 2 else 0.0) for i in range(20)}
    b = {f"q{i}": score(1.0) for i in range(20)} | {"abstain": score(0.0, evidence=False)}
    result = pair_result(
        a, b, rule="knowledge-hit-v1", checkpoint="final", metric="recall@5", resamples=200
    )
    assert result is not None
    assert result["n"] == 20
    assert result["delta"] == pytest.approx(0.5)
    assert result["wins"] == 10
    assert result["losses"] == 0
    cells = matrix(
        ["m1", "m2", "m3"],
        [
            {"a_model": "m1", "b_model": "m2", "result": {"delta": 0.5}},
            {"a_model": "m1", "b_model": "m2", "result": {"delta": 0.3}},
            {"a_model": "m2", "b_model": "m9", "result": {"delta": 0.1}},
            {"a_model": "m1", "b_model": "m3", "result": None},
        ],
    )
    assert cells[0][1] == {"delta": 0.4, "pairs": 2}
    assert cells[1][0] == {"delta": -0.4, "pairs": 2}
    assert cells[0][0] is None
    assert cells[0][2] is None
    assert cells[2][1] is None
    assert (
        pair_result({}, {}, rule="knowledge-hit-v1", checkpoint="final", metric="recall@5") is None
    )
