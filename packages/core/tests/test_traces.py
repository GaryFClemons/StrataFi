"""Traces (unit tier): spans from step latencies and ranked lists."""

import pytest

from stratafi_core.traces import build_trace, latency_profile, lost_at

LATENCY = {
    "keyword_ms": 2.0,
    "embed_ms": 5.5,
    "dense_ms": 1.5,
    "fusion_ms": 0.5,
    "rerank_ms": 30.0,
    "total_ms": 40.0,
}
LISTS = {
    "keyword": [("k1", 9.0), ("gold", 7.0)],
    "dense": [("d1", 0.9), ("d2", 0.8), ("gold", 0.7)],
    "fused": [("gold", 0.03), ("d1", 0.02), ("k1", 0.01)],
    "candidates": [("gold", 0.03), ("d1", 0.02), ("k1", 0.01)],
    "reranked": [("d1", 0.9), ("k1", 0.5), ("gold", 0.1)],
    "final": [("d1", 0.9), ("k1", 0.5)],
}
CONFIG = {
    "retrievers": {"dense": {"top_n": 50}, "keyword": {"top_n": 40}},
    "fusion": {"method": "rrf", "k": 60},
    "candidates": 50,
    "rerank": {"component": "rerank/gateway"},
    "top_k": 2,
}
MODELS = {
    "embed": {"alias": "embed-local", "revision": "r-embed"},
    "rerank": {"component": "rerank/gateway", "params": {"alias": "rerank-default"}},
}


def test_spans_follow_the_engine_and_start_where_the_last_ended() -> None:
    trace = build_trace(
        latency_ms=LATENCY, checkpoints=LISTS, relevant={"gold"}, config=CONFIG, models=MODELS
    )
    root, *steps = trace["spans"]
    assert root["id"] == "search"
    assert root["duration_ms"] == 40.0
    assert [span["id"] for span in steps] == [
        "keyword",
        "embed",
        "dense",
        "fuse",
        "rerank",
        "other",
    ]
    assert [span["start_ms"] for span in steps] == [0.0, 2.0, 7.5, 9.0, 9.5, 39.5]
    assert steps[-1]["duration_ms"] == 0.5
    for span in steps:
        assert span["parent_id"] == "search"
    assert steps[0]["attributes"] == {"top_n": 40}
    assert steps[1]["attributes"] == {"model": "embed-local", "revision": "r-embed"}
    assert steps[3]["attributes"] == {"method": "rrf", "candidates": 50, "k": 60}
    assert steps[4]["attributes"]["model"] == "rerank-default"
    assert [item["checkpoint"] for item in steps[3]["lists"]] == ["fused", "candidates"]


def test_lists_say_where_the_evidence_is_and_where_it_was_lost() -> None:
    trace = build_trace(latency_ms=LATENCY, checkpoints=LISTS, relevant={"gold"}, config=CONFIG)
    path = {entry["checkpoint"]: entry for entry in trace["path"]}
    assert path["dense"]["first_relevant_rank"] == 3
    assert path["fused"]["first_relevant_rank"] == 1
    assert path["reranked"]["found"] is True
    assert path["final"]["found"] is False
    assert all(entry["top"] == [] for entry in trace["path"])
    assert trace["lost_at_checkpoint"] == "final"
    rerank = next(span for span in trace["spans"] if span["id"] == "rerank")
    final = next(item for item in rerank["lists"] if item["checkpoint"] == "final")
    assert final["top"][0] == {"rank": 1, "chunk_id": "d1", "score": 0.9, "relevant": False}


def test_a_query_without_labels_marks_nothing_relevant() -> None:
    trace = build_trace(latency_ms=LATENCY, checkpoints=LISTS, relevant=None)
    for entry in trace["path"]:
        assert entry["found"] is None
        assert entry["first_relevant_rank"] is None
    assert trace["lost_at_checkpoint"] is None
    dense = next(span for span in trace["spans"] if span["id"] == "dense")
    assert dense["lists"][0]["top"][0]["relevant"] is None


def test_dense_only_searches_skip_the_keyword_span_and_the_remainder() -> None:
    latency = {
        "embed_ms": 1.0,
        "dense_ms": 1.0,
        "fusion_ms": 1.0,
        "rerank_ms": 1.0,
        "total_ms": 4.0,
    }
    trace = build_trace(latency_ms=latency, checkpoints={"dense": []}, relevant=set())
    assert [span["id"] for span in trace["spans"]] == ["search", "embed", "dense", "fuse", "rerank"]
    # A total shorter than its steps (rounding) never makes a negative remainder.
    short = build_trace(latency_ms={**latency, "total_ms": 3.9}, checkpoints={})
    assert short["total_ms"] == 4.0


@pytest.mark.parametrize(
    ("found", "expected"),
    [
        (
            {"dense": True, "keyword": False, "fused": True, "candidates": True, "final": False},
            "final",
        ),
        ({"dense": False, "keyword": True, "fused": False}, "fused"),
        ({"dense": False, "keyword": False, "fused": False}, None),
        ({"dense": True, "fused": True, "candidates": True, "reranked": True, "final": True}, None),
    ],
)
def test_lost_at_follows_the_evidence_downstream(found: dict, expected: str | None) -> None:
    assert (
        lost_at([{"checkpoint": key, "found": value} for key, value in found.items()]) == expected
    )


def test_a_profile_has_percentiles_histograms_and_the_slowest_queries() -> None:
    latencies = {f"q{i}": {"embed_ms": float(i), "total_ms": float(i * 2)} for i in range(1, 101)}
    profile = latency_profile(latencies, bins=10, slowest=3)
    assert profile["queries"] == 100
    total = profile["steps"]["total_ms"]
    assert total["n"] == 100
    assert total["max"] == 200.0
    assert total["p50"] == pytest.approx(101.0)
    assert total["min"] == 2.0
    assert total["p25"] < total["p50"] < total["p75"] < total["p95"]
    assert sum(total["histogram"]["counts"]) == 100
    assert len(total["histogram"]["edges"]) == 11
    assert [row["item_id"] for row in profile["slowest"]] == ["q100", "q99", "q98"]
    assert profile["slowest"][0]["steps"] == {"embed_ms": 100.0}
    assert "keyword_ms" not in profile["steps"]
    assert latency_profile({})["steps"] == {}
