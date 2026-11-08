"""Comparing up to six eval runs (unit tier), on synthetic scores."""

import pytest

from stratafi_core.comparison import RunScores, compare_many, curves, funnel, query_rows

RULE = "knowledge-hit-v1"


def score(recall: float, *, evidence: bool = True, fused: float | None = None) -> dict:
    final = {"recall@5": recall, "hit@5": float(recall > 0)}
    metrics = {RULE: {"final": final}}
    if fused is not None:
        metrics[RULE]["fused"] = {"recall@5": fused}
    return {"has_evidence": evidence, "metrics": metrics}


def summary(**overrides) -> dict:
    base = {
        "metrics": {
            RULE: {
                "dense": {"hit@all": {"mean": 0.8}, "recall@all": {"mean": 0.7}},
                "fused": {"hit@all": {"mean": 0.9}, "recall@all": {"mean": 0.85}},
                "final": {
                    "recall@1": {"mean": 0.4},
                    "recall@5": {"mean": 0.7},
                    "recall@all": {"mean": 0.8},
                    "hit@all": {"mean": 0.88},
                    "mrr@10": {"mean": 0.6},
                },
            }
        },
        "latency_ms": {"total_ms": {"p50": 10.0, "p95": 20.0, "mean": 11.0}},
        "where_lost": {"found": 50, "parse": 1, "retrieval": 6, "rerank": 3, "filtered": 0},
        "violations": {"avoid": {"count": 1, "of": 5}, "must_not_return": {"count": 0, "of": 60}},
        "findability": {"rate": 0.95},
        "slices": {
            "file_type": {
                "pdf": {"metrics": {RULE: {"final": {"recall@5": {"mean": 0.5}}}}},
            }
        },
    }
    base.update(overrides)
    return base


ITEMS = [f"q{i:02d}" for i in range(60)]


def run(identifier: str, recalls: list[float], **extra) -> RunScores:
    scores = {item: score(value) for item, value in zip(ITEMS, recalls, strict=True)}
    scores["abstain"] = score(0.0, evidence=False)
    return RunScores(identifier, scores, summary(**extra))


BASE = [1.0 if index % 3 else 0.0 for index in range(60)]  # 40 of 60 found
BETTER = [1.0 if index % 6 else 0.0 for index in range(60)]  # 50 of 60
SAME = list(BASE)


def test_metrics_are_paired_against_the_baseline_and_grouped_against_the_best() -> None:
    result = compare_many(
        [run("a", BASE), run("b", BETTER), run("c", SAME)],
        rule=RULE,
        checkpoint="final",
        metrics=["recall@5", "hit@5"],
        resamples=500,
    )
    assert result["paired"] == 60  # the abstention isn't paired
    row = result["table"][0]
    assert row["metric"] == "recall@5"
    assert row["n"] == 60
    a, b, c = row["cells"]
    assert a["mean"] == pytest.approx(40 / 60, abs=1e-6)
    assert b["mean"] == pytest.approx(50 / 60, abs=1e-6)
    assert a["low"] <= a["mean"] <= a["high"]
    # The baseline has no delta; the others do, with Holm across the two of them.
    assert "delta" not in a
    assert b["delta"] == pytest.approx(10 / 60, abs=1e-6)
    assert b["delta_low"] <= b["delta"] <= b["delta_high"]
    assert b["p_holm"] >= b["p_value"]
    assert (b["wins"], b["losses"], b["ties"]) == (10, 0, 50)
    assert c["delta"] == 0.0
    assert c["p_value"] == 1.0
    # b is best; a and c are behind it by the same 10 queries.
    assert row["best"] == 1
    assert b["group"] == "leader"
    assert b["behind_best"] == 0.0
    assert {a["group"], c["group"]} <= {"tied", "worse"}
    assert result["warnings"] == []


def test_another_baseline_moves_the_deltas() -> None:
    result = compare_many(
        [run("a", BASE), run("b", BETTER)],
        rule=RULE,
        checkpoint="final",
        metrics=["recall@5"],
        baseline=1,
        resamples=200,
    )
    a, b = result["table"][0]["cells"]
    assert "delta" not in b
    assert a["delta"] == pytest.approx(-10 / 60, abs=1e-6)
    assert result["baseline"] == 1


def test_curves_funnel_latency_where_lost_and_slices_come_from_each_summary() -> None:
    result = compare_many(
        [run("a", BASE), run("b", BETTER)],
        rule=RULE,
        checkpoint="final",
        metrics=["recall@5"],
        resamples=200,
    )
    recall = next(row for row in result["curves"] if row["metric"] == "recall")
    assert recall["ks"] == ["1", "5", "all"]
    assert recall["series"][0] == [0.4, 0.7, 0.8]
    assert result["funnel"]["checkpoints"] == ["dense", "fused", "final"]
    assert result["funnel"]["hit"][0] == [0.8, 0.9, 0.88]
    assert result["latency"]["steps"] == ["total_ms"]
    assert result["latency"]["p95"] == [[20.0], [20.0]]
    assert result["where_lost"]["counts"][0] == [50, 1, 6, 3, 0]
    assert result["violations"][0] == {"avoid": 1, "must_not_return": 0}
    assert result["findability"] == [0.95, 0.95]
    assert result["slices"] == [{"kind": "file_type", "label": "pdf", "values": [0.5, 0.5]}]


def test_queries_come_widest_spread_first() -> None:
    runs = [run("a", BASE), run("b", BETTER), run("c", SAME)]
    rows = query_rows(runs, RULE, "final", "recall@5")
    assert len(rows) == 60
    disagree = [row for row in rows if row["spread"] > 0]
    assert len(disagree) == 10
    assert rows[: len(disagree)] == disagree
    assert disagree[0]["values"] == [0.0, 1.0, 0.0]


def test_few_queries_warn_and_bad_inputs_are_refused() -> None:
    few = [RunScores(name, {"q": score(1.0)}, None) for name in ("a", "b")]
    result = compare_many(few, rule=RULE, checkpoint="final", metrics=["recall@5"], resamples=100)
    assert result["warnings"] == ["few_queries"]
    assert result["table"][0]["n"] == 1
    for runs, baseline, message in (
        (few[:1], 0, "two to six"),
        (few * 4, 0, "two to six"),
        (few, 2, "baseline"),
    ):
        with pytest.raises(ValueError, match=message):
            compare_many(
                runs, rule=RULE, checkpoint="final", metrics=["recall@5"], baseline=baseline
            )
    with pytest.raises(ValueError, match="name a metric"):
        compare_many(few, rule=RULE, checkpoint="final", metrics=[])


def test_runs_without_shared_queries_compare_nothing() -> None:
    a = RunScores("a", {"q1": score(1.0)}, None)
    b = RunScores("b", {"q2": score(1.0)}, None)
    result = compare_many([a, b], rule=RULE, checkpoint="final", metrics=["recall@5"])
    assert result["paired"] == 0
    assert result["table"][0]["best"] is None
    assert curves(None, RULE, "final") == {}
    assert funnel(None, RULE) == {}
