"""Comparing two eval runs query by query."""

from typing import Any

from stratafi_core.compare import compare_runs

RULE = "knowledge-hit-v1"


def score(recall: float, mrr: float = 0.5, *, evidence: bool = True) -> dict[str, Any]:
    if not evidence:
        return {"has_evidence": False, "metrics": {}}
    return {
        "has_evidence": True,
        "metrics": {RULE: {"final": {"recall@5": recall, "mrr@10": mrr}}},
    }


def test_identical_runs_compare_as_no_difference() -> None:
    runs = {f"q{n}": score(n % 2) for n in range(60)}
    result = compare_runs(runs, runs, rule=RULE, checkpoint="final", metrics=["recall@5"])
    (row,) = result["metrics"]
    assert result["paired"] == 60
    assert (row["delta"], row["low"], row["high"], row["p_value"]) == (0.0, 0.0, 0.0, 1.0)
    assert (row["wins"], row["losses"], row["ties"]) == (0, 0, 60)
    assert result["warnings"] == []


def test_a_better_run_wins_and_queries_show_where() -> None:
    a = {f"q{n}": score(0.0 if n < 30 else 1.0) for n in range(60)}
    b = {f"q{n}": score(1.0) for n in range(60)}
    a["q-abstain"] = b["q-abstain"] = score(0, evidence=False)
    a["q-only-a"] = score(1.0)
    result = compare_runs(a, b, rule=RULE, checkpoint="final", metrics=["recall@5", "mrr@10"])
    recall, mrr = result["metrics"]
    assert result["paired"] == 60
    assert recall["delta"] == 0.5
    assert recall["p_value"] < 0.01
    assert recall["p_holm"] >= recall["p_value"]
    assert (recall["wins"], recall["losses"], recall["ties"]) == (30, 0, 30)
    assert mrr["delta"] == 0.0
    assert mrr["p_holm"] == 1.0
    assert [row["delta"] for row in result["queries"]][-1] == 1.0
    assert {row["item_id"] for row in result["queries"]} == {f"q{n}" for n in range(60)}


def test_few_paired_queries_are_flagged() -> None:
    runs = {f"q{n}": score(1.0) for n in range(10)}
    result = compare_runs(runs, runs, rule=RULE, checkpoint="final", metrics=["recall@5"])
    assert result["warnings"] == ["few_queries"]
    empty = compare_runs({}, {}, rule=RULE, checkpoint="final", metrics=["recall@5"])
    assert empty["metrics"] == [{"metric": "recall@5", "n": 0}]
