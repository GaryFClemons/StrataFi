"""An experiment's results views: paired factor effects recover known effects in
synthetic data, in a full grid and in a partial design; the leaderboard, interactions, Pareto
frontier, file-type routing and where queries were lost."""

import itertools

import numpy as np
import pytest

from stratafi_core.analysis import (
    FactorInfo,
    TrialResult,
    factor_effects,
    file_types,
    guardrails,
    interactions,
    leaderboard,
    pareto,
    slice_metric,
    where_lost,
)
from stratafi_core.stats import min_detectable_difference, repeat_summary

QUERIES = [f"q{n:02d}" for n in range(80)]
CHUNK = FactorInfo("chunk.params.max_tokens", ("400", "600", "800"), stage="chunk")
RERANK = FactorInfo("rerank", ("rerank/none", "rerank/gateway"))
# The truth: each level's effect on every query's score, plus noise.
CHUNK_EFFECTS = (0.0, 0.08, -0.05)
RERANK_EFFECTS = (0.0, 0.12)


def synthetic(levels: tuple[int, int], *, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    difficulty = np.random.default_rng(99).uniform(0.2, 0.6, len(QUERIES))
    noise = rng.normal(0, 0.03, len(QUERIES))
    scores = difficulty + CHUNK_EFFECTS[levels[0]] + RERANK_EFFECTS[levels[1]] + noise
    queries = {query: float(score) for query, score in zip(QUERIES, scores, strict=True)}
    mean = float(np.mean(scores))
    return TrialResult(
        ordinal=levels[0] * 2 + levels[1],
        levels=levels,
        values={CHUNK.path: CHUNK.labels[levels[0]], RERANK.path: RERANK.labels[levels[1]]},
        objective={"mean": mean, "low": mean - 0.05, "high": mean + 0.05, "n": len(QUERIES)},
        queries=queries,
        measures={"latency_ms.p95": 100.0 + 50 * levels[1], "cost_per_1k_queries_usd": 0.1},
    )


GRID = [synthetic(levels, seed=n) for n, levels in enumerate(itertools.product(range(3), range(2)))]


def effect(effects: list[dict], path: str, level: int) -> dict:
    factor = next(item for item in effects if item["path"] == path)
    return next(row for row in factor["levels"] if row["level"] == level)


def test_effects_recover_known_effects_in_a_grid() -> None:
    effects = factor_effects(GRID, [CHUNK, RERANK], resamples=1000, seed=1)
    for level in (1, 2):
        row = effect(effects, CHUNK.path, level)
        truth = CHUNK_EFFECTS[level] - CHUNK_EFFECTS[0]
        assert row["pairs"] == 2  # one per reranker level
        assert row["n"] == len(QUERIES)
        assert row["low"] <= truth <= row["high"]
        assert row["delta"] == pytest.approx(truth, abs=0.02)
        assert row["p_value"] < 0.01
    rerank = effect(effects, RERANK.path, 1)
    assert rerank["pairs"] == 3
    assert rerank["low"] <= RERANK_EFFECTS[1] <= rerank["high"]
    assert effects[0]["reference"] == "400"


def test_effects_work_in_a_partial_design() -> None:
    partial = [trial for trial in GRID if trial.levels not in {(1, 1), (2, 0)}]
    effects = factor_effects(partial, [CHUNK, RERANK], resamples=1000, seed=1)
    middle = effect(effects, CHUNK.path, 1)
    assert middle["pairs"] == 1  # only (0, 0) → (1, 0) is left
    assert middle["low"] <= CHUNK_EFFECTS[1] <= middle["high"]
    last = effect(effects, CHUNK.path, 2)
    assert last["pairs"] == 1
    assert last["low"] <= CHUNK_EFFECTS[2] <= last["high"]
    alone = factor_effects(
        [trial for trial in GRID if trial.levels == (0, 0)], [CHUNK, RERANK], resamples=200
    )
    assert effect(alone, CHUNK.path, 1) == {"level": 1, "label": "600", "pairs": 0, "n": 0}


def test_no_effect_is_no_effect() -> None:
    flat = [
        TrialResult(
            ordinal=level,
            levels=(level, 0),
            values={},
            objective={"mean": 0.5},
            queries=dict.fromkeys(QUERIES, 0.5),
        )
        for level in range(3)
    ]
    row = effect(factor_effects(flat, [CHUNK, RERANK], resamples=200), CHUNK.path, 1)
    assert (row["delta"], row["low"], row["high"], row["p_value"]) == (0.0, 0.0, 0.0, 1.0)


def test_the_leaderboard_ranks_by_the_goal() -> None:
    board = leaderboard(GRID, goal="maximize")
    assert [row["rank"] for row in board] == list(range(1, 7))
    best = board[0]
    assert best["values"] == {CHUNK.path: "600", RERANK.path: "rerank/gateway"}
    assert board[0]["objective"]["mean"] >= board[-1]["objective"]["mean"]
    worst_first = leaderboard(GRID, goal="minimize")
    assert worst_first[0]["ordinal"] == board[-1]["ordinal"]
    assert leaderboard([TrialResult(9, (0, 0), {}, None, {})], goal="maximize") == []


def test_interactions_hold_each_cells_mean() -> None:
    (pair,) = interactions(GRID, [CHUNK, RERANK])
    assert (pair["rows"], pair["columns"]) == (CHUNK.path, RERANK.path)
    assert len(pair["cells"]) == 3
    assert all(len(row) == 2 for row in pair["cells"])
    cell = pair["cells"][1][1]
    assert cell["trials"] == 1
    assert cell["mean"] == pytest.approx(GRID[3].mean)


def test_the_pareto_frontier_is_the_undominated_trials() -> None:
    def trial(ordinal: int, mean: float, latency: float) -> TrialResult:
        return TrialResult(
            ordinal, (ordinal, 0), {}, {"mean": mean}, {}, measures={"latency_ms.p95": latency}
        )

    points = [trial(0, 0.7, 100), trial(1, 0.8, 200), trial(2, 0.75, 250), trial(3, 0.6, 90)]
    found = {
        row["ordinal"]: row["frontier"]
        for row in pareto(points, cost="latency_ms.p95", goal="maximize")
    }
    assert found == {0: True, 1: True, 2: False, 3: True}
    assert [row["ordinal"] for row in pareto(points, cost="latency_ms.p95", goal="maximize")] == [
        3,
        0,
        1,
        2,
    ]


def slices(pdf: float, markdown: float) -> dict:
    def one(mean: float) -> dict:
        estimate = {"mean": mean, "low": mean - 0.1, "high": mean + 0.1, "n": 10}
        return {
            "with_evidence": 10,
            "metrics": {"knowledge-hit-v1": {"final": {"recall@5": estimate}}},
        }

    return {"application/pdf": one(pdf), "text/markdown": one(markdown)}


def test_file_types_suggest_a_routing_table() -> None:
    parser = FactorInfo(
        'parse.route."application/pdf"',
        ("parse/markdown", "parse/docling"),
        stage="parse",
        levels=("parse/markdown", "parse/docling"),
    )
    trials = [
        TrialResult(0, (0, 0), {}, {"mean": 0.6}, {}, slices=slices(0.4, 0.9)),
        TrialResult(1, (0, 1), {}, {"mean": 0.62}, {}, slices=slices(0.42, 0.9)),
        TrialResult(2, (1, 0), {}, {"mean": 0.7}, {}, slices=slices(0.7, 0.88)),
        TrialResult(3, (1, 1), {}, {"mean": 0.72}, {}, slices=slices(0.75, 0.86)),
    ]
    table = file_types(
        trials, [parser, RERANK], rule="knowledge-hit-v1", metric="recall@5", goal="maximize"
    )
    assert [row["file_type"] for row in table["types"]] == ["application/pdf", "text/markdown"]
    assert table["types"][0]["trials"]["2"]["mean"] == 0.7
    assert table["best_trial"] == {"application/pdf": 3, "text/markdown": 0}
    (routing,) = table["routing"]
    assert routing["best"]["application/pdf"]["value"] == "parse/docling"
    assert routing["best"]["text/markdown"]["label"] == "parse/markdown"
    assert slice_metric("recall@5", 5) == "recall@5"
    assert slice_metric("precision@3", 5) == "recall@5"


def test_where_lost_counts_stages_and_finds_the_hardest_queries() -> None:
    trials = [
        TrialResult(
            0, (0, 0), {}, {"mean": 0.5}, {},
            where_lost={"found": 7, "parse": 1, "retrieval": 1, "rerank": 1, "filtered": 1},
            lost={"q1": "parse", "q2": "retrieval", "q3": "rerank"},
        ),
        TrialResult(
            1, (1, 0), {}, {"mean": 0.6}, {},
            where_lost={"found": 8, "parse": 1, "retrieval": 1, "rerank": 0},
            lost={"q1": "parse", "q2": "rerank"},
        ),
    ]  # fmt: skip
    found = where_lost(trials)
    assert found["trials"][0] == {
        "ordinal": 0,
        "values": {},
        "found": 7,
        "parse": 1,
        "retrieval": 1,
        "rerank": 1,
        "filtered": 1,
    }
    assert found["hardest"][:2] == [
        {"item_id": "q1", "missed_by": 2, "lost_at": {"parse": 2}},
        {"item_id": "q2", "missed_by": 2, "lost_at": {"retrieval": 1, "rerank": 1}},
    ]


# --- guardrails ------------------------------------------------------------------------


def shifted(ordinal: int, shift: float, *, cost: float, noise: float = 0.002) -> TrialResult:
    rng = np.random.default_rng(ordinal + 10)
    base = np.random.default_rng(7).uniform(0.3, 0.9, len(QUERIES))
    scores = base + shift + rng.normal(0, noise, len(QUERIES))
    mean = float(np.mean(scores))
    return TrialResult(
        ordinal=ordinal,
        levels=(ordinal, 0),
        values={},
        objective={"mean": mean, "low": mean - 0.04, "high": mean + 0.04, "n": len(QUERIES)},
        queries=dict(zip(QUERIES, scores.tolist(), strict=True)),
        measures={"cost_per_1k_queries_usd": cost, "latency_ms.p95": 100.0},
    )


def test_guardrails_group_against_the_leader_and_recommend_the_cheaper_equal() -> None:
    leader = shifted(0, 0.0, cost=0.5)
    wobble = [0.001 if n % 2 else -0.001 for n in range(len(QUERIES))]
    tied = TrialResult(
        ordinal=4,
        levels=(1, 1),
        values={},
        objective=leader.objective,
        queries={q: leader.queries[q] + w for q, w in zip(QUERIES, wobble, strict=True)},
        measures={"cost_per_1k_queries_usd": 0.3, "latency_ms.p95": 100.0},
    )
    trials = [
        leader,
        shifted(1, -0.003, cost=0.1),  # significantly worse, but within the margin, and cheaper
        shifted(2, -0.10, cost=0.0),  # clearly worse
        shifted(3, -0.03, cost=0.0, noise=0.2),  # noisy: not significantly worse, not equal
        tied,  # the same on average: tied, and equivalent
    ]
    found = guardrails(trials, goal="maximize", resamples=1000, seed=3)
    rows = {row["ordinal"]: row for row in found["trials"]}
    assert found["leader"] == 0
    assert rows[0]["group"] == "leader"
    assert (rows[4]["group"], rows[4]["equivalent"]) == ("tied", True)
    # Significance and equivalence are different questions.
    assert (rows[1]["group"], rows[1]["equivalent"]) == ("worse", True)
    assert (rows[2]["group"], rows[2]["equivalent"]) == ("worse", False)
    assert rows[2]["behind"] == pytest.approx(0.10, abs=0.01)
    assert (rows[3]["group"], rows[3]["equivalent"]) == ("tied", False)
    for row in found["trials"][1:]:
        assert row["p_holm"] >= row["p_value"]
    # The leader and trials 1 and 4 are within the margin; trial 1 costs least.
    assert found["recommended"] == 1
    hints = found["sample_size"]
    assert hints["queries"] == len(QUERIES)
    assert hints["warnings"] == []
    assert 0 < hints["min_detectable_difference"] < 0.1
    few = guardrails([shifted(0, 0, cost=0)], goal="maximize")
    assert few["recommended"] == 0
    assert few["sample_size"]["min_detectable_difference"] is None
    assert guardrails([], goal="maximize")["leader"] is None


def test_guardrails_follow_the_goal() -> None:
    trials = [shifted(0, 0.0, cost=0.1), shifted(1, -0.1, cost=0.1)]
    assert guardrails(trials, goal="minimize", resamples=200)["leader"] == 1


def test_detectable_differences_and_repeats() -> None:
    assert min_detectable_difference(0.3, 90) == pytest.approx(0.088594, abs=1e-6)
    assert min_detectable_difference(0.3, 1) is None
    summary = repeat_summary([0.5, 0.6, 0.7])
    assert summary is not None
    assert (summary.repeats, summary.mean, summary.sd) == (3, 0.6, 0.1)
    assert (summary.low, summary.high) == (0.5, 0.7)
    assert repeat_summary([0.4]).sd == 0.0  # type: ignore[union-attr]
    assert repeat_summary([]) is None
