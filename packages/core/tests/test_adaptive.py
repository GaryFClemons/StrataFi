"""Successive halving: the schedule, nested query subsets, pruning that respects
uncertainty, and the synthetic benchmark the step is done by."""

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from stratafi_core.adaptive import (
    Rung,
    halving_schedule,
    nominal_cost,
    prune,
    query_order,
    successive_halving,
)

QUERIES = [f"q{n:03d}" for n in range(90)]


def test_the_schedule_grows_by_eta_and_ends_on_every_query() -> None:
    assert halving_schedule(36, 90) == [Rung(10, 12), Rung(30, 4), Rung(90, 4)]
    assert halving_schedule(12, 63) == [Rung(10, 4), Rung(30, 2), Rung(63, 2)]
    assert halving_schedule(81, 90, eta=3, min_queries=5) == [
        Rung(5, 27),
        Rung(15, 9),
        Rung(45, 3),
        Rung(90, 3),
    ]
    assert halving_schedule(36, 90, eta=2, min_queries=30) == [
        Rung(30, 18),
        Rung(60, 9),
        Rung(90, 9),
    ]
    # One trial, or a first subset as big as the split: a grid.
    assert halving_schedule(1, 90) == [Rung(90, 1)]
    assert halving_schedule(5, 8) == [Rung(8, 5)]
    assert halving_schedule(0, 90) == halving_schedule(5, 0) == []


def test_the_nominal_cost_pays_each_query_once_per_trial() -> None:
    schedule = halving_schedule(36, 90)
    assert nominal_cost(schedule, 36) == {
        "cost": 36 * 10 + 12 * 20 + 4 * 60,
        "evaluations": 36 * 10 + 12 * 30 + 4 * 90,
        "grid": 36 * 90,
        "share": 0.2593,
    }
    assert nominal_cost([], 0)["share"] == 0.0


def test_rung_subsets_are_prefixes_of_one_seeded_order() -> None:
    order = query_order(QUERIES, 7)
    assert sorted(order) == QUERIES
    assert order != QUERIES
    assert query_order(list(reversed(QUERIES)), 7) == order
    assert query_order(QUERIES, 8) != order


def values(*rows: Sequence[float]) -> dict[int, list[float]]:
    return {trial: list(row) for trial, row in enumerate(rows)}


def test_pruning_drops_only_trials_clearly_behind_the_leader() -> None:
    leader = [1.0] * 18 + [0.0] * 2
    close = [1.0] * 17 + [0.0] * 3  # one query worse: it might still be the best
    middling = [1.0] * 15 + [0.0] * 5
    hopeless = [0.0] * 20
    decision = prune(values(close, hopeless, leader, middling), keep=1, resamples=500)
    assert decision.ranked == [2, 0, 3, 1]
    assert decision.kept == [2, 0, 3]
    assert decision.dropped == [1]
    assert decision.verdicts[2].against_leader is None  # the top ``keep`` stay as they are
    behind = decision.verdicts[1].against_leader
    assert behind is not None
    assert behind.high < 0
    assert decision.verdicts[3].as_dict()["against_leader"]["mean"] == -0.15
    # A wider rung keeps the top ``keep`` whatever the uncertainty says.
    assert prune(values(close, hopeless, leader), keep=3, resamples=500).kept == [2, 0, 1]
    # Minimizing mirrors it: the lowest value leads.
    costs = prune(values([0.1] * 20, [0.9] * 20, [0.0, 0.2] * 10), keep=1, goal="minimize")
    assert (sorted(costs.kept), costs.dropped) == ([0, 2], [1])


def test_pruning_needs_paired_values_and_keeps_everything_without_queries() -> None:
    with pytest.raises(ValueError, match="pairs"):
        prune({0: [1.0, 0.0], 1: [1.0]}, keep=1)
    assert prune({0: [], 1: []}, keep=1).kept == [0, 1]
    assert prune({}, keep=1).kept == []


def benchmark(seed: int, queries: int = 90) -> np.ndarray:
    """A 3x3x2x2 grid's per-query hits, like recall on single-evidence queries: each factor
    level adds to a trial's skill, each query has a difficulty every trial shares, and each
    (trial, query) has its own noise."""
    generator = np.random.default_rng(seed)
    levels = (3, 3, 2, 2)
    effects = [generator.normal(0, 0.35, size=count) for count in levels]
    combinations = list(itertools.product(*(range(count) for count in levels)))
    skill = np.array([sum(e[level] for e, level in zip(effects, combo, strict=True))
                      for combo in combinations])  # fmt: skip
    skill += generator.normal(0, 0.1, size=len(combinations))
    difficulty = generator.normal(0, 1, size=queries)
    noise = generator.normal(0, 0.6, size=(len(combinations), queries))
    return (skill[:, None] - difficulty[None, :] + noise > 0).astype(float)


def test_successive_halving_finds_the_grids_winner_at_half_the_cost() -> None:
    """On a synthetic benchmark, successive halving finds the grid's winner at <= 50% of
    the cost: 20 seeded 36-trial grids over 90 queries; it finds the best trial on every
    query in each of them, at less than half the grid's query evaluations on average."""
    shares, without_reuse = [], []
    for seed in range(20):
        hits = benchmark(seed)
        index = {query: column for column, query in enumerate(QUERIES)}

        def score(
            trials: Sequence[int],
            subset: Sequence[str],
            hits: np.ndarray = hits,
            index: Mapping[str, int] = index,
        ) -> Mapping[int, Sequence[float]]:
            columns = [index[query] for query in subset]
            return {trial: hits[trial, columns].tolist() for trial in trials}

        result = successive_halving(range(len(hits)), QUERIES, score, seed=seed)
        grid = hits.mean(axis=1)
        assert result.winner is not None
        assert grid[result.winner] == grid.max(), seed
        assert [rung.queries for rung in result.rungs] == [10, 30, 90]
        assert result.rungs[0].evaluated == list(range(36))
        assert len(result.rungs[-1].evaluated) < 36
        assert result.share < 1
        shares.append(result.share)
        without_reuse.append(result.share_without_reuse)
    assert np.mean(shares) <= 0.5
    # Without nested shards, the same search would score more of the grid's queries.
    assert np.mean(without_reuse) > np.mean(shares)
