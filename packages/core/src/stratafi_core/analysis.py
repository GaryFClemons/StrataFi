"""An experiment's results views.

Each view reads the finished trials (``TrialResult``): their levels, the objective per query,
their measures, their eval runs' file-type slices and where their misses were lost.

- ``leaderboard``: the trials, best objective first (the goal decides which way), with their
  intervals, constraints, latency and cost.
- ``factor_effects``: for each factor and each level but the first, the paired effect of
  switching from the first level to it, averaged over every pair of trials that differ in that
  factor only. Per query, the difference is averaged over the pairs; the bootstrap over queries
  gives the interval and a sign-flip test the p-value. Pairs are whatever the trials allow, so
  grids and partial designs (random search, exclusions) both work; a level with no pair has no
  effect.
- ``interactions``: for each pair of factors, the mean objective of the trials in each cell.
- ``pareto``: the objective against a cost (latency or cost per 1,000 queries), with the
  trials no other trial beats on both marked as the frontier.
- ``file_types``: each trial's objective in each gold file type's slice, and, for each factor
  on the parse stage, the level that does best per file type: a routing table to apply.
- ``where_lost``: per trial, how many queries were found or lost at parse, retrieval or rerank;
  and the queries most trials missed, with where.

Guardrails, in ``guardrails``:

- every trial is compared with the leader on the same queries (paired bootstrap interval and
  sign-flip test), and Holm's method adjusts the p-values across those comparisons; the trials
  not significantly worse than the leader share its group
- a trial whose difference from the leader lies wholly within the equivalence margin (0.02 on
  the objective by default) is practically equivalent, and among the leader and those, the
  cheapest is recommended: then the fastest, then the one with the fewest factors off their
  first level
- sample-size hints: the leader's interval width, the smallest paired difference the queries
  detect, and a warning when fewer than 50 queries decide

Numbers, IDs and labels only.
"""

import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stratafi_core.metrics import LOST
from stratafi_core.stats import (
    DEFAULT_ALPHA,
    DEFAULT_MARGIN,
    DEFAULT_RESAMPLES,
    FEW_QUERIES,
    bootstrap_mean,
    compare_paired,
    holm,
    min_detectable_difference,
    paired_randomization,
)

HEADLINE = ("hit", "recall", "mrr", "ndcg")


@dataclass(frozen=True)
class FactorInfo:
    """A factor as the views need it: its path, its levels' labels, and the stage it's on."""

    path: str
    labels: tuple[str, ...]
    stage: str = "query"
    levels: tuple[Any, ...] = ()


@dataclass(frozen=True)
class TrialResult:
    ordinal: int
    levels: tuple[int, ...]
    values: Mapping[str, str]
    objective: Mapping[str, Any] | None  # mean, low, high, n
    queries: Mapping[str, float]  # item ID → the objective's metric for that query
    measures: Mapping[str, Any] = field(default_factory=dict)
    feasible: bool | None = None
    constraints: Sequence[Mapping[str, Any]] = ()
    slices: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # file type → slice
    where_lost: Mapping[str, int] = field(default_factory=dict)
    lost: Mapping[str, str | None] = field(default_factory=dict)  # item ID → lost_at, misses
    eval_run_id: str | None = None

    @property
    def mean(self) -> float | None:
        value = (self.objective or {}).get("mean")
        return float(value) if isinstance(value, int | float) else None


def _better(goal: str) -> int:
    return 1 if goal == "maximize" else -1


def leaderboard(trials: Sequence[TrialResult], *, goal: str) -> list[dict[str, Any]]:
    """Trials with an objective, best first, ranked; ties keep plan order."""
    scored = [trial for trial in trials if trial.mean is not None]
    sign = _better(goal)
    scored.sort(key=lambda trial: (-sign * (trial.mean or 0.0), trial.ordinal))
    rows = []
    for rank, trial in enumerate(scored, start=1):
        rows.append(
            {
                "rank": rank,
                "ordinal": trial.ordinal,
                "values": dict(trial.values),
                "objective": dict(trial.objective or {}),
                "feasible": trial.feasible,
                "constraints": [dict(row) for row in trial.constraints],
                "latency_ms_p95": trial.measures.get("latency_ms.p95"),
                "cost_per_1k_queries_usd": trial.measures.get("cost_per_1k_queries_usd"),
                "ingest_cost_usd": trial.measures.get("ingest_cost_usd"),
                "eval_run_id": trial.eval_run_id,
            }
        )
    return rows


def _common_queries(trials: Sequence[TrialResult]) -> list[str]:
    if not trials:
        return []
    common = set(trials[0].queries)
    for trial in trials[1:]:
        common &= set(trial.queries)
    return sorted(common)


def factor_effects(
    trials: Sequence[TrialResult],
    factors: Sequence[FactorInfo],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Each factor's paired effects against its first level (the module docstring says how)."""
    by_levels = {trial.levels: trial for trial in trials if trial.queries}
    queries = _common_queries(list(by_levels.values()))
    effects = []
    for index, factor in enumerate(factors):
        rows = []
        for level in range(1, len(factor.labels)):
            pairs = []
            for levels, trial in by_levels.items():
                if levels[index] != 0:
                    continue
                other = by_levels.get((*levels[:index], level, *levels[index + 1 :]))
                if other is not None:
                    pairs.append((trial, other))
            row: dict[str, Any] = {
                "level": level,
                "label": factor.labels[level],
                "pairs": len(pairs),
                "n": 0,
            }
            if pairs and queries:
                differences = np.array(
                    [[b.queries[q] - a.queries[q] for a, b in pairs] for q in queries]
                ).mean(axis=1)
                estimate = bootstrap_mean(differences, resamples=resamples, seed=seed)
                if estimate is not None:
                    row.update(
                        n=estimate.n,
                        delta=estimate.mean,
                        low=estimate.low,
                        high=estimate.high,
                        p_value=paired_randomization(
                            np.zeros_like(differences), differences, resamples=resamples, seed=seed
                        ),
                    )
            rows.append(row)
        effects.append(
            {
                "path": factor.path,
                "reference": factor.labels[0],
                "levels": rows,
            }
        )
    return effects


def interactions(
    trials: Sequence[TrialResult], factors: Sequence[FactorInfo]
) -> list[dict[str, Any]]:
    """For each pair of factors, the mean objective of each cell's trials, and their count."""
    found = []
    for (i, a), (j, b) in itertools.combinations(enumerate(factors), 2):
        sums: dict[tuple[int, int], list[float]] = defaultdict(list)
        for trial in trials:
            if trial.mean is not None:
                sums[(trial.levels[i], trial.levels[j])].append(trial.mean)
        cells = [
            [
                {
                    "mean": round(float(np.mean(sums[(row, column)])), 6)
                    if sums[(row, column)]
                    else None,
                    "trials": len(sums[(row, column)]),
                }
                for column in range(len(b.labels))
            ]
            for row in range(len(a.labels))
        ]
        found.append(
            {"rows": a.path, "columns": b.path, "row_labels": list(a.labels),
             "column_labels": list(b.labels), "cells": cells}
        )  # fmt: skip
    return found


def pareto(trials: Sequence[TrialResult], *, cost: str, goal: str) -> list[dict[str, Any]]:
    """Each trial's objective against ``cost`` (a measure: lower is better), with the frontier:
    the trials that no other beats on both, without being worse on either."""
    sign = _better(goal)
    points = [
        (trial, float(trial.mean), float(trial.measures[cost]))
        for trial in trials
        if trial.mean is not None and isinstance(trial.measures.get(cost), int | float)
    ]
    rows = []
    for trial, objective, spend in points:
        dominated = any(
            sign * other_objective >= sign * objective
            and other_spend <= spend
            and (sign * other_objective > sign * objective or other_spend < spend)
            for other, other_objective, other_spend in points
            if other is not trial
        )
        rows.append(
            {
                "ordinal": trial.ordinal,
                "objective": objective,
                "cost": spend,
                "frontier": not dominated,
                "values": dict(trial.values),
            }
        )
    return sorted(rows, key=lambda row: (row["cost"], row["ordinal"]))


def slice_metric(objective_metric: str, k_eval: int) -> str:
    """The metric the file-type slices hold that stands for the objective: the objective
    itself when it's a headline metric (hit and recall at the eval set's k, MRR and nDCG at
    10), else recall at k."""
    headline = {f"hit@{k_eval}", f"recall@{k_eval}", "mrr@10", "ndcg@10"}
    return objective_metric if objective_metric in headline else f"recall@{k_eval}"


def file_types(
    trials: Sequence[TrialResult],
    factors: Sequence[FactorInfo],
    *,
    rule: str,
    metric: str,
    goal: str,
) -> dict[str, Any]:
    """Each gold file type's slice for every trial, the best trial per type, and per parse
    factor the level that does best per type (its trials' mean), as a routing table."""
    types = sorted({kind for trial in trials for kind in trial.slices})
    sign = _better(goal)
    rows = []
    best_trials: dict[str, int] = {}
    for kind in types:
        cells: dict[str, Any] = {}
        for trial in trials:
            estimate = (
                ((trial.slices.get(kind) or {}).get("metrics") or {}).get(rule, {}).get("final", {})
            ).get(metric)
            if isinstance(estimate, Mapping):
                cells[str(trial.ordinal)] = dict(estimate)
        if cells:
            best_trials[kind] = int(
                max(cells, key=lambda ordinal: (sign * cells[ordinal]["mean"], -int(ordinal)))
            )
        queries = max(
            (int((trial.slices.get(kind) or {}).get("with_evidence", 0)) for trial in trials),
            default=0,
        )
        rows.append(
            {
                "file_type": kind,
                "queries": queries,
                "few_queries": queries < FEW_QUERIES,
                "trials": cells,
            }
        )
    routing = []
    for index, factor in enumerate(factors):
        if factor.stage != "parse":
            continue
        best: dict[str, Any] = {}
        for kind in types:
            by_level: dict[int, list[float]] = defaultdict(list)
            for trial in trials:
                estimate = (
                    ((trial.slices.get(kind) or {}).get("metrics") or {})
                    .get(rule, {})
                    .get("final", {})
                    .get(metric)
                )
                if isinstance(estimate, Mapping):
                    by_level[trial.levels[index]].append(float(estimate["mean"]))
            if not by_level:
                continue
            level = max(by_level, key=lambda key: (sign * float(np.mean(by_level[key])), -key))
            best[kind] = {
                "level": level,
                "label": factor.labels[level],
                "value": factor.levels[level] if factor.levels else None,
                "mean": round(float(np.mean(by_level[level])), 6),
                "trials": len(by_level[level]),
            }
        routing.append({"path": factor.path, "best": best})
    return {"metric": metric, "types": rows, "best_trial": best_trials, "routing": routing}


def where_lost(trials: Sequence[TrialResult], *, hardest: int = 20) -> dict[str, Any]:
    """Per trial, its queries found and lost by stage; and the queries most trials missed."""
    per_trial = [
        {
            "ordinal": trial.ordinal,
            "values": dict(trial.values),
            **{bucket: int(trial.where_lost.get(bucket, 0)) for bucket in ("found", *LOST)},
            "filtered": int(trial.where_lost.get("filtered", 0)),
        }
        for trial in trials
        if trial.where_lost
    ]
    missed: dict[str, Counter[str]] = defaultdict(Counter)
    for trial in trials:
        for item_id, stage in trial.lost.items():
            if stage:
                missed[item_id][stage] += 1
    worst = sorted(missed.items(), key=lambda pair: (-sum(pair[1].values()), pair[0]))[:hardest]
    return {
        "trials": per_trial,
        "hardest": [
            {"item_id": item_id, "missed_by": sum(stages.values()), "lost_at": dict(stages)}
            for item_id, stages in worst
        ],
    }


def guardrails(
    trials: Sequence[TrialResult],
    *,
    goal: str,
    alpha: float = DEFAULT_ALPHA,
    margin: float = DEFAULT_MARGIN,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """Significance groups against the leader, practical equivalence, the recommended trial,
    and sample-size hints (the module docstring says how)."""
    scored = [trial for trial in trials if trial.mean is not None and trial.queries]
    board = leaderboard(scored, goal=goal)
    if not board:
        return {"leader": None, "trials": [], "recommended": None, "sample_size": None}
    by_ordinal = {trial.ordinal: trial for trial in scored}
    leader = by_ordinal[board[0]["ordinal"]]
    queries = _common_queries(scored)
    sign = _better(goal)
    rows: list[dict[str, Any]] = []
    spreads: list[float] = []
    for entry in board[1:]:
        trial = by_ordinal[entry["ordinal"]]
        # Positive: how much worse than the leader, whichever way the goal points.
        leader_values = [sign * leader.queries[q] for q in queries]
        trial_values = [sign * trial.queries[q] for q in queries]
        compared = compare_paired(trial_values, leader_values, resamples=resamples, seed=seed)
        if compared is None:
            continue
        spreads.append(
            float(np.std(np.subtract(leader_values, trial_values), ddof=1))
            if len(queries) > 1
            else 0.0
        )
        rows.append(
            {
                "ordinal": trial.ordinal,
                "behind": compared.delta,
                "low": compared.low,
                "high": compared.high,
                "p_value": compared.p_value,
            }
        )
    for row, adjusted in zip(rows, holm([row["p_value"] for row in rows]), strict=True):
        row["p_holm"] = adjusted
        row["group"] = "tied" if adjusted >= alpha else "worse"
        row["equivalent"] = -margin <= row["low"] and row["high"] <= margin
    rows.insert(
        0,
        {
            "ordinal": leader.ordinal,
            "behind": 0.0,
            "low": 0.0,
            "high": 0.0,
            "p_value": 1.0,
            "p_holm": 1.0,
            "group": "leader",
            "equivalent": True,
        },
    )
    candidates = [by_ordinal[row["ordinal"]] for row in rows if row["equivalent"]]

    def preference(trial: TrialResult) -> tuple[float, float, int, int]:
        cost = trial.measures.get("cost_per_1k_queries_usd")
        latency = trial.measures.get("latency_ms.p95")
        return (
            float(cost) if isinstance(cost, int | float) else math.inf,
            float(latency) if isinstance(latency, int | float) else math.inf,
            sum(1 for level in trial.levels if level != 0),
            trial.ordinal,
        )

    recommended = min(candidates, key=preference)
    n = len(queries)
    width = (leader.objective or {}).get("high", 0.0) - (leader.objective or {}).get("low", 0.0)
    spread = float(np.median(spreads)) if spreads else None
    return {
        "leader": leader.ordinal,
        "alpha": alpha,
        "margin": margin,
        "trials": rows,
        "recommended": recommended.ordinal,
        "sample_size": {
            "queries": n,
            "ci_width": round(float(width), 6),
            "min_detectable_difference": (
                min_detectable_difference(spread, n, alpha=alpha) if spread is not None else None
            ),
            "warnings": ["few_queries"] if n < FEW_QUERIES else [],
        },
    }
