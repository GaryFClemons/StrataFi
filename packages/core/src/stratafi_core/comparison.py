"""Comparing up to six eval runs, paired, against a baseline.

Eval runs of one eval set version asked the same queries, so ``compare_many`` pairs them as
``stratafi_core.compare`` pairs two, and adds what a comparison of several pipelines needs:

- **Per metric** (every metric here is better higher), on the queries with evidence that every
  run scored: each run's mean with its bootstrap interval; each run against the baseline (B minus
  the baseline, its paired interval, the sign-flip test's p-value, Holm's adjustment across the
  runs, wins, losses and ties); and its group against the best run, as the experiment
  leaderboard's: ``leader``, ``tied`` (not significantly worse after Holm's adjustment)
  or ``worse``.
- **From each run's own summary:** each metric at every k (curves), the share of queries with a
  relevant chunk anywhere in each checkpoint's list (the funnel, ``hit@all``) and its recall,
  latency per step, where queries were lost, violations, findability, and the first metric per
  file type and tag (slices).
- **Per query:** the first metric for each run, the queries the runs disagree on most first.

Numbers and item IDs only.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from stratafi_core.metrics import CHECKPOINTS, LATENCY_STEPS, METRICS
from stratafi_core.stats import (
    DEFAULT_ALPHA,
    DEFAULT_RESAMPLES,
    FEW_QUERIES,
    bootstrap_mean,
    compare_paired,
    holm,
)

MIN_RUNS = 2
MAX_RUNS = 6
MAX_QUERY_ROWS = 1000
LOST_BUCKETS = ("found", "parse", "retrieval", "rerank", "filtered")
_NAME_RE = re.compile(r"(?P<metric>[a-z]+)@(?P<k>\d{1,4}|all)")


@dataclass(frozen=True)
class RunScores:
    """One eval run's per-query scores (item ID → stored score) and its summary."""

    id: str
    scores: Mapping[str, Mapping[str, Any]]
    summary: Mapping[str, Any] | None


def score_value(score: Mapping[str, Any], rule: str, checkpoint: str, name: str) -> float | None:
    value = (score.get("metrics") or {}).get(rule, {}).get(checkpoint, {}).get(name)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _paired_items(runs: Sequence[RunScores]) -> list[str]:
    common = set(runs[0].scores)
    for run in runs[1:]:
        common &= set(run.scores)
    return sorted(
        item for item in common if all(run.scores[item].get("has_evidence") for run in runs)
    )


def _metric_row(
    runs: Sequence[RunScores],
    paired: Sequence[str],
    name: str,
    *,
    baseline: int,
    rule: str,
    checkpoint: str,
    resamples: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    values = [
        [score_value(run.scores[item], rule, checkpoint, name) for item in paired] for run in runs
    ]
    keep = [index for index in range(len(paired)) if all(v[index] is not None for v in values)]
    columns = [[float(v[index]) for index in keep] for v in values]  # type: ignore[arg-type]
    cells: list[dict[str, Any]] = []
    for column in columns:
        estimate = bootstrap_mean(column, resamples=resamples, seed=seed)
        cells.append(estimate.as_dict() if estimate is not None else {"n": 0})
    if not keep:
        return {"metric": name, "n": 0, "best": None, "cells": cells}
    against = [
        compare_paired(columns[baseline], column, resamples=resamples, seed=seed)
        if index != baseline
        else None
        for index, column in enumerate(columns)
    ]
    others = [index for index, result in enumerate(against) if result is not None]
    adjusted = holm([against[index].p_value for index in others])  # type: ignore[union-attr]
    for index, p_holm in zip(others, adjusted, strict=True):
        result = against[index]
        if result is None:  # pragma: no cover - others holds only compared runs
            continue
        cells[index].update(
            delta=result.delta,
            delta_low=result.low,
            delta_high=result.high,
            p_value=result.p_value,
            p_holm=p_holm,
            wins=result.wins,
            losses=result.losses,
            ties=result.ties,
        )
    means = [cell.get("mean") for cell in cells]
    best = max(range(len(cells)), key=lambda index: (means[index] or 0.0, -index))
    behind = [
        compare_paired(column, columns[best], resamples=resamples, seed=seed)
        if index != best
        else None
        for index, column in enumerate(columns)
    ]
    rest = [index for index, result in enumerate(behind) if result is not None]
    for index, p_holm in zip(
        rest,
        holm([behind[index].p_value for index in rest]),  # type: ignore[union-attr]
        strict=True,
    ):
        cells[index]["group"] = "tied" if p_holm >= alpha else "worse"
        cells[index]["behind_best"] = behind[index].delta  # type: ignore[union-attr]
    cells[best]["group"] = "leader"
    cells[best]["behind_best"] = 0.0
    return {"metric": name, "n": len(keep), "best": best, "cells": cells}


def _k_order(k: str) -> tuple[int, int]:
    return (1, 0) if k == "all" else (0, int(k))


def curves(summary: Mapping[str, Any] | None, rule: str, checkpoint: str) -> dict[str, dict]:
    """Each metric's mean at every k in a run's summary: metric → k → mean."""
    names = ((summary or {}).get("metrics") or {}).get(rule, {}).get(checkpoint, {})
    found: dict[str, dict[str, float]] = {}
    for name, estimate in names.items():
        match = _NAME_RE.fullmatch(name)
        if match is None or not isinstance(estimate, Mapping):
            continue
        mean = estimate.get("mean")
        if isinstance(mean, int | float):
            found.setdefault(match["metric"], {})[match["k"]] = float(mean)
    return {
        metric: dict(sorted(points.items(), key=lambda item: _k_order(item[0])))
        for metric, points in found.items()
    }


def funnel(summary: Mapping[str, Any] | None, rule: str) -> dict[str, dict[str, float | None]]:
    """For each checkpoint the run has: the share of queries with a relevant chunk anywhere in
    its list (``hit@all``) and its recall over the whole list."""
    checkpoints = ((summary or {}).get("metrics") or {}).get(rule, {})
    out: dict[str, dict[str, float | None]] = {}
    for checkpoint in CHECKPOINTS:
        names = checkpoints.get(checkpoint)
        if not names:
            continue
        out[checkpoint] = {
            key: (names.get(f"{key}@all") or {}).get("mean") for key in ("hit", "recall")
        }
    return out


def _slice_value(summary: Mapping[str, Any], rule: str, checkpoint: str, name: str) -> float | None:
    estimate = ((summary.get("metrics") or {}).get(rule) or {}).get(checkpoint, {}).get(name)
    mean = estimate.get("mean") if isinstance(estimate, Mapping) else None
    return float(mean) if isinstance(mean, int | float) else None


def slices(runs: Sequence[RunScores], rule: str, name: str) -> list[dict[str, Any]]:
    """The metric per file type and per tag, each run's own slice (slices hold the final
    list's headline metrics)."""
    labels: dict[tuple[str, str], list[float | None]] = {}
    for index, run in enumerate(runs):
        for kind, groups in ((run.summary or {}).get("slices") or {}).items():
            for label, summary in (groups or {}).items():
                row = labels.setdefault((kind, label), [None] * len(runs))
                row[index] = _slice_value(summary or {}, rule, "final", name)
    return [
        {"kind": kind, "label": label, "values": values}
        for (kind, label), values in sorted(labels.items())
        if any(value is not None for value in values)
    ]


def query_rows(
    runs: Sequence[RunScores], rule: str, checkpoint: str, name: str
) -> list[dict[str, Any]]:
    """The metric per query and run, for items with evidence in some run; the widest spread
    first, then by item."""
    items = sorted(
        {item for run in runs for item, score in run.scores.items() if score.get("has_evidence")}
    )
    rows = []
    for item in items:
        values = [
            score_value(run.scores[item], rule, checkpoint, name) if item in run.scores else None
            for run in runs
        ]
        known = [value for value in values if value is not None]
        spread = round(max(known) - min(known), 6) if known else 0.0
        rows.append({"item_id": item, "values": values, "spread": spread})
    rows.sort(key=lambda row: (-row["spread"], row["item_id"]))
    return rows[:MAX_QUERY_ROWS]


def compare_many(
    runs: Sequence[RunScores],
    *,
    rule: str,
    checkpoint: str,
    metrics: Sequence[str],
    baseline: int = 0,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """The comparison the module docstring describes. Raises ``ValueError`` for fewer than two
    or more than six runs, or a baseline that isn't one of them."""
    if not MIN_RUNS <= len(runs) <= MAX_RUNS:
        raise ValueError("compare two to six runs")
    if not 0 <= baseline < len(runs):
        raise ValueError("the baseline is one of the runs")
    if not metrics:
        raise ValueError("name a metric")
    paired = _paired_items(runs)
    table = [
        _metric_row(
            runs,
            paired,
            name,
            baseline=baseline,
            rule=rule,
            checkpoint=checkpoint,
            resamples=resamples,
            seed=seed,
            alpha=alpha,
        )
        for name in metrics
    ]
    summaries = [run.summary or {} for run in runs]
    per_run_curves = [curves(summary, rule, checkpoint) for summary in summaries]
    curve_rows = []
    for metric in METRICS:
        ks = sorted({k for found in per_run_curves for k in found.get(metric, {})}, key=_k_order)
        if ks:
            curve_rows.append(
                {
                    "metric": metric,
                    "ks": ks,
                    "series": [
                        [found.get(metric, {}).get(k) for k in ks] for found in per_run_curves
                    ],
                }
            )
    funnels = [funnel(summary, rule) for summary in summaries]
    checkpoints = [cp for cp in CHECKPOINTS if any(cp in found for found in funnels)]
    latency_steps = [
        step
        for step in LATENCY_STEPS
        if any(step in (s.get("latency_ms") or {}) for s in summaries)
    ]
    warnings = ["few_queries"] if len(paired) < FEW_QUERIES else []
    return {
        "baseline": baseline,
        "paired": len(paired),
        "table": table,
        "curves": curve_rows,
        "funnel": {
            "checkpoints": checkpoints,
            "hit": [[found.get(cp, {}).get("hit") for cp in checkpoints] for found in funnels],
            "recall": [
                [found.get(cp, {}).get("recall") for cp in checkpoints] for found in funnels
            ],
        },
        "latency": {
            "steps": latency_steps,
            **{
                stat: [
                    [
                        ((s.get("latency_ms") or {}).get(step) or {}).get(stat)
                        for step in latency_steps
                    ]
                    for s in summaries
                ]
                for stat in ("p50", "p95", "mean")
            },
        },
        "where_lost": {
            "buckets": list(LOST_BUCKETS),
            "counts": [
                [(s.get("where_lost") or {}).get(bucket) for bucket in LOST_BUCKETS]
                for s in summaries
            ],
        },
        "violations": [
            {
                kind: ((s.get("violations") or {}).get(kind) or {}).get("count")
                for kind in ("avoid", "must_not_return")
            }
            for s in summaries
        ],
        "findability": [(s.get("findability") or {}).get("rate") for s in summaries],
        "slices": slices(runs, rule, metrics[0]),
        "queries": query_rows(runs, rule, checkpoint, metrics[0]),
        "warnings": warnings,
    }
