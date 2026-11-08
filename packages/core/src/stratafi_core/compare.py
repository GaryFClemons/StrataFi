"""Comparing two eval runs, query by query.

Two eval runs of the same eval set version are paired: each query was asked of both
configurations. For each metric, ``compare_runs`` takes the queries with evidence that both runs
scored, in the same order, and reports B against A (``stratafi_core.stats.compare_paired``): the
means, B minus A with its bootstrap interval, a paired randomization test's p-value, and wins,
losses and ties. Holm's method adjusts the p-values across the metrics compared. Per query, the
first metric's values show which queries B won and lost.

Everything here is numbers and item IDs.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from stratafi_core.stats import DEFAULT_RESAMPLES, FEW_QUERIES, compare_paired, holm


def _value(score: Mapping[str, Any], rule: str, checkpoint: str, name: str) -> float | None:
    value = score.get("metrics", {}).get(rule, {}).get(checkpoint, {}).get(name)
    return float(value) if isinstance(value, int | float) else None


def compare_runs(
    a: Mapping[str, Mapping[str, Any]],
    b: Mapping[str, Mapping[str, Any]],
    *,
    rule: str,
    checkpoint: str,
    metrics: Sequence[str],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """B against A. ``a`` and ``b`` map item IDs to per-query scores (``QueryScore`` as
    stored); only items with evidence in both are paired."""
    paired = sorted(
        item_id
        for item_id in set(a) & set(b)
        if a[item_id].get("has_evidence") and b[item_id].get("has_evidence")
    )
    rows: list[dict[str, Any]] = []
    for name in metrics:
        values = [
            (_value(a[item], rule, checkpoint, name), _value(b[item], rule, checkpoint, name))
            for item in paired
        ]
        kept = [
            (first, second) for first, second in values if first is not None and second is not None
        ]
        result = compare_paired(
            [first for first, _ in kept],
            [second for _, second in kept],
            resamples=resamples,
            seed=seed,
        )
        rows.append({"metric": name, **(result.as_dict() if result else {"n": 0})})
    adjusted = holm([row.get("p_value", 1.0) for row in rows])
    for row, p_holm in zip(rows, adjusted, strict=True):
        if "p_value" in row:
            row["p_holm"] = p_holm
    queries = []
    if metrics:
        for item in paired:
            first = _value(a[item], rule, checkpoint, metrics[0])
            second = _value(b[item], rule, checkpoint, metrics[0])
            if first is not None and second is not None:
                queries.append(
                    {"item_id": item, "a": first, "b": second, "delta": round(second - first, 6)}
                )
        queries.sort(key=lambda row: (row["delta"], row["item_id"]))
    warnings = ["few_queries"] if len(paired) < FEW_QUERIES else []
    return {"paired": len(paired), "metrics": rows, "queries": queries, "warnings": warnings}
