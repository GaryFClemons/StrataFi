"""Traces: retrieval's steps for one query, as spans on a timeline.

A trace is built on request from what an eval run (or the search playground) already has: the
query's step latencies, its ranked lists at every checkpoint, and which chunks are relevant. The
retrieval engine (``stratafi_components.retrieval``) runs its steps one after another, in this
order, so each span starts where the one before it ended:

1. ``keyword``: BM25, giving the ``keyword`` list
2. ``embed``: the query's embedding (adapter included)
3. ``dense``: exact or ANN search, giving the ``dense`` list
4. ``fuse``: fusion, then the filter's second check, giving ``fused`` and ``candidates``
5. ``rerank``: the reranker over the candidates, giving ``reranked`` and, cut to ``top_k``,
   ``final``

Time the steps don't account for (the visibility masks before them, the final list after) is a
last ``other`` span. Each span carries its lists: their length, their first ten chunk IDs with
scores and relevance, the rank of their first relevant chunk, and whether the evidence is still
there. ``latency_profile`` summarizes a run's queries per step: percentiles, a histogram and the
slowest queries.

IDs and numbers only: texts stay in the chunks artifacts.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np

from stratafi_core.metrics import CHECKPOINTS

TOP = 10
SLOWEST = 10
STEPS: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    ("keyword", "Keyword search", "retriever", "keyword_ms", ("keyword",)),
    ("embed", "Embed the query", "embedding", "embed_ms", ()),
    ("dense", "Dense search", "retriever", "dense_ms", ("dense",)),
    ("fuse", "Fuse and filter", "fusion", "fusion_ms", ("fused", "candidates")),
    ("rerank", "Rerank", "reranker", "rerank_ms", ("reranked", "final")),
)
STEP_KEYS = tuple(step[3] for step in STEPS)
PROFILE_KEYS = (*STEP_KEYS, "total_ms")

Ranking = Sequence[Sequence[Any]]  # (chunk ID, score) pairs


def _round(value: float) -> float:
    return round(float(value), 3)


def list_summary(
    checkpoint: str, ranking: Ranking, relevant: Collection[str] | None
) -> dict[str, Any]:
    """One list: its length, first entries, and where its first relevant chunk is."""
    first = None
    if relevant:
        first = next(
            (rank for rank, entry in enumerate(ranking, start=1) if entry[0] in relevant), None
        )
    return {
        "checkpoint": checkpoint,
        "length": len(ranking),
        "first_relevant_rank": first,
        "found": first is not None if relevant is not None else None,
        "top": [
            {
                "rank": rank,
                "chunk_id": str(entry[0]),
                "score": round(float(entry[1]), 6),
                "relevant": (str(entry[0]) in relevant) if relevant is not None else None,
            }
            for rank, entry in enumerate(ranking[:TOP], start=1)
        ],
    }


def _model_label(record: Mapping[str, Any] | None) -> str | None:
    if not record:
        return None
    params = record.get("params") or {}
    return (
        record.get("alias") or params.get("alias") or params.get("model") or record.get("component")
    )


def _attributes(step: str, config: Mapping[str, Any], models: Mapping[str, Any]) -> dict[str, Any]:
    retrievers = config.get("retrievers") or {}
    if step == "keyword":
        return {"top_n": (retrievers.get("keyword") or {}).get("top_n")}
    if step == "embed":
        embed = models.get("embed") or {}
        attributes = {"model": _model_label(embed), "revision": embed.get("revision")}
        if adapter := models.get("adapter"):
            attributes["adapter"] = adapter.get("alias")
        return attributes
    if step == "dense":
        return {"top_n": (retrievers.get("dense") or {}).get("top_n")}
    if step == "fuse":
        fusion = config.get("fusion") or {}
        attributes = {"method": fusion.get("method"), "candidates": config.get("candidates")}
        if fusion.get("method") == "rrf":
            attributes["k"] = fusion.get("k")
        elif fusion.get("method") == "weighted":
            attributes["alpha"] = fusion.get("alpha")
        return attributes
    if step == "rerank":
        rerank = models.get("rerank") or {}
        return {
            "component": (config.get("rerank") or {}).get("component"),
            "model": _model_label(rerank),
            "revision": rerank.get("revision"),
            "top_k": config.get("top_k"),
        }
    return {}


def build_trace(
    *,
    latency_ms: Mapping[str, float],
    checkpoints: Mapping[str, Ranking],
    relevant: Collection[str] | None = None,
    config: Mapping[str, Any] | None = None,
    models: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One query's trace (the module docstring says how). ``relevant`` is None when the query
    has no labels (the playground): lists then don't say what's relevant."""
    config = config or {}
    models = models or {}
    relevant_set = set(relevant) if relevant is not None else None
    spans: list[dict[str, Any]] = []
    cursor = 0.0
    for step, name, kind, key, lists in STEPS:
        if key not in latency_ms:
            continue
        duration = max(float(latency_ms[key]), 0.0)
        spans.append(
            {
                "id": step,
                "parent_id": "search",
                "name": name,
                "kind": kind,
                "start_ms": _round(cursor),
                "duration_ms": _round(duration),
                "attributes": {
                    k: v for k, v in _attributes(step, config, models).items() if v is not None
                },
                "lists": [
                    list_summary(checkpoint, checkpoints[checkpoint], relevant_set)
                    for checkpoint in lists
                    if checkpoint in checkpoints
                ],
            }
        )
        cursor += duration
    total = float(latency_ms.get("total_ms", cursor))
    total = max(total, cursor)
    if total - cursor > 0.0005:
        spans.append(
            {
                "id": "other",
                "parent_id": "search",
                "name": "Visibility and the final list",
                "kind": "other",
                "start_ms": _round(cursor),
                "duration_ms": _round(total - cursor),
                "attributes": {},
                "lists": [],
            }
        )
    root = {
        "id": "search",
        "parent_id": None,
        "name": "Search",
        "kind": "search",
        "start_ms": 0.0,
        "duration_ms": _round(total),
        "attributes": {k: v for k, v in {"top_k": config.get("top_k")}.items() if v is not None},
        "lists": [],
    }
    path = [
        list_summary(checkpoint, checkpoints[checkpoint], relevant_set) | {"top": []}
        for checkpoint in CHECKPOINTS
        if checkpoint in checkpoints
    ]
    return {
        "total_ms": _round(total),
        "spans": [root, *spans],
        "path": path,
        "lost_at_checkpoint": lost_at(path) if relevant_set is not None else None,
    }


DOWNSTREAM = ("fused", "candidates", "reranked", "final")


def lost_at(path: Sequence[Mapping[str, Any]]) -> str | None:
    """The first checkpoint whose list dropped evidence the list feeding it had. The dense and
    keyword lists both feed fusion; the rest are a chain."""
    found = {entry["checkpoint"]: entry["found"] for entry in path}
    before = any(found.get(checkpoint) for checkpoint in ("dense", "keyword"))
    for checkpoint in DOWNSTREAM:
        if checkpoint not in found:
            continue
        if before and found[checkpoint] is False:
            return checkpoint
        before = bool(found[checkpoint])
    return None


def _stats(values: Sequence[float], bins: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    counts, edges = np.histogram(array, bins=bins, range=(0.0, float(array.max()) or 1.0))
    return {
        "n": int(array.size),
        "mean": _round(array.mean()),
        "min": _round(array.min()),
        "p25": _round(np.percentile(array, 25)),
        "p50": _round(np.percentile(array, 50)),
        "p75": _round(np.percentile(array, 75)),
        "p90": _round(np.percentile(array, 90)),
        "p95": _round(np.percentile(array, 95)),
        "p99": _round(np.percentile(array, 99)),
        "max": _round(array.max()),
        "histogram": {"edges": [_round(edge) for edge in edges], "counts": counts.tolist()},
    }


def latency_profile(
    latencies: Mapping[str, Mapping[str, float]], *, bins: int = 20, slowest: int = SLOWEST
) -> dict[str, Any]:
    """Per step, over a run's queries (item ID → step latencies): percentiles and a histogram;
    and the slowest queries by total time."""
    steps: dict[str, Any] = {}
    for key in PROFILE_KEYS:
        values = [float(latency[key]) for latency in latencies.values() if key in latency]
        if values:
            steps[key] = _stats(values, bins)
    ranked = sorted(
        ((item, latency) for item, latency in latencies.items() if "total_ms" in latency),
        key=lambda pair: (-float(pair[1]["total_ms"]), pair[0]),
    )
    return {
        "queries": len(latencies),
        "steps": steps,
        "slowest": [
            {
                "item_id": item,
                "total_ms": _round(latency["total_ms"]),
                "steps": {key: _round(latency[key]) for key in STEP_KEYS if key in latency},
            }
            for item, latency in ranked[:slowest]
        ],
    }
