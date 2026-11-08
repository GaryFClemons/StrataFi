"""Retrieval metrics: per query at every checkpoint, then aggregated with confidence intervals.

``score_query`` maps one eval item's evidence onto each checkpoint's ranked list
(``stratafi_core.relevance``) and computes, for every match rule and cutoff ``k``:

- ``hit@k``: at least one relevant chunk in the top k
- ``recall@k``: the share of the item's evidence items covered by the top k (union of chunks)
- ``precision@k``: relevant chunks in the top k, over k
- ``mrr@k``: the reciprocal rank of the first relevant chunk, 0 if none is in the top k
- ``ndcg@k``: each evidence item earns its gain once, at the first chunk that covers it (its
  grade under ``evidence-coverage-v1``, 1 under the binary rules); the ideal ranking has one
  fully covered item per position

``@all`` takes the whole list: ``recall@all`` of the candidate pool is the ceiling the reranker
can reach. Checkpoints are the retrieval engine's (``dense``, ``keyword``, ``fused``,
``candidates``, ``reranked``, ``final``).

Besides the metrics, each query records, from its final list:

- ``avoid_violation``: a version in ``avoid`` ranks above the first hit (or appears with none)
- ``must_not_return_violation``: a document in ``must_not_return`` is returned at all
- ``lost_at`` for a query that misses at the eval set's k: ``parse`` (its evidence can't be
  found in its document's text), ``retrieval`` (no relevant chunk reached the candidates) or
  ``rerank`` (one did, and was cut from the top k: by the reranker, or by the fused order when
  there is none). ``hidden`` says, per evidence item, that no version of it is visible to the
  asker on the as-of date (ACL, status or effective dates), so no retriever could return it; a
  retrieval loss whose evidence is all hidden counts as ``filtered`` in ``where_lost``.
- ``near_misses`` for an item with no evidence (abstention): what came back anyway

``aggregate`` averages over the items with evidence (violations over all), each mean with a
seeded bootstrap interval (``stratafi_core.stats``), plus the rerank uplift: final minus fused
on the same queries, with its own interval. All of it is numbers and IDs, never text.
"""

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from stratafi_core.evalsets import DocRef, EvalItem
from stratafi_core.relevance import (
    DEFAULT_THRESHOLD,
    EVIDENCE_COVERAGE,
    KNOWLEDGE_HIT,
    RULES,
    Judgement,
    Matcher,
    Passage,
)
from stratafi_core.stats import DEFAULT_RESAMPLES, bootstrap_means, percentile

CHECKPOINTS = ("dense", "keyword", "fused", "candidates", "reranked", "final")
METRICS = ("hit", "recall", "precision", "mrr", "ndcg")
ALL = "all"
LOST = ("parse", "retrieval", "rerank")
NEAR_MISSES = 3

Ranking = Sequence[tuple[str, float]]


def metric_name(metric: str, k: int | str) -> str:
    return f"{metric}@{k}"


def default_ks(*cutoffs: int) -> list[int]:
    """1, 3, 5 and 10, with the eval set's k and the config's top_k."""
    return sorted({1, 3, 5, 10, *cutoffs})


@dataclass(frozen=True)
class QueryScore:
    item_id: str
    has_evidence: bool
    # rule → checkpoint → "recall@5" → value
    metrics: dict[str, dict[str, dict[str, float]]]
    first_hit: dict[str, int | None]  # rule → rank in the final list, from 1
    found_in: dict[str, bool]  # checkpoint → a relevant chunk anywhere in it (knowledge-hit-v1)
    avoid_violation: bool | None  # None: the item avoids nothing
    must_not_return_violation: bool
    lost_at: str | None
    findable: list[bool | None] = field(default_factory=list)  # per evidence item
    hidden: list[bool] = field(default_factory=list)  # per evidence item: the filter hides it
    near_misses: list[dict[str, Any]] = field(default_factory=list)
    relevant: dict[str, list[str]] = field(default_factory=dict)  # rule → chunk IDs, any list

    def value(self, rule: str, checkpoint: str, name: str) -> float | None:
        return self.metrics.get(rule, {}).get(checkpoint, {}).get(name)


def _ndcg(judgements: Sequence[Judgement], rule: str, evidence: int, cutoff: int) -> float:
    credited: set[int] = set()
    dcg = 0.0
    for position, judgement in enumerate(judgements[:cutoff]):
        fresh = judgement.covered[rule] - credited
        if not fresh:
            continue
        if rule == EVIDENCE_COVERAGE:
            item = max(fresh, key=lambda index: (judgement.grades[index], -index))
            gain = judgement.grades[item]
        else:
            item, gain = min(fresh), 1.0
        credited.add(item)
        dcg += gain / math.log2(position + 2)
    ideal = sum(1 / math.log2(position + 2) for position in range(min(cutoff, evidence)))
    return dcg / ideal if ideal else 0.0


def list_metrics(
    judgements: Sequence[Judgement], rule: str, evidence: int, ks: Sequence[int]
) -> dict[str, float]:
    """Every metric at every cutoff for one ranked list under one rule."""
    values: dict[str, float] = {}
    for k in [*ks, ALL]:
        cutoff = len(judgements) if k == ALL else int(k)
        top = judgements[:cutoff]
        flags = [judgement.relevant(rule) for judgement in top]
        covered = set().union(*(judgement.covered[rule] for judgement in top)) if top else set()
        first = next((position for position, flag in enumerate(flags) if flag), None)
        values[metric_name("hit", k)] = float(first is not None)
        values[metric_name("recall", k)] = len(covered) / evidence
        values[metric_name("precision", k)] = sum(flags) / cutoff if cutoff else 0.0
        values[metric_name("mrr", k)] = 1 / (first + 1) if first is not None else 0.0
        values[metric_name("ndcg", k)] = _ndcg(top, rule, evidence, cutoff)
    return {name: round(value, 6) for name, value in values.items()}


def _matches(passage: Passage, refs: Sequence[DocRef]) -> bool:
    return any(
        passage.doc_key == ref.doc_key and (ref.version is None or passage.version == ref.version)
        for ref in refs
    )


def score_query(
    item: EvalItem,
    checkpoints: Mapping[str, Ranking],
    passages: Mapping[str, Passage],
    *,
    ks: Sequence[int],
    k_eval: int,
    threshold: float = DEFAULT_THRESHOLD,
    findable: Sequence[bool | None] | None = None,
    hidden: Sequence[bool] | None = None,
) -> QueryScore:
    """One query's metrics, violations and diagnosis (the module docstring says how)."""
    matcher = Matcher(item.expected, threshold)
    judged = {
        name: [matcher.judge(passages[chunk_id]) for chunk_id, _ in ranking]
        for name, ranking in checkpoints.items()
    }
    evidence = len(item.expected)
    final_ids = [chunk_id for chunk_id, _ in checkpoints.get("final", [])]
    final = judged.get("final", [])
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    first_hit: dict[str, int | None] = {}
    relevant: dict[str, list[str]] = {}
    if evidence:
        for rule in RULES:
            metrics[rule] = {
                name: list_metrics(judgements, rule, evidence, ks)
                for name, judgements in judged.items()
            }
            first = next((n for n, j in enumerate(final) if j.relevant(rule)), None)
            first_hit[rule] = first + 1 if first is not None else None
            relevant[rule] = sorted(
                {
                    chunk_id
                    for name, ranking in checkpoints.items()
                    for (chunk_id, _), judgement in zip(ranking, judged[name], strict=True)
                    if judgement.relevant(rule)
                }
            )
    found_in = {
        name: any(judgement.relevant(KNOWLEDGE_HIT) for judgement in judgements)
        for name, judgements in judged.items()
    }
    avoid_violation = None
    if item.avoid:
        avoided = [
            n for n, chunk_id in enumerate(final_ids) if _matches(passages[chunk_id], item.avoid)
        ]
        first = first_hit.get(KNOWLEDGE_HIT)
        avoid_violation = any(first is None or n + 1 < first for n in avoided)
    must_not = any(_matches(passages[chunk_id], item.must_not_return) for chunk_id in final_ids)
    known = [flag for flag in (findable or []) if flag is not None]
    lost_at = None
    if evidence and not metrics[KNOWLEDGE_HIT].get("final", {}).get(metric_name("hit", k_eval)):
        if known and not any(known):
            lost_at = "parse"
        elif not found_in.get("candidates", False):
            lost_at = "retrieval"
        else:
            lost_at = "rerank"
    near_misses = []
    if not evidence:
        for rank, (chunk_id, score) in enumerate(checkpoints.get("final", [])[:NEAR_MISSES], 1):
            passage = passages[chunk_id]
            near_misses.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "doc_key": passage.doc_key,
                    "version": passage.version,
                    "score": round(float(score), 6),
                }
            )
    return QueryScore(
        item_id=item.id,
        has_evidence=bool(evidence),
        metrics=metrics,
        first_hit=first_hit,
        found_in=found_in,
        avoid_violation=avoid_violation,
        must_not_return_violation=must_not,
        lost_at=lost_at,
        findable=list(findable or []),
        hidden=list(hidden or []),
        near_misses=near_misses,
        relevant=relevant,
    )


# --- aggregates -----------------------------------------------------------------------------

MetricFilter = Callable[[str, str, str], bool]


def _keys(scores: Sequence[QueryScore], wanted: MetricFilter | None) -> list[tuple[str, str, str]]:
    keys = sorted(
        {
            (rule, checkpoint, name)
            for score in scores
            for rule, by_checkpoint in score.metrics.items()
            for checkpoint, values in by_checkpoint.items()
            for name in values
        }
    )
    return [key for key in keys if wanted is None or wanted(*key)]


def _nest(estimates: Mapping[tuple[str, ...], Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for (*path, last), estimate in estimates.items():
        node = nested
        for part in path:
            node = node.setdefault(part, {})
        node[last] = estimate.as_dict() if estimate is not None else None
    return nested


def aggregate(
    scores: Sequence[QueryScore],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    wanted: MetricFilter | None = None,
) -> dict[str, Any]:
    """Means with bootstrap intervals over the queries with evidence (``wanted`` picks
    metrics); the rerank uplift; violations, where queries were lost, and findability."""
    with_evidence = [score for score in scores if score.has_evidence]
    keys = _keys(with_evidence, wanted)
    rows = [[score.value(*key) or 0.0 for key in keys] for score in with_evidence]
    estimates = bootstrap_means(rows, resamples=resamples, seed=seed) if keys else []
    uplift_keys = [
        (rule, name)
        for rule, checkpoint, name in keys
        if checkpoint == "final"
        and not name.endswith(f"@{ALL}")
        and (rule, "fused", name) in set(keys)
    ]
    uplift_rows = [
        [(score.value(rule, "final", name) or 0.0) - (score.value(rule, "fused", name) or 0.0)
         for rule, name in uplift_keys]
        for score in with_evidence
    ]  # fmt: skip
    uplift = bootstrap_means(uplift_rows, resamples=resamples, seed=seed) if uplift_keys else []
    avoiding = [score for score in scores if score.avoid_violation is not None]
    known = [flag for score in with_evidence for flag in score.findable if flag is not None]
    lost = Counter(score.lost_at or "found" for score in with_evidence)
    filtered = sum(
        1
        for score in with_evidence
        if score.lost_at == "retrieval" and score.hidden and all(score.hidden)
    )
    return {
        "queries": len(scores),
        "with_evidence": len(with_evidence),
        "abstention": len(scores) - len(with_evidence),
        "metrics": _nest(dict(zip(keys, estimates, strict=True))),
        "uplift": _nest(dict(zip(uplift_keys, uplift, strict=True))),
        "violations": {
            "avoid": {
                "count": sum(1 for score in avoiding if score.avoid_violation),
                "of": len(avoiding),
            },
            "must_not_return": {
                "count": sum(1 for score in scores if score.must_not_return_violation),
                "of": len(scores),
            },
        },
        "where_lost": {
            **{bucket: lost.get(bucket, 0) for bucket in ("found", *LOST)},
            "filtered": filtered,
        },
        "findability": {
            "findable": sum(known),
            "of": len(known),
            "rate": round(sum(known) / len(known), 6) if known else None,
        },
    }


# --- a whole eval run's summary -------------------------------------------------------------

LATENCY_STEPS = ("embed_ms", "keyword_ms", "dense_ms", "fusion_ms", "rerank_ms", "total_ms")


def headline(k_eval: int) -> MetricFilter:
    """The metrics slices report: the final list's hit and recall at the eval set's k, and MRR
    and nDCG at 10."""
    names = {
        metric_name("hit", k_eval),
        metric_name("recall", k_eval),
        metric_name("mrr", 10),
        metric_name("ndcg", 10),
    }
    return lambda rule, checkpoint, name: checkpoint == "final" and name in names


def _derived_seed(seed: int, *parts: str) -> int:
    return int.from_bytes(
        hashlib.sha256(":".join([str(seed), *parts]).encode()).digest()[:4], "big"
    )


def _rate(flags: Sequence[bool]) -> dict[str, Any]:
    return {
        "findable": sum(flags),
        "of": len(flags),
        "rate": round(sum(flags) / len(flags), 6) if flags else None,
    }


def summarize(
    scores: Sequence[QueryScore],
    *,
    k_eval: int,
    tags: Mapping[str, Sequence[str]],
    file_types: Mapping[str, Sequence[str]],
    latencies: Mapping[str, Mapping[str, float]],
    evidence_groups: Mapping[str, Sequence[tuple[str | None, str | None]]],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """An eval run's ``metrics``: the aggregates, slices by tag and by the gold documents' file
    type (headline metrics, each slice resampled with its own seed derived from ``seed``),
    latency percentiles per step, and findability by file type and by parser.
    ``evidence_groups`` gives each item's evidence items' (file type, parser)."""
    summary = aggregate(scores, resamples=resamples, seed=seed)
    summary["k_eval"] = k_eval
    slices: dict[str, dict[str, Any]] = {}
    for kind, labels in (("tag", tags), ("file_type", file_types)):
        groups: dict[str, list[QueryScore]] = {}
        for score in scores:
            for label in labels.get(score.item_id, ()):
                groups.setdefault(label, []).append(score)
        slices[kind] = {
            label: aggregate(
                members,
                resamples=resamples,
                seed=_derived_seed(seed, kind, label),
                wanted=headline(k_eval),
            )
            for label, members in sorted(groups.items())
        }
    summary["slices"] = slices
    summary["latency_ms"] = {
        step: {
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "mean": round(sum(values) / len(values), 3) if values else None,
        }
        for step in LATENCY_STEPS
        if (values := [latency[step] for latency in latencies.values() if step in latency])
    }
    by_type: dict[str, list[bool]] = {}
    by_parser: dict[str, list[bool]] = {}
    for score in scores:
        for flag, (file_type, parser) in zip(
            score.findable, evidence_groups.get(score.item_id, ()), strict=False
        ):
            if flag is None:
                continue
            by_type.setdefault(file_type or "unknown", []).append(flag)
            by_parser.setdefault(parser or "unknown", []).append(flag)
    summary["findability"]["by_file_type"] = {key: _rate(v) for key, v in sorted(by_type.items())}
    summary["findability"]["by_parser"] = {key: _rate(v) for key, v in sorted(by_parser.items())}
    return summary
