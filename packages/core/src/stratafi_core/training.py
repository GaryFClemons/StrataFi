"""Post-training: training sets mined from the lab's evidence, the query adapter's training,
and the records of training runs and trained models.

**Training sets**. ``mine`` turns a dev eval run into training records, one per item
(``stratafi.training-set.v1``, a line of a JSON Lines artifact):

- *Positives* are chunks of an expected document version, from the whole index, that cover the
  item's evidence by the chosen rule, best grade first: evidence that retrieval missed becomes
  training signal.
- *Hard negatives* are what the chosen checkpoints ranked within ``depth``, best rank first:
  chunks neither passage rule finds relevant, that cover no evidence item above
  ``negative_max_grade`` in any version (a copy of the evidence in another version would teach
  a model to tell equal texts apart), and that don't repeat a positive's text. Chunks of an
  ``avoid`` version or a ``must_not_return`` document come first, labelled with that reason.
- Items without evidence give negatives only (``abstention``), for rerankers and for teaching a
  generator to decline.
- *Leak guard*: a dev item whose query nearly duplicates a test item's (token Jaccard at or above
  ``duplicate_threshold``) is dropped, so the locked test split can still confirm a
  trained model. The share of test items whose gold documents appear among the positives is
  reported: the corpus is shared, which is domain adaptation, not leakage.
- A seeded share of items is held out (``validation``) for training's own early stopping.

``export_rows`` gives three views of a training set: ``sentence-transformers`` (anchor,
positive, negatives), ``cross-encoder`` (query, passage, label) and ``chat`` (the pinned
``grounded-qa`` prompt over gold and distractor passages, answered with the item's reference
answer and its citations, or with the refusal when the item has no evidence).

**Query adapters**. ``train_adapter`` learns a matrix ``W`` (``d`` by ``d``, from the
identity) so that ``W q`` ranks an item's positive above its hard negatives and the batch's
other positives: InfoNCE at ``temperature``, scored as the dense index scores (cosine or dot),
with Adam and an L2 pull toward the identity. Each epoch is measured on the validation items:
their loss, and recall@10 and MRR@10 of their positives over the whole index; the best epoch
by validation MRR is kept, and training stops after ``patience`` epochs without a better one.
Documents keep their vectors: only queries move.

Nothing here logs. Texts stay in the records and the artifacts they're written to.
"""

import io
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from stratafi_core.answers import NOT_IN_SOURCES, ContextPassage, answer_request, source_id
from stratafi_core.components import ComponentError
from stratafi_core.evalruns import RankedItem
from stratafi_core.evalsets import DocRef, EvalItem
from stratafi_core.hashing import cache_key, content_hash
from stratafi_core.manifest import ManifestEnvironment, ManifestHardware
from stratafi_core.relevance import (
    DEFAULT_THRESHOLD,
    EVIDENCE_COVERAGE,
    KNOWLEDGE_HIT,
    Matcher,
    Passage,
    coverage,
    normalize,
    tokens,
)

TRAINING_SET_SCHEMA = "stratafi.training-set.v1"
TRAINING_RUN_SCHEMA = "stratafi.training-run.v1"
MODEL_CARD_SCHEMA = "stratafi.model-card.v1"
MODEL_MANIFEST_SCHEMA = "stratafi.model.v1"
# Bump when mining's output could change for the same inputs and options.
MINER_VERSION = "1.0.0"
MINE_STAGE = "training-set"
TRAIN_STAGE = "train"
ADAPTER_FORMAT = "stratafi-query-adapter-v1"
ADAPTER_FILE = "adapter.npy"
VALIDATION_K = 10

ModelKind = Literal["query-adapter", "embedder", "reranker"]
MODEL_KINDS: tuple[str, ...] = ("query-adapter", "embedder", "reranker")
Checkpoint = Literal["dense", "keyword", "fused", "candidates", "reranked", "final"]
PositiveRule = Literal["evidence-coverage-v1", "knowledge-hit-v1"]
Reason = Literal["evidence", "retrieved", "avoid", "must_not_return", "abstention"]
RecordSplit = Literal["train", "validation"]
ExportFormat = Literal["sentence-transformers", "cross-encoder", "chat"]
EXPORT_FORMATS: tuple[str, ...] = ("sentence-transformers", "cross-encoder", "chat")
CHAT_PASSAGES = 8


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- training sets ---------------------------------------------------------------------------


class TrainingSetOptions(_M):
    """How a training set is mined; recorded with it and part of its key."""

    rule: PositiveRule = Field(
        default=EVIDENCE_COVERAGE, description="The rule a positive must pass"
    )
    max_positives: int = Field(default=3, ge=1, le=50, description="Positives per item, at most")
    max_negatives: int = Field(default=8, ge=0, le=200, description="Negatives per item, at most")
    checkpoints: tuple[Checkpoint, ...] = Field(
        default=("dense", "keyword", "candidates"),
        min_length=1,
        description="The ranked lists hard negatives come from",
    )
    depth: int = Field(default=20, ge=1, le=1000, description="How far down each list to look")
    negative_max_grade: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="A negative covers no evidence item at this grade or above, in any version",
    )
    abstentions: bool = Field(
        default=True, description="Keep items without evidence, with negatives only"
    )
    validation_share: float = Field(
        default=0.2, ge=0, le=0.5, description="Items held out for early stopping"
    )
    duplicate_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=1,
        description="Leak guard: drop a dev item this similar to a test item (token Jaccard)",
    )
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Seeds the validation split")


class TrainingPassage(_M):
    """One passage of a training record: a chunk with where it came from."""

    chunk_id: str
    doc_key: str | None = None
    version: str | None = None
    section: str | None = None
    text: str
    reason: Reason
    grade: float | None = Field(
        default=None, description="Positives: the evidence it covers; negatives: the most it covers"
    )
    rank: int | None = Field(default=None, ge=1, description="Its best rank in the lists mined")
    checkpoint: str | None = Field(default=None, description="The list that ranked it best")


class TrainingRecord(_M):
    """``stratafi.training-set.v1``: one eval item as training data, a line of a training set's
    JSON Lines artifact."""

    item_id: str
    split: RecordSplit
    query: str
    tags: list[str] = Field(default_factory=list)
    abstention: bool = Field(default=False, description="The item has no evidence: negatives only")
    reference_answer: str | None = None
    positives: list[TrainingPassage] = Field(default_factory=list)
    negatives: list[TrainingPassage] = Field(default_factory=list)


@dataclass
class MinedSet:
    records: list[TrainingRecord]
    stats: dict[str, Any]


def query_tokens(query: str) -> frozenset[str]:
    return frozenset(tokens(query))


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    if not first and not second:
        return 1.0
    return len(first & second) / len(first | second)


def near_duplicates(
    queries: Mapping[str, str], against: Sequence[str], threshold: float
) -> set[str]:
    """The IDs of ``queries`` whose tokens are at least ``threshold`` Jaccard-similar to one of
    ``against``'s."""
    others = [query_tokens(text) for text in against]
    if not others:
        return set()
    found = set()
    for item_id, text in queries.items():
        mine = query_tokens(text)
        if any(jaccard(mine, other) >= threshold for other in others):
            found.add(item_id)
    return found


def validation_ids(item_ids: Iterable[str], share: float, seed: int) -> set[str]:
    """A seeded ``share`` of the items, for validation."""
    ordered = sorted(set(item_ids))
    count = round(len(ordered) * share)
    return set(random.Random(seed).sample(ordered, count)) if count else set()  # noqa: S311


def _refers(refs: Sequence[DocRef], passage: Passage) -> bool:
    return any(
        passage.doc_key == ref.doc_key and (ref.version is None or passage.version == ref.version)
        for ref in refs
    )


def _best_ranks(
    ranked: RankedItem | None, checkpoints: Sequence[str], depth: int
) -> dict[str, tuple[int, str]]:
    """Each chunk's best rank (1-based) in the checkpoints' lists, and the first list with it."""
    found: dict[str, tuple[int, str]] = {}
    if ranked is None:
        return found
    for name in checkpoints:
        for rank, (chunk_id, _) in enumerate(ranked.checkpoints.get(name, [])[:depth], start=1):
            if chunk_id not in found or rank < found[chunk_id][0]:
                found[chunk_id] = (rank, name)
    return found


def _passage(
    passage: Passage,
    reason: Reason,
    *,
    grade: float | None = None,
    best: tuple[int, str] | None = None,
) -> TrainingPassage:
    return TrainingPassage(
        chunk_id=passage.chunk_id,
        doc_key=passage.doc_key,
        version=passage.version,
        section=passage.section,
        text=passage.text,
        reason=reason,
        grade=round(grade, 6) if grade is not None else None,
        rank=best[0] if best else None,
        checkpoint=best[1] if best else None,
    )


def mine(
    items: Sequence[EvalItem],
    ranked: Mapping[str, RankedItem],
    passages: Mapping[str, Passage],
    *,
    options: TrainingSetOptions,
    threshold: float = DEFAULT_THRESHOLD,
    test_items: Sequence[EvalItem] = (),
) -> MinedSet:
    """Training records from an eval run's ``items`` and their ``ranked`` lists, with
    ``passages`` holding every chunk of the run's indexes (the module docstring says how)."""
    by_doc: dict[str, list[str]] = defaultdict(list)
    for chunk_id in sorted(passages):
        doc_key = passages[chunk_id].doc_key
        if doc_key is not None:
            by_doc[doc_key].append(chunk_id)
    duplicates = near_duplicates(
        {item.id: item.query for item in items},
        [item.query for item in test_items],
        options.duplicate_threshold,
    )
    dropped: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    kept: list[TrainingRecord] = []
    positives_missed = negatives_overlapping = 0
    for item in items:
        if item.id in duplicates:
            dropped["near_duplicate"] += 1
            continue
        best = _best_ranks(ranked.get(item.id), options.checkpoints, options.depth)
        if not item.expected:
            if not options.abstentions:
                dropped["abstention"] += 1
                continue
            ordered = sorted(
                best.items(),
                key=lambda pair: (pair[1][0], options.checkpoints.index(pair[1][1]), pair[0]),
            )
            negatives = [
                _passage(passages[chunk_id], "abstention", best=where)
                for chunk_id, where in ordered
                if chunk_id in passages
            ][: options.max_negatives]
            if not negatives:
                dropped["no_negative"] += 1
                continue
            reasons["abstention"] += len(negatives)
            kept.append(
                TrainingRecord(
                    item_id=item.id,
                    split="train",
                    query=item.query,
                    tags=list(item.tags),
                    abstention=True,
                    reference_answer=item.reference_answer,
                    negatives=negatives,
                )
            )
            continue
        matcher = Matcher(item.expected, threshold)
        found: list[tuple[float, int, str]] = []
        for chunk_id in sorted({c for ref in item.expected for c in by_doc.get(ref.doc_key, [])}):
            judgement = matcher.judge(passages[chunk_id])
            if judgement.relevant(options.rule):
                grade = max(judgement.grades, default=0.0)
                found.append((-grade, best.get(chunk_id, (10**9, ""))[0], chunk_id))
        if not found:
            dropped["no_positive"] += 1
            continue
        found.sort()
        positives = [
            _passage(passages[chunk_id], "evidence", grade=-grade, best=best.get(chunk_id))
            for grade, _, chunk_id in found[: options.max_positives]
        ]
        positives_missed += sum(1 for passage in positives if passage.rank is None)
        evidence = [tokens(ref.evidence) for ref in item.expected if ref.evidence]
        # Evidence given by its section alone: that section in any version of the document.
        sections = {
            (ref.doc_key, normalize(ref.section))
            for ref in item.expected
            if ref.section and not ref.evidence
        }
        positive_ids = {passage.chunk_id for passage in positives}
        positive_texts = {normalize(passage.text) for passage in positives}
        candidates = []
        for chunk_id, where in best.items():
            passage = passages.get(chunk_id)
            if passage is None or chunk_id in positive_ids:
                continue
            judgement = matcher.judge(passage)
            if judgement.relevant(KNOWLEDGE_HIT) or judgement.relevant(EVIDENCE_COVERAGE):
                continue
            text_tokens = tokens(passage.text) if evidence else []
            grade = max(
                [*judgement.grades, *(coverage(wanted, text_tokens) for wanted in evidence)],
                default=0.0,
            )
            repeats = (
                normalize(passage.text) in positive_texts
                or (passage.doc_key, normalize(passage.section or "")) in sections
            )
            if grade >= options.negative_max_grade or repeats:
                negatives_overlapping += 1
                continue
            reason: Reason = (
                "avoid"
                if _refers(item.avoid, passage)
                else "must_not_return"
                if _refers(item.must_not_return, passage)
                else "retrieved"
            )
            trap = 0 if reason != "retrieved" else 1
            candidates.append((trap, where[0], where[1], chunk_id, reason, grade))
        candidates.sort(key=lambda row: (row[0], row[1], options.checkpoints.index(row[2]), row[3]))
        negatives = [
            _passage(passages[chunk_id], reason, grade=grade, best=(rank, name))
            for _, rank, name, chunk_id, reason, grade in candidates[: options.max_negatives]
        ]
        for passage in negatives:
            reasons[passage.reason] += 1
        kept.append(
            TrainingRecord(
                item_id=item.id,
                split="train",
                query=item.query,
                tags=list(item.tags),
                reference_answer=item.reference_answer,
                positives=positives,
                negatives=negatives,
            )
        )
    held_out = validation_ids(
        (record.item_id for record in kept), options.validation_share, options.seed
    )
    records = [
        record.model_copy(update={"split": "validation"}) if record.item_id in held_out else record
        for record in kept
    ]
    positive_docs = {p.doc_key for record in records for p in record.positives}
    tested = [item for item in test_items if item.expected]
    shared = sum(1 for item in tested if {ref.doc_key for ref in item.expected} & positive_docs)
    stats = {
        "items": len(items),
        "records": len(records),
        "pairs": sum(1 for record in records if record.positives),
        "abstentions": sum(1 for record in records if record.abstention),
        "validation": sum(1 for record in records if record.split == "validation"),
        "positives": sum(len(record.positives) for record in records),
        "positives_missed": positives_missed,
        "negatives": sum(len(record.negatives) for record in records),
        "negatives_by_reason": dict(sorted(reasons.items())),
        "negatives_overlapping": negatives_overlapping,
        "dropped": dict(sorted(dropped.items())),
        "leak_guard": {
            "test_items": len(test_items),
            "near_duplicates": dropped["near_duplicate"],
            "test_items_overlapping": shared,
            "test_overlap": round(shared / len(tested), 6) if tested else None,
        },
    }
    return MinedSet(records, stats)


def training_set_key(
    *,
    options: TrainingSetOptions,
    results_key: str,
    shard_keys: Sequence[str],
    indexes: Mapping[str, str],
    items_sha256: str,
    test_sha256: str,
) -> str:
    """The cache key of a training set's records."""
    return cache_key(
        stage=MINE_STAGE,
        component_id="train/miner",
        component_version=MINER_VERSION,
        params={"options": options.model_dump(mode="json"), "items_sha256": items_sha256},
        inputs=[results_key, *shard_keys, *(indexes[kind] for kind in sorted(indexes))],
        env={"test_sha256": test_sha256},
    )


def read_records(data: bytes) -> list[TrainingRecord]:
    return [TrainingRecord.model_validate_json(line) for line in data.splitlines() if line.strip()]


def records_bytes(records: Sequence[TrainingRecord]) -> bytes:
    return b"".join(record.model_dump_json().encode() + b"\n" for record in records)


# --- exports ---------------------------------------------------------------------------------


def _citations(passages: Sequence[ContextPassage]) -> str:
    return " ".join(f"[{passage.source_id}]" for passage in passages if passage.relevant)


def chat_example(record: TrainingRecord, *, day: str) -> dict[str, Any] | None:
    """One grounded-QA chat example (``grounded-qa-v2``), or None when the item has no target:
    its reference answer with the gold passages' citations, or the refusal for an item without
    evidence. Gold and distractor passages are interleaved in a seeded order."""
    if record.abstention:
        target: str | None = NOT_IN_SOURCES
    else:
        target = record.reference_answer
    if not target:
        return None
    chosen = [*record.positives, *record.negatives][:CHAT_PASSAGES]
    order = random.Random(record.item_id).sample(chosen, len(chosen))  # noqa: S311
    passages = [
        ContextPassage(
            source_id(passage.doc_key, passage.version, rank),
            passage.doc_key,
            passage.version,
            passage.section,
            passage.text,
            passage.reason == "evidence",
        )
        for rank, passage in enumerate(order, start=1)
    ]
    if not record.abstention:
        cited = _citations(passages)
        target = f"{target.strip()} {cited}".strip()
    messages = answer_request(record.query, passages, day=day).messages()
    return {
        "item_id": record.item_id,
        "split": record.split,
        "messages": [*messages, {"role": "assistant", "content": target}],
    }


def export_rows(
    records: Iterable[TrainingRecord],
    fmt: ExportFormat,
    *,
    day: str = "",
    split: Literal["train", "validation", "all"] = "all",
) -> Iterator[dict[str, Any]]:
    """A training set as rows of one export format (the module docstring says which)."""
    for record in records:
        if split != "all" and record.split != split:
            continue
        if fmt == "sentence-transformers":
            for positive in record.positives:
                yield {
                    "item_id": record.item_id,
                    "split": record.split,
                    "anchor": record.query,
                    "positive": positive.text,
                    "negatives": [negative.text for negative in record.negatives],
                }
        elif fmt == "cross-encoder":
            for passage, label in [
                *((positive, 1.0) for positive in record.positives),
                *((negative, 0.0) for negative in record.negatives),
            ]:
                yield {
                    "item_id": record.item_id,
                    "split": record.split,
                    "query": record.query,
                    "passage": passage.text,
                    "label": label,
                }
        else:
            example = chat_example(record, day=day)
            if example is not None:
                yield example


# --- query adapters ------------------------------------------------------------------------


class QueryAdapterParams(_M):
    """``train/query-adapter``'s parameters."""

    epochs: int = Field(default=30, ge=1, le=1000, description="Passes over the training items")
    batch_size: int = Field(
        default=32,
        ge=2,
        le=4096,
        description="Examples per step; the others' positives are negatives",
    )
    learning_rate: float = Field(default=0.001, gt=0, le=1, description="Adam's step size")
    temperature: float = Field(default=0.05, gt=0, le=1, description="InfoNCE's temperature")
    identity_weight: float = Field(
        default=0.01, ge=0, le=100, description="How hard the map is pulled toward the identity"
    )
    hard_negatives: int = Field(
        default=8, ge=0, le=200, description="Mined negatives per example, at most"
    )
    in_batch: bool = Field(default=True, description="Use the batch's other positives as negatives")
    patience: int = Field(
        default=5,
        ge=0,
        le=1000,
        description="Stop after this many epochs without a better validation MRR; 0 never stops",
    )
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Seeds the order of examples")


@dataclass(frozen=True)
class AdapterExamples:
    """Training examples as arrays: one per (item, positive)."""

    item_ids: list[str]
    queries: np.ndarray  # [n, d]
    positives: np.ndarray  # [n, d]
    positive_ids: list[str]
    negatives: np.ndarray  # [n, h, d]
    negative_mask: np.ndarray  # [n, h] bool
    relevant: list[frozenset[str]]  # each example's item's positive chunk IDs

    def __len__(self) -> int:
        return len(self.item_ids)

    def take(self, index: np.ndarray) -> "AdapterExamples":
        return AdapterExamples(
            [self.item_ids[i] for i in index],
            self.queries[index],
            self.positives[index],
            [self.positive_ids[i] for i in index],
            self.negatives[index],
            self.negative_mask[index],
            [self.relevant[i] for i in index],
        )


def _unit(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows / np.where(norms > 0, norms, 1.0)


def adapter_examples(
    records: Iterable[TrainingRecord],
    queries: Mapping[str, np.ndarray],
    vectors: Mapping[str, np.ndarray],
    *,
    hard_negatives: int,
    metric: str,
    dims: int,
) -> AdapterExamples:
    """Examples from records with positives, their query vectors (by item) and passage vectors
    (by chunk). Passages are scaled to length 1 for a cosine index, as it scores them."""
    item_ids, rows_q, rows_p, positive_ids, rows_h, masks, relevant = [], [], [], [], [], [], []
    width = max(hard_negatives, 1)
    for record in records:
        query = queries.get(record.item_id)
        if query is None:
            continue
        mine = frozenset(p.chunk_id for p in record.positives)
        negatives = [vectors[n.chunk_id] for n in record.negatives if n.chunk_id in vectors]
        negatives = negatives[:hard_negatives]
        block = np.zeros((width, dims), dtype=np.float64)
        mask = np.zeros(width, dtype=bool)
        for index, vector in enumerate(negatives):
            block[index], mask[index] = vector, True
        for positive in record.positives:
            if positive.chunk_id not in vectors:
                continue
            item_ids.append(record.item_id)
            rows_q.append(np.asarray(query, dtype=np.float64).reshape(-1))
            rows_p.append(np.asarray(vectors[positive.chunk_id], dtype=np.float64).reshape(-1))
            positive_ids.append(positive.chunk_id)
            rows_h.append(block)
            masks.append(mask)
            relevant.append(mine)
    count = len(item_ids)
    positives = np.asarray(rows_p).reshape(count, dims)
    negatives = np.asarray(rows_h).reshape(count, width, dims)
    if metric == "cosine":
        positives, negatives = _unit(positives), _unit(negatives)
    return AdapterExamples(
        item_ids,
        np.asarray(rows_q).reshape(count, dims),
        positives,
        positive_ids,
        negatives,
        np.asarray(masks, dtype=bool).reshape(count, width),
        relevant,
    )


def adapter_loss(
    weights: np.ndarray,
    batch: AdapterExamples,
    *,
    temperature: float,
    metric: str,
    in_batch: bool = True,
    identity_weight: float = 0.0,
) -> tuple[float, np.ndarray]:
    """InfoNCE of one batch under ``weights``, with its gradient: each example's positive against
    its hard negatives and, with ``in_batch``, the batch's other positives (except its item's own
    positives). Plus ``identity_weight`` times the squared distance from the identity."""
    size = len(batch)
    adapted = batch.queries @ weights.T
    norms = np.linalg.norm(adapted, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1e-12)
    scored = adapted / norms if metric == "cosine" else adapted
    if in_batch:
        # Another example's positive is a negative unless it's one of this item's positives.
        columns = {chunk_id: n for n, chunk_id in enumerate(dict.fromkeys(batch.positive_ids))}
        member = np.zeros((size, len(columns)), dtype=bool)
        for row, relevant in enumerate(batch.relevant):
            for chunk_id in relevant:
                if chunk_id in columns:
                    member[row, columns[chunk_id]] = True
        allowed = ~member[:, [columns[chunk_id] for chunk_id in batch.positive_ids]]
        np.fill_diagonal(allowed, True)
    else:
        allowed = np.eye(size, dtype=bool)
    allowed = np.concatenate([allowed, batch.negative_mask], axis=1)
    logits = (
        np.concatenate(
            [scored @ batch.positives.T, np.einsum("bd,bhd->bh", scored, batch.negatives)], axis=1
        )
        / temperature
    )
    logits = np.where(allowed, logits, -np.inf)
    top = logits.max(axis=1, keepdims=True)
    exp = np.where(allowed, np.exp(logits - top), 0.0)
    total = exp.sum(axis=1, keepdims=True)
    probs = exp / total
    diagonal = np.arange(size)
    loss = float(np.mean(np.log(total[:, 0]) + top[:, 0] - logits[diagonal, diagonal]))
    grad_logits = probs.copy()
    grad_logits[diagonal, diagonal] -= 1.0
    grad_logits /= size * temperature
    grad_scored = grad_logits[:, :size] @ batch.positives + np.einsum(
        "bh,bhd->bd", grad_logits[:, size:], batch.negatives
    )
    if metric == "cosine":
        along = np.sum(grad_scored * scored, axis=1, keepdims=True)
        grad_adapted = (grad_scored - scored * along) / norms
    else:
        grad_adapted = grad_scored
    gradient = grad_adapted.T @ batch.queries
    if identity_weight:
        offset = weights - np.eye(weights.shape[0])
        loss += identity_weight * float(np.sum(offset * offset))
        gradient = gradient + 2.0 * identity_weight * offset
    return loss, gradient


@dataclass(frozen=True)
class AdapterEpoch:
    epoch: int
    train_loss: float | None
    validation_loss: float | None
    recall_at_10: float | None
    mrr_at_10: float | None
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": _round(self.train_loss),
            "validation_loss": _round(self.validation_loss),
            "recall@10": _round(self.recall_at_10),
            "mrr@10": _round(self.mrr_at_10),
            "seconds": round(self.seconds, 3),
        }


@dataclass
class AdapterResult:
    weights: np.ndarray  # float32 [d, d], the best epoch's
    epochs: list[AdapterEpoch]
    best_epoch: int
    baseline: AdapterEpoch
    stopped_early: bool = False


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


@dataclass(frozen=True)
class Corpus:
    """Every passage vector of a dense index, scaled as the index scores them."""

    matrix: np.ndarray  # float32 [n, d]
    rows: dict[str, int]

    @classmethod
    def of(cls, matrix: np.ndarray, chunk_ids: Sequence[str], *, metric: str) -> "Corpus":
        vectors = np.asarray(matrix, dtype=np.float32)
        if metric == "cosine":
            vectors = _unit(vectors).astype(np.float32)
        return cls(vectors, {chunk_id: row for row, chunk_id in enumerate(chunk_ids)})


def retrieval_metrics(
    weights: np.ndarray,
    examples: AdapterExamples,
    corpus: Corpus,
    *,
    metric: str,
    k: int = VALIDATION_K,
) -> tuple[float | None, float | None]:
    """Recall@k and MRR@k of each item's positives over the whole corpus, with ``weights``."""
    rows, matrix = corpus.rows, corpus.matrix
    seen: dict[str, int] = {}
    for index, item_id in enumerate(examples.item_ids):
        seen.setdefault(item_id, index)
    if not seen:
        return None, None
    hits, reciprocal = [], []
    for index in seen.values():
        query = weights @ examples.queries[index]
        query = (_unit(query) if metric == "cosine" else query).astype(np.float32)
        scores = matrix @ query
        wanted = [rows[c] for c in examples.relevant[index] if c in rows]
        if not wanted:
            continue
        best = max(float(scores[row]) for row in wanted)
        rank = 1 + int(np.sum(scores > best))
        hits.append(1.0 if rank <= k else 0.0)
        reciprocal.append(1.0 / rank if rank <= k else 0.0)
    if not hits:
        return None, None
    return float(np.mean(hits)), float(np.mean(reciprocal))


def mean_loss(
    weights: np.ndarray,
    examples: AdapterExamples,
    *,
    params: QueryAdapterParams,
    metric: str,
) -> float:
    """The examples' loss in batches of ``batch_size``, in their order, without the pull."""
    total = 0.0
    for start in range(0, len(examples), params.batch_size):
        batch = examples.take(np.arange(start, min(start + params.batch_size, len(examples))))
        loss, _ = adapter_loss(
            weights,
            batch,
            temperature=params.temperature,
            metric=metric,
            in_batch=params.in_batch,
        )
        total += loss * len(batch)
    return total / len(examples)


def train_adapter(
    train: AdapterExamples,
    validation: AdapterExamples | None,
    *,
    params: QueryAdapterParams,
    metric: str,
    corpus: Corpus | None = None,
    checkpoint: Callable[[], None] = lambda: None,
    on_epoch: Callable[[AdapterEpoch], None] = lambda epoch: None,
    clock: Callable[[], float] = time.perf_counter,
) -> AdapterResult:
    """Learn a query adapter (the module docstring says how). ``checkpoint`` is called before
    every step, so a cancelled job stops within one batch."""
    timer = clock
    if len(train) == 0:
        raise ValueError("no training examples")
    dims = train.queries.shape[1]
    identity = np.eye(dims)
    weights = identity.copy()
    first, second = np.zeros_like(weights), np.zeros_like(weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(params.seed)
    held = validation if validation is not None and len(validation) else None

    def measure(epoch: int, train_loss: float | None, started: float) -> AdapterEpoch:
        loss = recall = mrr = None
        if held is not None:
            loss = mean_loss(weights, held, params=params, metric=metric)
            if corpus is not None and corpus.rows:
                recall, mrr = retrieval_metrics(weights, held, corpus, metric=metric)
        return AdapterEpoch(epoch, train_loss, loss, recall, mrr, timer() - started)

    baseline = measure(0, None, timer())
    best_weights, best_epoch = weights.copy(), 0
    best_score = (baseline.mrr_at_10 if baseline.mrr_at_10 is not None else -math.inf, 0.0)
    epochs: list[AdapterEpoch] = []
    waited, step, stopped = 0, 0, False
    for epoch in range(1, params.epochs + 1):
        started = timer()
        order = rng.permutation(len(train))
        total, seen = 0.0, 0
        for start in range(0, len(train), params.batch_size):
            checkpoint()
            batch = train.take(order[start : start + params.batch_size])
            loss, gradient = adapter_loss(
                weights,
                batch,
                temperature=params.temperature,
                metric=metric,
                in_batch=params.in_batch,
                identity_weight=params.identity_weight,
            )
            step += 1
            first = beta1 * first + (1 - beta1) * gradient
            second = beta2 * second + (1 - beta2) * gradient * gradient
            corrected = first / (1 - beta1**step)
            scale = np.sqrt(second / (1 - beta2**step)) + epsilon
            weights = weights - params.learning_rate * corrected / scale
            total += loss * len(batch)
            seen += len(batch)
        result = measure(epoch, total / seen, started)
        epochs.append(result)
        on_epoch(result)
        if held is None:
            best_weights, best_epoch = weights.copy(), epoch
            continue
        score = (
            result.mrr_at_10 if result.mrr_at_10 is not None else -math.inf,
            -(result.validation_loss or 0.0),
        )
        if score > best_score:
            best_score, best_weights, best_epoch, waited = score, weights.copy(), epoch, 0
        else:
            waited += 1
            if params.patience and waited >= params.patience:
                stopped = True
                break
    return AdapterResult(
        best_weights.astype(np.float32), epochs, best_epoch, baseline, stopped_early=stopped
    )


def adapter_bytes(weights: np.ndarray) -> bytes:
    """An adapter's file (``adapter.npy``): float32 ``[d, d]``."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(weights, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def load_adapter(data: bytes) -> np.ndarray:
    """An adapter's weights from its file; ``ComponentError`` when they aren't a square
    matrix."""
    try:
        weights = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError):
        raise ComponentError("model_corrupt") from None
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1] or weights.shape[0] == 0:
        raise ComponentError("model_corrupt")
    return np.ascontiguousarray(weights, dtype=np.float32)


def apply_adapter(weights: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Query vectors ``[n, d]`` through an adapter: ``W q`` for each."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != weights.shape[1]:
        raise ValueError("the adapter's dimensions don't match the vector's")
    return (matrix @ np.asarray(weights, dtype=np.float32).T).astype(np.float32)


# --- models and their records --------------------------------------------------------------


class BaseModelRef(_M):
    """What a model was trained from: a component and its model, revision and shape."""

    component: str = Field(description="The component that runs the base, e.g. embed/gateway")
    model: str = Field(description="Its model: a gateway alias or a catalog ID")
    revision: str | None = None
    dims: int | None = Field(default=None, description="Embedding dimensions, for embedders")
    metric: str | None = Field(default=None, description="How the dense index scores vectors")
    license: str | None = None


class ModelFile(_M):
    path: str = Field(description="Where the file goes in the model's directory")
    part: str = Field(description="The artifact part that holds it")
    sha256: str
    size: int


class ModelManifest(_M):
    """A model artifact's main object: what it is and which parts hold its files."""

    schema_: Literal["stratafi.model.v1"] = Field(default=MODEL_MANIFEST_SCHEMA, alias="schema")
    kind: ModelKind
    format: str = Field(description="stratafi-query-adapter-v1 or sentence-transformers")
    base: BaseModelRef
    files: list[ModelFile]

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def part_name(index: int) -> str:
    return f"f{index:03d}"


def model_key(
    *,
    trainer: str,
    trainer_version: str,
    params: Mapping[str, Any],
    training_set_key: str,
    base: BaseModelRef,
) -> str:
    """The cache key of a trained model: training the same thing twice reuses it."""
    return cache_key(
        stage=TRAIN_STAGE,
        component_id=trainer,
        component_version=trainer_version,
        params=dict(params),
        inputs=[training_set_key],
        env={"base": base.model_dump(mode="json", exclude={"license"})},
    )


class _R(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TrainingRunInfo(_R):
    id: str
    created_by: str
    created_at: datetime


class TrainingConfiguration(_R):
    trainer: dict[str, Any] = Field(description="The trainer's component, version and parameters")
    params_hash: str
    kind: ModelKind


class TrainingInputs(_R):
    training_set: dict[str, Any] = Field(
        description="ID, cache key, eval run, eval set (name, version, split) and counts"
    )
    base: BaseModelRef


class TrainingExecution(_R):
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = None
    error_code: str | None = None
    model_cached: bool = Field(default=False, description="An identical run's model was reused")


class TrainingMetrics(_R):
    baseline: dict[str, Any] | None = Field(
        default=None, description="The base model on the validation items, before training"
    )
    epochs: list[dict[str, Any]] = Field(default_factory=list)
    best_epoch: int | None = None
    stopped_early: bool = False
    examples: dict[str, int] = Field(default_factory=dict, description="Train and validation")


class TrainingOutput(_R):
    cache_key: str
    kind: str
    size_bytes: int
    files: int


class TrainingRunManifest(_R):
    """``stratafi.training-run.v1``: everything a training run recorded, metadata only."""

    schema_: Literal["stratafi.training-run.v1"] = Field(
        default=TRAINING_RUN_SCHEMA, alias="schema"
    )
    training_run: TrainingRunInfo
    configuration: TrainingConfiguration
    inputs: TrainingInputs
    environment: ManifestEnvironment
    hardware: ManifestHardware
    execution: TrainingExecution
    metrics: TrainingMetrics = Field(default_factory=TrainingMetrics)
    outputs: list[TrainingOutput] = Field(default_factory=list)

    def fingerprint(self) -> str:
        return content_hash(self.model_dump(mode="json", by_alias=True))


class ModelCardModel(_R):
    id: str
    name: str
    version: int
    kind: ModelKind
    created_by: str
    created_at: datetime


class ModelCardUse(_R):
    how: str = Field(description="Where the model is named to use it")
    lab_only: bool = True
    needs: list[str] = Field(default_factory=list, description="Settings and extras it needs")


class ModelCard(_R):
    """``stratafi.model-card.v1``: what a trained model is, what made it, and its limits."""

    schema_: Literal["stratafi.model-card.v1"] = Field(default=MODEL_CARD_SCHEMA, alias="schema")
    model: ModelCardModel
    base: BaseModelRef
    trainer: dict[str, Any] = Field(description="Component, version and parameters")
    training_set: dict[str, Any] = Field(
        description="ID, eval set, split, counts and the leak guard's findings"
    )
    training: dict[str, Any] = Field(
        description="The run, its baseline, best epoch and that epoch's validation metrics"
    )
    use: ModelCardUse
    limits: list[str] = Field(default_factory=list)
    artifact: dict[str, Any] = Field(description="Cache key, size and files")


def model_use(kind: str, model_id: str) -> ModelCardUse:
    """How a model of ``kind`` is used, and what it needs."""
    if kind == "query-adapter":
        return ModelCardUse(
            how=f"retrieval config: retrievers.dense.adapter: {model_id}",
            needs=["models.in_process"],
        )
    component = "embed/sentence-transformers" if kind == "embedder" else "rerank/cross-encoder"
    return ModelCardUse(
        how=f"{component} with model: trained/{model_id}",
        needs=["models.in_process", "the sentence-transformers extra"],
    )


def model_limits(kind: str, stats: Mapping[str, Any]) -> list[str]:
    """What the card says not to rely on."""
    limits = [
        "Trained on one eval set's dev split: confirm it once on the test split before relying "
        "on it.",
    ]
    if kind == "query-adapter":
        limits += [
            "Lab only: Stratafi's own retrieval applies query adapters; a search service "
            "elsewhere needs the adapter added before it can use one.",
            "Fits one embedding model, revision and dimension count: a new revision needs a new "
            "adapter.",
        ]
    else:
        limits.append("Lab only: production can't run it until the model gateway serves it.")
    pairs = int(stats.get("pairs") or 0)
    if pairs < 200:
        limits.append(
            f"Small training set ({pairs} items with evidence): expect overfitting; compare "
            "against the base with a paired test before choosing it."
        )
    leak = stats.get("leak_guard") or {}
    if leak.get("test_overlap"):
        limits.append(
            "The test split's queries share documents with the training positives: a gain on "
            "the test split measures adaptation to this corpus."
        )
    return limits
