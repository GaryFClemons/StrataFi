"""Eval runs: a retrieval config over a pipeline run's indexes, scored on an eval set.

An eval run is planned like a pipeline run: its work has content-addressed keys, so the
same work is never done twice.

- **Sample.** An eval run scores its split's items, or with ``sample`` the first ``size`` of
  them in a seeded order (successive halving's rungs): the same seed gives nested
  subsets, so a bigger sample's first shards are a smaller one's.
- **Shards.** The items are cut into shards of ``shard_size`` queries. A shard's key hashes the
  retrieval engine's version, the resolved retrieval config, the indexes' keys, the corpus
  entries' metadata (what filters read), the embedder and reranker with their revisions, and
  each item's query, asker and as-of date. Its artifact (``ranked_lists``) holds every query's
  ranked list at every checkpoint, with scores and latencies.
- **Results.** Scoring maps the evidence onto those lists and aggregates. Its key hashes the
  scorer's version, the shards' keys, every item in full (evidence, avoid, must-not-return,
  tags, split), the options, and the cleaned documents findability reads. Its artifact
  (``eval_results``) holds the per-query scores and the aggregates, so an identical eval run
  reuses everything and finishes in one step.

``stratafi.eval-run.v1`` is the run's record: its configuration, inputs, models, environment,
hardware, execution, cost, egress and outputs, metadata only, as the run manifest is for a
pipeline run. Its hash is the eval run's fingerprint.
"""

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from stratafi_core.adaptive import query_order
from stratafi_core.evalsets import EvalItem, asker_of
from stratafi_core.hashing import cache_key, content_hash, sha256_hex
from stratafi_core.manifest import (
    ManifestCost,
    ManifestEgress,
    ManifestEnvironment,
    ManifestHardware,
    ManifestModel,
)
from stratafi_core.relevance import DEFAULT_THRESHOLD
from stratafi_core.stats import DEFAULT_RESAMPLES

EVAL_RUN_SCHEMA = "stratafi.eval-run.v1"
# Bump when retrieval's output could change (the engine) or scoring's could (the scorer).
ENGINE_VERSION = "1.0.0"
SCORER_VERSION = "1.0.0"
SHARD_STAGE = "eval-shard"
RESULTS_STAGE = "eval-results"
EvalSplit = Literal["dev", "test", "all"]
EvalStatus = Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]


class EvalSample(BaseModel):
    """The first ``size`` of the split's items in a seeded order (every item without one)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Seeds the order")
    size: int | None = Field(default=None, ge=1, le=100_000, description="Items scored")


class EvalOptions(BaseModel):
    """How an eval run scores; recorded with it and part of its results' key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shard_size: int = Field(default=25, ge=1, le=1000, description="Queries per shard job")
    coverage_threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        gt=0,
        le=1,
        description="evidence-coverage-v1: the grade at which a chunk covers evidence",
    )
    resamples: int = Field(
        default=DEFAULT_RESAMPLES, ge=100, le=20_000, description="Bootstrap resamples"
    )
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Seeds the bootstrap")
    sample: EvalSample | None = Field(
        default=None, description="Score a seeded sample of the split's items instead of all"
    )

    def dump(self) -> dict[str, Any]:
        """What an eval run records: without ``sample`` when there's none, so the options of
        eval runs from before samples existed are the same."""
        return self.model_dump(mode="json", exclude_none=True)


def sample_items(items: Sequence[EvalItem], sample: EvalSample | None) -> list[EvalItem]:
    """The items an eval run scores (the module docstring says how)."""
    if sample is None:
        return list(items)
    by_id = {item.id: item for item in items}
    return [by_id[item_id] for item_id in query_order(list(by_id), sample.seed)[: sample.size]]


def shards(items: Sequence[EvalItem], size: int) -> list[list[EvalItem]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def query_material(item: EvalItem, day: date | None) -> dict[str, Any]:
    """What retrieval reads of an item: never its evidence, so relabelling reuses retrieval."""
    asker = asker_of(item)
    return {
        "id": item.id,
        "query_sha256": sha256_hex(item.query.encode("utf-8")),
        "asker": asker.model_dump(mode="json", exclude={"name"}),
        "day": day.isoformat() if day else None,
    }


def shard_key(
    *,
    retrieval: Mapping[str, Any],
    indexes: Mapping[str, str],
    entries_hash: str,
    models: Mapping[str, Any],
    items: Sequence[EvalItem],
    day: date | None,
) -> str:
    """The cache key of one shard's ranked lists."""
    return cache_key(
        stage=SHARD_STAGE,
        component_id="eval/retrieval-engine",
        component_version=ENGINE_VERSION,
        params={"retrieval": dict(retrieval), "items": [query_material(i, day) for i in items]},
        inputs=[indexes[kind] for kind in sorted(indexes)],
        env={"entries": entries_hash, "models": dict(models)},
    )


def results_key(
    *,
    shard_keys: Sequence[str],
    items: Sequence[EvalItem],
    options: EvalOptions,
    k_eval: int,
    ks: Sequence[int],
    findability_inputs: Mapping[str, str],
) -> str:
    """The cache key of an eval run's scores and aggregates."""
    return cache_key(
        stage=RESULTS_STAGE,
        component_id="eval/scorer",
        component_version=SCORER_VERSION,
        params={
            "options": options.model_dump(mode="json", exclude={"shard_size"}, exclude_none=True),
            "k_eval": k_eval,
            "ks": list(ks),
            "items_sha256": content_hash([item.model_dump(mode="json") for item in items]),
        },
        inputs=list(shard_keys),
        env={"findability": dict(sorted(findability_inputs.items()))},
    )


# --- the record ---------------------------------------------------------------------------


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class EvalRunInfo(_M):
    id: str
    split: EvalSplit
    test_read: bool = Field(description="The run read the locked test split")
    created_by: str
    created_at: datetime


class EvalConfiguration(_M):
    retrieval: dict[str, Any] = Field(description="The resolved retrieval config")
    retrieval_hash: str
    retrieval_ref: dict[str, Any] = Field(description="The saved config: id, name and version")
    options: dict[str, Any]
    k_eval: int = Field(description="The cutoff the eval set's own rule uses")
    ks: list[int]
    engine_version: str
    scorer_version: str


class EvalInputs(_M):
    pipeline_run: dict[str, Any] = Field(description="The run searched: id, manifest hash, corpus")
    indexes: dict[str, str] = Field(description="Index kind → its artifact's cache key")
    eval_set: dict[str, Any] = Field(description="Name, version, hash, split, items, as_of")


class EvalShardRecord(_M):
    ordinal: int
    cache_key: str
    items: int
    status: str
    cache_hit: bool
    duration_ms: float | None = None
    error_code: str | None = None


class EvalExecution(_M):
    status: EvalStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = None
    error_code: str | None = None
    results_cached: bool = False
    shards: list[EvalShardRecord] = Field(default_factory=list)


class EvalOutput(_M):
    cache_key: str
    kind: str
    size_bytes: int


class EvalRunManifest(_M):
    """``stratafi.eval-run.v1``."""

    schema_: Literal["stratafi.eval-run.v1"] = Field(default=EVAL_RUN_SCHEMA, alias="schema")
    eval_run: EvalRunInfo
    configuration: EvalConfiguration
    inputs: EvalInputs
    models: list[ManifestModel] = Field(default_factory=list)
    environment: ManifestEnvironment
    hardware: ManifestHardware
    execution: EvalExecution
    cost: ManifestCost = Field(default_factory=ManifestCost)
    egress: list[ManifestEgress] = Field(default_factory=list)
    outputs: list[EvalOutput] = Field(default_factory=list)

    def fingerprint(self) -> str:
        return content_hash(self.model_dump(mode="json", by_alias=True))


# --- a shard's artifact -------------------------------------------------------------------


class RankedItem(_M):
    """One query's retrieval: a line of a ``ranked_lists`` artifact. Chunk IDs and scores
    only; texts stay in the chunks artifacts."""

    item_id: str
    checkpoints: dict[str, list[tuple[str, float]]]
    latency_ms: dict[str, float]
    embedding_revision: str | None = None
    rerank_revision: str | None = None
    visible_chunks: int = 0
