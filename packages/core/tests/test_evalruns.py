"""Eval run planning: identical eval runs get identical keys, so they reuse
cached results; anything that changes retrieval or scoring changes the right key, and nothing
else; a seeded sample scores nested subsets whose shards are shared."""

from datetime import date
from typing import Any

import pytest

from stratafi_core.evalruns import (
    EvalOptions,
    EvalSample,
    results_key,
    sample_items,
    shard_key,
    shards,
)
from stratafi_core.evalsets import Asker, EvalItem

STAFF = Asker(name="staff", user="u-halcyon-staff", roles=["staff"], groups=["all-staff"])
EXEC = Asker(name="exec", roles=["exec"], groups=["all-staff", "managers", "executives"])

ITEMS = [
    EvalItem(
        id=f"q{n}",
        query=f"Question {n}?",
        asker=STAFF,
        expected=[{"doc_key": "funds", "version": "3", "evidence": f"Phrase {n}"}],
        tags=["funds"],
        split="dev",
    )
    for n in range(7)
]
RETRIEVAL = {"schema": "stratafi.retrieval.v1", "name": "x", "top_k": 8}
INDEXES = {"dense": "a" * 64, "keyword": "b" * 64}
MODELS = {"embed": {"component": "embed/gateway", "revision": "r1"}, "rerank": {"revision": "r2"}}
DAY = date(2026, 11, 2)


def shard(**changes: Any) -> str:
    arguments: dict[str, Any] = {
        "retrieval": RETRIEVAL,
        "indexes": INDEXES,
        "entries_hash": "c" * 64,
        "models": MODELS,
        "items": ITEMS,
        "day": DAY,
    }
    return shard_key(**{**arguments, **changes})


def results(**changes: Any) -> str:
    arguments: dict[str, Any] = {
        "shard_keys": [shard()],
        "items": ITEMS,
        "options": EvalOptions(),
        "k_eval": 5,
        "ks": [1, 3, 5, 8, 10],
        "findability_inputs": {"funds@3": "d" * 64},
    }
    return results_key(**{**arguments, **changes})


def test_an_identical_eval_run_gets_the_same_keys() -> None:
    """An identical eval run reuses cached results: its shards and results keys are the
    ones the first run stored its artifacts under."""
    assert shard() == shard()
    assert results() == results()


@pytest.mark.parametrize(
    "change",
    [
        {"retrieval": {**RETRIEVAL, "top_k": 5}},
        {"indexes": {**INDEXES, "dense": "e" * 64}},
        {"entries_hash": "f" * 64},
        {"models": {**MODELS, "rerank": {"revision": "r3"}}},
        {"items": [ITEMS[0].model_copy(update={"query": "Another question?"}), *ITEMS[1:]]},
        {"items": [ITEMS[0].model_copy(update={"asker": EXEC}), *ITEMS[1:]]},
        {"day": date(2027, 1, 2)},
    ],
)
def test_what_retrieval_reads_changes_the_shard_key(change: dict[str, Any]) -> None:
    assert shard(**change) != shard()


def test_relabelling_evidence_reuses_retrieval_but_not_scoring() -> None:
    relabelled = [ITEMS[0].model_copy(update={"tags": ["other"], "split": "test"}), *ITEMS[1:]]
    assert shard(items=relabelled) == shard()
    assert results(items=relabelled) != results()
    renamed = [ITEMS[0].model_copy(update={"asker": STAFF.model_copy(
        update={"name": "teller"})}), *ITEMS[1:]]  # fmt: skip
    assert shard(items=renamed) == shard()


@pytest.mark.parametrize(
    "change",
    [
        {"options": EvalOptions(coverage_threshold=0.8)},
        {"options": EvalOptions(seed=1)},
        {"options": EvalOptions(resamples=500)},
        {"k_eval": 3},
        {"ks": [1, 5]},
        {"findability_inputs": {"funds@3": "0" * 64}},
        {"shard_keys": [shard(day=date(2027, 1, 2))]},
    ],
)
def test_what_scoring_reads_changes_the_results_key(change: dict[str, Any]) -> None:
    assert results(**change) != results()


def test_the_shard_size_cuts_the_items_but_not_the_scoring() -> None:
    assert [len(batch) for batch in shards(ITEMS, 3)] == [3, 3, 1]
    assert results(options=EvalOptions(shard_size=3)) == results()


def test_existing_keys_do_not_change() -> None:
    """Eval runs from before samples existed keep their cached shards and results."""
    assert shard() == "2f1bba164e9b819477b8d66db03fa7260eee351c9b49c6e71c721579ad59c45c"
    assert results() == "ad29e240e00e7d4e07d7989a6f8a6703fe0936f5dbd39f7090d6b6e7452beef0"
    assert EvalOptions().dump() == {
        "shard_size": 25,
        "coverage_threshold": 0.7,
        "resamples": 2000,
        "seed": 0,
    }


def test_a_sample_scores_nested_prefixes_of_one_seeded_order() -> None:
    every = sample_items(ITEMS, EvalSample(seed=4))
    assert sorted(item.id for item in every) == sorted(item.id for item in ITEMS)
    small = sample_items(ITEMS, EvalSample(seed=4, size=2))
    bigger = sample_items(ITEMS, EvalSample(seed=4, size=4))
    assert bigger[:2] == small
    assert every[:4] == bigger
    assert sample_items(ITEMS, None) == ITEMS
    assert sample_items(ITEMS, EvalSample(seed=5, size=4)) != bigger
    # Shards cut two at a time: the bigger sample's first shard is the smaller one's.
    assert shard(items=shards(bigger, 2)[0]) == shard(items=shards(small, 2)[0])
    # The sample is part of what scoring reads.
    sampled = EvalOptions(sample=EvalSample(seed=4, size=2))
    assert results(options=sampled) != results()
    assert sampled.dump()["sample"] == {"seed": 4, "size": 2}
