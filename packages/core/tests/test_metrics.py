"""Retrieval metrics: every metric against hand-computed values, violations, where
a query was lost, and aggregates whose intervals reproduce for a seed."""

import math

import pytest

from stratafi_core.evalsets import EvalItem
from stratafi_core.metrics import aggregate, default_ks, score_query
from stratafi_core.relevance import DOC, EVIDENCE_COVERAGE, KNOWLEDGE_HIT, Passage

E1 = "Business days are Monday through Friday, except federal holidays"
E2 = "A teller may place a large-deposit hold without approval"

ITEM = EvalItem.model_validate(
    {
        "id": "q1",
        "query": "When is a deposit made, and who approves holds?",
        "expected": [
            {"doc_key": "funds", "version": "3", "section": "2 Day of deposit", "evidence": E1},
            {"doc_key": "funds", "version": "3", "section": "4.3 Approval", "evidence": E2},
        ],
        "avoid": [{"doc_key": "funds", "version": "2"}],
        "must_not_return": [{"doc_key": "board"}],
    }
)

PASSAGES = {
    "old": Passage("old", "funds", "2", "2 Day of deposit", f"Old rules. {E1}."),
    "p1": Passage("p1", "funds", "3", "2 Day of deposit", f"{E1}. Deposits after 5 p.m. wait."),
    "other": Passage("other", "fees", "5", "1 Fees", "A $30 fee applies to returned items."),
    "p2": Passage("p2", "funds", "3", "4.3 Approval", f"{E2}; others need a supervisor."),
    "same-doc": Passage("same-doc", "funds", "3", "3 Standard", "Cash is available at once."),
    "board": Passage("board", "board", "2", "Limits", "Indirect loans are capped."),
}


def ranked(*ids: str) -> list[tuple[str, float]]:
    return [(chunk_id, 1.0 - n / 10) for n, chunk_id in enumerate(ids)]


def lists(final: tuple[str, ...], **others: tuple[str, ...]) -> dict[str, list[tuple[str, float]]]:
    checkpoints = {name: ranked(*ids) for name, ids in others.items()}
    checkpoints["final"] = ranked(*final)
    checkpoints.setdefault("candidates", ranked(*final))
    checkpoints.setdefault("fused", ranked(*final))
    return checkpoints


def test_every_metric_matches_hand_computed_values() -> None:
    """Every metric matches hand-computed fixtures."""
    score = score_query(
        ITEM,
        lists(("old", "p1", "other", "p2")),
        PASSAGES,
        ks=[1, 2, 3, 4],
        k_eval=3,
    )
    final = score.metrics[KNOWLEDGE_HIT]["final"]
    # Relevant at ranks 2 (evidence 1) and 4 (evidence 2), of two evidence items.
    assert [final[f"hit@{k}"] for k in (1, 2, 3, 4)] == [0.0, 1.0, 1.0, 1.0]
    assert [final[f"recall@{k}"] for k in (1, 2, 3, 4)] == [0.0, 0.5, 0.5, 1.0]
    assert [final[f"precision@{k}"] for k in (1, 2, 3, 4)] == [0.0, 0.5, round(1 / 3, 6), 0.5]
    assert [final[f"mrr@{k}"] for k in (1, 2, 3, 4)] == [0.0, 0.5, 0.5, 0.5]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    assert final["ndcg@4"] == round(dcg / (1 + 1 / math.log2(3)), 6)
    assert final["ndcg@2"] == round((1 / math.log2(3)) / (1 + 1 / math.log2(3)), 6)
    assert final["recall@all"] == 1.0
    assert score.first_hit == {KNOWLEDGE_HIT: 2, EVIDENCE_COVERAGE: 2, DOC: 2}
    # doc-v1 also counts the "same-doc" passage if it appears; coverage grades are 1 here.
    assert score.metrics[DOC]["final"]["precision@4"] == 0.5
    assert score.metrics[EVIDENCE_COVERAGE]["final"]["ndcg@4"] == final["ndcg@4"]
    # The avoided version ranks above the first hit; nothing must-not-return came back.
    assert score.avoid_violation is True
    assert score.must_not_return_violation is False
    assert score.lost_at is None


def test_graded_gains_feed_ndcg_and_credit_each_evidence_item_once() -> None:
    partial = Passage("partial", "funds", "3", "Other", "Business days are Monday to Friday.")
    passages = {**PASSAGES, "partial": partial}
    score = score_query(
        ITEM,
        lists(("partial", "p1", "p1b")),
        {**passages, "p1b": PASSAGES["p1"]},
        ks=[1, 3],
        k_eval=3,
    )
    coverage = score.metrics[EVIDENCE_COVERAGE]["final"]
    grade = 6 / 9  # business days are monday friday ... of 9 evidence tokens: below 0.7
    assert grade < 0.7
    # p1 (rank 2) is the first to cover evidence 1; its copy at rank 3 earns nothing more.
    assert coverage["ndcg@3"] == round((1 / math.log2(3)) / (1 + 1 / math.log2(3)), 6)
    assert score.metrics[DOC]["final"]["hit@1"] == 1.0
    assert score.metrics[KNOWLEDGE_HIT]["final"]["hit@1"] == 0.0


def test_where_a_missed_query_was_lost() -> None:
    misses = lists(("other", "same-doc"), candidates=("other", "same-doc", "p1"))
    assert score_query(ITEM, misses, PASSAGES, ks=[2], k_eval=2).lost_at == "rerank"
    nowhere = lists(("other",), candidates=("other", "same-doc"), dense=("p1",))
    lost = score_query(ITEM, nowhere, PASSAGES, ks=[2], k_eval=2)
    assert lost.lost_at == "retrieval"
    assert lost.found_in == {"dense": True, "candidates": False, "fused": False, "final": False}
    parse = score_query(ITEM, nowhere, PASSAGES, ks=[2], k_eval=2, findable=[False, False])
    assert parse.lost_at == "parse"
    partly = score_query(ITEM, nowhere, PASSAGES, ks=[2], k_eval=2, findable=[False, True])
    assert partly.lost_at == "retrieval"
    hidden = score_query(ITEM, nowhere, PASSAGES, ks=[2], k_eval=2, hidden=[True, True])
    assert (hidden.lost_at, hidden.hidden) == ("retrieval", [True, True])
    assert aggregate([hidden], resamples=100)["where_lost"]["filtered"] == 1


def test_violations_and_abstention_near_misses() -> None:
    returned = score_query(ITEM, lists(("p1", "board")), PASSAGES, ks=[2], k_eval=2)
    assert returned.must_not_return_violation is True
    assert returned.avoid_violation is False
    abstain = EvalItem(id="q2", query="Is there a crypto policy?")
    score = score_query(abstain, lists(("other", "board", "p1", "p2")), PASSAGES, ks=[5], k_eval=5)
    assert score.has_evidence is False
    assert score.metrics == {}
    assert score.lost_at is None
    assert score.avoid_violation is None
    assert [miss["doc_key"] for miss in score.near_misses] == ["fees", "board", "funds"]
    assert score.near_misses[0] == {
        "rank": 1, "chunk_id": "other", "doc_key": "fees", "version": "5", "score": 1.0
    }  # fmt: skip


def scores() -> list:
    found = score_query(ITEM, lists(("p1", "p2")), PASSAGES, ks=[1, 2], k_eval=2)
    half = score_query(ITEM, lists(("p1", "other")), PASSAGES, ks=[1, 2], k_eval=2)
    lost = score_query(
        ITEM, lists(("other",), candidates=("other", "p1")), PASSAGES, ks=[1, 2], k_eval=2
    )
    abstain_item = EvalItem(id="q9", query="Crypto?", must_not_return=[{"doc_key": "board"}])
    abstain = score_query(abstain_item, lists(("board",)), PASSAGES, ks=[1, 2], k_eval=2)
    return [found, half, lost, abstain]


def test_aggregates_average_queries_with_evidence_and_count_the_rest() -> None:
    result = aggregate(scores(), resamples=500, seed=3)
    assert (result["queries"], result["with_evidence"], result["abstention"]) == (4, 3, 1)
    recall = result["metrics"][KNOWLEDGE_HIT]["final"]["recall@2"]
    assert recall["mean"] == pytest.approx((1.0 + 0.5 + 0.0) / 3, abs=1e-6)
    assert recall["n"] == 3
    assert recall["low"] <= recall["mean"] <= recall["high"]
    assert result["where_lost"] == {
        "found": 2, "parse": 0, "retrieval": 0, "rerank": 1, "filtered": 0
    }  # fmt: skip
    assert result["violations"]["must_not_return"] == {"count": 1, "of": 4}
    assert result["violations"]["avoid"] == {"count": 0, "of": 3}
    uplift = result["uplift"][KNOWLEDGE_HIT]["recall@2"]
    assert uplift["mean"] == pytest.approx(0.0)  # these finals are their fused lists


def test_intervals_reproduce_for_a_seed() -> None:
    """CIs reproduce for a given seed."""
    first = aggregate(scores(), resamples=800, seed=11)
    again = aggregate(scores(), resamples=800, seed=11)
    other = aggregate(scores(), resamples=800, seed=12)
    assert first == again
    path = ("metrics", KNOWLEDGE_HIT, "final", "mrr@2")
    pick = lambda result: result[path[0]][path[1]][path[2]][path[3]]  # noqa: E731
    assert pick(first) == pick(again)
    assert pick(first)["mean"] == pick(other)["mean"]


def test_aggregates_can_pick_their_metrics() -> None:
    result = aggregate(
        scores(), resamples=200, wanted=lambda rule, checkpoint, name: name == "hit@1"
    )
    assert set(result["metrics"]) == {KNOWLEDGE_HIT, EVIDENCE_COVERAGE, DOC}
    assert set(result["metrics"][KNOWLEDGE_HIT]["final"]) == {"hit@1"}


def test_default_cutoffs_include_the_sets_k_and_the_configs_top_k() -> None:
    assert default_ks(5, 8) == [1, 3, 5, 8, 10]
    assert default_ks(20) == [1, 3, 5, 10, 20]
