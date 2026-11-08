"""Post-training's core: mining training sets from evidence, the leak guard,
the exports, the query adapter's loss, gradient and training, and the records."""

from datetime import UTC, datetime

import numpy as np
import pytest

from stratafi_core.answers import NOT_IN_SOURCES, citations_in
from stratafi_core.evalruns import RankedItem
from stratafi_core.evalsets import EvalItem
from stratafi_core.manifest import ManifestEnvironment, ManifestHardware
from stratafi_core.relevance import EVIDENCE_COVERAGE, KNOWLEDGE_HIT, Matcher, Passage
from stratafi_core.training import (
    AdapterExamples,
    BaseModelRef,
    Corpus,
    ModelCard,
    ModelCardModel,
    QueryAdapterParams,
    TrainingConfiguration,
    TrainingExecution,
    TrainingInputs,
    TrainingRecord,
    TrainingRunInfo,
    TrainingRunManifest,
    TrainingSetOptions,
    adapter_examples,
    adapter_loss,
    apply_adapter,
    export_rows,
    jaccard,
    mine,
    model_key,
    model_limits,
    model_use,
    query_tokens,
    read_records,
    records_bytes,
    retrieval_metrics,
    train_adapter,
    validation_ids,
)

EVIDENCE = "An account is a new account during the first 30 calendar days after it is opened"


def passage(chunk_id: str, doc_key: str, version: str, section: str, text: str) -> Passage:
    return Passage(chunk_id, doc_key, version, section, text)


def corpus() -> dict[str, Passage]:
    found = [
        passage("v3-new", "funds", "3", "5.1 New accounts", f"Rules. {EVIDENCE}. Holds apply."),
        passage("v3-new-b", "funds", "3", "5.1 New accounts", f"Also: {EVIDENCE} in branches."),
        passage("v3-std", "funds", "3", "3 Standard", "Deposits are available the next day."),
        passage("v2-new", "funds", "2", "5.1 New accounts", f"Old rules. {EVIDENCE}."),
        passage("v2-std", "funds", "2", "3 Standard", "Old deposits wait two days for funds."),
        passage("fees", "fees", "1", "Fees", "A wire costs ten dollars at the counter."),
        passage("cards", "cards", "1", "Cards", "Lost cards are replaced in a week."),
        passage("secret", "secret", "1", "Secret", "Internal memo about check holds."),
    ]
    return {item.chunk_id: item for item in found}


def item(item_id: str, query: str, **fields: object) -> EvalItem:
    return EvalItem.model_validate({"id": item_id, "query": query, **fields})


NEW_ACCOUNT = item(
    "new-account",
    "What is the hold on a check for a brand new account?",
    expected=[
        {"doc_key": "funds", "version": "3", "section": "5.1 New accounts", "evidence": EVIDENCE}
    ],
    avoid=[{"doc_key": "funds", "version": "2"}],
    must_not_return=[{"doc_key": "secret"}],
    split="dev",
)
NOWHERE = item("nowhere", "Can I open an account on the moon?", expected=[], split="dev")


def ranked(item_id: str, **checkpoints: list[str]) -> RankedItem:
    return RankedItem(
        item_id=item_id,
        checkpoints={
            name: [(chunk_id, 1.0 - rank / 100) for rank, chunk_id in enumerate(ids)]
            for name, ids in checkpoints.items()
        },
        latency_ms={},
    )


RANKED = {
    "new-account": ranked(
        "new-account",
        dense=["fees", "v2-new", "v3-std", "secret", "v2-std", "v3-new", "cards"],
        keyword=["v3-std", "cards"],
        candidates=["v2-std", "fees"],
    ),
    "nowhere": ranked("nowhere", dense=["cards", "fees", "v3-std"], keyword=["fees"]),
}


def test_positives_come_from_the_index_and_negatives_in_rank_order() -> None:
    mined = mine([NEW_ACCOUNT, NOWHERE], RANKED, corpus(), options=TrainingSetOptions())
    record = next(r for r in mined.records if r.item_id == "new-account")
    # Both chunks of v3 that hold the evidence, the retrieved one first among equal grades;
    # v3-new-b was never retrieved: a positive retrieval missed.
    assert [p.chunk_id for p in record.positives] == ["v3-new", "v3-new-b"]
    assert record.positives[0].rank == 6
    assert record.positives[0].checkpoint == "dense"
    assert record.positives[1].rank is None
    assert all(p.reason == "evidence" and p.grade == 1.0 for p in record.positives)
    assert mined.stats["positives_missed"] == 1
    # Traps first (avoid, must_not_return), then by best rank across the lists; v2-new repeats
    # the evidence in another version, so it's no negative at all.
    assert [(n.chunk_id, n.reason) for n in record.negatives] == [
        ("v2-std", "avoid"),
        ("secret", "must_not_return"),
        ("fees", "retrieved"),
        ("v3-std", "retrieved"),
        ("cards", "retrieved"),
    ]
    assert [(n.rank, n.checkpoint) for n in record.negatives] == [
        (1, "candidates"),
        (4, "dense"),
        (1, "dense"),
        (1, "keyword"),
        (2, "keyword"),
    ]
    assert mined.stats["negatives_overlapping"] == 1
    # Nothing relevant by either passage rule is ever a negative.
    matcher = Matcher(NEW_ACCOUNT.expected)
    for negative in record.negatives:
        judged = matcher.judge(corpus()[negative.chunk_id])
        assert not judged.relevant(KNOWLEDGE_HIT)
        assert not judged.relevant(EVIDENCE_COVERAGE)


def test_items_without_evidence_give_negatives_only() -> None:
    mined = mine([NOWHERE], RANKED, corpus(), options=TrainingSetOptions(max_negatives=2))
    [record] = mined.records
    assert record.abstention
    assert record.positives == []
    assert [(n.chunk_id, n.rank) for n in record.negatives] == [("cards", 1), ("fees", 1)]
    assert {n.reason for n in record.negatives} == {"abstention"}
    skipped = mine([NOWHERE], RANKED, corpus(), options=TrainingSetOptions(abstentions=False))
    assert skipped.records == []
    assert skipped.stats["dropped"] == {"abstention": 1}


def test_limits_depth_and_the_knowledge_hit_rule() -> None:
    options = TrainingSetOptions(max_positives=1, max_negatives=2, depth=2, checkpoints=("dense",))
    [record] = mine([NEW_ACCOUNT], RANKED, corpus(), options=options).records
    assert [p.chunk_id for p in record.positives] == ["v3-new"]
    # Within depth 2 of the dense list: fees, then v2-new, which repeats the evidence.
    assert [n.chunk_id for n in record.negatives] == ["fees"]
    by_section = TrainingSetOptions(rule="knowledge-hit-v1")
    [hit] = mine([NEW_ACCOUNT], RANKED, corpus(), options=by_section).records
    assert {p.chunk_id for p in hit.positives} == {"v3-new", "v3-new-b"}


def test_evidence_no_chunk_holds_drops_the_item() -> None:
    lost = item(
        "lost",
        "Where is the fee table?",
        expected=[{"doc_key": "fees", "version": "1", "evidence": "a table that the parser lost"}],
    )
    mined = mine([lost], {}, corpus(), options=TrainingSetOptions())
    assert mined.records == []
    assert mined.stats["dropped"] == {"no_positive": 1}


def test_the_leak_guard_drops_near_duplicates_of_test_queries() -> None:
    copy = item(
        "copy",
        "how long is the hold on a check deposited into a brand new account today",
        expected=NEW_ACCOUNT.model_dump()["expected"],
    )
    test = item(
        "held-out",
        "How long is the hold on a check deposited into a brand new account?",
        expected=NEW_ACCOUNT.model_dump()["expected"],
        split="test",
    )
    assert jaccard(query_tokens(copy.query), query_tokens(test.query)) >= 0.8
    mined = mine(
        [copy, NOWHERE],
        RANKED,
        corpus(),
        options=TrainingSetOptions(),
        test_items=[test],
    )
    assert [r.item_id for r in mined.records] == ["nowhere"]
    assert mined.stats["leak_guard"] == {
        "test_items": 1,
        "near_duplicates": 1,
        "test_items_overlapping": 0,
        "test_overlap": 0.0,
    }
    kept = mine([NEW_ACCOUNT], RANKED, corpus(), options=TrainingSetOptions(), test_items=[test])
    assert kept.stats["leak_guard"]["test_overlap"] == 1.0
    loose = TrainingSetOptions(duplicate_threshold=1.0)
    kept_copy = mine([copy], RANKED, corpus(), options=loose, test_items=[test])
    assert [r.item_id for r in kept_copy.records] == ["copy"]


def test_the_validation_split_is_seeded() -> None:
    ids = [f"item-{n}" for n in range(50)]
    first = validation_ids(ids, 0.2, seed=3)
    assert len(first) == 10
    assert first == validation_ids(reversed(ids), 0.2, seed=3)
    assert first != validation_ids(ids, 0.2, seed=4)
    assert validation_ids(ids, 0.0, seed=3) == set()
    items = [
        item(f"q{n}", f"question {n}", expected=NEW_ACCOUNT.model_dump()["expected"])
        for n in range(10)
    ]
    mined = mine(items, {}, corpus(), options=TrainingSetOptions(validation_share=0.3, seed=1))
    assert sum(r.split == "validation" for r in mined.records) == 3
    assert mined.stats["validation"] == 3


def test_records_round_trip() -> None:
    mined = mine([NEW_ACCOUNT, NOWHERE], RANKED, corpus(), options=TrainingSetOptions())
    assert read_records(records_bytes(mined.records)) == mined.records


def test_the_exports() -> None:
    answered = NEW_ACCOUNT.model_copy(
        update={"reference_answer": "It's a new account for 30 days."}
    )
    records = mine([answered, NOWHERE], RANKED, corpus(), options=TrainingSetOptions()).records
    pairs = list(export_rows(records, "sentence-transformers"))
    assert len(pairs) == 2  # one row per positive
    assert set(pairs[0]) == {"item_id", "split", "anchor", "positive", "negatives"}
    assert pairs[0]["anchor"] == answered.query
    assert len(pairs[0]["negatives"]) == 5
    scored = list(export_rows(records, "cross-encoder"))
    assert [row["label"] for row in scored].count(1.0) == 2
    assert len(scored) == 2 + 5 + 3
    chats = list(export_rows(records, "chat", day="2026-11-02"))
    assert len(chats) == 2
    first, second = chats
    roles = [message["role"] for message in first["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "Today's date is 2026-11-02." in first["messages"][1]["content"]
    answer = first["messages"][2]["content"]
    assert answer.startswith("It's a new account for 30 days.")
    # It cites exactly the gold passages, by the IDs the context gave them.
    cited = citations_in(answer)
    assert len(cited) == 2
    assert all(f"[{source}]" in first["messages"][1]["content"] for source in cited)
    assert second["messages"][2]["content"] == NOT_IN_SOURCES
    # Without a reference answer an item with evidence has no target.
    plain = mine([NEW_ACCOUNT], RANKED, corpus(), options=TrainingSetOptions()).records
    assert list(export_rows(plain, "chat")) == []
    assert list(export_rows(records, "cross-encoder", split="validation")) == []


# --- the query adapter -------------------------------------------------------------------


def random_examples(seed: int, *, n: int = 6, d: int = 5, h: int = 3) -> AdapterExamples:
    rng = np.random.default_rng(seed)
    mask = rng.random((n, h)) > 0.3
    return AdapterExamples(
        item_ids=[f"i{index // 2}" for index in range(n)],
        queries=rng.normal(size=(n, d)),
        positives=rng.normal(size=(n, d)),
        positive_ids=[f"p{index}" for index in range(n)],
        negatives=rng.normal(size=(n, h, d)),
        negative_mask=mask,
        relevant=[
            frozenset({f"p{index - index % 2}", f"p{index - index % 2 + 1}"}) for index in range(n)
        ],
    )


@pytest.mark.parametrize("metric", ["cosine", "dot"])
@pytest.mark.parametrize("in_batch", [True, False])
def test_the_gradient_matches_finite_differences(metric: str, in_batch: bool) -> None:
    batch = random_examples(7)
    rng = np.random.default_rng(1)
    weights = np.eye(5) + 0.1 * rng.normal(size=(5, 5))
    options = {"temperature": 0.3, "metric": metric, "in_batch": in_batch, "identity_weight": 0.05}
    _, gradient = adapter_loss(weights, batch, **options)
    numeric = np.zeros_like(weights)
    step = 1e-6
    for i in range(5):
        for j in range(5):
            up, down = weights.copy(), weights.copy()
            up[i, j] += step
            down[i, j] -= step
            numeric[i, j] = (
                adapter_loss(up, batch, **options)[0] - adapter_loss(down, batch, **options)[0]
            ) / (2 * step)
    assert np.allclose(gradient, numeric, atol=1e-6, rtol=1e-4)


def test_in_batch_negatives_skip_the_items_own_positives() -> None:
    batch = random_examples(3, n=2)
    # Both examples are one item with two positives: neither is the other's negative, so
    # with no hard negatives the loss is zero.
    empty = AdapterExamples(
        batch.item_ids,
        batch.queries,
        batch.positives,
        batch.positive_ids,
        batch.negatives,
        np.zeros_like(batch.negative_mask),
        batch.relevant,
    )
    loss, gradient = adapter_loss(np.eye(5), empty, temperature=0.1, metric="cosine")
    assert loss == pytest.approx(0.0)
    assert np.allclose(gradient, 0.0)


def rotated_problem(seed: int, *, items: int, d: int = 12) -> tuple[AdapterExamples, Corpus]:
    """Queries that land on their passages only after a fixed rotation."""
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
    passages = rng.normal(size=(items * 3, d))
    ids = [f"c{index}" for index in range(len(passages))]
    queries = {f"q{n}": rotation.T @ passages[n] + 0.05 * rng.normal(size=d) for n in range(items)}
    records = [
        TrainingRecord(
            item_id=f"q{n}",
            split="train",
            query="(vectors only)",
            positives=[{"chunk_id": f"c{n}", "text": "-", "reason": "evidence"}],
            negatives=[
                {"chunk_id": f"c{(n + k) % len(ids)}", "text": "-", "reason": "retrieved"}
                for k in (1, 2, 3)
            ],
        )
        for n in range(items)
    ]
    vectors = dict(zip(ids, passages, strict=True))
    examples = adapter_examples(
        records, queries, vectors, hard_negatives=3, metric="cosine", dims=d
    )
    return examples, Corpus.of(passages, ids, metric="cosine")


def test_training_lowers_the_loss_and_lifts_validation_retrieval() -> None:
    examples, corpus_ = rotated_problem(0, items=40)
    train, validation = examples.take(np.arange(30)), examples.take(np.arange(30, 40))
    params = QueryAdapterParams(epochs=60, learning_rate=0.02, temperature=0.1, patience=0, seed=5)
    seen = []
    result = train_adapter(
        train, validation, params=params, metric="cosine", corpus=corpus_, on_epoch=seen.append
    )
    assert [epoch.epoch for epoch in seen] == list(range(1, 61))
    assert seen[-1].train_loss < seen[0].train_loss
    final_recall, final_mrr = retrieval_metrics(
        result.weights, validation, corpus_, metric="cosine"
    )
    assert result.baseline.mrr_at_10 is not None
    assert final_mrr is not None
    assert final_mrr > result.baseline.mrr_at_10 + 0.2
    assert final_recall == pytest.approx(1.0)
    assert result.weights.dtype == np.float32
    assert result.weights.shape == (12, 12)
    # The same seed gives the same adapter.
    again = train_adapter(train, validation, params=params, metric="cosine", corpus=corpus_)
    assert np.array_equal(result.weights, again.weights)
    assert result.best_epoch == again.best_epoch


def test_patience_stops_early_and_keeps_the_best_epoch() -> None:
    examples, corpus_ = rotated_problem(1, items=12)
    params = QueryAdapterParams(epochs=200, learning_rate=0.5, patience=2, seed=0)
    train, validation = examples.take(np.arange(8)), examples.take(np.arange(8, 12))
    result = train_adapter(train, validation, params=params, metric="cosine", corpus=corpus_)
    assert result.stopped_early
    assert len(result.epochs) < 200
    assert result.best_epoch == len(result.epochs) - 2 or result.best_epoch == 0


def test_a_checkpoint_stops_training() -> None:
    examples, _ = rotated_problem(2, items=8)

    class Stop(Exception):
        pass

    calls = []

    def checkpoint() -> None:
        calls.append(1)
        if len(calls) == 3:
            raise Stop

    with pytest.raises(Stop):
        train_adapter(
            examples,
            None,
            params=QueryAdapterParams(batch_size=2),
            metric="dot",
            checkpoint=checkpoint,
        )


def test_applying_an_adapter() -> None:
    weights = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert apply_adapter(weights, np.array([1.0, 0.0, 0.0])).tolist() == [[0.0, 3.0, 6.0]]
    with pytest.raises(ValueError, match="dimensions"):
        apply_adapter(weights, np.ones((1, 4)))


# --- records -----------------------------------------------------------------------------


BASE = BaseModelRef(component="embed/gateway", model="embed-default", revision="abc", dims=384)


def test_a_models_key_follows_what_made_it() -> None:
    key = model_key(
        trainer="train/query-adapter",
        trainer_version="1.0.0",
        params={"epochs": 3},
        training_set_key="a" * 64,
        base=BASE,
    )
    assert len(key) == 64
    same = model_key(
        trainer="train/query-adapter",
        trainer_version="1.0.0",
        params={"epochs": 3},
        training_set_key="a" * 64,
        base=BASE.model_copy(update={"license": "MIT"}),
    )
    assert same == key  # the license describes the base; it doesn't change the model
    for change in (
        {"params": {"epochs": 4}},
        {"training_set_key": "b" * 64},
        {"base": BASE.model_copy(update={"revision": "def"})},
        {"trainer_version": "1.0.1"},
    ):
        arguments = {
            "trainer": "train/query-adapter",
            "trainer_version": "1.0.0",
            "params": {"epochs": 3},
            "training_set_key": "a" * 64,
            "base": BASE,
            **change,
        }
        assert model_key(**arguments) != key  # type: ignore[arg-type]


def test_the_run_record_and_the_card() -> None:
    now = datetime(2026, 11, 5, tzinfo=UTC)
    manifest = TrainingRunManifest(
        training_run=TrainingRunInfo(id="run", created_by="person", created_at=now),
        configuration=TrainingConfiguration(
            trainer={"component": "train/query-adapter", "version": "1.0.0", "params": {}},
            params_hash="0" * 64,
            kind="query-adapter",
        ),
        inputs=TrainingInputs(training_set={"id": "set"}, base=BASE),
        environment=ManifestEnvironment(stratafi_version="0.1.0", python="3.12", os="linux"),
        hardware=ManifestHardware(),
        execution=TrainingExecution(status="succeeded"),
    )
    assert manifest.fingerprint() == manifest.model_copy(deep=True).fingerprint()
    assert manifest.model_dump(by_alias=True)["schema"] == "stratafi.training-run.v1"
    use = model_use("query-adapter", "m1")
    assert "retrievers.dense.adapter: m1" in use.how
    assert use.needs == ["models.in_process"]
    assert model_use("reranker", "m2").how == "rerank/cross-encoder with model: trained/m2"
    assert "the sentence-transformers extra" in model_use("embedder", "m3").needs
    limits = model_limits("query-adapter", {"pairs": 40, "leak_guard": {"test_overlap": 0.5}})
    assert any("40 items" in limit for limit in limits)
    assert any("test split" in limit for limit in limits)
    card = ModelCard(
        model=ModelCardModel(
            id="m1", name="halcyon-adapter", version=1, kind="query-adapter", created_by="p",
            created_at=now,
        ),
        base=BASE,
        trainer={"component": "train/query-adapter"},
        training_set={"id": "set"},
        training={"run": "run"},
        use=use,
        limits=limits,
        artifact={"cache_key": "0" * 64},
    )  # fmt: skip
    assert card.model_dump(by_alias=True)["schema"] == "stratafi.model-card.v1"
