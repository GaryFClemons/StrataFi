"""Trainers and trained models in components: the ``train/`` kind, the
adapter trainer's output, the torch trainers' refusals, ``trained/<model ID>`` in the
in-process embedder and reranker, and adapters on query vectors.

The torch trainers' fine-tuning runs only with the ``sentence-transformers`` extra
and ``STRATAFI_TEST_HF_MODELS=1``, since it downloads two small models from Hugging Face. It
needs no endpoint, so it isn't in pytest's ``model`` tier, which waits for a gateway."""

import importlib.util
import os
import sys
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from stratafi_components import default_registry
from stratafi_components.embed import FakeHashEmbed
from stratafi_components.embedders import SentenceTransformersEmbed
from stratafi_components.index import DenseIndex
from stratafi_components.rerankers import CrossEncoderRerank
from stratafi_components.retrieval import (
    AdaptedQueryEmbedder,
    QueryEmbedder,
    with_adapter,
)
from stratafi_components.trainers import (
    CrossEncoderTrainer,
    QueryAdapterTrainer,
    SentenceTransformersTrainer,
    TrainingData,
)
from stratafi_core.artifacts import Chunk, IndexManifest
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.training import (
    ADAPTER_FILE,
    BaseModelRef,
    Corpus,
    TrainingRecord,
    load_adapter,
)

MODEL_ID = str(uuid.UUID(int=7))
BASE = BaseModelRef(component="embed/fake-hash", model="fake-hash", revision="r1", dims=12)


def test_trainers_are_lab_only_components_of_the_train_kind() -> None:
    trainers = {component.info.id: component for component in default_registry().for_stage("train")}
    assert set(trainers) == {
        "train/query-adapter",
        "train/sentence-transformers",
        "train/cross-encoder",
    }
    assert all(component.info.lab_only for component in trainers.values())
    assert trainers["train/query-adapter"].info.resource_class == "cpu"
    assert trainers["train/sentence-transformers"].info.resource_class == "gpu"
    assert {c.model_kind for c in trainers.values()} == {"query-adapter", "embedder", "reranker"}
    assert QueryAdapterTrainer.unavailable() is None


def rotated(seed: int, *, items: int = 30, dims: int = 12) -> TrainingData:
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(dims, dims)))
    passages = rng.normal(size=(items * 2, dims))
    ids = [f"c{n}" for n in range(len(passages))]
    records = [
        TrainingRecord(
            item_id=f"q{n}",
            split="validation" if n % 5 == 0 else "train",
            query="(vectors)",
            positives=[{"chunk_id": f"c{n}", "text": "-", "reason": "evidence"}],
            negatives=[
                {"chunk_id": f"c{(n + k) % len(ids)}", "text": "-", "reason": "retrieved"}
                for k in (1, 2)
            ],
        )
        for n in range(items)
    ]
    return TrainingData(
        records=records,
        base=BASE,
        queries={f"q{n}": rotation.T @ passages[n] for n in range(items)},
        vectors=dict(zip(ids, passages, strict=True)),
        corpus=Corpus.of(passages, ids, metric="cosine"),
        metric="cosine",
    )


def test_the_adapter_trainer_writes_its_weights_and_reports_each_epoch() -> None:
    trainer = QueryAdapterTrainer()
    params = trainer.resolve_params({"epochs": 30, "learning_rate": 0.02, "patience": 0})
    with pytest.raises(ComponentError) as error:
        trainer.train(rotated(0), params, StageContext())
    assert error.value.code == "in_process_disabled"
    seen: list[dict[str, Any]] = []
    context = StageContext(services={"in_process": True})
    try:
        output = trainer.train(rotated(0), params, context, on_epoch=seen.append)
        assert [epoch["epoch"] for epoch in seen] == list(range(1, 31))
        assert set(seen[0]) == {
            "epoch",
            "train_loss",
            "validation_loss",
            "recall@10",
            "mrr@10",
            "seconds",
        }
        assert output.format == "stratafi-query-adapter-v1"
        assert [path.name for path in output.files()] == [ADAPTER_FILE]
        weights = load_adapter(output.files()[0].read_bytes())
        assert weights.shape == (12, 12)
        assert output.examples == {"train": 24, "validation": 6}
        assert output.baseline is not None
        best = next(epoch for epoch in output.epochs if epoch["epoch"] == output.best_epoch)
        assert best["mrr@10"] > output.baseline["mrr@10"]
    finally:
        context.close()
    nothing = rotated(1)
    nothing.records = [record for record in nothing.records if record.split == "validation"]
    with pytest.raises(ComponentError) as error:
        trainer.train(nothing, params, StageContext(services={"in_process": True}))
    assert error.value.code == "no_training_examples"


@pytest.mark.parametrize("trainer", [SentenceTransformersTrainer, CrossEncoderTrainer])
def test_the_torch_trainers_refuse_before_loading_anything(
    trainer: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = trainer.resolve_params({})
    data = TrainingData(records=[], base=trainer().base_model(params))
    with pytest.raises(ComponentError) as error:
        trainer().train(data, params, StageContext())
    assert error.value.code == "in_process_disabled"
    monkeypatch.setattr(trainer, "unavailable", classmethod(lambda cls: "needs the extra"))
    with pytest.raises(ComponentError) as error:
        trainer().train(data, params, StageContext(services={"in_process": True}))
    assert error.value.code == "extra_missing"
    monkeypatch.setattr(trainer, "unavailable", classmethod(lambda cls: None))
    unknown = trainer.resolve_params({"model": "someone/unknown-model"})
    with pytest.raises(ComponentError) as error:
        trainer().train(data, unknown, StageContext(services={"in_process": True}))
    assert error.value.code == "model_not_cataloged"


def test_a_torch_trainers_base_is_its_catalog_model() -> None:
    base = SentenceTransformersTrainer().base_model(
        SentenceTransformersTrainer.resolve_params({"model": "BAAI/bge-small-en-v1.5"})
    )
    assert base is not None
    assert (base.component, base.model, base.revision, base.dims, base.license) == (
        "embed/sentence-transformers",
        "BAAI/bge-small-en-v1.5",
        "main",
        384,
        "MIT",
    )
    reranker = CrossEncoderTrainer().base_model(CrossEncoderTrainer.resolve_params({}))
    assert reranker is not None
    assert reranker.component == "rerank/cross-encoder"
    assert QueryAdapterTrainer().base_model(QueryAdapterTrainer.resolve_params({})) is None


# --- trained models in use ------------------------------------------------------------------


@dataclass
class Registered:
    id: uuid.UUID
    kind: str
    model_key: str
    base: dict[str, Any]
    name: str = "halcyon-model"
    version: int = 1


@dataclass
class FakeModels:
    """``services["trained_models"]``, in memory."""

    models: dict[str, Registered] = field(default_factory=dict)
    withdrawn: set[str] = field(default_factory=set)
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    materialized: list[Path] = field(default_factory=list)

    def ready(self, model_id: str, kind: str | None = None) -> Registered:
        model = self.models.get(model_id)
        if model is None:
            raise ComponentError("model_not_found")
        if kind is not None and model.kind != kind:
            raise ComponentError("model_kind_mismatch")
        if model_id in self.withdrawn:
            raise ComponentError("model_withdrawn")
        return model

    def adapter(self, model_id: str) -> tuple[Registered, np.ndarray]:
        return self.ready(model_id, "query-adapter"), self.weights[model_id]

    def materialize(self, model_id: str, kind: str, directory: Path) -> tuple[Registered, Any]:
        model = self.ready(model_id, kind)
        (directory / "config.json").write_text("{}")
        self.materialized.append(directory)
        return model, None


class StandInModel:
    loaded: ClassVar[list[tuple]] = []

    def __init__(self, name: str, **options: Any) -> None:
        StandInModel.loaded.append((name, Path(name).is_dir(), options.get("revision")))
        self.tokenizer = None

    def encode(self, texts, batch_size=32, convert_to_numpy=True):
        return np.ones((len(texts), 384), dtype=np.float32)


class StandInCrossEncoder:
    loaded: ClassVar[list[tuple]] = []

    def __init__(self, name: str, **options: Any) -> None:
        StandInCrossEncoder.loaded.append((name, Path(name).is_dir()))

    def predict(self, pairs, batch_size=32):
        return np.arange(len(pairs), dtype=float)


@pytest.fixture
def stand_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StandInModel  # type: ignore[attr-defined]
    module.CrossEncoder = StandInCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    for component in (SentenceTransformersEmbed, CrossEncoderRerank):
        monkeypatch.setattr(component, "unavailable", classmethod(lambda cls: None))
        monkeypatch.setattr(component, "_models", {})
    StandInModel.loaded, StandInCrossEncoder.loaded = [], []


def test_a_trained_embedder_loads_from_its_files_and_is_revisioned_by_its_artifact(
    stand_ins: None,
) -> None:
    models = FakeModels()
    models.models[MODEL_ID] = Registered(
        uuid.UUID(MODEL_ID), "embedder", "a" * 64, {"model": "BAAI/bge-small-en-v1.5"}
    )
    context = StageContext(services={"in_process": True, "trained_models": models})
    params = SentenceTransformersEmbed.resolve_params({"model": f"trained/{MODEL_ID}"})
    component = SentenceTransformersEmbed()
    assert component.revision_for(params, context) == f"trained/{MODEL_ID}@{'a' * 16}"
    chunk = Chunk(
        chunk_id=f"{1:032x}",
        document_id="d-1",
        ordinal=0,
        text="wire callbacks",
        index_text="wire callbacks",
        char_start=0,
        char_end=14,
    )
    vectors = component.embed([chunk], params, expected_revision=None, context=context)
    assert vectors.sidecar.revision == f"trained/{MODEL_ID}@{'a' * 16}"
    # Loaded from a directory of its files, which is gone once it's loaded; the base's catalog
    # entry gave the query instruction.
    [(path, was_dir, revision)] = StandInModel.loaded
    assert was_dir
    assert revision is None
    assert not Path(path).exists()
    assert vectors.sidecar.prefix == ""
    models.withdrawn.add(MODEL_ID)
    with pytest.raises(ComponentError) as error:
        component.embed([chunk], params, expected_revision=None, context=context)
    assert error.value.code == "model_withdrawn"
    with pytest.raises(ComponentError) as error:
        component.embed([chunk], params, expected_revision=None, context=StageContext(
            services={"in_process": True}
        ))  # fmt: skip
    assert error.value.code == "trained_models_not_configured"


def test_a_trained_reranker_loads_from_its_files(stand_ins: None) -> None:
    models = FakeModels()
    models.models[MODEL_ID] = Registered(
        uuid.UUID(MODEL_ID), "reranker", "b" * 64, {"model": "BAAI/bge-reranker-base"}
    )
    context = StageContext(services={"in_process": True, "trained_models": models})
    params = CrossEncoderRerank.resolve_params({"model": f"trained/{MODEL_ID}"})
    scores = CrossEncoderRerank().rerank("q", ["a", "b"], params, context)
    assert scores == [0.0, 1.0]
    assert CrossEncoderRerank().revision_for(params, context) == f"trained/{MODEL_ID}@{'b' * 16}"
    assert [was_dir for _, was_dir in StandInCrossEncoder.loaded] == [True]
    # An embedder isn't a reranker.
    models.models[MODEL_ID].kind = "embedder"
    with pytest.raises(ComponentError) as error:
        CrossEncoderRerank().rerank("q", ["a"], params, context)
    assert error.value.code == "model_kind_mismatch"


def dense_index(dims: int, revision: str) -> DenseIndex:
    manifest = IndexManifest.model_validate(
        {
            "kind": "dense",
            "component": "index/exact",
            "version": "1.0.0",
            "chunk_ids": ["c1"],
            "document_ids": ["d1"],
            "chunks": [],
            "model": "fake-hash",
            "revision": revision,
            "dims": dims,
            "metric": "cosine",
            "dtype": "float32",
        }
    )
    return DenseIndex(manifest, np.ones((1, dims), dtype=np.float32))


def test_an_adapter_moves_query_vectors_and_must_fit_the_index() -> None:
    component = FakeHashEmbed()
    params = component.resolve_params({"dimensions": 12})
    models = FakeModels()
    models.models[MODEL_ID] = Registered(
        uuid.UUID(MODEL_ID),
        "query-adapter",
        "c" * 64,
        {"model": "fake-hash", "revision": component.revision_for(params, StageContext())},
    )
    models.weights[MODEL_ID] = 2 * np.eye(12, dtype=np.float32)
    context = StageContext(services={"in_process": True, "trained_models": models})
    plain = QueryEmbedder(component, params, context)
    revision = component.revision_for(params, context)
    adapted = with_adapter(plain, MODEL_ID, dense=dense_index(12, revision))
    assert isinstance(adapted, AdaptedQueryEmbedder)
    assert adapted.key(revision) == (*plain.key(revision), "adapter", "c" * 64)
    vector, got = adapted("wire callbacks")
    original, _ = plain("wire callbacks")
    assert got == revision
    assert np.allclose(vector, 2 * original)
    for index in (dense_index(16, revision), dense_index(12, "another-revision")):
        with pytest.raises(ComponentError) as error:
            with_adapter(plain, MODEL_ID, dense=index)
        assert error.value.code == "adapter_mismatch"
    off = QueryEmbedder(component, params, StageContext(services={"trained_models": models}))
    with pytest.raises(ComponentError) as error:
        with_adapter(off, MODEL_ID, dense=dense_index(12, revision))
    assert error.value.code == "in_process_disabled"


# --- the model tier: real fine-tuning, on CPU ----------------------------------------------

needs_models = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("sentence_transformers") is None
    or os.environ.get("STRATAFI_TEST_HF_MODELS") != "1",
    reason="needs the sentence-transformers extra and STRATAFI_TEST_HF_MODELS=1 (downloads)",
)

TOPICS = [
    ("wire transfer callback", "Call the customer back on the phone on file before any wire."),
    ("new account check hold", "A new account's check deposits wait up to nine business days."),
    ("lost debit card", "Report a lost card at once; we replace it within five days."),
    ("overdraft fee amount", "Each overdraft costs thirty dollars, at most three a day."),
    ("safe deposit box access", "Only listed renters may open a safe deposit box."),
    ("dormant account rules", "An account with no activity for a year becomes dormant."),
    ("loan payment due date", "Loan payments are due on the first of each month."),
    ("notary service hours", "Notaries are available weekdays from nine to four."),
]


def topic_records() -> list[TrainingRecord]:
    records = []
    for index, (query, passage) in enumerate(TOPICS):
        negatives = [TOPICS[(index + k) % len(TOPICS)][1] for k in (1, 2, 3)]
        records.append(
            TrainingRecord(
                item_id=f"t{index}",
                split="validation" if index % 4 == 0 else "train",
                query=f"What is the rule for {query}?",
                positives=[{"chunk_id": f"p{index}", "text": passage, "reason": "evidence"}],
                negatives=[
                    {
                        "chunk_id": f"p{(index + k) % len(TOPICS)}",
                        "text": text,
                        "reason": "retrieved",
                    }
                    for k, text in zip((1, 2, 3), negatives, strict=True)
                ],
            )
        )
    return records


@needs_models
@pytest.mark.parametrize(
    ("trainer", "model"),
    [
        (SentenceTransformersTrainer, "sentence-transformers/all-MiniLM-L6-v2"),
        (CrossEncoderTrainer, "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ],
)
def test_a_catalog_model_fine_tunes_on_cpu(trainer: type, model: str) -> None:
    component = trainer()
    params = trainer.resolve_params(
        {"model": model, "epochs": 2, "batch_size": 4, "patience": 0, "device": "cpu"}
    )
    base = component.base_model(params)
    assert base is not None
    seen: list[dict[str, Any]] = []
    context = StageContext(services={"in_process": True})
    try:
        output = component.train(
            TrainingData(records=topic_records(), base=base), params, context, on_epoch=seen.append
        )
        assert [epoch["epoch"] for epoch in seen] == [1, 2]
        assert all(epoch["train_loss"] is not None for epoch in seen)
        assert output.baseline is not None
        assert output.baseline["mrr@10"] is not None
        names = {path.relative_to(output.directory).as_posix() for path in output.files()}
        assert "config.json" in names
        assert any(name.endswith(".safetensors") for name in names)
        assert "README.md" not in names
        # What was saved loads back as the component that uses it would load it.
        from sentence_transformers import CrossEncoder, SentenceTransformer

        loaded = (
            SentenceTransformer(str(output.directory), device="cpu")
            if trainer is SentenceTransformersTrainer
            else CrossEncoder(str(output.directory), device="cpu")
        )
        assert loaded is not None
    finally:
        context.close()


def test_experiments_sweep_adapters_and_trained_models() -> None:
    from stratafi_core.experiments import parse_experiment, plan_trials
    from stratafi_core.specs import resolve_pipeline, resolve_retrieval
    from stratafi_testing.lab import baseline_pipeline, hybrid_retrieval

    registry = default_registry()
    spec = parse_experiment(
        {
            "schema": "stratafi.experiment.v1",
            "name": "adapter-sweep",
            "corpus": "halcyon@1",
            "eval_set": {"name": "halcyon-retrieval", "version": 1, "split": "dev"},
            "base": {"pipeline": "baseline-v1", "retrieval": "hybrid-v1"},
            "objective": {"metric": "recall@5", "checkpoint": "final", "goal": "maximize"},
            "factors": {
                "retrievers.dense.adapter": [None, MODEL_ID],
                "rerank.params.model": [
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    f"trained/{MODEL_ID}",
                ],
            },
            "budget": {"max_trials": 10},
        }
    )
    base = hybrid_retrieval(rerank={"component": "rerank/cross-encoder"})
    plan = plan_trials(
        spec,
        resolve_pipeline(baseline_pipeline(), registry),
        resolve_retrieval(base, registry),
        registry,
    )
    assert plan.issues == []
    assert [trial.retrieval.retrievers.dense.adapter for trial in plan.trials] == [
        None,
        None,
        MODEL_ID,
        MODEL_ID,
    ]
    assert len({trial.retrieval_hash for trial in plan.trials}) == 4
    # Without an adapter the config hashes as it did before adapters existed.
    assert "adapter" not in str(plan.trials[0].retrieval.model_dump(by_alias=True))
