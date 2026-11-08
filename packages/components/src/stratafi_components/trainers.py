"""Trainers: components of the ``train`` kind, which post-train a
retrieval model on a training set (``stratafi.training-set.v1``) and leave its files in a
directory for the worker to store.

- ``train/query-adapter`` (numpy, ``cpu``): a linear map over an embedder's query vectors
  (``stratafi_core.training.train_adapter``). It reads vectors only: the worker embeds each
  item's query with the training set's embedder and hands over the dense index's passage
  vectors, which also make the corpus validation is measured on.
- ``train/sentence-transformers`` (torch, ``gpu``): fine-tunes a catalog bi-encoder. Each step
  scores a batch's queries against every positive and hard negative in the batch
  (MultipleNegativesRankingLoss: cross-entropy over scaled cosine similarities); another
  example's positive that is also this item's positive is masked, never a negative.
- ``train/cross-encoder`` (torch, ``gpu``): fine-tunes a catalog cross-encoder with binary
  cross-entropy on (query, passage) pairs: positives 1, negatives 0.

Every trainer is lab only: it refuses while ``models.in_process`` is off, the torch
ones without the ``sentence-transformers`` extra, and a base whose license isn't permissive
unless ``licenses.allowed`` names it. Torch trainers run plain loops, so a step checks for
cancellation and each epoch reports its training loss and validation metrics. Validation for a
torch trainer ranks each validation item's positives among every passage of the training set
(the pool); the adapter ranks them over the whole dense index. The best epoch by validation
MRR@10 is kept, and ``patience`` epochs without a better one stop training. Nothing logs text.
"""

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.embedders import (
    BY_MODEL,
    IN_PROCESS_SERVICE,
    PERMISSIVE,
    CatalogModel,
)
from stratafi_components.rerankers import RERANKER_MODELS
from stratafi_core.components import (
    Component,
    ComponentError,
    ComponentInfo,
    StageContext,
)
from stratafi_core.training import (
    ADAPTER_FILE,
    ADAPTER_FORMAT,
    AdapterEpoch,
    BaseModelRef,
    Corpus,
    QueryAdapterParams,
    TrainingRecord,
    adapter_bytes,
    adapter_examples,
    train_adapter,
)

EpochCallback = Callable[[dict[str, Any]], None]
CATALOG_MODEL = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
REVISION = r"^[A-Za-z0-9._-]{1,64}$"
VALIDATION_K = 10
# Files a sentence-transformers save writes that the model card replaces.
SKIPPED_FILES = frozenset({"README.md"})


@dataclass
class TrainingData:
    """What a trainer trains on: the training set's records and the base model. An adapter
    also gets each item's query vector and the passages' vectors, by chunk ID, with the dense
    index's metric and the corpus validation ranks over."""

    records: list[TrainingRecord]
    base: BaseModelRef
    queries: Mapping[str, np.ndarray] = field(default_factory=dict)
    vectors: Mapping[str, np.ndarray] = field(default_factory=dict)
    corpus: Corpus | None = None
    metric: str = "cosine"

    def split(self, name: str) -> list[TrainingRecord]:
        return [record for record in self.records if record.split == name]


@dataclass
class TrainedFiles:
    """A trainer's output: its files in ``directory``, their format, and its metrics."""

    directory: Path
    format: str
    epochs: list[dict[str, Any]]
    best_epoch: int
    baseline: dict[str, Any] | None
    stopped_early: bool = False
    examples: dict[str, int] = field(default_factory=dict)

    def files(self) -> list[Path]:
        return sorted(
            path
            for path in self.directory.rglob("*")
            if path.is_file() and path.relative_to(self.directory).as_posix() not in SKIPPED_FILES
        )


class TrainComponent(Component):
    """A trainer: a training set and a base model → a model's files."""

    model_kind: ClassVar[str]

    def base_model(self, params: BaseModel) -> BaseModelRef | None:
        """The base the parameters name; None when it comes from the training set (adapters)."""
        return None

    def train(
        self,
        data: TrainingData,
        params: BaseModel,
        context: StageContext,
        *,
        on_epoch: EpochCallback = lambda epoch: None,
    ) -> TrainedFiles:
        raise NotImplementedError

    @staticmethod
    def require_in_process(context: StageContext) -> None:
        if not context.services.get(IN_PROCESS_SERVICE, False):
            raise ComponentError("in_process_disabled")


def _epoch_dict(epoch: AdapterEpoch) -> dict[str, Any]:
    return epoch.as_dict()


# --- train/query-adapter ---------------------------------------------------------------------


class QueryAdapterTrainer(TrainComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="train/query-adapter",
        stage="train",
        version="1.0.0",
        title="Query adapter (numpy)",
        description="Learns a linear map over an embedder's query vectors from the training "
        "set's positives and hard negatives (InfoNCE, Adam, pulled toward the identity). "
        "Documents keep their vectors; any embedder works. Lab only.",
        resource_class="cpu",
        license="MIT",
        lab_only=True,
    )
    Params: ClassVar[type[BaseModel]] = QueryAdapterParams
    model_kind: ClassVar[str] = "query-adapter"

    def train(
        self,
        data: TrainingData,
        params: BaseModel,
        context: StageContext,
        *,
        on_epoch: EpochCallback = lambda epoch: None,
    ) -> TrainedFiles:
        options = cast(QueryAdapterParams, params)
        self.require_in_process(context)
        dims = data.base.dims
        if not dims:
            raise ComponentError("base_dimensions_missing")

        def examples(split: str):
            return adapter_examples(
                data.split(split),
                data.queries,
                data.vectors,
                hard_negatives=options.hard_negatives,
                metric=data.metric,
                dims=dims,
            )

        train, validation = examples("train"), examples("validation")
        if len(train) == 0:
            raise ComponentError("no_training_examples")
        result = train_adapter(
            train,
            validation if len(validation) else None,
            params=options,
            metric=data.metric,
            corpus=data.corpus,
            checkpoint=context.checkpoint,
            on_epoch=lambda epoch: on_epoch(_epoch_dict(epoch)),
        )
        directory = context.scratch_dir() / "model"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ADAPTER_FILE).write_bytes(adapter_bytes(result.weights))
        context.metric("train.examples", len(train))
        return TrainedFiles(
            directory=directory,
            format=ADAPTER_FORMAT,
            epochs=[epoch.as_dict() for epoch in result.epochs],
            best_epoch=result.best_epoch,
            baseline=result.baseline.as_dict() if len(validation) else None,
            stopped_early=result.stopped_early,
            examples={"train": len(train), "validation": len(validation)},
        )


# --- the torch trainers ----------------------------------------------------------------------


class _TorchParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: str = Field(default="main", pattern=REVISION, description="A branch, tag or commit")
    epochs: int = Field(default=3, ge=1, le=100, description="Passes over the training items")
    batch_size: int = Field(default=16, ge=2, le=1024, description="Examples per step")
    learning_rate: float = Field(default=2e-5, gt=0, le=1, description="AdamW's step size")
    warmup_ratio: float = Field(
        default=0.1, ge=0, le=0.5, description="Share of steps the learning rate warms up over"
    )
    weight_decay: float = Field(default=0.01, ge=0, le=1, description="AdamW's weight decay")
    patience: int = Field(
        default=2,
        ge=0,
        le=100,
        description="Stop after this many epochs without a better validation MRR; 0 never stops",
    )
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Seeds order and dropout")
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto", description="Where to train: auto picks a GPU when there is one"
    )


class SentenceTransformersTrainParams(_TorchParams):
    model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        pattern=CATALOG_MODEL,
        description="A model from the embedder catalog (Hugging Face ID)",
    )
    hard_negatives: int = Field(
        default=3, ge=0, le=50, description="Mined negatives per example, at most"
    )
    max_seq_length: int = Field(default=256, ge=16, le=8192, description="Tokens per text")
    scale: float = Field(
        default=20.0, gt=0, le=100, description="Similarity scale (1 / temperature)"
    )


class CrossEncoderTrainParams(_TorchParams):
    model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        pattern=CATALOG_MODEL,
        description="A model from the reranker catalog",
    )
    max_length: int = Field(default=512, ge=16, le=8192, description="Tokens per pair")
    negatives_per_positive: int = Field(
        default=4, ge=0, le=50, description="Mined negatives per item and positive, at most"
    )


def _metrics_at(ranks: Sequence[int | None], k: int = VALIDATION_K) -> tuple[float, float]:
    """Recall@k and MRR@k from each item's first relevant rank (None: not ranked)."""
    if not ranks:
        return 0.0, 0.0
    recall = sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)
    mrr = sum(1.0 / rank for rank in ranks if rank is not None and rank <= k) / len(ranks)
    return recall, mrr


class _TorchTrainer(TrainComponent):
    requires: ClassVar[tuple[str, ...]] = ("sentence_transformers", "torch")
    extra: ClassVar[str | None] = "sentence-transformers"
    catalog: ClassVar[Mapping[str, Any]]
    uses: ClassVar[str]  # the component that runs what it trains

    def base_model(self, params: BaseModel) -> BaseModelRef | None:
        options = cast(Any, params)
        entry = self.catalog.get(options.model)
        return BaseModelRef(
            component=self.uses,
            model=options.model,
            revision=options.revision,
            dims=getattr(entry, "dims", None),
            license=getattr(entry, "license", None),
        )

    def _check(self, params: BaseModel, context: StageContext) -> Any:
        self.require_in_process(context)
        if self.unavailable() is not None:
            raise ComponentError("extra_missing")
        options = cast(Any, params)
        entry = self.catalog.get(options.model)
        if entry is None:
            raise ComponentError("model_not_cataloged")
        if entry.license not in PERMISSIVE and entry.license not in context.services.get(
            "licenses", ()
        ):
            raise ComponentError("license_not_allowed")
        return entry

    @staticmethod
    def _device(params: _TorchParams) -> str:
        import torch

        if params.device == "cuda" and not torch.cuda.is_available():
            raise ComponentError("cuda_unavailable")
        if params.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return params.device

    @staticmethod
    def _seed(seed: int) -> None:
        import torch

        random.seed(seed)
        np.random.seed(seed)  # libraries that sample read the global generator
        torch.manual_seed(seed)

    @staticmethod
    def _optimizer(model: Any, params: _TorchParams, steps: int) -> tuple[Any, Any]:
        import torch

        decay, plain = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            (plain if "bias" in name or "LayerNorm" in name else decay).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": params.weight_decay},
                {"params": plain, "weight_decay": 0.0},
            ],
            lr=params.learning_rate,
        )
        warmup = max(1, int(steps * params.warmup_ratio))

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            return max(0.0, (steps - step) / max(1, steps - warmup))

        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    def _fit(
        self,
        *,
        model: Any,
        params: _TorchParams,
        examples: list[Any],
        step: Callable[[list[Any]], Any],
        validate: Callable[[], tuple[float, float] | None],
        context: StageContext,
        on_epoch: EpochCallback,
        directory: Path,
    ) -> TrainedFiles:
        """The loop both torch trainers share: seeded batches, AdamW with warmup and linear
        decay, gradient clipping, a validation after each epoch, and the best epoch saved."""
        import time

        import torch

        if not examples:
            raise ComponentError("no_training_examples")
        rng = random.Random(params.seed)  # noqa: S311
        steps = params.epochs * math.ceil(len(examples) / params.batch_size)
        optimizer, scheduler = self._optimizer(model, params, steps)
        best_dir = directory / "best"
        started = time.perf_counter()
        measured = validate()
        baseline = None
        best_score = -math.inf
        best_epoch, waited, stopped = 0, 0, False
        if measured is not None:
            baseline = {
                "epoch": 0,
                "train_loss": None,
                "validation_loss": None,
                "recall@10": round(measured[0], 6),
                "mrr@10": round(measured[1], 6),
                "seconds": round(time.perf_counter() - started, 3),
            }
            best_score = measured[1]
            model.save(str(best_dir))
        epochs: list[dict[str, Any]] = []
        for epoch in range(1, params.epochs + 1):
            started = time.perf_counter()
            model.train()
            order = rng.sample(examples, len(examples))
            total = 0.0
            for start in range(0, len(order), params.batch_size):
                context.checkpoint()
                batch = order[start : start + params.batch_size]
                loss = step(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total += float(loss.detach().cpu()) * len(batch)
            model.eval()
            measured = validate()
            result = {
                "epoch": epoch,
                "train_loss": round(total / len(order), 6),
                "validation_loss": None,
                "recall@10": round(measured[0], 6) if measured else None,
                "mrr@10": round(measured[1], 6) if measured else None,
                "seconds": round(time.perf_counter() - started, 3),
            }
            epochs.append(result)
            on_epoch(result)
            if measured is None or measured[1] > best_score:
                best_score = measured[1] if measured else best_score
                best_epoch, waited = epoch, 0
                model.save(str(best_dir))
            else:
                waited += 1
                if params.patience and waited >= params.patience:
                    stopped = True
                    break
        return TrainedFiles(
            directory=best_dir,
            format="sentence-transformers",
            epochs=epochs,
            best_epoch=best_epoch,
            baseline=baseline,
            stopped_early=stopped,
        )


def _features(model: Any, texts: Sequence[Any], device: str) -> Any:
    from sentence_transformers import util

    prepare = getattr(model, "preprocess", None) or model.tokenize
    return util.batch_to_device(prepare(list(texts)), device)


class SentenceTransformersTrainer(_TorchTrainer):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="train/sentence-transformers",
        stage="train",
        version="1.0.0",
        title="Embedder fine-tuning (sentence-transformers)",
        description="Fine-tunes a catalog embedder on the training set: each query against "
        "every positive and hard negative in its batch (MultipleNegativesRankingLoss). Lab "
        "only; needs the sentence-transformers extra; a GPU helps.",
        resource_class="gpu",
        license="Apache-2.0 (models: per catalog entry)",
        lab_only=True,
    )
    Params: ClassVar[type[BaseModel]] = SentenceTransformersTrainParams
    model_kind: ClassVar[str] = "embedder"
    catalog: ClassVar[Mapping[str, Any]] = BY_MODEL
    uses: ClassVar[str] = "embed/sentence-transformers"

    def train(
        self,
        data: TrainingData,
        params: BaseModel,
        context: StageContext,
        *,
        on_epoch: EpochCallback = lambda epoch: None,
    ) -> TrainedFiles:
        options = cast(SentenceTransformersTrainParams, params)
        entry = cast(CatalogModel, self._check(options, context))
        import torch
        from sentence_transformers import SentenceTransformer

        device = self._device(options)
        self._seed(options.seed)
        try:
            model = SentenceTransformer(
                options.model,
                revision=options.revision,
                device=device,
                trust_remote_code=entry.trust_remote_code,
            )
        except Exception:
            raise ComponentError("model_unavailable") from None
        model.max_seq_length = min(options.max_seq_length, entry.max_tokens)
        query_prefix, document_prefix = entry.query_prefix, entry.document_prefix
        examples = [
            (
                record.item_id,
                query_prefix + record.query,
                positive.chunk_id,
                document_prefix + positive.text,
                frozenset(p.chunk_id for p in record.positives),
                [document_prefix + n.text for n in record.negatives[: options.hard_negatives]],
            )
            for record in data.split("train")
            for positive in record.positives
        ]
        held = [record for record in data.split("validation") if record.positives]
        pool: dict[str, str] = {}
        for record in data.records:
            for passage in [*record.positives, *record.negatives]:
                pool.setdefault(passage.chunk_id, document_prefix + passage.text)
        pool_ids = sorted(pool)
        rows = {chunk_id: row for row, chunk_id in enumerate(pool_ids)}

        def embed(texts: Sequence[str]) -> Any:
            return model.encode(
                list(texts),
                batch_size=options.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        def validate() -> tuple[float, float] | None:
            if not held:
                return None
            with torch.no_grad():
                passages = embed([pool[chunk_id] for chunk_id in pool_ids])
                queries = embed([query_prefix + record.query for record in held])
                scores = (queries @ passages.T).cpu().numpy()
            ranks = []
            for row, record in zip(scores, held, strict=True):
                wanted = [rows[p.chunk_id] for p in record.positives]
                best = max(float(row[index]) for index in wanted)
                ranks.append(1 + int(np.sum(row > best)))
            return _metrics_at(ranks)

        def step(batch: list[Any]) -> Any:
            anchors = model(_features(model, [example[1] for example in batch], device))
            documents = [example[3] for example in batch]
            owners = [example[2] for example in batch]
            for example in batch:
                documents += example[5]
                owners += [""] * len(example[5])
            encoded = model(_features(model, documents, device))
            first = torch.nn.functional.normalize(anchors["sentence_embedding"], dim=-1)
            second = torch.nn.functional.normalize(encoded["sentence_embedding"], dim=-1)
            scores = first @ second.T * options.scale
            # Another example's positive is masked when it's one of this item's positives.
            mask = torch.zeros_like(scores, dtype=torch.bool)
            for row, example in enumerate(batch):
                for column, owner in enumerate(owners):
                    if column != row and owner and owner in example[4]:
                        mask[row, column] = True
            scores = scores.masked_fill(mask, -1e4)
            labels = torch.arange(len(batch), device=scores.device)
            return torch.nn.functional.cross_entropy(scores, labels)

        output = self._fit(
            model=model,
            params=options,
            examples=examples,
            step=step,
            validate=validate,
            context=context,
            on_epoch=on_epoch,
            directory=context.scratch_dir(),
        )
        output.examples = {"train": len(examples), "validation": len(held)}
        context.metric("train.examples", len(examples))
        return output


class CrossEncoderTrainer(_TorchTrainer):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="train/cross-encoder",
        stage="train",
        version="1.0.0",
        title="Reranker fine-tuning (cross-encoder)",
        description="Fine-tunes a catalog cross-encoder on the training set's (query, passage) "
        "pairs: positives 1, mined negatives 0, binary cross-entropy. Lab only; needs the "
        "sentence-transformers extra; a GPU helps.",
        resource_class="gpu",
        license="Apache-2.0 (models: per catalog entry)",
        lab_only=True,
    )
    Params: ClassVar[type[BaseModel]] = CrossEncoderTrainParams
    model_kind: ClassVar[str] = "reranker"
    catalog: ClassVar[Mapping[str, Any]] = RERANKER_MODELS
    uses: ClassVar[str] = "rerank/cross-encoder"

    def train(
        self,
        data: TrainingData,
        params: BaseModel,
        context: StageContext,
        *,
        on_epoch: EpochCallback = lambda epoch: None,
    ) -> TrainedFiles:
        options = cast(CrossEncoderTrainParams, params)
        entry = self._check(options, context)
        import torch
        from sentence_transformers import CrossEncoder

        device = self._device(options)
        self._seed(options.seed)
        try:
            model = CrossEncoder(
                options.model,
                revision=options.revision,
                max_length=min(options.max_length, entry.max_tokens),
                device=device,
            )
        except Exception:
            raise ComponentError("model_unavailable") from None
        examples: list[tuple[str, str, float]] = []
        for record in data.split("train"):
            negatives = record.negatives[: options.negatives_per_positive * len(record.positives)]
            examples += [(record.query, p.text, 1.0) for p in record.positives]
            examples += [(record.query, n.text, 0.0) for n in negatives]
        held = [r for r in data.split("validation") if r.positives and r.negatives]

        def logits(pairs: Sequence[tuple[str, str]]) -> Any:
            if hasattr(model, "preprocess"):
                features = _features(model, [list(pair) for pair in pairs], device)
                return model(features)["scores"].reshape(-1)
            encoded = model.tokenizer(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                padding=True,
                truncation=True,
                max_length=model.max_length,
                return_tensors="pt",
            ).to(device)
            return model.model(**encoded).logits.reshape(-1)

        def validate() -> tuple[float, float] | None:
            if not held:
                return None
            ranks = []
            with torch.no_grad():
                for record in held:
                    candidates = [*record.positives, *record.negatives]
                    scored = (
                        logits([(record.query, passage.text) for passage in candidates])
                        .float()
                        .cpu()
                        .numpy()
                    )
                    best = max(float(scored[i]) for i in range(len(record.positives)))
                    ranks.append(1 + int(np.sum(scored > best)))
            return _metrics_at(ranks)

        loss_fn = torch.nn.BCEWithLogitsLoss()

        def step(batch: list[tuple[str, str, float]]) -> Any:
            predicted = logits([(query, passage) for query, passage, _ in batch])
            labels = torch.tensor([label for _, _, label in batch], device=predicted.device)
            return loss_fn(predicted.float(), labels.float())

        output = self._fit(
            model=model,
            params=options,
            examples=examples,
            step=step,
            validate=validate,
            context=context,
            on_epoch=on_epoch,
            directory=context.scratch_dir(),
        )
        output.examples = {"train": len(examples), "validation": len(held)}
        context.metric("train.examples", len(examples))
        return output


COMPONENTS: list[type[Component]] = [
    QueryAdapterTrainer,
    SentenceTransformersTrainer,
    CrossEncoderTrainer,
]
