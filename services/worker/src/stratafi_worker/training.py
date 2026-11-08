"""The worker's post-training jobs.

``training_set.build`` mines a training set from a succeeded dev eval run
(``stratafi_core.training.mine``): the eval run's items, their ranked lists, every chunk of the
run's indexes as a passage, and the eval set's test queries for the leak guard. The records go
into one artifact (``training_set``, JSON Lines) whose key hashes the miner, the options and
those inputs, recorded as derived from the eval run's results, ranked lists and indexes, so a
purge of any document under them purges it. An identical set reuses the artifact.

``training_run.start`` trains a model with a ``train/`` component. An adapter's base is the
training set's embedder: each item's query is embedded with it, as eval runs embed queries, and
the passages' vectors come from the dense index, which is also the corpus validation ranks
over. The model's key hashes the trainer, its parameters, the training set and the base, so an
identical run reuses the stored model. Each epoch's metrics become the run's progress; a run
asked to cancel stops at its next checkpoint. The model's files go into an artifact (parts plus
a manifest) derived from the training set, the registry gets a version of the run's name with
its card, and the run gets its record (``stratafi.training-run.v1``).

Queries, passages and answers never reach logs, progress, records or error codes.
"""

import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
from sqlalchemy.engine import Engine

from stratafi_components.embed import EmbedComponent
from stratafi_components.index import DenseIndex
from stratafi_components.retrieval import EntryMeta, QueryEmbedder, section_of
from stratafi_components.trainers import TrainComponent, TrainedFiles, TrainingData
from stratafi_core.components import Cancelled, ComponentError, ComponentRegistry, StageContext
from stratafi_core.evalruns import EvalOptions, RankedItem
from stratafi_core.hashing import content_hash
from stratafi_core.relevance import Passage
from stratafi_core.settings import Settings
from stratafi_core.training import (
    BaseModelRef,
    Corpus,
    ModelCard,
    TrainingRunManifest,
    TrainingSetOptions,
    mine,
    model_key,
    model_limits,
    model_use,
    read_records,
    records_bytes,
    training_set_key,
)
from stratafi_store import corpora, evalruns, evalsets
from stratafi_store import training as training_store
from stratafi_store.artifacts import ArtifactNotFound, ArtifactStore, artifact_key
from stratafi_store.evalruns import artifact_exists
from stratafi_store.lineage import ArtifactRecord, get_artifact, record_artifact
from stratafi_worker.environment import environment, hardware
from stratafi_worker.evals import IndexCache, _items_for
from stratafi_worker.runs import ServicesFor
from stratafi_worker.worker import HandlerRegistry, JobCancelled, JobContext


class _Base:
    def __init__(
        self,
        engine: Engine,
        store: ArtifactStore,
        registry: ComponentRegistry,
        services: ServicesFor,
        indexes: IndexCache,
    ) -> None:
        self.engine = engine
        self.store = store
        self.registry = registry
        self.services = services
        self.indexes = indexes

    def _read(self, workspace_id: uuid.UUID, key: str) -> bytes:
        try:
            return self.store.get(artifact_key(workspace_id, key))
        except ArtifactNotFound:
            raise ComponentError("artifact_missing") from None


def _dense_info(index: Any) -> dict[str, Any]:
    manifest = index.manifest
    return {
        "model": manifest.model,
        "revision": manifest.revision,
        "dims": manifest.dims,
        "metric": manifest.metric,
        "exact": isinstance(index, DenseIndex),
    }


class TrainingSetBuilder(_Base):
    """The ``training_set.build`` job."""

    def __call__(self, job: JobContext) -> Mapping[str, Any]:
        set_id = uuid.UUID(str(job.payload["training_set_id"]))
        with self.engine.begin() as conn:
            training_set = training_store.begin_training_set(conn, set_id)
        if training_set is None:
            return {"skipped": True}
        try:
            return self._build(training_set, job)
        except ComponentError as error:
            if error.retryable and job.job.attempts < job.job.max_attempts:
                raise
            with self.engine.begin() as conn:
                training_store.finish_training_set(
                    conn, training_set, status="failed", error_code=error.code
                )
            return {"failed": error.code}

    def _build(
        self, training_set: training_store.TrainingSet, job: JobContext
    ) -> Mapping[str, Any]:
        options = TrainingSetOptions.model_validate(training_set.options)
        workspace = training_set.workspace_id
        with self.engine.connect() as conn:
            evaluation = evalruns.get_eval_run(conn, workspace, training_set.eval_run_id)
            if evaluation is None or evaluation.status != "succeeded":
                raise ComponentError("eval_run_not_succeeded")
            if evaluation.split != "dev":
                raise ComponentError("not_dev_split")
            version = evalsets.get_version(
                conn, workspace, version_id=evaluation.eval_set_version_id
            )
            if version is None:
                raise ComponentError("eval_set_missing")
            eval_set = evalsets.stored_eval_set(conn, version)
            shard_list = evalruns.shards_of(conn, evaluation.id)
            corpus_version = uuid.UUID(evaluation.context["pipeline_run"]["corpus_version_id"])
            metas = [
                EntryMeta.from_entry(entry)
                for entry in corpora.version_entries(conn, workspace, corpus_version)
            ]
        eval_options = EvalOptions.model_validate(evaluation.options)
        items = _items_for(eval_set, evaluation.split, eval_options)
        tests = [item for item in eval_set.items if item.split == "test"]
        indexes: dict[str, str] = dict(evaluation.context["indexes"])
        shard_keys = [shard.cache_key for shard in shard_list]
        key = training_set_key(
            options=options,
            results_key=evaluation.results_key or "",
            shard_keys=shard_keys,
            indexes=indexes,
            items_sha256=content_hash([item.model_dump(mode="json") for item in items]),
            test_sha256=content_hash(sorted(item.query for item in tests)),
        )
        loaded = {kind: self.indexes.get(workspace, ikey, kind) for kind, ikey in indexes.items()}
        context: dict[str, Any] = {
            "eval_run_id": str(evaluation.id),
            "eval_set": {
                "id": str(version.eval_set_id),
                "name": eval_set.name,
                "version": version.version,
                "version_id": str(version.id),
                "split": evaluation.split,
            },
            "pipeline_run": evaluation.context["pipeline_run"],
            "indexes": indexes,
            "day": evaluation.context.get("day"),
            "models": evaluation.context.get("models", {}),
        }
        if "dense" in loaded:
            context["dense"] = _dense_info(loaded["dense"].index)
        with self.engine.connect() as conn:
            cached = artifact_exists(conn, workspace, key)
            found = get_artifact(conn, workspace, key) if cached else None
        if found is not None:
            stats = dict(found.summary)
        else:
            ranked: dict[str, RankedItem] = {}
            for shard in shard_list:
                job.checkpoint()
                for line in self._read(workspace, shard.cache_key).splitlines():
                    if line.strip():
                        item = RankedItem.model_validate_json(line)
                        ranked[item.item_id] = item
            by_document: dict[str, EntryMeta] = {}
            for meta in sorted(metas, key=lambda entry: entry.name):
                by_document.setdefault(meta.document_id, meta)
            passages: dict[str, Passage] = {}
            for index in loaded.values():
                for chunk_id, chunk in index.chunks.items():
                    meta = by_document.get(chunk.document_id)
                    passages.setdefault(
                        chunk_id,
                        Passage(
                            chunk_id=chunk_id,
                            doc_key=meta.doc_key if meta else None,
                            version=meta.version if meta else None,
                            section=section_of(chunk),
                            text=chunk.text,
                        ),
                    )
            job.checkpoint()
            mined = mine(
                items,
                ranked,
                passages,
                options=options,
                threshold=eval_options.coverage_threshold,
                test_items=tests,
            )
            stats = mined.stats
            payload = records_bytes(mined.records)
            info = self.store.put(
                artifact_key(workspace, key), payload, media_type="application/x-ndjson"
            )
            with self.engine.begin() as conn:
                record_artifact(
                    conn,
                    ArtifactRecord(
                        workspace_id=workspace,
                        cache_key=key,
                        kind="training_set",
                        storage_uri=self.store.uri(artifact_key(workspace, key)),
                        size_bytes=info.size,
                        content_sha256=info.sha256,
                        media_type="application/x-ndjson",
                        summary=stats,
                    ),
                    [k for k in [evaluation.results_key, *shard_keys, *indexes.values()] if k],
                )
        with self.engine.begin() as conn:
            training_store.finish_training_set(
                conn,
                training_set,
                status="succeeded",
                cache_key=key,
                cached=cached,
                context=context,
                stats=stats,
            )
        return {"records": int(stats.get("records", 0)), "cached": cached}


class TrainingRunner(_Base):
    """The ``training_run.start`` job."""

    def __init__(self, *args: Any, settings: Settings, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.settings = settings

    def __call__(self, job: JobContext) -> Mapping[str, Any]:
        run_id = uuid.UUID(str(job.payload["training_run_id"]))
        with self.engine.begin() as conn:
            run = training_store.begin_training_run(conn, run_id)
        if run is None:
            return {"skipped": True}
        context = StageContext(
            str(run.workspace_id),
            services=self.services(run.workspace_id),
            cancelled=job.stop,
            rate_limit=lambda name, tokens: job.provider(name, tokens=tokens),
        )
        try:
            return self._train(run, job, context)
        except (Cancelled, JobCancelled):
            # Asked to cancel: the run ends. A worker that shuts down or loses its lease hands
            # the job back instead, and the run trains again from the start.
            if job.cancel_requested or self._cancelling(run):
                with self.engine.begin() as conn:
                    training_store.finish_training_run(conn, run, status="cancelled")
            raise JobCancelled(str(job.job.id)) from None
        except ComponentError as error:
            if error.retryable and job.job.attempts < job.job.max_attempts:
                raise
            with self.engine.begin() as conn:
                training_store.finish_training_run(
                    conn, run, status="failed", error_code=error.code
                )
            return {"failed": error.code}
        finally:
            context.close()

    def _cancelling(self, run: training_store.TrainingRun) -> bool:
        with self.engine.connect() as conn:
            current = training_store.get_training_run(conn, run.workspace_id, run.id)
        return current is not None and current.status in ("cancelling", "cancelled")

    def _trainer(self, run: training_store.TrainingRun) -> tuple[TrainComponent, Any]:
        component = self.registry.get(run.trainer)()
        if not isinstance(component, TrainComponent):
            raise ComponentError("not_a_trainer")
        return component, component.resolve_params(run.params)

    def _train(
        self, run: training_store.TrainingRun, job: JobContext, context: StageContext
    ) -> Mapping[str, Any]:
        started = time.monotonic()
        trainer, params = self._trainer(run)
        workspace = run.workspace_id
        with self.engine.connect() as conn:
            training_set = training_store.get_training_set(conn, workspace, run.training_set_id)
        if training_set is None or training_set.status != "succeeded" or not training_set.cache_key:
            raise ComponentError("training_set_not_ready")
        dense = None
        if trainer.model_kind == "query-adapter":
            dense_key = training_set.context.get("indexes", {}).get("dense")
            embed = training_set.context.get("models", {}).get("embed")
            if dense_key is None or embed is None:
                raise ComponentError("dense_index_missing")
            dense = self.indexes.get(workspace, dense_key, "dense")
            if not isinstance(dense.index, DenseIndex):
                raise ComponentError("dense_index_unsupported")
            info = _dense_info(dense.index)
            base = BaseModelRef(
                component=str(embed["component"]),
                model=str(info["model"]),
                revision=info["revision"],
                dims=info["dims"],
                metric=info["metric"],
            )
        else:
            found = trainer.base_model(params)
            if found is None:
                raise ComponentError("base_missing")
            base = found
        resolved = trainer.dump_params(params)
        key = model_key(
            trainer=trainer.info.id,
            trainer_version=trainer.info.version,
            params=resolved,
            training_set_key=training_set.cache_key,
            base=base,
        )
        with self.engine.begin() as conn:
            training_store.record_progress(
                conn, run.id, {"epoch": 0, "epochs": _epochs(resolved)}, base=_dump(base)
            )
            cached = artifact_exists(conn, workspace, key)
        if cached:
            metrics, files_count, size = self._stored(workspace, key)
        else:
            records = read_records(self._read(workspace, training_set.cache_key))
            data = TrainingData(records=records, base=base)
            if dense is not None:
                self._vectors(data, dense, training_set, context, job)
            job.checkpoint()
            epochs = _epochs(resolved)

            def on_epoch(result: dict[str, Any]) -> None:
                job.progress(done=int(result["epoch"]), total=epochs, force=True)
                with self.engine.begin() as conn:
                    status = training_store.record_progress(
                        conn, run.id, {**result, "epochs": epochs}
                    )
                if status == "cancelling":
                    job.stop.set()

            output = trainer.train(data, params, context, on_epoch=on_epoch)
            metrics = _metrics(output)
            paths = output.files()
            manifest, size, info = training_store.put_model(
                self.store,
                workspace,
                key,
                output.directory,
                paths,
                kind=trainer.model_kind,
                format=output.format,
                base=base,
            )
            files_count = len(manifest.files)
            with self.engine.begin() as conn:
                record_artifact(
                    conn,
                    ArtifactRecord(
                        workspace_id=workspace,
                        cache_key=key,
                        kind="model",
                        storage_uri=self.store.uri(artifact_key(workspace, key)),
                        size_bytes=size,
                        content_sha256=info.sha256,
                        media_type="application/json",
                        # Numbers only: what a run that reuses the model reports.
                        summary={
                            "kind": trainer.model_kind,
                            "files": files_count,
                            "metrics": metrics,
                        },
                    ),
                    [training_set.cache_key],
                )
        finished = datetime.now(UTC)
        manifest_record = self._record(
            run,
            training_set,
            trainer,
            resolved,
            base,
            metrics,
            key=key,
            cached=cached,
            size=size,
            files_count=files_count,
            started=started,
            finished=finished,
        )
        card_base = base
        if base.license is None and trainer.model_kind == "query-adapter":
            card_base = base.model_copy(update={"license": "per embedder"})
        with self.engine.begin() as conn:
            current = training_store.get_training_run(conn, workspace, run.id)
            if current is not None and current.status == "cancelling":
                raise Cancelled()
            model = training_store.register_model(
                conn,
                run=run,
                model_key=key,
                base=_dump(base),
                card=lambda model_id, version, now: _card(
                    model_id,
                    version,
                    now,
                    run=run,
                    training_set=training_set,
                    trainer=trainer,
                    params=resolved,
                    base=card_base,
                    metrics=metrics,
                    key=key,
                    size=size,
                    files_count=files_count,
                ),
                created_by=run.created_by,
            )
            training_store.finish_training_run(
                conn,
                run,
                status="succeeded",
                base=_dump(base),
                metrics=metrics,
                manifest=manifest_record.model_dump(mode="json", by_alias=True),
                manifest_hash=manifest_record.fingerprint(),
                model_key=key,
                model_cached=cached,
                model_id=model.id,
            )
        return {"model": str(model.id), "cached": cached}

    def _vectors(
        self,
        data: TrainingData,
        dense: Any,
        training_set: training_store.TrainingSet,
        context: StageContext,
        job: JobContext,
    ) -> None:
        """An adapter's vectors: each item's query through the training set's embedder, and
        every passage's from the dense index, which is also validation's corpus."""
        index = cast(DenseIndex, dense.index)
        embed = training_set.context["models"]["embed"]
        component = cast(EmbedComponent, self.registry.get(embed["component"])())
        embedder = QueryEmbedder(component, component.resolve_params(embed["params"]), context)
        revision = index.manifest.revision
        queries: dict[str, np.ndarray] = {}
        # Only items with positives make examples; abstentions train rerankers, not adapters.
        wanted = [record for record in data.records if record.positives]
        for done, record in enumerate(wanted, start=1):
            job.checkpoint()
            vector, got = embedder(record.query)
            if got != revision:
                raise ComponentError("revision_mismatch")
            queries[record.item_id] = np.asarray(vector, dtype=np.float32).reshape(-1)
            job.progress(done=done, total=len(wanted), phase="embedding")
        rows = {chunk_id: row for row, chunk_id in enumerate(index.manifest.chunk_ids)}
        data.queries = queries
        data.vectors = {chunk_id: index.matrix[row] for chunk_id, row in rows.items()}
        data.metric = index.manifest.metric
        data.corpus = Corpus.of(index.matrix, index.manifest.chunk_ids, metric=data.metric)

    def _stored(self, workspace_id: uuid.UUID, key: str) -> tuple[dict[str, Any], int, int]:
        """A stored model's training metrics, files and size, from its artifact."""
        with self.engine.connect() as conn:
            found = get_artifact(conn, workspace_id, key)
        if found is None:
            raise ComponentError("artifact_missing")
        summary = dict(found.summary)
        metrics = dict(summary.get("metrics") or {"epochs": [], "best_epoch": None})
        return metrics, int(summary.get("files", 0)), found.size_bytes

    def _record(
        self,
        run: training_store.TrainingRun,
        training_set: training_store.TrainingSet,
        trainer: TrainComponent,
        params: Mapping[str, Any],
        base: BaseModelRef,
        metrics: Mapping[str, Any],
        *,
        key: str,
        cached: bool,
        size: int,
        files_count: int,
        started: float,
        finished: datetime,
    ) -> TrainingRunManifest:
        return TrainingRunManifest.model_validate(
            {
                "training_run": {
                    "id": str(run.id),
                    "created_by": run.created_by,
                    "created_at": run.created_at,
                },
                "configuration": {
                    "trainer": {
                        "component": trainer.info.id,
                        "version": trainer.info.version,
                        "params": dict(params),
                    },
                    "params_hash": content_hash(dict(params)),
                    "kind": trainer.model_kind,
                },
                "inputs": {
                    "training_set": _set_summary(training_set),
                    "base": base,
                },
                "environment": environment(),
                "hardware": hardware([trainer.info.resource_class]),
                "execution": {
                    "status": "succeeded",
                    "started_at": run.started_at,
                    "finished_at": finished,
                    "duration_ms": round((time.monotonic() - started) * 1000, 3),
                    "model_cached": cached,
                },
                "metrics": metrics,
                "outputs": [
                    {"cache_key": key, "kind": "model", "size_bytes": size, "files": files_count}
                ],
            }
        )


def _dump(base: BaseModelRef) -> dict[str, Any]:
    return base.model_dump(mode="json", exclude_none=True)


def _epochs(params: Mapping[str, Any]) -> int:
    return int(params.get("epochs") or 0)


def _metrics(output: TrainedFiles) -> dict[str, Any]:
    return {
        "baseline": output.baseline,
        "epochs": output.epochs,
        "best_epoch": output.best_epoch,
        "stopped_early": output.stopped_early,
        "examples": dict(output.examples),
    }


def _set_summary(training_set: training_store.TrainingSet) -> dict[str, Any]:
    stats = training_set.stats or {}
    return {
        "id": str(training_set.id),
        "name": training_set.name,
        "cache_key": training_set.cache_key,
        "eval_run_id": str(training_set.eval_run_id),
        "eval_set": training_set.context.get("eval_set", {}),
        "counts": {
            name: stats[name]
            for name in ("records", "pairs", "abstentions", "validation", "positives", "negatives")
            if name in stats
        },
        "leak_guard": stats.get("leak_guard", {}),
    }


def _best(metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    best = metrics.get("best_epoch")
    if not best:
        return metrics.get("baseline")
    return next((epoch for epoch in metrics.get("epochs", []) if epoch["epoch"] == best), None)


def _card(
    model_id: uuid.UUID,
    version: int,
    now: datetime,
    *,
    run: training_store.TrainingRun,
    training_set: training_store.TrainingSet,
    trainer: TrainComponent,
    params: Mapping[str, Any],
    base: BaseModelRef,
    metrics: Mapping[str, Any],
    key: str,
    size: int,
    files_count: int,
) -> dict[str, Any]:
    card = ModelCard.model_validate(
        {
            "model": {
                "id": str(model_id),
                "name": run.name,
                "version": version,
                "kind": trainer.model_kind,
                "created_by": run.created_by,
                "created_at": now,
            },
            "base": base,
            "trainer": {
                "component": trainer.info.id,
                "version": trainer.info.version,
                "params": dict(params),
            },
            "training_set": _set_summary(training_set),
            "training": {
                "run_id": str(run.id),
                "best_epoch": metrics.get("best_epoch"),
                "epochs_run": len(metrics.get("epochs", [])),
                "stopped_early": metrics.get("stopped_early", False),
                "baseline": metrics.get("baseline"),
                "best": _best(metrics),
            },
            "use": model_use(trainer.model_kind, str(model_id)),
            "limits": model_limits(trainer.model_kind, training_set.stats or {}),
            "artifact": {"cache_key": key, "size_bytes": size, "files": files_count},
        }
    )
    return card.model_dump(mode="json", by_alias=True)


def register_training_handlers(
    handlers: HandlerRegistry,
    *,
    engine: Engine,
    store: ArtifactStore,
    registry: ComponentRegistry,
    services: ServicesFor,
    settings: Settings,
    indexes: IndexCache | None = None,
) -> None:
    """Register ``training_set.build`` and ``training_run.start``."""
    shared = indexes or IndexCache(store, engine=engine)
    handlers.register(
        training_store.BUILD_JOB, TrainingSetBuilder(engine, store, registry, services, shared)
    )
    handlers.register(
        training_store.TRAIN_JOB,
        TrainingRunner(engine, store, registry, services, shared, settings=settings),
    )
