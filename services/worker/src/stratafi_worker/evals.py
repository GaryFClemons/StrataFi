"""The worker's eval jobs: ``eval.start`` plans an eval run, ``eval.shard`` searches
a batch of its queries, ``eval.finish`` scores and records it.

``eval.start`` resolves the eval run's inputs: the pipeline run's indexes and corpus entries,
the retrieval config, the eval set version's split, and the embedder's and reranker's revisions
(the reranker's from the model gateway). It plans the shards and the results with their cache
keys (``stratafi_core.evalruns``) and hands them to ``EvalOrchestrator.start``: shards whose
ranked lists exist are cached, the rest get ``eval.shard`` jobs.

``eval.shard`` runs the retrieval engine (``stratafi_components.retrieval``) for each query of
its shard, as the query's asker on the eval set's as-of date, and stores every checkpoint's
ranked list (``ranked_lists``, JSONL). Indexes, query vectors and rerank scores are cached for
the worker process, so a shard rarely reloads what the one before it loaded.

``eval.finish`` maps evidence onto the ranked lists, scores every query, checks findability in
the gold documents' cleaned text, aggregates with bootstrap intervals, slices by tag and file
type, and records the results (``eval_results``), the per-query rows and the run's record
(``stratafi.eval-run.v1``). If the results artifact exists already, an identical eval run reuses
it.

Queries and texts never reach a log, an event, a summary or the record: only IDs and numbers.
"""

import dataclasses
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from stratafi_components.embed import EmbedComponent
from stratafi_components.index import DenseIndex, IndexComponent
from stratafi_components.rerank import RerankComponent
from stratafi_components.retrieval import (
    TRAINED_MODELS_SERVICE,
    AdaptedQueryEmbedder,
    EntryMeta,
    LoadedIndex,
    QueryEmbedder,
    Reranker,
    RetrievalCache,
    RetrievalEngine,
    Visibility,
    load_index,
    section_of,
    visible_until,
    with_adapter,
)
from stratafi_core.artifacts import Chunk, ParsedDocument
from stratafi_core.components import Cancelled, ComponentError, ComponentRegistry, StageContext
from stratafi_core.evalruns import (
    ENGINE_VERSION,
    SCORER_VERSION,
    EvalOptions,
    EvalRunManifest,
    RankedItem,
    results_key,
    sample_items,
    shard_key,
    shards,
)
from stratafi_core.evalsets import EvalItem, EvalSet, asker_of
from stratafi_core.hashing import content_hash
from stratafi_core.metrics import QueryScore, default_ks, metric_name, score_query, summarize
from stratafi_core.relevance import KNOWLEDGE_HIT, Passage, document_findability
from stratafi_core.settings import Settings
from stratafi_core.specs import RetrievalConfig, dump_document, resolve_retrieval
from stratafi_store import corpora, evalsets, runs, specs
from stratafi_store.artifacts import (
    ArtifactNotFound,
    ArtifactStore,
    artifact_key,
    artifact_part_key,
)
from stratafi_store.evalruns import (
    FINISH_JOB,
    SHARD_JOB,
    START_JOB,
    EvalOrchestrator,
    EvalRun,
    EvalShard,
    ShardPlan,
    artifact_exists,
    get_eval_run,
    shards_of,
)
from stratafi_store.hooks import finish_hooks
from stratafi_store.jobs import JobQueue
from stratafi_store.lineage import ArtifactRecord
from stratafi_store.tables import artifacts
from stratafi_worker.environment import environment, hardware
from stratafi_worker.runs import ServicesFor
from stratafi_worker.worker import HandlerRegistry, JobCancelled, JobContext

logger = logging.getLogger("stratafi.worker.evals")

CACHED_INDEXES = 8


class IndexCache:
    """The worker process's last few loaded indexes, by workspace and cache key."""

    def __init__(
        self, store: ArtifactStore, size: int = CACHED_INDEXES, engine: Any = None
    ) -> None:
        self.store = store
        self.engine = engine  # for indexes that live in Postgres
        self.size = size
        self._items: OrderedDict[tuple[uuid.UUID, str], LoadedIndex] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, workspace_id: uuid.UUID, cache_key: str, kind: str) -> LoadedIndex:
        key = (workspace_id, cache_key)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]

        def read(artifact: str, part: str | None) -> bytes:
            if part is None:
                return self.store.get(artifact_key(workspace_id, artifact))
            return self.store.get(artifact_part_key(workspace_id, artifact, part))

        try:
            loaded = load_index(read, cache_key, kind, engine=self.engine)
        except ArtifactNotFound:
            raise ComponentError("artifact_missing") from None
        with self._lock:
            self._items[key] = loaded
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return loaded


@dataclass(frozen=True)
class Inputs:
    """What an eval run reads, as ``eval.start`` resolved it."""

    run: EvalRun
    config: RetrievalConfig
    eval_set: EvalSet
    items: list[EvalItem]
    entries: list[EntryMeta]
    day: date | None


def _items_for(eval_set: EvalSet, split: str, options: EvalOptions) -> list[EvalItem]:
    """The split's items, or the options' seeded sample of them, in the order they're scored."""
    items = [item for item in eval_set.items if split == "all" or item.split == split]
    return sample_items(items, options.sample)


def _metas(conn: sa.Connection, workspace_id: uuid.UUID, corpus_version_id: uuid.UUID):
    return [
        EntryMeta.from_entry(entry)
        for entry in corpora.version_entries(conn, workspace_id, corpus_version_id)
    ]


def _cleaned_keys(conn: sa.Connection, pipeline: runs.Run, metas: Sequence[EntryMeta]):
    """Each entry's cleaned artifact and parser, by the entry's (doc_key, version)."""
    by_name = {meta.name: meta for meta in metas}
    names = {entry.ordinal: entry.name for entry in runs.entries_of(conn, pipeline.id)}
    found: dict[str, dict[str, str]] = {}
    for task in runs.tasks_of(conn, pipeline.id):
        if task.stage not in ("clean", "parse") or task.status not in runs.DONE_TASK_STATUSES:
            continue
        meta = by_name.get(names.get(task.ordinal, ""))
        if meta is None or meta.doc_key is None:
            continue
        label = f"{meta.doc_key}@{meta.version or ''}"
        found.setdefault(label, {})
        if task.stage == "clean":
            found[label]["cleaned"] = task.cache_key
        else:
            found[label]["parser"] = task.component_id
        found[label]["media_type"] = meta.media_type or "unknown"
    return found


class EvalStarter:
    """The ``eval.start`` job."""

    def __init__(
        self,
        engine: Engine,
        registry: ComponentRegistry,
        orchestrator: EvalOrchestrator,
        services: ServicesFor,
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.orchestrator = orchestrator
        self.services = services

    def __call__(self, job: JobContext) -> Mapping[str, Any]:
        eval_run_id = uuid.UUID(str(job.payload["eval_run_id"]))
        with self.engine.connect() as conn:
            run = get_eval_run(conn, None, eval_run_id)
            if run is None or run.status != "queued":
                return {"skipped": True}
            try:
                context, results, plans = self._plan(conn, run)
            except ComponentError as error:
                if error.retryable and job.job.attempts < job.job.max_attempts:
                    raise
                self.orchestrator.fail_run(eval_run_id, error.code)
                return {"failed": error.code}
        started = self.orchestrator.start(
            eval_run_id, context=context, results_key=results, plans=plans
        )
        cached = sum(1 for plan in plans if plan.cached)
        return {
            "status": started.status if started else "missing",
            "shards": len(plans),
            "cached": cached,
        }

    def _plan(
        self, conn: sa.Connection, run: EvalRun
    ) -> tuple[dict[str, Any], str, list[ShardPlan]]:
        workspace = run.workspace_id
        pipeline = runs.get_run(conn, workspace, run.pipeline_run_id)
        if pipeline is None or pipeline.status != "succeeded":
            raise ComponentError("run_not_ready")
        stored = specs.get_by_id(conn, "retrieval", workspace, run.retrieval_config_id)
        version = evalsets.get_version(conn, workspace, version_id=run.eval_set_version_id)
        if stored is None or version is None:
            raise ComponentError("inputs_missing")
        config = resolve_retrieval(stored.document, self.registry)
        index_tasks = {
            cast(type[IndexComponent], self.registry.get(task.component_id)).kind: task
            for task in runs.tasks_of(conn, pipeline.id, stage="index")
            if task.status in runs.DONE_TASK_STATUSES
        }
        wanted = [
            kind
            for kind, retriever in (
                ("dense", config.retrievers.dense),
                ("keyword", config.retrievers.keyword),
            )
            if retriever is not None
        ]
        if any(kind not in index_tasks for kind in wanted):
            raise ComponentError("index_missing")
        indexes = {kind: index_tasks[kind].cache_key for kind in wanted}
        eval_set = evalsets.stored_eval_set(conn, version)
        options = EvalOptions.model_validate(run.options)
        items = _items_for(eval_set, run.split, options)
        if not items:
            raise ComponentError("no_items")
        metas = _metas(conn, workspace, pipeline.corpus_version_id)
        entries_hash = content_hash([dataclasses.asdict(meta) for meta in metas])
        spec = runs.run_spec(conn, pipeline)
        context = StageContext(str(workspace), services=self.services(workspace))
        rerank = cast(RerankComponent, self.registry.get(config.rerank.component)())
        rerank_params = rerank.resolve_params(config.rerank.params)
        models: dict[str, Any] = {
            "rerank": {
                "component": config.rerank.component,
                "version": rerank.info.version,
                "params": rerank.dump_params(rerank_params),
                "revision": rerank.revision_for(rerank_params, context),
                "egress": rerank.info.egress,
            }
        }
        if "dense" in indexes:
            embedder = cast(EmbedComponent, self.registry.get(spec.stages.embed.component)())
            embed_params = embedder.resolve_params(spec.stages.embed.params)
            models["embed"] = {
                "component": spec.stages.embed.component,
                "version": embedder.info.version,
                "params": embedder.dump_params(embed_params),
                "alias": embedder.model_name(embed_params),
                "revision": (pipeline.model_env.get("embed") or {}).get("revision"),
                "egress": embedder.info.egress,
            }
            adapter = config.retrievers.dense.adapter if config.retrievers.dense else None
            if adapter is not None:
                models["adapter"] = _adapter_model(context, adapter, models["embed"])
        day = eval_set.as_of or run.created_at.astimezone(UTC).date()
        retrieval = dump_document(config)
        planned = []
        for batch in shards(items, options.shard_size):
            key = shard_key(
                retrieval=retrieval,
                indexes=indexes,
                entries_hash=entries_hash,
                models=models,
                items=batch,
                day=day,
            )
            planned.append((key, [item.id for item in batch]))
        gold = {f"{ref.doc_key}@{ref.version or ''}" for item in items for ref in item.expected}
        cleaned = {
            label: info
            for label, info in _cleaned_keys(conn, pipeline, metas).items()
            if label in gold
        }
        k_eval = eval_set.defaults.k
        ks = default_ks(k_eval, config.top_k)
        results = results_key(
            shard_keys=[key for key, _ in planned],
            items=items,
            options=options,
            k_eval=k_eval,
            ks=ks,
            findability_inputs={label: info.get("cleaned", "") for label, info in cleaned.items()},
        )
        plans = [ShardPlan(key, ids, artifact_exists(conn, workspace, key)) for key, ids in planned]
        summary = {
            "retrieval": retrieval,
            "retrieval_ref": {"id": str(stored.id), "name": stored.name, "version": stored.version},
            "indexes": indexes,
            "entries_hash": entries_hash,
            "models": models,
            "day": day.isoformat(),
            "k_eval": k_eval,
            "ks": ks,
            "eval_set": {
                "id": str(version.eval_set_id),
                "name": eval_set.name,
                "version": version.version,
                "version_id": str(version.id),
                "document_hash": version.document_hash,
                "split": run.split,
                "items": len(items),
                "as_of": eval_set.as_of.isoformat() if eval_set.as_of else None,
            },
            "pipeline_run": {
                "id": str(pipeline.id),
                "manifest_hash": pipeline.manifest_hash,
                "pipeline_hash": pipeline.spec_hash,
                "corpus_version_id": str(pipeline.corpus_version_id),
            },
            "documents": cleaned,
        }
        return summary, results, plans


def _adapter_model(
    context: StageContext, model_id: str, embed: Mapping[str, Any]
) -> dict[str, Any]:
    """A retrieval config's trained query adapter, checked against the run's embedder: what the
    plan records of it, its artifact's key included, so its ranked lists are keyed by it."""
    if not context.services.get("in_process", False):
        raise ComponentError("in_process_disabled")
    model = context.service(TRAINED_MODELS_SERVICE).ready(model_id, "query-adapter")
    if model.base.get("revision") != embed.get("revision"):
        raise ComponentError("adapter_mismatch")
    return {
        "component": "train/query-adapter",
        "model_id": str(model.id),
        "model_key": model.model_key,
        "alias": f"{model.name}@{model.version}",
        "revision": model.model_key[:16],
        "egress": "none",
    }


def _engine_for(
    run: EvalRun,
    *,
    entries: list[EntryMeta],
    registry: ComponentRegistry,
    indexes: IndexCache,
    context: StageContext,
    cache: RetrievalCache,
) -> tuple[RetrievalEngine, dict[str, Chunk]]:
    plan = run.context
    config = RetrievalConfig.model_validate(plan["retrieval"])
    loaded = {
        kind: indexes.get(run.workspace_id, key, kind) for kind, key in plan["indexes"].items()
    }
    chunks: dict[str, Chunk] = {}
    for item in loaded.values():
        chunks.update(item.chunks)
    embedder: QueryEmbedder | None = None
    if "embed" in plan["models"]:
        model = plan["models"]["embed"]
        component = cast(EmbedComponent, registry.get(model["component"])())
        embedder = QueryEmbedder(component, component.resolve_params(model["params"]), context)
        adapter = plan["models"].get("adapter")
        if adapter is not None and "dense" in loaded:
            embedder = with_adapter(
                embedder,
                adapter["model_id"],
                dense=cast(DenseIndex, loaded["dense"].index),
            )
            if cast(AdaptedQueryEmbedder, embedder).adapter_key != adapter["model_key"]:
                raise ComponentError("adapter_changed")
    rerank_model = plan["models"]["rerank"]
    rerank = cast(RerankComponent, registry.get(rerank_model["component"])())
    reranker = Reranker(
        rerank, rerank.resolve_params(rerank_model["params"]), context, rerank_model["revision"]
    )
    engine = RetrievalEngine(
        config,
        entries=entries,
        chunks=chunks,
        dense=loaded["dense"].index if "dense" in loaded else None,  # type: ignore[arg-type]
        keyword=loaded["keyword"].index if "keyword" in loaded else None,  # type: ignore[arg-type]
        embedder=embedder,
        reranker=reranker,
        cache=cache,
    )
    return engine, chunks


def _ms(started: float) -> float:
    return (time.monotonic() - started) * 1000


class ShardRunner:
    """The ``eval.shard`` job."""

    def __init__(
        self,
        engine: Engine,
        store: ArtifactStore,
        registry: ComponentRegistry,
        orchestrator: EvalOrchestrator,
        services: ServicesFor,
        indexes: IndexCache,
        cache: RetrievalCache,
        cancel_jobs: Callable[[list[uuid.UUID], uuid.UUID], None],
    ) -> None:
        self.engine = engine
        self.store = store
        self.registry = registry
        self.orchestrator = orchestrator
        self.services = services
        self.indexes = indexes
        self.cache = cache
        self.cancel_jobs = cancel_jobs

    def __call__(self, job: JobContext) -> Mapping[str, Any]:
        shard_id = uuid.UUID(str(job.payload["shard_id"]))
        begun = self.orchestrator.begin_shard(shard_id, job.job.attempts)
        if begun is None:
            return {"skipped": True}
        shard, run = begun
        context = StageContext(
            workspace_id=str(run.workspace_id),
            services=self.services(run.workspace_id),
            cancelled=job.stop,
            rate_limit=lambda name, tokens: job.provider(name, tokens=tokens),
        )
        started = time.monotonic()
        try:
            payload, searched = self._search(shard, run, context, job)
        except (Cancelled, JobCancelled):
            if job.cancel_requested:
                self.orchestrator.cancel_shard(shard_id)
            raise JobCancelled(str(job.job.id)) from None
        except ComponentError as error:
            if error.retryable and job.job.attempts < job.job.max_attempts:
                raise
            _, jobs_to_cancel = self.orchestrator.fail_shard(
                shard_id, error.code, duration_ms=_ms(started)
            )
            self.cancel_jobs(jobs_to_cancel, run.workspace_id)
            return {"failed": error.code}
        except Exception as error:
            if job.job.attempts < job.job.max_attempts:
                raise
            logger.warning("eval shard failed", exc_info=error)
            _, jobs_to_cancel = self.orchestrator.fail_shard(
                shard_id, type(error).__name__, duration_ms=_ms(started)
            )
            self.cancel_jobs(jobs_to_cancel, run.workspace_id)
            return {"failed": type(error).__name__}
        finally:
            context.close()
        key = artifact_key(run.workspace_id, shard.cache_key)
        info = self.store.put(key, payload, media_type="application/x-ndjson")
        record = ArtifactRecord(
            workspace_id=run.workspace_id,
            cache_key=shard.cache_key,
            kind="ranked_lists",
            storage_uri=self.store.uri(key),
            size_bytes=info.size,
            content_sha256=info.sha256,
            media_type="application/x-ndjson",
            summary={"items": searched, "checkpoints": 6},
        )
        metrics = {name: value for name, value in context.metrics.items()}
        self.orchestrator.complete_shard(
            shard_id,
            record=record,
            inputs=list(run.context["indexes"].values()),
            metrics=metrics,
            duration_ms=_ms(started),
        )
        return {"items": searched}

    def _search(
        self, shard: EvalShard, run: EvalRun, context: StageContext, job: JobContext
    ) -> tuple[bytes, int]:
        with self.engine.connect() as conn:
            version = evalsets.get_version(
                conn, run.workspace_id, version_id=run.eval_set_version_id
            )
            if version is None:
                raise ComponentError("eval_set_missing")
            stored = {item.item_id: item for item in evalsets.list_items(conn, version.id)}
            corpus_version = uuid.UUID(run.context["pipeline_run"]["corpus_version_id"])
            entries = _metas(conn, run.workspace_id, corpus_version)
        engine, _ = _engine_for(
            run,
            entries=entries,
            registry=self.registry,
            indexes=self.indexes,
            context=context,
            cache=self.cache,
        )
        day = date.fromisoformat(run.context["day"]) if run.context.get("day") else None
        lines = []
        for done, item_id in enumerate(shard.item_ids, start=1):
            context.checkpoint()
            item = stored[item_id].model()
            result = engine.search(item.query, asker=asker_of(item), day=day)
            ranked = RankedItem(
                item_id=item.id,
                checkpoints={
                    name: [(chunk_id, round(float(score), 6)) for chunk_id, score in ranking]
                    for name, ranking in result.checkpoints.items()
                },
                latency_ms=result.latency_ms,
                embedding_revision=result.embedding_revision,
                rerank_revision=result.rerank_revision,
                visible_chunks=result.visible_chunks,
            )
            lines.append(ranked.model_dump_json().encode() + b"\n")
            job.progress(done=done, total=len(shard.item_ids))
        return b"".join(lines), len(lines)


def _passages(
    ranked: Mapping[str, RankedItem], chunks: Mapping[str, Chunk], metas: Sequence[EntryMeta]
) -> dict[str, Passage]:
    by_document = {}
    for meta in sorted(metas, key=lambda entry: entry.name):
        by_document.setdefault(meta.document_id, meta)
    passages: dict[str, Passage] = {}
    for item in ranked.values():
        for ranking in item.checkpoints.values():
            for chunk_id, _ in ranking:
                if chunk_id in passages:
                    continue
                chunk = chunks[chunk_id]
                meta = by_document.get(chunk.document_id)
                passages[chunk_id] = Passage(
                    chunk_id=chunk_id,
                    doc_key=meta.doc_key if meta else None,
                    version=meta.version if meta else None,
                    section=section_of(chunk),
                    text=chunk.text,
                )
    return passages


def result_row(
    item: EvalItem,
    ordinal: int,
    shard: int,
    score: QueryScore,
    *,
    k_eval: int,
    file_types: list[str],
    latency: Mapping[str, float],
) -> dict[str, Any]:
    hit = score.value(KNOWLEDGE_HIT, "final", metric_name("hit", k_eval))
    return {
        "item_id": item.id,
        "ordinal": ordinal,
        "shard": shard,
        "split": item.split or "dev",
        "tags": list(item.tags),
        "file_types": file_types,
        "has_evidence": score.has_evidence,
        "hit": bool(hit) if score.has_evidence else None,
        "first_hit_rank": score.first_hit.get(KNOWLEDGE_HIT),
        "lost_at": score.lost_at,
        "avoid_violation": score.avoid_violation,
        "must_not_return_violation": score.must_not_return_violation,
        "score": {**dataclasses.asdict(score), "latency_ms": dict(latency)},
    }


class EvalFinisher:
    """The ``eval.finish`` job."""

    def __init__(
        self,
        engine: Engine,
        store: ArtifactStore,
        orchestrator: EvalOrchestrator,
        indexes: IndexCache,
        settings: Settings,
        resource_classes: list[str],
    ) -> None:
        self.engine = engine
        self.store = store
        self.orchestrator = orchestrator
        self.indexes = indexes
        self.settings = settings
        self.resource_classes = resource_classes

    def __call__(self, job: JobContext) -> Mapping[str, Any]:
        eval_run_id = uuid.UUID(str(job.payload["eval_run_id"]))
        run = self.orchestrator.begin_finish(eval_run_id)
        if run is None:
            return {"skipped": True}
        try:
            return self._finish(run, job)
        except ComponentError as error:
            if error.retryable and job.job.attempts < job.job.max_attempts:
                raise
            self.orchestrator.fail_run(eval_run_id, error.code)
            return {"failed": error.code}

    def _read(self, run: EvalRun, key: str) -> bytes:
        try:
            return self.store.get(artifact_key(run.workspace_id, key))
        except ArtifactNotFound:
            raise ComponentError("artifact_missing") from None

    def _finish(self, run: EvalRun, job: JobContext) -> Mapping[str, Any]:
        plan = run.context
        with self.engine.connect() as conn:
            version = evalsets.get_version(
                conn, run.workspace_id, version_id=run.eval_set_version_id
            )
            if version is None:
                raise ComponentError("eval_set_missing")
            eval_set = evalsets.stored_eval_set(conn, version)
            shard_list = shards_of(conn, run.id)
            corpus_version = uuid.UUID(plan["pipeline_run"]["corpus_version_id"])
            metas = _metas(conn, run.workspace_id, corpus_version)
            results_cached = bool(run.results_key) and artifact_exists(
                conn, run.workspace_id, run.results_key or ""
            )
            sizes = dict(
                conn.execute(
                    sa.select(artifacts.c.cache_key, artifacts.c.size_bytes).where(
                        artifacts.c.workspace_id == run.workspace_id,
                        artifacts.c.cache_key.in_([shard.cache_key for shard in shard_list]),
                    )
                ).all()
            )
        options = EvalOptions.model_validate(run.options)
        items = _items_for(eval_set, run.split, options)
        shard_of = {item_id: shard.ordinal for shard in shard_list for item_id in shard.item_ids}
        k_eval = int(plan["k_eval"])
        documents: dict[str, dict[str, str]] = plan.get("documents", {})
        record = None
        egress: set[str] = set()
        if results_cached:
            stored = json.loads(self._read(run, run.results_key or ""))
            rows, metrics = stored["rows"], stored["metrics"]
        else:
            rows, metrics, egress = self._score(
                run, items, shard_list, shard_of, metas, documents, options, k_eval, job
            )
            payload = json.dumps({"rows": rows, "metrics": metrics}, separators=(",", ":")).encode()
            key = artifact_key(run.workspace_id, run.results_key or "")
            info = self.store.put(key, payload, media_type="application/json")
            record = ArtifactRecord(
                workspace_id=run.workspace_id,
                cache_key=run.results_key or "",
                kind="eval_results",
                storage_uri=self.store.uri(key),
                size_bytes=info.size,
                content_sha256=info.sha256,
                media_type="application/json",
                summary={"queries": len(rows), "with_evidence": metrics.get("with_evidence", 0)},
            )
        manifest = self._manifest(run, shard_list, sizes, egress, results_cached, record)
        self.orchestrator.finish(
            run.id,
            results=rows,
            metrics=metrics,
            manifest=manifest.model_dump(mode="json", by_alias=True),
            manifest_hash=manifest.fingerprint(),
            record=record,
            inputs=[shard.cache_key for shard in shard_list],
            results_cached=results_cached,
        )
        return {"queries": len(rows), "cached": results_cached}

    def _score(
        self,
        run: EvalRun,
        items: list[EvalItem],
        shard_list: list[EvalShard],
        shard_of: Mapping[str, int],
        metas: list[EntryMeta],
        documents: Mapping[str, Mapping[str, str]],
        options: EvalOptions,
        k_eval: int,
        job: JobContext,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
        """The rows and metrics, and the documents whose chunks this run's shards sent to a
        remote reranker (its egress)."""
        plan = run.context
        ranked: dict[str, RankedItem] = {}
        sent: set[str] = set()
        remote = plan["models"]["rerank"].get("egress", "none") != "none"
        for shard in shard_list:
            for line in self._read(run, shard.cache_key).splitlines():
                if line.strip():
                    item = RankedItem.model_validate_json(line)
                    ranked[item.item_id] = item
                    if remote and shard.status == "succeeded":
                        sent.update(
                            chunk_id for chunk_id, _ in item.checkpoints.get("candidates", [])
                        )
        chunks: dict[str, Chunk] = {}
        for kind, key in plan["indexes"].items():
            chunks.update(self.indexes.get(run.workspace_id, key, kind).chunks)
        passages = _passages(ranked, chunks, metas)
        texts: dict[tuple[str, str | None], str] = {}
        for label, info in documents.items():
            if info.get("cleaned"):
                doc_key, _, doc_version = label.partition("@")
                cleaned = ParsedDocument.model_validate_json(self._read(run, info["cleaned"]))
                texts[(doc_key, doc_version or None)] = cleaned.text
        types_by_doc = {
            (meta.doc_key, meta.version): meta.media_type or "unknown"
            for meta in metas
            if meta.doc_key
        }
        config = RetrievalConfig.model_validate(plan["retrieval"])
        day = date.fromisoformat(plan["day"]) if plan.get("day") else None
        ends = visible_until(metas)
        scores, rows = [], []
        tags, file_types, latencies, groups = {}, {}, {}, {}
        for ordinal, item in enumerate(items):
            job.checkpoint()
            findable = document_findability(
                item.expected, texts, threshold=options.coverage_threshold
            )
            visibility = Visibility.for_asker(config, asker_of(item), day)
            hidden = [
                not any(
                    meta.doc_key == ref.doc_key
                    and (ref.version is None or meta.version == ref.version)
                    and visibility.allows(meta, ends.get(meta.name))
                    for meta in metas
                )
                for ref in item.expected
            ]
            score = score_query(
                item,
                ranked[item.id].checkpoints,
                passages,
                ks=plan["ks"],
                k_eval=k_eval,
                threshold=options.coverage_threshold,
                findable=[flag.findable if flag is not None else None for flag in findable],
                hidden=hidden,
            )
            types = sorted(
                {types_by_doc.get((ref.doc_key, ref.version), "unknown") for ref in item.expected}
            )
            scores.append(score)
            tags[item.id] = item.tags
            file_types[item.id] = types
            latencies[item.id] = ranked[item.id].latency_ms
            groups[item.id] = [
                (
                    types_by_doc.get((ref.doc_key, ref.version)),
                    documents.get(f"{ref.doc_key}@{ref.version or ''}", {}).get("parser"),
                )
                for ref in item.expected
            ]
            rows.append(
                result_row(
                    item,
                    ordinal,
                    shard_of[item.id],
                    score,
                    k_eval=k_eval,
                    file_types=types,
                    latency=ranked[item.id].latency_ms,
                )
            )
        metrics = summarize(
            scores,
            k_eval=k_eval,
            tags=tags,
            file_types=file_types,
            latencies=latencies,
            evidence_groups=groups,
            resamples=options.resamples,
            seed=options.seed,
        )
        metrics["cost"] = self._cost(shard_list, len(items))
        documents_sent = {chunks[chunk_id].document_id for chunk_id in sent if chunk_id in chunks}
        return rows, metrics, documents_sent

    def _cost(self, shard_list: list[EvalShard], queries: int) -> dict[str, Any]:
        calls = sum(int(shard.metrics.get("gateway.calls", 0)) for shard in shard_list)
        tokens = sum(int(shard.metrics.get("tokens", 0)) for shard in shard_list)
        price = self.settings.models.prices.get("model_gateway", 0.0)
        estimated = tokens / 1000 * price
        return {
            "providers": {"model_gateway": {"calls": calls, "tokens": tokens}},
            "estimated": round(estimated, 6),
            "per_1k_queries": round(estimated / queries * 1000, 6) if queries else None,
            "price_table": dict(self.settings.models.prices),
        }

    def _manifest(
        self,
        run: EvalRun,
        shard_list: list[EvalShard],
        sizes: Mapping[str, int],
        egress_documents: set[str],
        results_cached: bool,
        record: ArtifactRecord | None,
    ) -> EvalRunManifest:
        plan = run.context
        finished = datetime.now(UTC)
        models = []
        for role, model in sorted(plan["models"].items()):
            models.append(
                {
                    "role": role,
                    "component": model["component"],
                    "alias": model.get("alias") or model.get("params", {}).get("alias"),
                    "revision": model.get("revision"),
                }
            )
        ran = [shard for shard in shard_list if shard.status == "succeeded"]
        calls = sum(int(shard.metrics.get("gateway.calls", 0)) for shard in ran)
        tokens = sum(int(shard.metrics.get("tokens", 0)) for shard in ran)
        price_table = dict(self.settings.models.prices)
        egress = []
        if egress_documents:
            egress.append({"provider": "model_gateway", "documents": sorted(egress_documents)})
        outputs = [
            {
                "cache_key": shard.cache_key,
                "kind": "ranked_lists",
                "size_bytes": int(sizes.get(shard.cache_key, 0)),
            }
            for shard in shard_list
        ]
        if record is not None:
            outputs.append(
                {
                    "cache_key": record.cache_key,
                    "kind": record.kind,
                    "size_bytes": record.size_bytes,
                }
            )
        return EvalRunManifest.model_validate(
            {
                "eval_run": {
                    "id": str(run.id),
                    "split": run.split,
                    "test_read": run.test_read,
                    "created_by": run.created_by,
                    "created_at": run.created_at,
                },
                "configuration": {
                    "retrieval": plan["retrieval"],
                    "retrieval_hash": run.retrieval_hash,
                    "retrieval_ref": plan["retrieval_ref"],
                    "options": run.options,
                    "k_eval": plan["k_eval"],
                    "ks": plan["ks"],
                    "engine_version": ENGINE_VERSION,
                    "scorer_version": SCORER_VERSION,
                },
                "inputs": {
                    "pipeline_run": plan["pipeline_run"],
                    "indexes": plan["indexes"],
                    "eval_set": plan["eval_set"],
                },
                "models": models,
                "environment": environment(),
                "hardware": hardware(self.resource_classes),
                "execution": {
                    "status": "succeeded",
                    "started_at": run.started_at,
                    "finished_at": finished,
                    "duration_ms": round((finished - run.started_at).total_seconds() * 1000, 3)
                    if run.started_at
                    else None,
                    "results_cached": results_cached,
                    "shards": [
                        {
                            "ordinal": shard.ordinal,
                            "cache_key": shard.cache_key,
                            "items": len(shard.item_ids),
                            "status": shard.status,
                            "cache_hit": shard.cache_hit,
                            "duration_ms": shard.duration_ms,
                            "error_code": shard.error_code,
                        }
                        for shard in shard_list
                    ],
                },
                "cost": {
                    "providers": {"model_gateway": {"calls": calls, "tokens": tokens}}
                    if calls
                    else {},
                    "currency": "USD",
                    "estimated": round(tokens / 1000 * price_table.get("model_gateway", 0.0), 6),
                    "price_table": price_table,
                },
                "egress": egress,
                "outputs": outputs,
            }
        )


def register_eval_handlers(
    handlers: HandlerRegistry,
    *,
    engine: Engine,
    queue: JobQueue,
    store: ArtifactStore,
    registry: ComponentRegistry,
    services: ServicesFor,
    settings: Settings,
) -> EvalOrchestrator:
    """Register ``eval.start``, ``eval.shard`` and ``eval.finish``; returns the orchestrator,
    whose ``reconcile`` the worker runs with its heartbeats."""
    orchestrator = EvalOrchestrator(
        engine,
        queue,
        max_attempts=settings.worker.max_attempts,
        on_finish=finish_hooks(queue, max_attempts=settings.worker.max_attempts),
    )
    indexes = IndexCache(store, engine=engine)
    cache = RetrievalCache()
    resource_classes = [name for name, count in settings.worker.slots.items() if count > 0]

    def cancel_jobs(job_ids: list[uuid.UUID], workspace_id: uuid.UUID) -> None:
        for job_id in job_ids:
            queue.cancel(job_id, workspace_id)

    handlers.register(START_JOB, EvalStarter(engine, registry, orchestrator, services))
    handlers.register(
        SHARD_JOB,
        ShardRunner(engine, store, registry, orchestrator, services, indexes, cache, cancel_jobs),
    )
    handlers.register(
        FINISH_JOB,
        EvalFinisher(engine, store, orchestrator, indexes, settings, resource_classes),
    )
    return orchestrator
