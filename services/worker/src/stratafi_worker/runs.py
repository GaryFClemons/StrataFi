"""The worker's run jobs: ``run.start`` plans a run, ``stage.run`` runs one stage
task.

``run.start`` resolves the run's spec, its corpus version's entries (and the sample), and the
embedding model's revision through the model gateway, so the revision is in every embed key.
It writes the manifest's first half (configuration, inputs, models, environment, hardware), then
hands the run to ``stratafi_store.runs.Orchestrator.start``.

``stage.run`` loads the task's inputs from the artifact store, runs its component with a
``StageContext`` (the services for the workspace, the limits, the job's cancellation and rate
limits), writes the outputs under the task's cache key, and records them; the orchestrator then
advances the entry. A component error that retrying can't fix, or the last attempt of one that
can, fails the task and stops its entry, without failing the job: the job did its work.
"""

import json
import logging
import random
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from stratafi_components.chunk import ChunkComponent, EnrichComponent
from stratafi_components.clean import (
    NEAR_DUPLICATE,
    CleanComponent,
    Corpus,
    CorpusDocument,
    run_clean,
)
from stratafi_components.embed import (
    EmbedComponent,
    describe,
    vectors_from_bytes,
    vectors_to_bytes,
)
from stratafi_components.extract import ExtractComponent
from stratafi_components.index import MANIFEST_PART, IndexComponent, IndexInput
from stratafi_components.models import GatewayProvider
from stratafi_components.parse import ParseComponent
from stratafi_components.parse_quality import parse_quality
from stratafi_components.parsers.hosted import hosted_services
from stratafi_components.parsers.tika import TikaClient
from stratafi_components.scan import ScanInput, run_scan
from stratafi_core.artifacts import Chunk, ParsedDocument, RawDocument
from stratafi_core.components import (
    Cancelled,
    ComponentError,
    ComponentRegistry,
    StageContext,
)
from stratafi_core.identity import HEADER as IDENTITY_HEADER
from stratafi_core.identity import service_tokens
from stratafi_core.observability import STAGE_DURATION, STAGE_TASKS
from stratafi_core.planner import EntryInput, stages_until
from stratafi_core.settings import LimitSettings, Settings
from stratafi_core.specs import (
    PipelineSpec,
    SpecInvalid,
    component_refs,
    dump_document,
    resolve_pipeline,
    spec_hash,
)
from stratafi_store.artifacts import (
    ArtifactStore,
    artifact_key,
    artifact_part_key,
    quarantine_key,
)
from stratafi_store.corpora import get_version, planner_entries, version_entries
from stratafi_store.hooks import finish_hooks
from stratafi_store.jobs import JobQueue
from stratafi_store.lineage import ArtifactRecord
from stratafi_store.runs import (
    ARTIFACT_KINDS,
    STAGE_JOB,
    START_JOB,
    Orchestrator,
    Run,
    RunEntry,
    Task,
    get_entry,
    get_run,
    run_spec,
)
from stratafi_store.tables import pipeline_specs, workspaces
from stratafi_store.training import ModelFiles, WorkspaceModels
from stratafi_worker.environment import environment, hardware
from stratafi_worker.worker import HandlerRegistry, JobCancelled, JobContext

logger = logging.getLogger("stratafi.worker.runs")

ServicesFor = Callable[[uuid.UUID], Mapping[str, Any]]


# --- services ---------------------------------------------------------------------------------


class Services:
    """The clients a workspace's components get: clamd and the model gateway, the gateway
    authenticated as Stratafi's service identity in the workspace's tenant.
    Clients are made once per workspace and shared by its tasks; each request gets a fresh,
    short-lived token, minted with the development key."""

    def __init__(
        self, engine: Engine, settings: Settings, store: ArtifactStore | None = None
    ) -> None:
        self.engine = engine
        self.settings = settings
        # Trained models' files, for embedders, rerankers and adapters.
        self.models = ModelFiles(engine, store) if store is not None else None
        identity = settings.identity
        self.tokens = service_tokens(
            signing_key=identity.service_signing_key, name=identity.service_name
        )
        self._clients: dict[uuid.UUID, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _tenant(self, workspace_id: uuid.UUID) -> str | None:
        with self.engine.connect() as conn:
            return conn.execute(
                sa.select(workspaces.c.external_tenant_id).where(workspaces.c.id == workspace_id)
            ).scalar_one_or_none()

    def _make(self, workspace_id: uuid.UUID) -> dict[str, Any]:
        tenant = self._tenant(workspace_id)
        tokens = self.tokens

        def headers() -> dict[str, str]:
            if tokens is None or tenant is None:
                return {}
            token = tokens.token(tenant)
            return {IDENTITY_HEADER: token} if token else {}

        services: dict[str, Any] = {}
        if clamd := self.settings.services.clamd:
            host, _, port = clamd.rpartition(":")
            services["clamd"] = (host or "127.0.0.1", int(port))
        if url := self.settings.models.gateway_url:
            services["models"] = GatewayProvider(url, headers=headers)
        if url := self.settings.services.tika_url:
            services["tika"] = TikaClient(url)
        # Hosted providers (only allow-listed ones get access), licenses and lab switches.
        services.update(hosted_services(self.settings))
        # Postgres-backed indexes (index/postgres-fts, index/pgvector-hnsw) keep their tables here.
        services["database"] = self.engine
        if key := self.settings.clean.redaction_key:
            services["redaction_key"] = key.get_secret_value().encode()
        if self.models is not None:
            services["trained_models"] = WorkspaceModels(self.models, workspace_id)
        return services

    def __call__(self, workspace_id: uuid.UUID) -> Mapping[str, Any]:
        with self._lock:
            if workspace_id not in self._clients:
                self._clients[workspace_id] = self._make(workspace_id)
            return self._clients[workspace_id]

    def close(self) -> None:
        with self._lock:
            clients, self._clients = self._clients, {}
        for services in clients.values():
            for client in services.values():
                if hasattr(client, "close"):
                    client.close()


# --- run.start --------------------------------------------------------------------------------


def select_sample(entries: list[EntryInput], sample: Mapping[str, Any] | None) -> list[EntryInput]:
    """The entries a run works on: named ones, or ``size`` chosen by ``seed``, in name order."""
    if not sample:
        return entries
    names = sample.get("entries")
    if names:
        wanted = set(names)
        return [entry for entry in entries if entry.name in wanted]
    size = sample.get("size")
    if size is None or size >= len(entries):
        return entries
    chosen = random.Random(sample.get("seed") or 0).sample(range(len(entries)), int(size))  # noqa: S311
    return [entries[index] for index in sorted(chosen)]


@dataclass(frozen=True)
class CorpusEntries:
    """A run's corpus version as the planner and the manifest see it."""

    entries: list[EntryInput]  # by name; a duplicate's bytes match an earlier name's
    corpus: dict[str, Any]  # the version's IDs, number and manifest hash
    documents: list[dict[str, Any]]  # each entry's manifest row
    refused: frozenset[str]  # names of documents intake refused


def corpus_entries(conn: sa.Connection, run: Run) -> CorpusEntries:
    version = get_version(conn, run.workspace_id, version_id=run.corpus_version_id)
    if version is None:
        raise ComponentError("corpus_version_missing")
    stored = version_entries(conn, run.workspace_id, version.id)
    entries = planner_entries(stored)
    documents, refused = [], set()
    for entry in stored:
        if entry.status == "refused":
            refused.add(entry.name)
        documents.append(
            {
                "entry": entry.name,
                "document_id": str(entry.document_id),
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
                "media_type": entry.media_type,
                "traits": entry.traits,
            }
        )
    corpus = {
        "corpus_id": str(version.corpus_id),
        "corpus_version_id": str(version.id),
        "version": version.version,
        "manifest_hash": version.manifest_hash,
    }
    return CorpusEntries(entries, corpus, documents, frozenset(refused))


def embed_model(spec: PipelineSpec, registry: ComponentRegistry, revision: str) -> dict[str, Any]:
    """The manifest's record of the embedding model, as far as the run knows it when it starts;
    what the gateway served and the dimensions come from the vectors when it ends."""
    component = cast(EmbedComponent, registry.get(spec.stages.embed.component)())
    params = component.resolve_params(spec.stages.embed.params)
    dumped = component.dump_params(params)
    return {
        "role": "embed",
        "component": component.info.id,
        "alias": component.model_name(params),
        "revision": revision,
        "dtype": "float32",
        "prefixes": {
            "document": dumped.get("document_prefix", ""),
            "query": dumped.get("query_prefix", ""),
        },
        # How the chunker counted the tokens of the texts this model embeds.
        "tokenizer": spec.stages.chunk.params.get("token_estimator"),
    }


def start_manifest(
    run: Run,
    spec: PipelineSpec,
    spec_ref: Mapping[str, Any],
    corpus: CorpusEntries,
    selected: list[EntryInput],
    models: list[dict[str, Any]],
    registry: ComponentRegistry,
    *,
    price_table: Mapping[str, float],
    resource_classes: list[str],
) -> dict[str, Any]:
    """The manifest's first half, written when the run starts: the run, its configuration,
    inputs, models, environment and hardware, and the price table its cost will use."""
    components = []
    for path, ref in component_refs(spec):
        component = registry.get(ref.component)
        components.append(
            {
                "path": path,
                "id": ref.component,
                "version": component.info.version,
                "params": ref.params,
                "params_hash": component.params_hash(component.resolve_params(ref.params)),
            }
        )
    names = {entry.name for entry in selected}
    return {
        "schema": "stratafi.run-manifest.v1",
        "run": {
            "id": str(run.id),
            "kind": run.kind,
            "mode": run.mode,
            "until": run.until_stage,
            "rerun_of": str(run.rerun_of) if run.rerun_of else None,
            "rerun_from": run.rerun_from,
            "created_by": run.created_by,
            "created_at": run.created_at.isoformat(),
        },
        "configuration": {
            "pipeline": dump_document(spec),
            "pipeline_hash": spec_hash(spec),
            "pipeline_ref": dict(spec_ref),
            # A pipeline run has no retrieval config; evaluation runs add one.
            "retrieval": None,
            "retrieval_hash": None,
            "components": components,
        },
        "inputs": {
            "corpus": corpus.corpus,
            "documents": [row for row in corpus.documents if row["entry"] in names],
            "sample": dict(run.sample) if run.sample else None,
            "eval_set": None,
        },
        "models": models,
        "environment": environment(),
        "hardware": hardware(resource_classes),
        "cost": {"price_table": dict(price_table)},
    }


class RunStarter:
    """The ``run.start`` job."""

    def __init__(
        self,
        engine: Engine,
        registry: ComponentRegistry,
        orchestrator: Orchestrator,
        services: ServicesFor,
        settings: Settings,
        resource_classes: list[str],
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.orchestrator = orchestrator
        self.services = services
        self.settings = settings
        self.resource_classes = resource_classes

    def __call__(self, job: JobContext) -> Mapping[str, Any]:
        run_id = uuid.UUID(str(job.payload["run_id"]))
        with self.engine.connect() as conn:
            run = get_run(conn, None, run_id)
            if run is None or run.status != "queued":
                return {"skipped": True}
            document = conn.execute(
                sa.select(pipeline_specs).where(pipeline_specs.c.id == run.pipeline_spec_id)
            ).one()
            try:
                corpus = corpus_entries(conn, run)
            except ComponentError as error:
                self.orchestrator.fail_run(run_id, error.code)
                return {"failed": error.code}
        try:
            spec = resolve_pipeline(document.document, self.registry)
        except SpecInvalid:
            self.orchestrator.fail_run(run_id, "invalid_spec")
            return {"failed": "invalid_spec"}
        selected = select_sample(corpus.entries, run.sample)
        model_env: dict[str, dict[str, Any]] = {}
        models: list[dict[str, Any]] = []
        if "embed" in stages_until(run.until_stage):
            # The revision is in every embed key, so a model swap at the gateway never reuses
            # vectors from the old one.
            component = cast(EmbedComponent, self.registry.get(spec.stages.embed.component)())
            context = StageContext(services=self.services(run.workspace_id))
            try:
                revision = component.revision_for(
                    component.resolve_params(spec.stages.embed.params), context
                )
            except ComponentError as error:
                if error.retryable and job.job.attempts < job.job.max_attempts:
                    raise
                self.orchestrator.fail_run(run_id, error.code)
                return {"failed": error.code}
            model_env["embed"] = {"revision": revision}
            models.append(embed_model(spec, self.registry, revision))
        manifest = start_manifest(
            run,
            spec,
            {"id": str(document.id), "name": document.name, "version": document.version},
            corpus,
            selected,
            models,
            self.registry,
            price_table=self.settings.models.prices,
            resource_classes=self.resource_classes,
        )
        started = self.orchestrator.start(
            run_id,
            entries=selected,
            model_env=model_env,
            manifest=manifest,
            refused=corpus.refused,
        )
        return {"status": started.status if started else "missing", "entries": len(selected)}


# --- stage.run --------------------------------------------------------------------------------


def _chunks_from(data: bytes) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in data.splitlines() if line.strip()]


def _chunks_to(chunks: list[Chunk]) -> bytes:
    return b"".join(chunk.model_dump_json().encode() + b"\n" for chunk in chunks)


def _component_params(component: type, params: Mapping[str, Any], *drop: str) -> BaseModel:
    return component.resolve_params({k: v for k, v in params.items() if k not in drop})  # type: ignore[attr-defined]


class StageRunner:
    """The ``stage.run`` job."""

    def __init__(
        self,
        engine: Engine,
        store: ArtifactStore,
        registry: ComponentRegistry,
        orchestrator: Orchestrator,
        services: ServicesFor,
        limits: LimitSettings,
    ) -> None:
        self.engine = engine
        self.store = store
        self.registry = registry
        self.orchestrator = orchestrator
        self.services = services
        self.limits = limits
        # MinHash signatures by (parse key, parameters), shared by the clean tasks of a process.
        self.signatures: dict[tuple[str, str], Any] = {}

    def __call__(self, job: JobContext) -> Mapping[str, Any]:
        task_id = uuid.UUID(str(job.payload["task_id"]))
        begun = self.orchestrator.begin_task(task_id, job.job.attempts)
        if begun is None:
            return {"skipped": True}
        task, run = begun
        context = StageContext(
            workspace_id=str(run.workspace_id),
            services=self.services(run.workspace_id),
            limits=self.limits,
            cancelled=job.stop,
            rate_limit=lambda name, tokens: job.provider(name, tokens=tokens),
        )
        started = time.monotonic()
        try:
            record, inputs = self._execute(task, run, context)
        except (Cancelled, JobCancelled):
            if job.cancel_requested:
                self.orchestrator.cancel_task(task_id)
            raise JobCancelled(str(job.job.id)) from None
        except ComponentError as error:
            if error.retryable and job.job.attempts < job.job.max_attempts:
                raise
            self.orchestrator.fail_task(task_id, error.code, duration_ms=_ms(started))
            STAGE_TASKS.labels(task.stage, "failed").inc()
            return {"failed": error.code}
        except Exception as error:
            if job.job.attempts < job.job.max_attempts:
                raise
            logger.warning("stage task failed", exc_info=error)
            self.orchestrator.fail_task(task_id, type(error).__name__, duration_ms=_ms(started))
            STAGE_TASKS.labels(task.stage, "failed").inc()
            return {"failed": type(error).__name__}
        finally:
            context.close()
        # Throughput: one document's stage done.
        STAGE_TASKS.labels(task.stage, "succeeded").inc()
        STAGE_DURATION.labels(task.stage).observe(time.monotonic() - started)
        self.orchestrator.complete_task(
            task_id,
            record=record,
            inputs=inputs,
            metrics=context.metrics,
            duration_ms=_ms(started),
        )
        return {"stage": task.stage}

    # --- inputs and outputs -------------------------------------------------------------------

    def _read(self, run: Run, key: str, part: str | None = None) -> bytes:
        workspace = run.workspace_id
        return self.store.get(
            artifact_part_key(workspace, key, part) if part else artifact_key(workspace, key)
        )

    def _document_file(self, run: Run, entry: RunEntry, context: StageContext):
        path = context.scratch_dir() / "document"
        with (
            self.store.open(quarantine_key(run.workspace_id, entry.sha256)) as source,
            path.open("wb") as sink,
        ):
            while chunk := source.read(1024 * 1024):
                sink.write(chunk)
        return path

    def _record(
        self,
        run: Run,
        task: Task,
        payload: bytes,
        media_type: str,
        summary: Mapping[str, Any],
        parts: Mapping[str, bytes] | None = None,
    ) -> ArtifactRecord:
        key = artifact_key(run.workspace_id, task.cache_key)
        for part, data in (parts or {}).items():
            self.store.put(
                artifact_part_key(run.workspace_id, task.cache_key, part),
                data,
                media_type="application/json",
            )
        info = self.store.put(key, payload, media_type=media_type)
        return ArtifactRecord(
            workspace_id=run.workspace_id,
            cache_key=task.cache_key,
            kind=ARTIFACT_KINDS[task.stage],
            storage_uri=self.store.uri(key),
            size_bytes=info.size,
            content_sha256=info.sha256,
            media_type=media_type,
            summary=dict(summary),
        )

    def _entry(self, run: Run, task: Task) -> RunEntry:
        with self.engine.connect() as conn:
            entry = get_entry(conn, run.id, task.ordinal)
        if entry is None:
            raise ComponentError("entry_missing")
        return entry

    def _execute(
        self, task: Task, run: Run, context: StageContext
    ) -> tuple[ArtifactRecord, list[str]]:
        with self.engine.connect() as conn:
            spec = run_spec(conn, run)
        stage = task.stage
        if stage == "index":
            return self._index(task, run, context)
        entry = self._entry(run, task)
        if stage == "extract":
            component = cast(ExtractComponent, self.registry.get(task.component_id)())
            raw = component.extract(
                self._document_file(run, entry, context),
                entry.name,
                document_id=str(entry.document_id),
                expected_sha256=entry.sha256,
                params=component.resolve_params(task.params),
                context=context,
            )
            summary = raw.model_dump(mode="json", exclude={"schema_", "document_id", "sha256"})
            # The document itself is the root artifact (recorded at intake).
            return self._json(run, task, raw, summary), [entry.sha256]
        raw_key = task.inputs[0]
        if stage == "scan":
            raw = RawDocument.model_validate_json(self._read(run, raw_key))
            report = run_scan(
                ScanInput(self._document_file(run, entry, context), raw, entry.duplicate),
                spec.stages.scan,
                self.registry,
                context,
            )
            summary = {
                "outcome": report.outcome,
                "decision": report.decision,
                "findings": report.findings,
            }
            return self._json(run, task, report, summary), [raw_key]
        if stage == "parse":
            raw = RawDocument.model_validate_json(self._read(run, raw_key))
            parser = cast(ParseComponent, self.registry.get(task.component_id)())
            document = parser.parse(
                self._document_file(run, entry, context),
                raw,
                parser.resolve_params(task.params),
                context,
            )
            summary = {
                "elements": len(document.elements),
                "pages": len(document.pages),
                "chars": len(document.text),
                **parse_quality(document),
                "ocr_pages": int(context.metrics.get("ocr_pages", 0)),
                "warnings": list(document.warnings),
            }
            return self._json(run, task, document, summary), [raw_key]
        if stage == "clean":
            parsed = ParsedDocument.model_validate_json(self._read(run, raw_key))
            rules = []
            for ref in spec.stages.clean:
                rule = cast(CleanComponent, self.registry.get(ref.component)())
                rules.append((ref.component, rule, rule.resolve_params(ref.params)))
            cleaned = run_clean(parsed, rules, context, self._corpus(run, task, raw_key))
            summary: dict[str, Any] = {
                "edits": len(cleaned.edits or []),
                "chars": len(cleaned.text),
                "redactions": int(context.metrics.get("redactions", 0)),
            }
            if cleaned.language:
                summary["language"] = cleaned.language
            if NEAR_DUPLICATE in cleaned.warnings:
                summary["excluded"] = NEAR_DUPLICATE
            return self._json(run, task, cleaned, summary), list(task.inputs)
        if stage == "chunk":
            cleaned = ParsedDocument.model_validate_json(self._read(run, raw_key))
            component_type = self.registry.get(task.component_id)
            chunker = cast(ChunkComponent, component_type())
            chunks = chunker.chunk(
                cleaned,
                _component_params(component_type, task.params, "index_text", "label"),
                cache_key=task.cache_key,
                label=task.params.get("label"),
                index_text=task.params["index_text"],
                context=context,
            )
            return self._chunks(run, task, chunks), [raw_key]
        if stage == "enrich":
            enricher = cast(EnrichComponent, self.registry.get(task.component_id)())
            chunks = enricher.enrich(
                _chunks_from(self._read(run, raw_key)),
                enricher.resolve_params(task.params),
                context,
            )
            return self._chunks(run, task, chunks), [raw_key]
        embedder = cast(EmbedComponent, self.registry.get(task.component_id)())
        vectors = embedder.embed(
            _chunks_from(self._read(run, raw_key)),
            embedder.resolve_params(task.params),
            expected_revision=task.env.get("revision"),
            context=context,
        )
        payload, sidecar = vectors_to_bytes(vectors)
        record = self._record(
            run, task, payload, "application/x-npy", describe(vectors), {"sidecar": sidecar}
        )
        return record, [raw_key]

    def _corpus(self, run: Run, task: Task, own: str) -> Corpus | None:
        """The corpus a corpus-wide clean rule compares with: the other parsed documents."""
        if "corpus" not in task.params:
            return None
        if len(self.signatures) > 10_000:
            self.signatures.clear()
        ranks = {item["key"]: tuple(item["rank"]) for item in task.params["corpus"]}

        def loader(key: str):
            return lambda: ParsedDocument.model_validate_json(self._read(run, key)).text

        return Corpus(
            key=own,
            rank=ranks.get(own, ()),
            others=[
                CorpusDocument(key, rank, loader(key)) for key, rank in ranks.items() if key != own
            ],
            cache=self.signatures,
        )

    def _json(
        self, run: Run, task: Task, model: BaseModel, summary: Mapping[str, Any]
    ) -> ArtifactRecord:
        payload = model.model_dump_json(by_alias=True).encode()
        return self._record(run, task, payload, "application/json", summary)

    def _chunks(self, run: Run, task: Task, chunks: list[Chunk]) -> ArtifactRecord:
        tokens = [next(iter(chunk.tokens.values()), 0) for chunk in chunks]
        summary = {
            "chunk_count": len(chunks),
            "tokens_min": min(tokens, default=0),
            "tokens_max": max(tokens, default=0),
        }
        return self._record(run, task, _chunks_to(chunks), "application/x-ndjson", summary)

    def _index(
        self, task: Task, run: Run, context: StageContext
    ) -> tuple[ArtifactRecord, list[str]]:
        component = cast(IndexComponent, self.registry.get(task.component_id)())
        inputs: list[IndexInput] = []
        if component.kind == "dense":
            pairs = list(zip(task.inputs[::2], task.inputs[1::2], strict=True))
            for chunks_key, vectors_key in pairs:
                context.checkpoint()
                vectors = vectors_from_bytes(
                    self._read(run, vectors_key), self._read(run, vectors_key, "sidecar")
                )
                inputs.append(
                    IndexInput(chunks_key, _chunks_from(self._read(run, chunks_key)), vectors)
                )
        else:
            for chunks_key in task.inputs:
                context.checkpoint()
                inputs.append(IndexInput(chunks_key, _chunks_from(self._read(run, chunks_key))))
        built = component.build(inputs, component.resolve_params(task.params), context)
        manifest = json.dumps(built.manifest.model_dump(mode="json", by_alias=True)).encode()
        summary = {
            "index_kind": built.manifest.kind,
            "chunk_count": len(built.manifest.chunk_ids),
            "dims": built.manifest.dims,
            "revision": built.manifest.revision,
        }
        record = self._record(
            run, task, built.payload, built.media_type, summary, {MANIFEST_PART: manifest}
        )
        return record, list(task.inputs)


def _ms(started: float) -> float:
    return (time.monotonic() - started) * 1000


def register_run_handlers(
    handlers: HandlerRegistry,
    *,
    engine: Engine,
    queue: JobQueue,
    store: ArtifactStore,
    registry: ComponentRegistry,
    services: ServicesFor,
    settings: Settings,
) -> Orchestrator:
    """Register ``run.start`` and ``stage.run``; returns the orchestrator, whose ``reconcile``
    the worker runs with its heartbeats."""
    orchestrator = Orchestrator(
        engine,
        queue,
        registry,
        max_attempts=settings.worker.max_attempts,
        on_finish=finish_hooks(queue, max_attempts=settings.worker.max_attempts),
    )
    resource_classes = [name for name, count in settings.worker.slots.items() if count > 0]
    handlers.register(
        START_JOB,
        RunStarter(engine, registry, orchestrator, services, settings, resource_classes),
    )
    handlers.register(
        STAGE_JOB,
        StageRunner(engine, store, registry, orchestrator, services, settings.limits),
    )
    return orchestrator
