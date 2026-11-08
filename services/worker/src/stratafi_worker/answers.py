"""The worker's answer-level jobs: ``answer_run.start`` answers, judges and
scores an eval run's queries, and ``questions.draft`` drafts questions from a pipeline run's
passages for the review queue.

**Answer runs.** Each query's context is its eval run's final passages, as the knowledge
service would return them, each with a source ID (``funds-availability-v3-p1``) and whether it
covers the query's evidence (the eval set's ``knowledge-hit-v1`` rule). The generator answers
with the ``grounded-qa`` prompt; a judge, when the run has one, scores faithfulness and, for an
item with a reference answer, correctness; ``stratafi_core.answers.score_answer`` adds
abstention and citations. The answers and their passages go into one artifact whose key hashes
the eval results, the generator and judge with their revisions and the pinned prompts, so an
identical run reuses it; the database gets numbers and flags.

**Drafting.** A seeded sample of the run's passages of at least 25 words, each asked for one
question with the ``draft-questions`` prompt; a question survives only with evidence copied from
its passage.

Answers, passages and questions never reach logs or error codes: a failure is its code.
"""

import hashlib
import json
import random
import uuid
from collections.abc import Mapping
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Engine

from stratafi_components.generate import GenerateComponent
from stratafi_components.index import IndexComponent
from stratafi_components.retrieval import EntryMeta, section_of
from stratafi_core.answers import (
    GROUNDED_QA,
    JUDGE_CORRECTNESS,
    JUDGE_FAITHFULNESS,
    AnswerScore,
    ContextPassage,
    GenerationUsage,
    Verdicts,
    answer_request,
    correctness_request,
    draft_request,
    faithfulness_request,
    parse_judge_score,
    parse_questions,
    score_answer,
    source_id,
    summarize,
    words,
)
from stratafi_core.components import ComponentError, ComponentRegistry, StageContext
from stratafi_core.evalruns import EvalOptions, RankedItem
from stratafi_core.hashing import cache_key
from stratafi_core.relevance import KNOWLEDGE_HIT, Matcher
from stratafi_core.settings import Settings
from stratafi_store import answers, corpora, evalruns, evalsets, questions, runs
from stratafi_store.artifacts import ArtifactNotFound, ArtifactStore, artifact_key
from stratafi_store.evalruns import artifact_exists
from stratafi_store.lineage import ArtifactRecord, record_artifact
from stratafi_worker.evals import IndexCache, _items_for, _passages
from stratafi_worker.runs import ServicesFor
from stratafi_worker.worker import HandlerRegistry, JobContext

ANSWERS_STAGE = "answers"
MIN_PASSAGE_WORDS = 25


class ComponentRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component: str
    params: dict[str, Any] = Field(default_factory=dict)


class AnswerConfig(BaseModel):
    """An answer run's configuration, as ``answer_runs.config`` records it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generator: ComponentRef
    judge: ComponentRef | None = None
    passages: int = Field(default=8, ge=1, le=50)
    prompts: dict[str, dict[str, str]] = Field(default_factory=dict)


class DraftConfig(BaseModel):
    """A question batch's configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generator: ComponentRef
    count: int = Field(default=20, ge=1, le=100)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    prompt: dict[str, str] = Field(default_factory=dict)


def answer_key(item_id: str, answer: str) -> str:
    """What a label attaches to: the item and its answer."""
    return hashlib.sha256(f"{item_id}\x00{answer}".encode()).hexdigest()


def _generator(
    registry: ComponentRegistry, ref: ComponentRef
) -> tuple[GenerateComponent, BaseModel]:
    component = registry.get(ref.component)()
    if not isinstance(component, GenerateComponent):
        raise ComponentError("not_a_generator")
    return component, component.resolve_params(ref.params)


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


class AnswerRunner(_Base):
    """The ``answer_run.start`` job."""

    def __call__(self, job: JobContext) -> Mapping[str, Any]:
        run_id = uuid.UUID(str(job.payload["answer_run_id"]))
        with self.engine.begin() as conn:
            run = answers.begin(conn, run_id)
        if run is None:
            return {"skipped": True}
        try:
            return self._run(run, job)
        except ComponentError as error:
            if error.retryable and job.job.attempts < job.job.max_attempts:
                raise
            with self.engine.begin() as conn:
                answers.finish(conn, run, status="failed", error_code=error.code)
            return {"failed": error.code}

    def _run(self, run: answers.AnswerRun, job: JobContext) -> Mapping[str, Any]:
        config = AnswerConfig.model_validate(run.config)
        with self.engine.connect() as conn:
            evaluation = evalruns.get_eval_run(conn, run.workspace_id, run.eval_run_id)
            if evaluation is None or evaluation.status != "succeeded":
                raise ComponentError("eval_run_not_succeeded")
            version = evalsets.get_version(
                conn, run.workspace_id, version_id=evaluation.eval_set_version_id
            )
            if version is None:
                raise ComponentError("eval_set_missing")
            eval_set = evalsets.stored_eval_set(conn, version)
            shard_list = evalruns.shards_of(conn, evaluation.id)
            corpus_version = uuid.UUID(evaluation.context["pipeline_run"]["corpus_version_id"])
            metas = [
                EntryMeta.from_entry(entry)
                for entry in corpora.version_entries(conn, run.workspace_id, corpus_version)
            ]
        context = StageContext(str(run.workspace_id), services=self.services(run.workspace_id))
        generator, generator_params = _generator(self.registry, config.generator)
        judge = _generator(self.registry, config.judge) if config.judge else None
        revisions = {
            "generator": generator.revision_for(generator_params, context),
            "judge": judge[0].revision_for(judge[1], context) if judge else None,
        }
        key = cache_key(
            stage=ANSWERS_STAGE,
            component_id=generator.info.id,
            component_version=generator.info.version,
            params={
                "config": config.model_dump(mode="json"),
                "judge_version": judge[0].info.version if judge else None,
            },
            inputs=[evaluation.results_key or ""],
            env={"revisions": revisions},
        )
        with self.engine.connect() as conn:
            cached = artifact_exists(conn, run.workspace_id, key)
        usage = GenerationUsage()
        if cached:
            rows = json.loads(self._read(run.workspace_id, key))["items"]
        else:
            options = EvalOptions.model_validate(evaluation.options)
            items = _items_for(eval_set, evaluation.split, options)
            ranked: dict[str, RankedItem] = {}
            for shard in shard_list:
                for line in self._read(run.workspace_id, shard.cache_key).splitlines():
                    if line.strip():
                        found = RankedItem.model_validate_json(line)
                        ranked[found.item_id] = found
            chunks = {}
            for kind, index_key in evaluation.context["indexes"].items():
                chunks.update(self.indexes.get(run.workspace_id, index_key, kind).chunks)
            passages = _passages(ranked, chunks, metas)
            day = str(evaluation.context.get("day") or "")
            threshold = float(evaluation.options.get("coverage_threshold", 0.7))
            rows = []
            for item in items:
                job.checkpoint()
                matcher = Matcher(item.expected, threshold)
                final = ranked[item.id].checkpoints.get("final", [])[: config.passages]
                given = [
                    ContextPassage(
                        source_id(passage.doc_key, passage.version, rank),
                        passage.doc_key,
                        passage.version,
                        passage.section,
                        passage.text,
                        matcher.judge(passage).relevant(KNOWLEDGE_HIT),
                    )
                    for rank, (chunk_id, _) in enumerate(final, start=1)
                    if (passage := passages.get(chunk_id)) is not None
                ]
                generated = generator.generate(
                    answer_request(item.query, given, day=day), generator_params, context
                )
                usage.add(GROUNDED_QA, generated.prompt_tokens, generated.completion_tokens)
                answer = generated.text.strip()
                verdicts = Verdicts()
                if judge is not None:
                    reply = judge[0].generate(
                        faithfulness_request(item.query, given, answer), judge[1], context
                    )
                    usage.add(JUDGE_FAITHFULNESS, reply.prompt_tokens, reply.completion_tokens)
                    correctness = None
                    if item.reference_answer:
                        graded = judge[0].generate(
                            correctness_request(item.query, item.reference_answer, answer),
                            judge[1],
                            context,
                        )
                        usage.add(JUDGE_CORRECTNESS, graded.prompt_tokens, graded.completion_tokens)
                        correctness = parse_judge_score(graded.text)
                    verdicts = Verdicts(parse_judge_score(reply.text), correctness)
                score = score_answer(
                    item_id=item.id,
                    expected=[(ref.doc_key, ref.version) for ref in item.expected],
                    reference=item.reference_answer,
                    answer=answer,
                    passages=given,
                    verdicts=verdicts,
                )
                rows.append(
                    {
                        "item_id": item.id,
                        "answer": answer,
                        "answer_key": answer_key(item.id, answer),
                        "passages": [
                            {**passage.describe(), "text": passage.text} for passage in given
                        ],
                        "score": score.as_dict(),
                    }
                )
            payload = json.dumps({"items": rows}, separators=(",", ":")).encode()
            info = self.store.put(
                artifact_key(run.workspace_id, key), payload, media_type="application/json"
            )
            with self.engine.begin() as conn:
                record_artifact(
                    conn,
                    ArtifactRecord(
                        workspace_id=run.workspace_id,
                        cache_key=key,
                        kind="answers",
                        storage_uri=self.store.uri(artifact_key(run.workspace_id, key)),
                        size_bytes=info.size,
                        content_sha256=info.sha256,
                        media_type="application/json",
                        summary={"queries": len(rows)},
                    ),
                    [evaluation.results_key] if evaluation.results_key else [],
                )
        scores = [AnswerScore(**row["score"]) for row in rows]
        options = EvalOptions.model_validate(evaluation.options)
        metrics = summarize(scores, resamples=options.resamples, seed=options.seed)
        results = [
            answers.AnswerResult(row["item_id"], ordinal, row["answer_key"], row["score"])
            for ordinal, row in enumerate(rows)
        ]
        with self.engine.begin() as conn:
            answers.finish(
                conn,
                run,
                status="succeeded",
                results=results,
                metrics=metrics,
                results_key=key,
                results_cached=cached,
                usage={
                    "calls": usage.calls,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "by_prompt": usage.by_prompt,
                    "revisions": revisions,
                    "egress": sorted(
                        {generator.info.egress, *([judge[0].info.egress] if judge else [])}
                    ),
                },
            )
        return {"queries": len(rows), "cached": cached}


class QuestionDrafter(_Base):
    """The ``questions.draft`` job."""

    def __call__(self, job: JobContext) -> Mapping[str, Any]:
        batch_id = uuid.UUID(str(job.payload["batch_id"]))
        with self.engine.begin() as conn:
            batch = questions.begin_batch(conn, batch_id)
        if batch is None:
            return {"skipped": True}
        try:
            return self._draft(batch, job)
        except ComponentError as error:
            if error.retryable and job.job.attempts < job.job.max_attempts:
                raise
            with self.engine.begin() as conn:
                questions.finish_batch(conn, batch, status="failed", error_code=error.code)
            return {"failed": error.code}

    def _draft(self, batch: questions.QuestionBatch, job: JobContext) -> Mapping[str, Any]:
        config = DraftConfig.model_validate(batch.config)
        if batch.pipeline_run_id is None:  # nothing to draft from
            raise ComponentError("run_not_succeeded")
        with self.engine.connect() as conn:
            pipeline = runs.get_run(conn, batch.workspace_id, batch.pipeline_run_id)
            if pipeline is None or pipeline.status != "succeeded":
                raise ComponentError("run_not_succeeded")
            index_tasks = [
                task
                for task in runs.tasks_of(conn, pipeline.id, stage="index")
                if task.status in runs.DONE_TASK_STATUSES
            ]
            metas = [
                EntryMeta.from_entry(entry)
                for entry in corpora.version_entries(
                    conn, batch.workspace_id, pipeline.corpus_version_id
                )
            ]
        if not index_tasks:
            raise ComponentError("index_missing")
        task = index_tasks[0]
        kind = cast(type[IndexComponent], self.registry.get(task.component_id)).kind
        chunks = self.indexes.get(batch.workspace_id, task.cache_key, kind).chunks
        by_document: dict[str, EntryMeta] = {}
        for meta in sorted(metas, key=lambda entry: entry.name):
            by_document.setdefault(meta.document_id, meta)
        eligible = sorted(
            chunk_id
            for chunk_id, chunk in chunks.items()
            if len(words(chunk.text)) >= MIN_PASSAGE_WORDS
            and by_document.get(chunk.document_id) is not None
        )
        order = random.Random(config.seed).sample(eligible, len(eligible))  # noqa: S311
        context = StageContext(str(batch.workspace_id), services=self.services(batch.workspace_id))
        generator, params = _generator(self.registry, config.generator)
        drafts: list[dict[str, Any]] = []
        seen: set[str] = set()
        asked = dropped = 0
        for chunk_id in order:
            if len(drafts) == config.count:
                break
            job.checkpoint()
            chunk = chunks[chunk_id]
            meta = by_document[chunk.document_id]
            doc_key = meta.doc_key or meta.name
            reply = generator.generate(draft_request(doc_key, chunk.text, count=1), params, context)
            asked += 1
            found = parse_questions(reply.text, chunk.text, limit=1)
            if not found or found[0].question.lower() in seen:
                dropped += 1
                continue
            seen.add(found[0].question.lower())
            drafts.append(
                {
                    "document_id": uuid.UUID(chunk.document_id),
                    "chunk_id": chunk_id,
                    "doc_key": doc_key,
                    "version": meta.version,
                    "section": section_of(chunk),
                    "query": found[0].question,
                    "evidence": found[0].evidence,
                }
            )
        with self.engine.begin() as conn:
            questions.finish_batch(
                conn,
                batch,
                status="succeeded",
                drafts=drafts,
                counts={"passages": asked, "drafted": len(drafts), "dropped": dropped},
            )
        return {"drafted": len(drafts), "dropped": dropped}


def register_answer_handlers(
    handlers: HandlerRegistry,
    *,
    engine: Engine,
    store: ArtifactStore,
    registry: ComponentRegistry,
    services: ServicesFor,
    settings: Settings,
    indexes: IndexCache | None = None,
) -> None:
    """Register ``answer_run.start`` and ``questions.draft``."""
    shared = indexes or IndexCache(store, engine=engine)
    handlers.register(answers.START_JOB, AnswerRunner(engine, store, registry, services, shared))
    handlers.register(
        questions.DRAFT_JOB, QuestionDrafter(engine, store, registry, services, shared)
    )
