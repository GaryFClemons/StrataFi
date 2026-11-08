"""The Stratafi API's bodies as Pydantic models.

Generated from contracts/openapi.json by scripts/export_sdk_models.py: don't edit, regenerate.
Models ignore fields they don't know, so an older SDK reads a newer API's answers.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class AddedOut(ApiModel):
    version: EvalSetVersionOut
    added: list[CandidateOut]


class AgreementOut(ApiModel):
    dimensions: dict[str, dict[str, Any]] = Field(
        description="Per dimension: labels, agreement with its 95% CI, Cohen's kappa, rates"
    )


class AliasOut(ApiModel):
    kind: Literal["trained_model", "knowledge_profile"]
    name: str
    alias: str
    target_id: uuid.UUID = Field(description="The trained model or profile version it names")
    version: int | None = Field(description="That version's number")
    moved_from: uuid.UUID | None = Field(
        default=None, description="The version it named before, when setting it moved it"
    )


class AliasesOut(ApiModel):
    items: list[AliasOut]


class AnnotationIn(ApiModel):
    "What to change; fields left out stay as they are."

    title: str | None = Field(
        default=None, description="Up to 120 characters; empty or null clears it"
    )
    note: str | None = Field(
        default=None, description="Markdown, up to 20,000 characters; empty or null clears it"
    )
    set_tags: dict[str, str] | None = Field(
        default=None,
        description="Tags to add or replace: keys of letters, digits and _ . - / :, not starting with stratafi.; values up to 500 characters",
    )
    delete_tags: list[str] | None = Field(default=None)
    archived: bool | None = Field(
        default=None, description="true hides it from lists and searches; false restores it"
    )


class AnnotationOut(ApiModel):
    title: str | None
    tags: dict[str, str]
    note: str | None
    archived: bool
    archived_at: datetime.datetime | None
    updated_at: datetime.datetime | None = Field(description="Null until the first change")


class AnswerDetailOut(ApiModel):
    item_id: str
    answer: str
    passages: list[dict[str, Any]] = Field(description="Source ID, document, section, text")
    score: dict[str, Any]
    labels: list[LabelOut]


class AnswerResultOut(ApiModel):
    item_id: str
    ordinal: int
    answer_key: str = Field(description="The hash of the item and its answer: labels attach here")
    score: dict[str, Any] = Field(description="Abstention, citations, reference F1, judge scores")
    labels: list[LabelOut]


class AnswerResultsOut(ApiModel):
    items: list[AnswerResultOut]


class AnswerRunIn(ApiModel):
    eval_run_id: uuid.UUID = Field(description="A succeeded eval run: its queries and passages")
    generator: GeneratorIn
    judge: GeneratorIn | None = Field(
        default=None, description="Scores faithfulness and correctness; none: only the graders"
    )
    passages: int = Field(default=8, description="Final passages the answer reads")


class AnswerRunOut(ApiModel):
    id: uuid.UUID
    eval_run_id: uuid.UUID
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    config: dict[str, Any] = Field(description="Generator, judge, pinned prompts and passages")
    results_cached: bool = Field(description="An identical earlier run's answers were reused")
    metrics: dict[str, Any] | None = Field(description="Each metric's mean with its 95% CI")
    usage: dict[str, Any] = Field(description="Calls, tokens, model revisions and egress")
    error_code: str | None
    created_by: str
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None


class AnswerRunsOut(ApiModel):
    items: list[AnswerRunOut]


class ApprovalOut(ApiModel):
    decision: Literal["approved", "rejected"]
    decided_by: str
    decided_at: datetime.datetime
    self_review: bool = Field(default=False, description="Its author decided it")


class ArtifactInputOut(ApiModel):
    key: str
    role: str
    ordinal: int


class ArtifactOut(ApiModel):
    cache_key: str
    kind: str
    size_bytes: int
    content_sha256: str
    media_type: str
    summary: dict[str, Any]
    created_at: datetime.datetime | None
    inputs: list[ArtifactInputOut] = Field(description="What it was made from, in order")


class Asker(ApiModel):
    "Who asks: what ACL filtering sees of a person."

    name: str | None = Field(default=None, description="A name in the set's askers table")
    user: str | None = Field(default=None, description="Their user ID, for ACLs that list users")
    roles: list[str] | None = Field(default=None)
    groups: list[str] | None = Field(default=None)


class AskerIn(ApiModel):
    name: str | None = Field(
        default=None,
        description="A label for the asker, such as an eval set's asker name; it grants nothing",
    )
    user: str | None = Field(default=None, description="Their user ID, for ACLs that list users")
    roles: list[str] | None = Field(default=None)
    groups: list[str] | None = Field(default=None, description="The ACL groups they belong to")


class AuditEventOut(ApiModel):
    id: int
    occurred_at: datetime.datetime
    actor_kind: str
    actor_subject: str
    action: str
    outcome: str
    resource_type: str | None
    resource_id: str | None
    request_id: str | None
    details: dict[str, Any]


class AuthorIn(ApiModel):
    eval_set: str = Field(description="The set: created if it's new")
    query: str = Field(description="The question")
    evidence: EvidenceRef = Field(description="The passage the question is about")
    asker: str | None = Field(default=None)
    tags: list[str] | None = Field(default=None)


class AuthorOut(ApiModel):
    eval_set: EvalSetOut
    version: EvalSetVersionOut
    item_id: str


class AutomationOut(ApiModel):
    suite: str
    version: int
    events: list[str] = Field(description="The events that run it: corpus_version_frozen...")
    schedules: list[ScheduleOut] = Field(description="Its cron schedules and their next times")
    time_zone: str


class BatchIn(ApiModel):
    run_id: uuid.UUID = Field(description="A succeeded pipeline run: its passages")
    generator: GeneratorIn
    count: int = Field(default=20, description="Questions to draft")
    seed: int = Field(default=0, description="Seeds the passages' sample")


class BatchOut(ApiModel):
    id: uuid.UUID
    eval_set_id: uuid.UUID
    run_id: uuid.UUID = Field(description="The pipeline run drafted from")
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    config: dict[str, Any]
    counts: dict[str, Any] = Field(description="Passages asked, questions drafted and dropped")
    error_code: str | None
    created_by: str
    created_at: datetime.datetime
    finished_at: datetime.datetime | None


class BatchesOut(ApiModel):
    items: list[BatchOut]


class CancelOut(ApiModel):
    outcome: Literal["cancelled", "requested", "finished"]
    job: JobOut


class CandidateOut(ApiModel):
    id: uuid.UUID
    batch_id: uuid.UUID
    status: Literal["proposed", "accepted", "rejected", "added"]
    query: str
    evidence: str = Field(description="The sentence that answers it, from its passage")
    drafted_query: str = Field(description="The query as the generator drafted it")
    edited: bool
    doc_key: str
    version: str | None
    section: str | None
    chunk_id: str | None
    split: Literal["dev", "test"]
    tags: list[str]
    decided_by: str | None
    decided_at: datetime.datetime | None
    item_id: str | None = Field(description="Its item, once added to a version")
    added_version_id: uuid.UUID | None


class CandidatesOut(ApiModel):
    items: list[CandidateOut]


class CatalogModelOut(ApiModel):
    where: Literal["gateway", "in_process", "hosted"]
    component: str
    model: str = Field(description="An alias, a Hugging Face ID, or provider:model")
    title: str
    dims: int | None
    max_tokens: int | None
    license: str
    query_prefix: str = Field(default="")
    document_prefix: str = Field(default="")
    matryoshka: list[int] | None = Field(default=None)
    available: bool
    reason: str | None = Field(default=None)


class CatalogOut(ApiModel):
    models: list[CatalogModelOut]


class CellOut(ApiModel):
    mean: float | None = Field(default=None)
    low: float | None = Field(default=None)
    high: float | None = Field(default=None)
    n: int = Field(default=0)
    delta: float | None = Field(default=None, description="This run minus the baseline")
    delta_low: float | None = Field(default=None)
    delta_high: float | None = Field(default=None)
    p_value: float | None = Field(default=None, description="Paired randomization test")
    p_holm: float | None = Field(default=None, description="Holm-adjusted across the runs")
    wins: int | None = Field(default=None)
    losses: int | None = Field(default=None)
    ties: int | None = Field(default=None)
    group: Literal["leader", "tied", "worse"] | None = Field(
        default=None, description="Against the best run, after Holm's adjustment"
    )
    behind_best: float | None = Field(default=None)


class Change(ApiModel):
    path: str
    op: Literal["added", "removed", "changed"]
    before: Any | None = Field(default=None)
    after: Any | None = Field(default=None)


class ChangeOut(ApiModel):
    path: str
    op: Literal["added", "removed", "changed"]
    before: Any | None = Field(default=None)
    after: Any | None = Field(default=None)


class ChatIn(ApiModel):
    messages: list[ChatMessage] = Field(description="The conversation; the last is the question")
    model: str = Field(default="chat-default", description="A chat model on the gateway")
    run_id: uuid.UUID | None = Field(
        default=None, description="A run whose indexes to search; none chats with the model alone"
    )
    mode: Literal["dense", "keyword", "hybrid"] = Field(default="hybrid")
    retrieval_config_id: uuid.UUID | None = Field(
        default=None, description="A saved retrieval config: replaces mode and top_k"
    )
    retrieval_config: dict[str, Any] | None = Field(
        default=None, description="An unsaved stratafi.retrieval.v1 document"
    )
    top_k: int = Field(default=5, description="Passages to give the model")
    as_of: datetime.date | None = Field(
        default=None,
        description="Search the versions in effect on this day (today by default), so a superseded or future version never answers",
    )
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=600)
    seed: int | None = Field(default=None)
    system_prompt: str | None = Field(default=None)


class ChatMessage(ApiModel):
    role: Literal["user", "assistant"]
    content: str


class ChatModelOut(ApiModel):
    id: str
    revision: str | None = Field(default=None)
    served_model: str | None = Field(default=None)
    parameter_size: str | None = Field(default=None)
    quantization: str | None = Field(default=None)
    context_length: int | None = Field(default=None)


class ChatModelsOut(ApiModel):
    models: list[ChatModelOut]
    default: str | None = Field(description="chat-default, when the gateway serves it")


class Check(ApiModel):
    status: Literal["ok", "failed", "not_configured"]
    code: str | None = Field(default=None)


class CheckOut(ApiModel):
    id: str
    type: str
    result: Literal["pass", "fail", "error", "skipped"]
    code: str | None
    value: float | None
    threshold: dict[str, Any]
    baseline_value: float | None
    detail: dict[str, Any]
    evidence: list[dict[str, Any]]
    duration_ms: float | None


class ChunkOut(ApiModel):
    chunk_id: str
    document_id: str
    ordinal: int
    text: str
    index_text: str
    char_start: int = Field(description="Offset into the cleaned document's text")
    char_end: int
    page_start: int | None
    page_end: int | None
    heading_path: list[str]
    tokens: dict[str, int]
    element_ids: list[str]
    context: str | None = Field(default=None, description="The enricher's context line")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="The chunker's notes: section, table and header_repeated, a parent section",
    )


class ChunkStage(ApiModel):
    component: str = Field(default="chunk/sections")
    params: dict[str, Any] | None = Field(default=None)
    index_text: str = Field(
        default="{title}\n{heading_path}\n{text}",
        description="What each chunk's index text holds: {title}, {heading_path}, {section}, {context} and {text}, one per line; lines that come out empty are dropped",
    )


class CodeFileOut(ApiModel):
    path: str
    sha256: str
    bytes: int


class CompareManyOut(ApiModel):
    runs: list[ComparedRunOut]
    baseline: int = Field(description="The baseline's index")
    rule: str
    checkpoint: str
    metrics: list[str]
    paired: int = Field(description="Queries with evidence every run scored")
    resamples: int
    seed: int
    table: list[MetricRowOut]
    curves: list[CurveOut]
    funnel: FunnelOut
    latency: LatencyOut
    where_lost: WhereLostOut
    violations: list[dict[str, int | None]]
    findability: list[float | None]
    slices: list[SliceOut]
    params: list[ParamDiffOut] = Field(description="The parameters that differ, per run")
    queries: list[QueryRowOut] = Field(
        description="The first metric per query and run; the widest spread first"
    )
    warnings: list[str] = Field(
        description="few_queries: fewer than 50 paired; different_splits: the runs read different splits, and only shared items pair"
    )


class CompareOut(ApiModel):
    a: RunRef
    b: RunRef
    rule: str
    checkpoint: str
    paired: int = Field(description="Queries with evidence that both runs scored")
    resamples: int
    seed: int
    metrics: list[MetricDelta]
    queries: list[QueryDelta] = Field(description="The first metric per query, worst for B first")
    retrieval_diff: list[ChangeOut] = Field(description="How B's retrieval config differs")
    pipeline_diff: list[ChangeOut] = Field(description="How B's pipeline spec differs")
    warnings: list[str] = Field(
        description="few_queries: fewer than 50 paired, so the intervals are wide; different_splits: the runs read different splits, and only shared items pair"
    )


class ComparedRunOut(ApiModel):
    id: uuid.UUID
    label: str = Field(description="Its title, or its config and pipeline")
    title: str | None
    tags: dict[str, str]
    split: str
    created_at: datetime.datetime
    run_id: uuid.UUID = Field(description="The pipeline run it searched")
    retrieval_config: dict[str, Any] = Field(description="id, name and version")
    pipeline_spec: dict[str, Any] = Field(description="id, name and version")
    corpus: dict[str, Any] = Field(description="Name and version")
    k_eval: int | None
    models: dict[str, str | None] = Field(
        description="Role → the model it used: an alias, a catalog model or a trained model"
    )


class ComponentList(ApiModel):
    items: list[ComponentOut]


class ComponentOut(ApiModel):
    id: str
    stage: Literal[
        "extract",
        "scan",
        "parse",
        "clean",
        "chunk",
        "enrich",
        "embed",
        "index",
        "rerank",
        "generate",
        "train",
    ]
    version: str
    title: str
    description: str
    accepts: list[str]
    resource_class: Literal["cpu", "gpu", "io"]
    deterministic: bool
    license: str
    platforms: list[str]
    egress: str
    services: list[str] | None = Field(
        default=None,
        description="Stores and services it uses besides the artifact store and the database ",
    )
    lab_only: bool
    baseline: bool
    params_schema: dict[str, Any]
    available: bool = Field(description="This installation can run it")
    unavailable_reason: str | None = Field(
        description="Why it can't, such as a missing optional extra"
    )


class ComponentRef(ApiModel):
    'A component and its parameters. In a document, a bare ID means "with defaults".'

    component: str
    params: dict[str, Any] | None = Field(default=None)


class ConfirmIn(ApiModel):
    confirm_test_read: Literal[True] = Field(
        description="This reads the locked test split: counted and audited"
    )
    reason: Literal["run_failed", "candidate_changed", "other"] | None = Field(
        default=None, description="Needed for every read after the experiment's first; logged"
    )
    note: str | None = Field(default=None, description="Kept with the confirmation, never logged")


class ConfirmationOut(ApiModel):
    id: uuid.UUID
    ordinal: int = Field(description="Its place among the experiment's reads of the test split")
    trial_id: uuid.UUID
    trial_ordinal: int
    values: dict[str, str]
    eval_run_id: uuid.UUID
    status: Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]
    reason: str | None
    note: str | None
    dev: dict[str, Any] | None = Field(description="The objective on dev, with its 95% CI")
    test: dict[str, Any] | None = Field(description="The objective on test, once it's scored")
    created_by: str
    created_at: datetime.datetime


class ConfirmationsOut(ApiModel):
    items: list[ConfirmationOut]
    test_reads: int = Field(description="How often the eval set version's test split was read")


class CorpusDetail(ApiModel):
    id: uuid.UUID
    name: str
    description: str
    created_by: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    entries: int = Field(description="Entries in the working set")
    total_bytes: int
    latest_version: VersionOut | None
    duplicate_groups: list[list[str]] = Field(
        description="Working-set names that share content; scan/dedupe keeps each group's first"
    )


class CorpusIn(ApiModel):
    name: str
    description: str = Field(default="")


class CorpusOut(ApiModel):
    id: uuid.UUID
    name: str
    description: str
    created_by: str
    created_at: datetime.datetime
    updated_at: datetime.datetime


class Count(ApiModel):
    count: int


class CreatedTokenOut(ApiModel):
    id: uuid.UUID
    name: str
    prefix: str = Field(description="The token's first characters, to tell tokens apart")
    scopes: list[str]
    active: bool
    created_by: str
    created_at: datetime.datetime
    expires_at: datetime.datetime
    revoked_at: datetime.datetime | None
    last_used_at: datetime.datetime | None
    token: str = Field(description="The token. Shown this once: keep it in a secret store")


class CreatedWebhookOut(ApiModel):
    id: uuid.UUID
    url: str
    suite: str | None
    events: list[str]
    enabled: bool
    created_by: str
    created_at: datetime.datetime
    secret: str = Field(description="Signs every delivery. Shown this once: keep it now")


class CurveOut(ApiModel):
    metric: str
    ks: list[str]
    series: list[list[float | None]] = Field(description="One per run: the mean at each k")


class DeliveriesOut(ApiModel):
    items: list[DeliveryOut]


class DeliveryOut(ApiModel):
    id: uuid.UUID
    event: str
    test_run_id: uuid.UUID | None
    status: Literal["pending", "delivered", "failed"]
    attempts: int
    response_status: int | None
    error_code: str | None
    created_at: datetime.datetime
    delivered_at: datetime.datetime | None


class DenseRetrieverSettings(ApiModel):
    top_n: int = Field(default=50, description="Results each retriever returns")
    adapter: str | None = Field(
        default=None, description="A trained query adapter's model ID, applied to query vectors"
    )


class DiffHunkOut(ApiModel):
    rule: str
    element_id: str
    start: int = Field(description="Offset into the rule's input text")
    end: int
    kind: str = Field(description="The edit's replacement kind")
    before: str = Field(description="The edited span with context; redacted values masked")
    after: str
    redaction: bool


class DiffOut(ApiModel):
    cache_key: str
    parsed_key: str = Field(description="The parsed document the edits apply to")
    rules: list[RuleCountOut] = Field(description="Edits by rule, in the order they ran")
    hunks: list[DiffHunkOut]
    total_edits: int
    truncated: bool = Field(description="More edits than hunks shown")
    before_chars: int
    after_chars: int
    language: str | None
    warnings: list[str]


class DocumentDetail(ApiModel):
    id: uuid.UUID
    name: str = Field(description="The name it first arrived with")
    sha256: str
    size_bytes: int
    extension: str
    media_type: str = Field(description="What its bytes are")
    kind: str
    mismatch: bool = Field(description="Its bytes disagree with its extension")
    known: bool = Field(description="Its extension is one Stratafi lists")
    traits: dict[str, Any]
    status: str = Field(description="stored, or refused (an archive that broke a limit)")
    reason_code: str | None
    created_at: datetime.datetime
    origins: list[OriginOut]


class DocumentOut(ApiModel):
    id: uuid.UUID
    name: str = Field(description="The name it first arrived with")
    sha256: str
    size_bytes: int
    extension: str
    media_type: str = Field(description="What its bytes are")
    kind: str
    mismatch: bool = Field(description="Its bytes disagree with its extension")
    known: bool = Field(description="Its extension is one Stratafi lists")
    traits: dict[str, Any]
    status: str = Field(description="stored, or refused (an archive that broke a limit)")
    reason_code: str | None
    created_at: datetime.datetime


class EntriesIn(ApiModel):
    entries: list[EntryIn]


class EntryIn(ApiModel):
    document_id: uuid.UUID
    name: str | None = Field(default=None, description="Defaults to the document's name")


class EntryOut(ApiModel):
    name: str
    document_id: uuid.UUID
    sha256: str
    size_bytes: int
    media_type: str
    traits: dict[str, Any]
    status: str
    reason_code: str | None
    doc_key: str | None = Field(default=None)
    doc_version: str | None = Field(default=None)
    collection: str | None = Field(default=None)
    acl: dict[str, Any] | None = Field(default=None)
    effective_from: datetime.date | None = Field(default=None)
    effective_to: datetime.date | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)


class EvalConfiguration(ApiModel):
    retrieval: dict[str, Any] = Field(description="The resolved retrieval config")
    retrieval_hash: str
    retrieval_ref: dict[str, Any] = Field(description="The saved config: id, name and version")
    options: dict[str, Any]
    k_eval: int = Field(description="The cutoff the eval set's own rule uses")
    ks: list[int]
    engine_version: str
    scorer_version: str


class EvalExecution(ApiModel):
    status: Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]
    started_at: datetime.datetime | None = Field(default=None)
    finished_at: datetime.datetime | None = Field(default=None)
    duration_ms: float | None = Field(default=None)
    error_code: str | None = Field(default=None)
    results_cached: bool = Field(default=False)
    shards: list[EvalShardRecord] | None = Field(default=None)


class EvalInputs(ApiModel):
    pipeline_run: dict[str, Any] = Field(description="The run searched: id, manifest hash, corpus")
    indexes: dict[str, str] = Field(description="Index kind → its artifact's cache key")
    eval_set: dict[str, Any] = Field(description="Name, version, hash, split, items, as_of")


class EvalItemOut(ApiModel):
    id: str
    ordinal: int
    query: str
    asker: dict[str, Any]
    expected: list[dict[str, Any]]
    avoid: list[dict[str, Any]]
    must_not_return: list[dict[str, Any]]
    near_misses: list[dict[str, Any]]
    reference_answer: str | None
    tags: list[str]
    split: Literal["dev", "test"]


class EvalOptions(ApiModel):
    "How an eval run scores; recorded with it and part of its results' key."

    shard_size: int = Field(default=25, description="Queries per shard job")
    coverage_threshold: float = Field(
        default=0.7, description="evidence-coverage-v1: the grade at which a chunk covers evidence"
    )
    resamples: int = Field(default=2000, description="Bootstrap resamples")
    seed: int = Field(default=0, description="Seeds the bootstrap")
    sample: EvalSample | None = Field(
        default=None, description="Score a seeded sample of the split's items instead of all"
    )


class EvalOutput(ApiModel):
    cache_key: str
    kind: str
    size_bytes: int


class EvalRunIn(ApiModel):
    run_id: uuid.UUID = Field(description="A succeeded pipeline run with its indexes")
    retrieval_config_id: uuid.UUID = Field(description="A saved retrieval config version")
    eval_set_version_id: uuid.UUID = Field(description="A frozen eval set version")
    split: Literal["dev", "test", "all"] = Field(
        default="dev", description="dev for tuning; test only to confirm"
    )
    confirm_test_read: bool = Field(
        default=False,
        description="Reading the locked test split is a confirmation, counted and audited",
    )
    options: EvalOptions | None = Field(default=None)


class EvalRunInfo(ApiModel):
    id: str
    split: Literal["dev", "test", "all"]
    test_read: bool = Field(description="The run read the locked test split")
    created_by: str
    created_at: datetime.datetime


class EvalRunManifest(ApiModel):
    "``stratafi.eval-run.v1``."

    schema_: Literal["stratafi.eval-run.v1"] = Field(default="stratafi.eval-run.v1", alias="schema")
    eval_run: EvalRunInfo
    configuration: EvalConfiguration
    inputs: EvalInputs
    models: list[ManifestModel] | None = Field(default=None)
    environment: ManifestEnvironment
    hardware: ManifestHardware
    execution: EvalExecution
    cost: ManifestCost | None = Field(default=None)
    egress: list[ManifestEgress] | None = Field(default=None)
    outputs: list[EvalOutput] | None = Field(default=None)


class EvalRunOut(ApiModel):
    id: uuid.UUID
    status: Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]
    run_id: uuid.UUID = Field(description="The pipeline run searched")
    retrieval_config_id: uuid.UUID
    eval_set_version_id: uuid.UUID
    split: Literal["dev", "test", "all"]
    options: dict[str, Any]
    retrieval_hash: str
    test_read: bool
    results_cached: bool = Field(description="An identical earlier run's results were reused")
    retrieval_ref: dict[str, Any] | None = Field(description="The config's name and version")
    eval_set: dict[str, Any] | None = Field(description="The eval set's name, version, items")
    k_eval: int | None = Field(description="The cutoff the eval set's own rule uses")
    counts: dict[str, Any]
    metrics: dict[str, Any] | None = Field(
        description="Aggregates with 95% intervals, slices, latency, findability and cost"
    )
    manifest_hash: str | None
    error_code: str | None
    created_by: str
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None


class EvalRunSearchIn(ApiModel):
    filter: str = Field(
        default="",
        description="Clauses joined by `and`: `metrics.recall@5 > 0.8 and params.rerank.component = 'rerank/none' and tags.owner = 'ana'`",
    )
    order_by: list[str] | None = Field(
        default=None,
        description="Keys with ASC or DESC, such as `metrics.recall@5 DESC`; newest first by default",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Keys whose values each item carries, such as `params.chunk.component`",
    )
    include_archived: bool = Field(default=False)
    limit: int = Field(default=50)
    cursor: str | None = Field(default=None)


class EvalRunSearchKeys(ApiModel):
    metrics: list[str]
    params: list[str]
    tags: list[str]


class EvalRunSearchOut(ApiModel):
    items: list[TrackedRunOut]
    next_cursor: str | None
    total: int = Field(description="Runs the filter matched")
    scanned: int = Field(description="Runs searched: the newest 5,000 at most")
    keys: EvalRunSearchKeys = Field(
        description="Every metric, parameter and tag key the matches have"
    )


class EvalSample(ApiModel):
    "The first ``size`` of the split's items in a seeded order (every item without one)."

    seed: int = Field(default=0, description="Seeds the order")
    size: int | None = Field(default=None, description="Items scored")


class EvalSetImport(ApiModel):
    name: str = Field(description="The eval set: created if it's new")
    description: str | None = Field(default=None)
    format: Literal["yaml", "jsonl", "csv"] = Field(
        default="yaml",
        description="yaml: stratafi.eval-set.v1, or the retrieval YAML format unchanged; jsonl: one item per line; csv: one row per evidence item",
    )
    content: str
    as_of: datetime.date | None = Field(
        default=None, description='Overrides the document\'s "today"'
    )
    k: int | None = Field(default=None, description="Overrides defaults.k")
    askers: dict[str, Any] | None = Field(
        default=None,
        description="Overrides the askers table: who each asker name in the items is (a CSV has no table of its own)",
    )
    test_fraction: float = Field(
        default=0.3, description="The share of items to put in the locked test split"
    )
    seed: int = Field(default=0, description="Seeds the stratified split")
    resplit: bool = Field(default=False, description="Split items that name a split too")
    freeze: bool = Field(default=True, description="Freeze the version at once")


class EvalSetOut(ApiModel):
    id: uuid.UUID
    name: str
    description: str
    created_by: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    latest_version: EvalSetVersionOut | None


class EvalSetVersionOut(ApiModel):
    id: uuid.UUID
    eval_set_id: uuid.UUID
    version: int
    document: dict[str, Any] = Field(description="The set's own fields: as_of, defaults, askers")
    document_hash: str
    item_count: int
    split_counts: dict[str, int]
    split_params: dict[str, Any] | None = Field(description="How the split was made: seed, share")
    source: dict[str, Any] = Field(description="Where the items came from: format and counts")
    frozen: bool
    frozen_at: datetime.datetime | None
    test_locked: bool = Field(description="Evaluating the test split is a counted confirmation")
    test_reads: int = Field(description="How often the test split has been evaluated")
    last_test_read_at: datetime.datetime | None
    created_by: str
    created_at: datetime.datetime


class EvalShardRecord(ApiModel):
    ordinal: int
    cache_key: str
    items: int
    status: str
    cache_hit: bool
    duration_ms: float | None = Field(default=None)
    error_code: str | None = Field(default=None)


class EvidenceRef(ApiModel):
    "Gold evidence: a document version, a section, and a phrase from it."

    doc_key: str
    version: str | None = Field(default=None, description="None: any version")
    section: str | None = Field(default=None)
    evidence: str | None = Field(default=None)
    source_uri: str | None = Field(default=None)


class ExperimentIn(ApiModel):
    "An experiment as a document, or as YAML text: one of the two."

    document: dict[str, Any] | None = Field(
        default=None, description="The stratafi.experiment.v1 document"
    )
    content: str | None = Field(default=None, description="The same, as YAML")


class ExperimentOut(ApiModel):
    id: uuid.UUID
    name: str
    status: Literal[
        "planned",
        "running",
        "stopping",
        "succeeded",
        "stopped",
        "failed",
        "cancelling",
        "cancelled",
    ]
    document: dict[str, Any] = Field(description="The stratafi.experiment.v1 it runs")
    document_hash: str
    corpus_version_id: uuid.UUID
    eval_set_version_id: uuid.UUID
    pipeline_spec_id: uuid.UUID = Field(description="The base pipeline spec version")
    retrieval_config_id: uuid.UUID = Field(description="The base retrieval config version")
    plan: dict[str, Any]
    spent: dict[str, Any] = Field(description="Estimated cost so far, and hours since the start")
    counts: dict[str, Any] = Field(description="Trials by status")
    stop_reason: str | None = Field(description="The budget that stopped it")
    error_code: str | None
    created_by: str
    started_by: str | None
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None


class ExtractStage(ApiModel):
    component: ComponentRef | str | None = Field(
        default=None,
        description="Reads each document's stored bytes, checks its hash and detects its type",
    )
    unpack: list[ComponentRef | str] | None = Field(
        default=None, description="Unpackers applied when documents arrive"
    )


class FeatureIn(ApiModel):
    enabled: bool


class FeatureOut(ApiModel):
    name: Literal["custom_checks"]
    enabled: bool
    blocked: str | None = Field(
        description="Why it can't be turned on here: sandbox_not_configured or sandbox_not_reviewed; null when it can"
    )
    review: str | None = Field(description="The security review it was turned on under")
    changed_by: str | None
    changed_at: datetime.datetime | None


class FeaturesOut(ApiModel):
    items: list[FeatureOut]


class FileTypeOut(ApiModel):
    media_type: str
    short: str = Field(description="The name parse routes use, e.g. pdf")
    family: str
    title: str
    extensions: list[str]


class FileTypesOut(ApiModel):
    types: list[FileTypeOut]
    traits: list[TraitOut]


class Filters(ApiModel):
    collections: list[str] | None = Field(default=None)
    current_only: bool = Field(
        default=True, description="Only versions in effect on the as-of date"
    )
    statuses: list[str] | None = Field(default=None)


class FolderImport(ApiModel):
    root: str = Field(description="A name from intake.folder_roots")
    path: str = Field(default="", description="A folder below the root")
    corpus_id: uuid.UUID | None = Field(default=None)


class FunnelOut(ApiModel):
    checkpoints: list[str]
    hit: list[list[float | None]] = Field(
        description="Per run and checkpoint: the share of queries with a relevant chunk anywhere"
    )
    recall: list[list[float | None]]


class Fusion(ApiModel):
    method: Literal["rrf", "weighted", "none"] = Field(default="rrf")
    k: int = Field(default=60, description="RRF's k")
    alpha: float = Field(default=0.5, description="Weighted fusion: the dense weight")


class GeneratorIn(ApiModel):
    component: str = Field(description="A generate/ component, such as generate/gateway")
    params: dict[str, Any] | None = Field(default=None)


class Health(ApiModel):
    status: Literal["ok", "unavailable"]
    version: str
    checks: dict[str, Check]


class HistogramOut(ApiModel):
    edges: list[float]
    counts: list[int]


class HistoryOut(ApiModel):
    suite: str
    runs: list[dict[str, Any]] = Field(
        description="Finished runs, oldest first, with each check's result and value"
    )


class Hit(ApiModel):
    rank: int
    chunk_id: str
    score: float
    document_id: str
    entry: str | None = Field(description="The document's name in the run's corpus version")
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    text: str
    highlights: list[tuple[int, int]] | None = Field(
        default=None, description="Where the asked evidence sits in the text"
    )


class ImportOut(ApiModel):
    eval_set: EvalSetOut
    version: EvalSetVersionOut
    created: bool = Field(description="False: the same set is already this version")
    report: ImportReport


class ImportReport(ApiModel):
    items: int
    derived_tags: int = Field(description="Items whose tags were derived")
    assigned_splits: int = Field(description="Items this import split")
    splits: dict[str, int]


class IndexStage(ApiModel):
    dense: ComponentRef | str | None = Field(default=None)
    keyword: ComponentRef | str | None = Field(default=None)


class IngestOut(ApiModel):
    source_id: uuid.UUID
    files: list[IngestedOut]
    added_to_corpus: int
    counts: dict[str, int]


class IngestedOut(ApiModel):
    name: str
    document: DocumentOut
    deduplicated: bool = Field(description="These bytes were already in the workspace")
    refused: dict[str, str] | None = Field(
        default=None, description="A refused archive: its outcome and reason"
    )
    members: list[IngestedOut] | None = Field(default=None)


class IntakeSourcesOut(ApiModel):
    folder_roots: list[str] = Field(description="Names of server folders (extract/folder)")
    s3_sources: list[str] = Field(description="Names of buckets (extract/s3)")
    url_import: bool = Field(description="Whether URLs can be fetched (extract/url)")
    unpack_archives: bool
    unpack_email: bool


class Issue(ApiModel):
    path: str
    code: str


class IssueOut(ApiModel):
    path: str
    code: str


class JobOut(ApiModel):
    id: uuid.UUID
    kind: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    resource_class: Literal["cpu", "gpu", "io"]
    priority: int
    attempts: int
    max_attempts: int
    cancel_requested: bool
    progress: dict[str, Any] | None
    error_code: str | None
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None


class LabelIn(ApiModel):
    item_id: str
    dimension: Literal["faithful", "correct", "abstained"]
    label: bool


class LabelOut(ApiModel):
    dimension: Literal["faithful", "correct", "abstained"]
    label: bool
    labeled_by: str
    labeled_at: datetime.datetime


class LatencyOut(ApiModel):
    steps: list[str]
    p50: list[list[float | None]]
    p95: list[list[float | None]]
    mean: list[list[float | None]]


class LatencyProfileOut(ApiModel):
    eval_run_id: uuid.UUID
    queries: int
    steps: dict[str, StepProfileOut] = Field(description="By step: embed_ms, ..., total_ms")
    slowest: list[SlowQueryOut]


class Liveness(ApiModel):
    status: Literal["ok"]


class ManifestComponent(ApiModel):
    path: str = Field(description="Where the spec names it, e.g. stages.chunk")
    id: str
    version: str
    params: dict[str, Any]
    params_hash: str


class ManifestConfiguration(ApiModel):
    pipeline: dict[str, Any] = Field(description="The resolved pipeline spec")
    pipeline_hash: str
    pipeline_ref: dict[str, Any] | None = Field(
        default=None, description="The saved spec it came from: name and version"
    )
    retrieval: dict[str, Any] | None = Field(default=None)
    retrieval_hash: str | None = Field(default=None)
    components: list[ManifestComponent] | None = Field(default=None)


class ManifestCorpus(ApiModel):
    corpus_id: str
    corpus_version_id: str
    version: int
    manifest_hash: str


class ManifestCost(ApiModel):
    providers: dict[str, ProviderUsage] | None = Field(default=None)
    currency: str = Field(default="USD")
    estimated: float | None = Field(default=None)
    price_table: dict[str, float] | None = Field(
        default=None, description="Price per 1,000 tokens by provider, at run time"
    )


class ManifestDocument(ApiModel):
    entry: str = Field(description="The document's name in the corpus version")
    document_id: str
    sha256: str
    size_bytes: int
    media_type: str
    traits: dict[str, Any] | None = Field(default=None)


class ManifestEgress(ApiModel):
    provider: str
    documents: list[str] | None = Field(default=None, description="Document IDs")


class ManifestEnvironment(ApiModel):
    stratafi_version: str
    commit: str | None = Field(default=None)
    python: str
    libraries: dict[str, str] | None = Field(default=None)
    os: str
    container_image: str | None = Field(default=None)


class ManifestExecution(ApiModel):
    status: Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]
    started_at: datetime.datetime | None = Field(default=None)
    finished_at: datetime.datetime | None = Field(default=None)
    duration_ms: float | None = Field(default=None)
    error_code: str | None = Field(default=None)
    stages: list[StageExecution] | None = Field(default=None)
    tasks: list[TaskExecution] | None = Field(default=None)


class ManifestGpu(ApiModel):
    model: str
    memory_bytes: int | None = Field(default=None)
    driver: str | None = Field(default=None)


class ManifestHardware(ApiModel):
    host: str | None = Field(default=None, description="The worker host's name, if recorded")
    cpu_model: str | None = Field(default=None)
    cpu_cores: int | None = Field(default=None)
    ram_bytes: int | None = Field(default=None)
    gpus: list[ManifestGpu] | None = Field(default=None)
    resource_classes: list[str] | None = Field(default=None)


class ManifestImport(ApiModel):
    root: str = Field(description="A name from intake.folder_roots")
    path: str = Field(default="", description="A folder below the root that holds manifest.yaml")
    corpus_id: uuid.UUID = Field(description="The corpus whose working set the entries join")


class ManifestInputs(ApiModel):
    corpus: ManifestCorpus | None = Field(default=None)
    documents: list[ManifestDocument] | None = Field(default=None)
    sample: ManifestSample | None = Field(default=None)
    eval_set: dict[str, Any] | None = Field(default=None, description="Null for a pipeline run")


class ManifestModel(ApiModel):
    role: str = Field(description="embed, rerank, ...")
    component: str
    alias: str | None = Field(default=None)
    served_model: str | None = Field(default=None)
    revision: str | None = Field(default=None)
    dimensions: int | None = Field(default=None)
    dtype: str | None = Field(default=None)
    prefixes: dict[str, str] | None = Field(default=None)
    tokenizer: str | None = Field(default=None)


class ManifestOutput(ApiModel):
    cache_key: str
    kind: str
    stage: str
    size_bytes: int
    summary: dict[str, Any] | None = Field(default=None)


class ManifestRun(ApiModel):
    id: str
    kind: Literal["pipeline"] = Field(default="pipeline")
    mode: Literal["interactive", "batch"] = Field(default="batch")
    until: str | None = Field(default=None, description="The last stage the run goes to")
    rerun_of: str | None = Field(default=None)
    rerun_from: str | None = Field(default=None, description="The stage a re-run started from")
    created_by: str
    created_at: datetime.datetime


class ManifestSample(ApiModel):
    size: int | None = Field(default=None)
    seed: int | None = Field(default=None)
    entries: list[str] | None = Field(default=None)


class MapEdgeOut(ApiModel):
    id: str
    source: str
    target: str
    kind: str = Field(description="flow, read, write or call")
    label: str
    sends_text: bool = Field(description="It sends document or query text out of the lab")


class MapNodeOut(ApiModel):
    id: str
    kind: str = Field(description="input, stage, query, output, datastore or service")
    lane: str = Field(description="services, ingest, query or stores")
    label: str
    column: int
    row: int
    stage: str | None
    component: str | None
    version: str | None
    params: dict[str, Any]
    params_hash: str | None
    description: str | None
    egress: str | None
    lab_only: bool
    noop: bool = Field(description="It passes its input through: enrich/none, rerank/none")
    configured: bool | None = Field(description="This deployment configures the store or service")
    backend: str | None = Field(description="A datastore's kind: postgres, s3, filesystem, files")
    details: dict[str, Any]
    overlay: dict[str, Any] | None = Field(
        default=None, description="A run's numbers for this node, on a run's or eval run's map"
    )


class MapOut(ApiModel):
    schema_: str = Field(alias="schema")
    pipeline: dict[str, Any] | None = Field(description="The pipeline spec: id, name, version")
    retrieval: dict[str, Any] | None = Field(description="The retrieval config: id, name, version")
    nodes: list[MapNodeOut]
    edges: list[MapEdgeOut]
    summary: dict[str, Any] = Field(
        description="Stages, datastores, services, where text goes, and lab-only components"
    )
    mermaid: str = Field(description="The map as a Mermaid flowchart")
    run_id: uuid.UUID | None = Field(default=None)
    eval_run_id: uuid.UUID | None = Field(default=None)


class MatrixCellOut(ApiModel):
    delta: float = Field(description="The column's model minus the row's, averaged over pairs")
    pairs: int


class Me(ApiModel):
    subject: str
    tenant: str
    roles: list[str]
    scopes: list[str] | None = Field(
        default=None, description="An API token's scopes; none for people, whose roles decide"
    )
    workspace: WorkspaceOut


class MetricDelta(ApiModel):
    metric: str
    n: int = Field(description="Queries paired for this metric")
    mean_a: float | None = Field(default=None)
    mean_b: float | None = Field(default=None)
    delta: float | None = Field(default=None, description="B minus A")
    low: float | None = Field(default=None, description="The 95% interval of the delta")
    high: float | None = Field(default=None)
    p_value: float | None = Field(default=None, description="Paired randomization test")
    p_holm: float | None = Field(default=None, description="Holm-adjusted across the metrics")
    wins: int | None = Field(default=None, description="Queries where B scored higher")
    losses: int | None = Field(default=None)
    ties: int | None = Field(default=None)


class MetricRowOut(ApiModel):
    metric: str
    n: int = Field(description="Queries paired for this metric")
    best: int | None = Field(description="The best run's index")
    cells: list[CellOut] = Field(description="One per run, in the runs' order")


class ModelComparisonOut(ApiModel):
    eval_set_version_id: uuid.UUID
    role: str
    metric: str
    rule: str
    checkpoint: str
    split: str | None
    eval_runs: int = Field(description="Succeeded eval runs considered")
    models: list[ModelRowOut] = Field(description="Best first")
    pairs: list[ModelPairOut]
    matrix: list[list[MatrixCellOut | None]] = Field(
        description="matrix[i][j]: model j against model i, in the models' order"
    )
    warnings: list[str] = Field(
        description="no_controlled_pairs: no two runs differ only in the model; few_queries: a pair shares fewer than 50 queries"
    )


class ModelPairOut(ApiModel):
    a_eval_run_id: uuid.UUID
    b_eval_run_id: uuid.UUID
    a_model: str
    b_model: str
    result: dict[str, Any] | None = Field(
        description="B against A on shared queries: means, delta with its interval, p-value, wins, losses and ties"
    )


class ModelRowOut(ApiModel):
    key: str = Field(description="role|component|name|revision")
    label: str
    role: str
    component: str
    name: str = Field(description="A gateway alias, a catalog model or a trained model")
    revision: str | None
    runs: int
    best_eval_run_id: uuid.UUID | None
    best: dict[str, Any] | None = Field(description="The best run's mean, low, high and n")
    latency_p95_ms: float | None = Field(description="The median of its runs' p95 latency")
    egress: str | None


class NamesIn(ApiModel):
    names: list[str]


class NeighbourOut(ApiModel):
    chunk_id: str
    score: float
    document_id: str
    entry: str | None
    text: str


class NeighboursOut(ApiModel):
    chunk_id: str
    neighbours: list[NeighbourOut]


class OriginOut(ApiModel):
    source_id: uuid.UUID
    source_kind: str
    name: str
    parent_document_id: uuid.UUID | None
    member_path: str | None
    created_at: datetime.datetime


class PageAuditEventOut(ApiModel):
    items: list[AuditEventOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageChunkOut(ApiModel):
    items: list[ChunkOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageCorpusOut(ApiModel):
    items: list[CorpusOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageDocumentOut(ApiModel):
    items: list[DocumentOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageEntryOut(ApiModel):
    items: list[EntryOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageEvalItemOut(ApiModel):
    items: list[EvalItemOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageEvalRunOut(ApiModel):
    items: list[EvalRunOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageEvalSetOut(ApiModel):
    items: list[EvalSetOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageEvalSetVersionOut(ApiModel):
    items: list[EvalSetVersionOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageExperimentOut(ApiModel):
    items: list[ExperimentOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageJobOut(ApiModel):
    items: list[JobOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PagePipelineSpecVersion(ApiModel):
    items: list[PipelineSpecVersion]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageProfileOut(ApiModel):
    items: list[ProfileOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageQueryResultOut(ApiModel):
    items: list[QueryResultOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageRetrievalConfigVersion(ApiModel):
    items: list[RetrievalConfigVersion]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageRunEntryOut(ApiModel):
    items: list[RunEntryOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageRunOut(ApiModel):
    items: list[RunOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageSuiteOut(ApiModel):
    items: list[SuiteOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageTaskOut(ApiModel):
    items: list[TaskOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageTestRunOut(ApiModel):
    items: list[TestRunOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class PageVersionOut(ApiModel):
    items: list[VersionOut]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


class ParamDiffOut(ApiModel):
    key: str
    values: list[Any]


class ParseStage(ApiModel):
    route: dict[str, ComponentRef | str] | None = Field(
        default=None,
        description='File type → parser. Keys are "default", a media type, or a short type with traits such as "pdf[scanned]"',
    )


class PassageOut(ApiModel):
    rank: int
    chunk_id: str
    score: float
    document_id: str
    entry: str | None
    doc_key: str | None
    version: str | None
    section: str | None
    heading_path: list[str]
    text: str
    relevant: dict[str, bool] = Field(description="By match rule")
    grades: list[float] = Field(description="evidence-coverage-v1, per evidence item")
    highlights: list[list[int]] = Field(
        description="Evidence spans in text: [start, end) character offsets"
    )


class PipelineSpec(ApiModel):
    "``stratafi.pipeline.v1``."

    schema_: Literal["stratafi.pipeline.v1"] = Field(alias="schema")
    name: str
    description: str = Field(default="")
    stages: Stages | None = Field(default=None)


class PipelineSpecValidation(ApiModel):
    valid: bool
    issues: list[Issue]
    spec_hash: str | None = Field(default=None)
    document: PipelineSpec | None = Field(default=None)


class PipelineSpecVersion(ApiModel):
    id: uuid.UUID
    name: str
    version: int
    spec_hash: str
    document_hash: str
    created_by: str
    created_at: datetime.datetime
    document: PipelineSpec


class PlanOut(ApiModel):
    valid: bool
    issues: list[IssueOut]
    document: dict[str, Any] | None = Field(description="The experiment, every default filled in")
    plan: dict[str, Any] | None = Field(
        description="Factors, pinned inputs, unique work per stage, estimate and budget check"
    )
    trials: list[PlannedTrialOut]


class PlannedTrialOut(ApiModel):
    ordinal: int
    grid_index: int
    levels: list[int]
    values: dict[str, str] = Field(description="Factor path → the level's label")
    pipeline_hash: str
    retrieval_hash: str
    evaluated: bool = Field(description="An identical eval run has finished: its results reuse")


class ProblemDetails(ApiModel):
    "An RFC 9457 problem. ``code``, ``errors`` and ``issues`` are Stratafi's extension\nmembers."

    type: str
    title: str
    status: int
    detail: str
    instance: str | None = Field(default=None)
    request_id: str | None = Field(default=None)
    code: str | None = Field(default=None)
    errors: list[str] | None = Field(default=None, description="Paths of the invalid fields")
    issues: list[ProblemIssue] | None = Field(
        default=None, description="Problems in a submitted document: paths and codes"
    )


class ProblemIssue(ApiModel):
    "One problem inside a document the client sent: a dotted path and a code, never a value."

    path: str
    code: str


class ProfileDiffOut(ApiModel):
    against: dict[str, Any] | None = Field(description="The version compared with: ID, version")
    changes: list[ChangeOut] = Field(description="Pipeline and retrieval, by path")


class ProfileFromTrialIn(ApiModel):
    experiment_id: uuid.UUID
    trial_id: uuid.UUID = Field(description="A trial with a dev result and a test confirmation")
    name: str
    description: str = Field(default="")


class ProfileImportIn(ApiModel):
    "A stratafi.knowledge-profile.v1, as a document or as YAML or JSON text: one of the two."

    document: dict[str, Any] | None = Field(default=None)
    content: str | None = Field(default=None)


class ProfileOut(ApiModel):
    id: uuid.UUID
    name: str
    version: int
    status: Literal["draft", "approved", "rejected"]
    source: Literal["trial", "import"]
    document: dict[str, Any] = Field(description="The stratafi.knowledge-profile.v1")
    document_hash: str = Field(description="The profile's hash; its review doesn't change it")
    experiment_id: uuid.UUID | None
    trial_id: uuid.UUID | None
    approvals: list[ApprovalOut]
    created: bool | None = Field(default=None, description="False: the latest version holds it")
    created_by: str
    created_at: datetime.datetime
    aliases: list[str] | None = Field(
        default=None, description="The aliases that name this version"
    )


class ProjectionOut(ApiModel):
    method: Literal["pca", "umap"]
    points: list[ProjectionPoint]
    total: int = Field(description="Vectors in the index; points is a seeded sample")
    explained_variance: list[float] | None = Field(default=None, description="PCA only")


class ProjectionPoint(ApiModel):
    chunk_id: str
    x: float
    y: float
    document_id: str
    entry: str | None
    media_type: str | None


class ProviderUsage(ApiModel):
    calls: int = Field(default=0)
    tokens: int = Field(default=0)


class PurgeOut(ApiModel):
    document_id: uuid.UUID
    status: Literal["withdrawn"]
    artifacts: int = Field(description="Artifacts derived from it that went, with their parts")
    tables: int = Field(description="Postgres index tables built with it that went")
    working_sets: int = Field(description="Corpora whose working set lost it")
    candidates: int = Field(description="Question candidates drafted from it that went")
    models: int = Field(
        default=0, description="Trained models withdrawn: their training data quoted it"
    )
    objects: int = Field(description="Stored objects deleted")


class QueryDelta(ApiModel):
    item_id: str
    a: float
    b: float
    delta: float


class QueryDetailOut(ApiModel):
    result: QueryResultOut
    item: dict[str, Any] = Field(description="The eval item: query, asker, evidence, tags")
    checkpoints: dict[str, list[PassageOut]] = Field(
        description="Each checkpoint's list, its first 20 passages"
    )
    lengths: dict[str, int] = Field(description="Each checkpoint's full length")


class QueryResultOut(ApiModel):
    item_id: str
    ordinal: int
    split: str
    tags: list[str]
    file_types: list[str]
    has_evidence: bool
    hit: bool | None = Field(description="knowledge-hit-v1 at the eval set's k, in the final list")
    first_hit_rank: int | None
    lost_at: Literal["parse", "retrieval", "rerank"] | None = Field(
        description="Where a miss lost its evidence"
    )
    avoid_violation: bool | None
    must_not_return_violation: bool
    score: dict[str, Any] = Field(description="Every metric at every checkpoint, and diagnoses")


class QueryRowOut(ApiModel):
    item_id: str
    values: list[float | None]
    spread: float


class RagPipelineOut(ApiModel):
    pipeline_spec: dict[str, Any] = Field(description="id, name and version")
    retrieval_config: dict[str, Any] = Field(description="id, name and version")
    eval_runs: int
    eval_sets: list[str] = Field(description="The eval sets it was measured on: name@version")
    latest_eval_run_id: uuid.UUID
    latest_at: datetime.datetime
    best: dict[str, Any] | None = Field(
        description="Its best eval run on the headline recall: id, metric, mean, eval set"
    )
    components: list[str] = Field(description="Every component it uses, ingest and query")
    services: list[str] = Field(description="The services it calls, from its map")
    datastores: list[str]
    sends_text_to: list[str]


class RagPipelinesOut(ApiModel):
    items: list[RagPipelineOut]


class RerunIn(ApiModel):
    from_stage: Literal[
        "extract", "scan", "parse", "clean", "chunk", "enrich", "embed", "index"
    ] = Field(description="This stage and every later one run again")


class ResultsOut(ApiModel):
    objective: dict[str, Any] = Field(description="The metric, rule, checkpoint and goal shown")
    k_eval: int = Field(description="The eval set's cutoff")
    trials: int = Field(description="Trials with results")
    queries: int = Field(description="Queries with evidence every trial scored")
    leaderboard: list[dict[str, Any]] = Field(description="Best first, ranked")
    effects: list[dict[str, Any]] = Field(
        description="Per factor, each level's paired effect against the first, with a 95% CI"
    )
    interactions: list[dict[str, Any]] = Field(description="Per pair of factors, cell means")
    pareto: dict[str, list[dict[str, Any]]] = Field(
        description="The objective against p95 latency and cost per 1,000 queries; frontiers"
    )
    file_types: dict[str, Any] = Field(
        description="Each gold file type's slice per trial, and per parse factor a routing table"
    )
    where_lost: dict[str, Any] = Field(
        description="Per trial, queries found and lost by stage; the queries most trials missed"
    )
    guardrails: dict[str, Any] = Field(
        description="Against the leader: Holm-adjusted groups, practical equivalence within the margin, the recommended trial, and sample-size hints"
    )


class RetrievalConfig(ApiModel):
    "``stratafi.retrieval.v1``."

    schema_: Literal["stratafi.retrieval.v1"] = Field(alias="schema")
    name: str
    description: str = Field(default="")
    retrievers: Retrievers | None = Field(default=None)
    fusion: Fusion | None = Field(default=None)
    candidates: int = Field(default=50, description="Fused results the reranker sees")
    filters: Filters | None = Field(default=None)
    rerank: ComponentRef | str | None = Field(default=None)
    top_k: int = Field(default=8, description="Passages returned")


class RetrievalConfigValidation(ApiModel):
    valid: bool
    issues: list[Issue]
    spec_hash: str | None = Field(default=None)
    document: RetrievalConfig | None = Field(default=None)


class RetrievalConfigVersion(ApiModel):
    id: uuid.UUID
    name: str
    version: int
    spec_hash: str
    document_hash: str
    created_by: str
    created_at: datetime.datetime
    document: RetrievalConfig


class RetrieverSettings(ApiModel):
    top_n: int = Field(default=50, description="Results each retriever returns")


class Retrievers(ApiModel):
    dense: DenseRetrieverSettings | None = Field(default=None)
    keyword: RetrieverSettings | None = Field(default=None)


class RuleCountOut(ApiModel):
    rule: str
    edits: int


class RunCounts(ApiModel):
    tasks: dict[str, int] | None = Field(default=None, description="Tasks by status")
    stages: dict[str, dict[str, int]] | None = Field(
        default=None, description="Tasks by stage and status"
    )
    entries: dict[str, int] | None = Field(default=None, description="Entries by state")


class RunEntryOut(ApiModel):
    ordinal: int
    name: str
    document_id: uuid.UUID
    sha256: str
    duplicate: bool
    state: Literal["active", "done", "excluded", "unsupported_type", "failed", "cancelled"]
    stop_code: str | None


class RunIn(ApiModel):
    pipeline_spec_id: uuid.UUID = Field(description="A saved pipeline spec version")
    corpus_version_id: uuid.UUID = Field(description="A frozen corpus version")
    until: (
        Literal["extract", "scan", "parse", "clean", "chunk", "enrich", "embed", "index"] | None
    ) = Field(default=None, description="The last stage; every stage if none")
    sample: SampleIn | None = Field(default=None)
    mode: Literal["interactive", "batch"] = Field(
        default="batch", description="Interactive runs' jobs go ahead of batch runs' jobs"
    )


class RunManifest(ApiModel):
    "``stratafi.run-manifest.v1``."

    schema_: Literal["stratafi.run-manifest.v1"] = Field(
        default="stratafi.run-manifest.v1", alias="schema"
    )
    run: ManifestRun
    configuration: ManifestConfiguration
    inputs: ManifestInputs
    models: list[ManifestModel] | None = Field(default=None)
    environment: ManifestEnvironment
    hardware: ManifestHardware
    execution: ManifestExecution
    cost: ManifestCost | None = Field(default=None)
    egress: list[ManifestEgress] | None = Field(default=None)
    outputs: list[ManifestOutput] | None = Field(default=None)


class RunOut(ApiModel):
    id: uuid.UUID
    kind: Literal["pipeline"]
    mode: Literal["interactive", "batch"]
    status: Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]
    pipeline_spec_id: uuid.UUID
    corpus_version_id: uuid.UUID
    until: Literal["extract", "scan", "parse", "clean", "chunk", "enrich", "embed", "index"] | None
    sample: dict[str, Any] | None
    rerun_of: uuid.UUID | None
    rerun_from: (
        Literal["extract", "scan", "parse", "clean", "chunk", "enrich", "embed", "index"] | None
    )
    spec_hash: str
    manifest_hash: str | None = Field(description="The finished run's fingerprint")
    counts: RunCounts
    error_code: str | None
    created_by: str
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None


class RunRef(ApiModel):
    id: uuid.UUID
    split: Literal["dev", "test", "all"]
    retrieval_ref: dict[str, Any] | None
    eval_set: dict[str, Any] | None
    pipeline_run_id: uuid.UUID
    created_at: datetime.datetime


class S3Import(ApiModel):
    source: str = Field(description="A name from intake.s3_sources")
    prefix: str = Field(default="", description="A prefix below the source's")
    corpus_id: uuid.UUID | None = Field(default=None)


class SampleIn(ApiModel):
    "Which entries a run works on: ``size`` chosen at random by ``seed``, or ``entries`` by\nname. Without a sample, a run works on every entry."

    size: int | None = Field(default=None)
    seed: int | None = Field(default=None, description="Defaults to 0")
    entries: list[str] | None = Field(default=None)


class ScanStage(ApiModel):
    checks: list[ComponentRef | str] | None = Field(
        default=None,
        description="Checks run in order on each raw document, before any parser opens it",
    )
    on_outcome: dict[str, Literal["continue", "flag", "exclude"]] | None = Field(
        default=None,
        description="What each outcome does to the document: continue, flag or exclude",
    )


class ScheduleOut(ApiModel):
    cron: str
    time_zone: str
    next_fire_at: datetime.datetime
    last_fired_at: datetime.datetime | None
    last_test_run_id: uuid.UUID | None


class SearchIn(ApiModel):
    run_id: uuid.UUID = Field(description="A run whose indexes to search")
    query: str
    mode: Literal["dense", "keyword", "hybrid"] = Field(default="hybrid")
    top_k: int = Field(default=10)
    retrieval_config_id: uuid.UUID | None = Field(
        default=None, description="A saved retrieval config: replaces mode and top_k"
    )
    retrieval_config: dict[str, Any] | None = Field(
        default=None,
        description="An unsaved stratafi.retrieval.v1 document, validated like a saved one",
    )
    evidence: list[str] | None = Field(
        default=None,
        description="An eval item's evidence phrases: each hit says where they sit in its text",
    )
    asker: AskerIn | None = Field(
        default=None, description="Filter by ACL as this asker; without one, nothing is hidden"
    )
    as_of: datetime.date | None = Field(
        default=None, description="Filter by effective dates as of this day"
    )


class SearchOut(ApiModel):
    mode: Literal["dense", "keyword", "hybrid"]
    revision: str | None = Field(description="The embedding revision a dense query used")
    results: list[Hit]
    checkpoints: dict[str, list[Hit]] = Field(
        description="Each list before the cut: dense, keyword, and fused for hybrid; with a retrieval config, also candidates and reranked"
    )
    trace: TraceBody | None = Field(
        default=None,
        description="The search's steps on a timeline, each with its lists; the playground's query has no labels, so nothing is marked relevant",
    )


class SignInOut(ApiModel):
    dev_sign_in: bool = Field(
        default=False, description="Whether the development sign-in route is on"
    )


class SliceOut(ApiModel):
    kind: str = Field(description="file_type or tag")
    label: str
    values: list[float | None]


class SlowQueryOut(ApiModel):
    item_id: str
    total_ms: float
    steps: dict[str, float]


class SpanOut(ApiModel):
    id: str = Field(description="search, keyword, embed, dense, fuse, rerank or other")
    parent_id: str | None
    name: str
    kind: str
    start_ms: float = Field(description="When it started, from the search's start")
    duration_ms: float
    attributes: dict[str, Any]
    lists: list[TraceListOut] = Field(description="The lists this step made")


class SpecDiff(ApiModel):
    name: str
    from_version: int
    to_version: int
    changes: list[Change]


class SplitIn(ApiModel):
    test_fraction: float = Field(default=0.3)
    seed: int = Field(default=0)


class StageExecution(ApiModel):
    stage: str
    tasks: int = Field(default=0)
    succeeded: int = Field(default=0)
    cached: int = Field(default=0)
    skipped: int = Field(default=0)
    failed: int = Field(default=0)
    cancelled: int = Field(default=0)
    duration_ms: float = Field(default=0.0)
    throughput_per_second: float | None = Field(default=None)
    metrics: dict[str, float] | None = Field(default=None)


class Stages(ApiModel):
    extract: ExtractStage | None = Field(default=None)
    scan: ScanStage | None = Field(default=None)
    parse: ParseStage | None = Field(default=None)
    clean: list[ComponentRef | str] | None = Field(
        default=None, description="Cleaning rules, applied in order"
    )
    chunk: ChunkStage | None = Field(default=None)
    enrich: ComponentRef | str | None = Field(default=None)
    embed: ComponentRef | str | None = Field(default=None)
    index: IndexStage | None = Field(default=None)


class StartIn(ApiModel):
    confirm: Literal[True] = Field(description="The plan was reviewed: start it")


class StartedOut(ApiModel):
    id: uuid.UUID
    suite_id: uuid.UUID
    suite_name: str
    suite_version: int
    status: Literal[
        "queued", "running", "evaluating", "passed", "failed", "error", "cancelling", "cancelled"
    ]
    outcome: Literal["passed", "failed"] | None = Field(
        description="passed when no check failed or errored"
    )
    trigger: Literal["api", "schedule", "corpus_version_frozen", "target_changed", "ci"]
    triggers: list[str] = Field(description="Every trigger combined into this run")
    resolved: dict[str, Any] = Field(
        description="Target, corpus version, eval set version and baseline, as fixed IDs"
    )
    pipeline_run_id: uuid.UUID | None
    eval_run_id: uuid.UUID | None
    baseline_pipeline_run_id: uuid.UUID | None
    baseline_eval_run_id: uuid.UUID | None
    baseline_test_run_id: uuid.UUID | None
    summary: dict[str, Any] = Field(description="Checks by result")
    checks: list[CheckOut] | None = Field(
        default=None, description="Each check's result, once the run has finished"
    )
    record_hash: str | None
    code: list[CodeFileOut] | None = Field(
        default=None,
        description="The custom checks' code the run used: each file's path, SHA-256 and size",
    )
    error_code: str | None
    created_by: str
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None
    coalesced: bool = Field(description="The request joined a run already waiting for the suite")


class StepProfileOut(ApiModel):
    n: int
    mean: float
    min: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    max: float
    histogram: HistogramOut


class SuiteIn(ApiModel):
    "A suite as a document, or as YAML text: one of the two."

    document: dict[str, Any] | None = Field(
        default=None, description="The stratafi.test-suite.v1 document"
    )
    content: str | None = Field(default=None, description="The same, as YAML")
    code: dict[str, str] | None = Field(
        default=None,
        description="Custom checks' Python files by the path the suite names: added to the suite's code, replacing a file of the same path",
    )


class SuiteOut(ApiModel):
    id: uuid.UUID
    name: str
    version: int
    document: dict[str, Any] = Field(description="The stratafi.test-suite.v1, resolved")
    spec_hash: str = Field(description="Its configuration, without name and description")
    document_hash: str
    checks: int
    created_by: str
    created_at: datetime.datetime


class TaskExecution(ApiModel):
    stage: str
    entry: str | None = Field(default=None, description="None for corpus-level tasks")
    component: str
    component_version: str
    cache_key: str | None = Field(default=None)
    status: Literal[
        "pending", "queued", "running", "succeeded", "cached", "skipped", "failed", "cancelled"
    ]
    cache_hit: bool = Field(default=False)
    attempts: int = Field(default=0)
    started_at: datetime.datetime | None = Field(default=None)
    finished_at: datetime.datetime | None = Field(default=None)
    duration_ms: float | None = Field(default=None)
    peak_memory_bytes: int | None = Field(default=None)
    error_code: str | None = Field(default=None)
    metrics: dict[str, float] | None = Field(default=None)


class TaskOut(ApiModel):
    id: uuid.UUID
    ordinal: int = Field(description="The entry's ordinal; -1 for corpus-level tasks")
    entry: str | None = Field(description="The entry's name; none for corpus-level tasks")
    stage: Literal["extract", "scan", "parse", "clean", "chunk", "enrich", "embed", "index"]
    component_id: str
    component_version: str
    cache_key: str = Field(description="The key of the artifact it makes")
    inputs: list[str] = Field(description="Input artifacts' keys (extract: the document's hash)")
    params: dict[str, Any]
    env: dict[str, Any]
    route: str | None
    resource_class: Literal["cpu", "gpu", "io"]
    status: Literal[
        "pending", "queued", "running", "succeeded", "cached", "skipped", "failed", "cancelled"
    ]
    cache_hit: bool
    job_id: uuid.UUID | None
    attempts: int
    error_code: str | None
    metrics: dict[str, Any]
    summary: dict[str, Any] | None = Field(
        description="Its artifact's summary, once it has one: metadata only"
    )
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None
    duration_ms: float | None


class TestRunIn(ApiModel):
    suite: str | None = Field(
        default=None, description="A saved suite: name, name@N or name@latest"
    )
    document: dict[str, Any] | None = Field(
        default=None, description="A suite to save (as its name's next version) and run"
    )
    content: str | None = Field(default=None, description="The same, as YAML")
    code: dict[str, str] | None = Field(
        default=None,
        description="Custom checks' Python files by the path the suite names: added to the suite's code, replacing a file of the same path",
    )
    trigger: Literal["api", "ci"] = Field(default="api", description="Who asked: a person or CI")


class TestRunOut(ApiModel):
    id: uuid.UUID
    suite_id: uuid.UUID
    suite_name: str
    suite_version: int
    status: Literal[
        "queued", "running", "evaluating", "passed", "failed", "error", "cancelling", "cancelled"
    ]
    outcome: Literal["passed", "failed"] | None = Field(
        description="passed when no check failed or errored"
    )
    trigger: Literal["api", "schedule", "corpus_version_frozen", "target_changed", "ci"]
    triggers: list[str] = Field(description="Every trigger combined into this run")
    resolved: dict[str, Any] = Field(
        description="Target, corpus version, eval set version and baseline, as fixed IDs"
    )
    pipeline_run_id: uuid.UUID | None
    eval_run_id: uuid.UUID | None
    baseline_pipeline_run_id: uuid.UUID | None
    baseline_eval_run_id: uuid.UUID | None
    baseline_test_run_id: uuid.UUID | None
    summary: dict[str, Any] = Field(description="Checks by result")
    checks: list[CheckOut] | None = Field(
        default=None, description="Each check's result, once the run has finished"
    )
    record_hash: str | None
    code: list[CodeFileOut] | None = Field(
        default=None,
        description="The custom checks' code the run used: each file's path, SHA-256 and size",
    )
    error_code: str | None
    created_by: str
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None


class TokenIn(ApiModel):
    name: str = Field(description="Such as ci-main")
    scopes: list[Literal["read", "runs:write", "tests:run"]]
    expires_in_days: int = Field(default=90)


class TokenOut(ApiModel):
    id: uuid.UUID
    name: str
    prefix: str = Field(description="The token's first characters, to tell tokens apart")
    scopes: list[str]
    active: bool
    created_by: str
    created_at: datetime.datetime
    expires_at: datetime.datetime
    revoked_at: datetime.datetime | None
    last_used_at: datetime.datetime | None


class TokensOut(ApiModel):
    items: list[TokenOut]


class TraceBody(ApiModel):
    total_ms: float
    spans: list[SpanOut] = Field(description="The search, then its steps in the order they ran")
    path: list[TraceListOut] = Field(
        description="Every checkpoint's list, in order: where the evidence is, without entries"
    )
    lost_at_checkpoint: str | None = Field(
        description="The first list that dropped evidence the list before it had"
    )


class TraceEntryOut(ApiModel):
    rank: int
    chunk_id: str
    score: float
    relevant: bool | None = Field(description="Null when the query has no labels")


class TraceListOut(ApiModel):
    checkpoint: str
    length: int
    first_relevant_rank: int | None
    found: bool | None = Field(description="A relevant chunk is in the list; null without labels")
    top: list[TraceEntryOut] = Field(description="The first ten entries")


class TraceOut(ApiModel):
    total_ms: float
    spans: list[SpanOut] = Field(description="The search, then its steps in the order they ran")
    path: list[TraceListOut] = Field(
        description="Every checkpoint's list, in order: where the evidence is, without entries"
    )
    lost_at_checkpoint: str | None = Field(
        description="The first list that dropped evidence the list before it had"
    )
    eval_run_id: uuid.UUID
    item_id: str
    rule: str = Field(description="The match rule that marks relevant chunks")
    has_evidence: bool
    hit: bool | None
    lost_at: str | None = Field(description="Where the eval run says a miss was lost")
    embedding_revision: str | None
    rerank_revision: str | None
    visible_chunks: int = Field(description="Chunks the asker could see on the as-of date")


class TrackedRunOut(ApiModel):
    id: uuid.UUID
    label: str = Field(description="Its title, or its config and pipeline")
    status: str
    split: str
    title: str | None
    tags: dict[str, str]
    archived: bool
    created_by: str
    created_at: datetime.datetime
    finished_at: datetime.datetime | None
    run_id: uuid.UUID = Field(description="The pipeline run searched")
    retrieval_config_id: uuid.UUID
    pipeline_spec_id: uuid.UUID | None
    eval_set_version_id: uuid.UUID
    attributes: dict[str, Any]
    headline: dict[str, float | None] = Field(
        description="hit and recall at the eval set's k, MRR and nDCG at 10, final list"
    )
    latency_p95_ms: float | None
    values: dict[str, Any] = Field(description="The requested columns' values")


class TrainedModelOut(ApiModel):
    id: uuid.UUID
    name: str
    version: int
    kind: Literal["query-adapter", "embedder", "reranker"]
    training_run_id: uuid.UUID
    model_key: str
    base: dict[str, Any]
    status: Literal["ready", "withdrawn"]
    stored: bool = Field(description="Its artifact is still stored: a purge removes it")
    card: dict[str, Any] = Field(description="Its card, stratafi.model-card.v1")
    withdrawn_by: str | None
    withdrawn_at: datetime.datetime | None
    created_by: str
    created_at: datetime.datetime
    aliases: list[str] | None = Field(
        default=None, description="The aliases that name this version"
    )


class TrainedModelsOut(ApiModel):
    items: list[TrainedModelOut]


class TrainerIn(ApiModel):
    component: str = Field(description="A train/ component, such as train/query-adapter")
    params: dict[str, Any] | None = Field(default=None)


class TrainingPassage(ApiModel):
    "One passage of a training record: a chunk with where it came from."

    chunk_id: str
    doc_key: str | None = Field(default=None)
    version: str | None = Field(default=None)
    section: str | None = Field(default=None)
    text: str
    reason: Literal["evidence", "retrieved", "avoid", "must_not_return", "abstention"]
    grade: float | None = Field(
        default=None, description="Positives: the evidence it covers; negatives: the most it covers"
    )
    rank: int | None = Field(default=None, description="Its best rank in the lists mined")
    checkpoint: str | None = Field(default=None, description="The list that ranked it best")


class TrainingRecord(ApiModel):
    "``stratafi.training-set.v1``: one eval item as training data, a line of a training set's\nJSON Lines artifact."

    item_id: str
    split: Literal["train", "validation"]
    query: str
    tags: list[str] | None = Field(default=None)
    abstention: bool = Field(default=False, description="The item has no evidence: negatives only")
    reference_answer: str | None = Field(default=None)
    positives: list[TrainingPassage] | None = Field(default=None)
    negatives: list[TrainingPassage] | None = Field(default=None)


class TrainingRecordsOut(ApiModel):
    items: list[TrainingRecord]
    total: int
    offset: int


class TrainingRunIn(ApiModel):
    training_set_id: uuid.UUID = Field(description="A succeeded training set")
    name: str = Field(description="The model's name; each run of it is a new version")
    trainer: TrainerIn


class TrainingRunOut(ApiModel):
    id: uuid.UUID
    training_set_id: uuid.UUID
    name: str
    kind: Literal["query-adapter", "embedder", "reranker"]
    trainer: str
    trainer_version: str
    params: dict[str, Any] = Field(description="The trainer's parameters, resolved")
    base: dict[str, Any] = Field(description="The base model, once the run has started")
    status: Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]
    progress: dict[str, Any] = Field(description="The epoch it's at, with that epoch's metrics")
    metrics: dict[str, Any] | None = Field(
        description="Baseline, every epoch's loss and validation metrics, the epoch kept"
    )
    manifest_hash: str | None
    model_key: str | None
    model_cached: bool = Field(description="An identical earlier run's model was reused")
    model_id: uuid.UUID | None = Field(description="The model it registered")
    error_code: str | None
    created_by: str
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None


class TrainingRunsOut(ApiModel):
    items: list[TrainingRunOut]


class TrainingSetIn(ApiModel):
    eval_run_id: uuid.UUID = Field(description="A succeeded eval run on the dev split")
    name: str | None = Field(default=None, description="Defaults to the eval set's name")
    options: TrainingSetOptions | None = Field(default=None)


class TrainingSetOptions(ApiModel):
    "How a training set is mined; recorded with it and part of its key."

    rule: Literal["evidence-coverage-v1", "knowledge-hit-v1"] = Field(
        default="evidence-coverage-v1", description="The rule a positive must pass"
    )
    max_positives: int = Field(default=3, description="Positives per item, at most")
    max_negatives: int = Field(default=8, description="Negatives per item, at most")
    checkpoints: list[Literal["dense", "keyword", "fused", "candidates", "reranked", "final"]] = (
        Field(
            default_factory=lambda: ["dense", "keyword", "candidates"],
            description="The ranked lists hard negatives come from",
        )
    )
    depth: int = Field(default=20, description="How far down each list to look")
    negative_max_grade: float = Field(
        default=0.3,
        description="A negative covers no evidence item at this grade or above, in any version",
    )
    abstentions: bool = Field(
        default=True, description="Keep items without evidence, with negatives only"
    )
    validation_share: float = Field(default=0.2, description="Items held out for early stopping")
    duplicate_threshold: float = Field(
        default=0.8,
        description="Leak guard: drop a dev item this similar to a test item (token Jaccard)",
    )
    seed: int = Field(default=0, description="Seeds the validation split")


class TrainingSetOut(ApiModel):
    id: uuid.UUID
    name: str
    eval_run_id: uuid.UUID
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    options: dict[str, Any]
    cache_key: str | None = Field(description="The records' artifact")
    cached: bool = Field(description="An identical earlier set's records were reused")
    context: dict[str, Any] = Field(
        description="What it was mined from: eval set, indexes, embedder, dense index"
    )
    stats: dict[str, Any] | None = Field(
        description="Records, positives, negatives by reason, what was dropped, the leak guard"
    )
    error_code: str | None
    created_by: str
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None


class TrainingSetsOut(ApiModel):
    items: list[TrainingSetOut]


class TraitOut(ApiModel):
    name: str
    kind: str = Field(description="flag (true or false; can route a parser) or count")
    applies_to: list[str] = Field(description="Short type names")
    description: str


class TrialOut(ApiModel):
    id: uuid.UUID
    ordinal: int
    grid_index: int
    levels: list[int]
    values: dict[str, str] = Field(description="Factor path → the level's label")
    status: Literal["pending", "running", "succeeded", "failed", "skipped", "cancelled", "pruned"]
    pipeline_hash: str
    retrieval_hash: str
    pipeline_spec_id: uuid.UUID | None
    retrieval_config_id: uuid.UUID | None
    run_id: uuid.UUID | None = Field(
        description="Its pipeline run, shared by its pipeline's trials"
    )
    eval_run_id: uuid.UUID | None
    objective: dict[str, Any] | None = Field(description="The objective's mean with its 95% CI")
    measures: dict[str, Any]
    constraints: list[dict[str, Any]]
    feasible: bool | None = Field(description="Every constraint passed")
    error_code: str | None
    rung: int = Field(description="Successive halving: the rung it's at, or reached")
    rungs: list[dict[str, Any]] = Field(
        description="Successive halving: each finished rung's sample, eval run, estimate and verdict"
    )
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None


class TrialsOut(ApiModel):
    items: list[TrialOut]


class UrlImport(ApiModel):
    url: str = Field(description="An http or https URL")
    corpus_id: uuid.UUID | None = Field(default=None)


class ValidationOut(ApiModel):
    valid: bool
    issues: list[IssueOut]
    document: dict[str, Any] | None = Field(description="The suite, every default filled in")
    needs: dict[str, bool] = Field(
        description="What its checks need: a pipeline run, an eval run, a baseline, and the workspace's custom_checks feature"
    )


class VersionOut(ApiModel):
    id: uuid.UUID
    corpus_id: uuid.UUID
    version: int
    manifest_hash: str
    document_count: int
    total_bytes: int
    created_by: str
    frozen_at: datetime.datetime


class WebhookIn(ApiModel):
    url: str = Field(description="An http or https URL")
    suite: str | None = Field(
        default=None, description="Only this suite's runs; every suite's if none"
    )
    events: list[Literal["test_run.finished"]] | None = Field(default=None)


class WebhookOut(ApiModel):
    id: uuid.UUID
    url: str
    suite: str | None
    events: list[str]
    enabled: bool
    created_by: str
    created_at: datetime.datetime


class WebhooksOut(ApiModel):
    items: list[WebhookOut]


class WhereLostOut(ApiModel):
    buckets: list[str]
    counts: list[list[int | None]]


class WorkOut(ApiModel):
    planned: dict[str, Any] = Field(description="The plan's unique work per stage")
    done: dict[str, dict[str, int]] = Field(
        description="The experiment's pipeline runs' tasks by stage and status"
    )


class WorkspaceOut(ApiModel):
    id: uuid.UUID
    slug: str
    name: str


class stratafiapiroutesprofilesDecisionIn(ApiModel):
    decision: Literal["approved", "rejected"]


class stratafiapiroutesquestionsDecisionIn(ApiModel):
    decision: Literal["accepted", "rejected"]
    query: str | None = Field(default=None)
    evidence: str | None = Field(default=None)
    split: Literal["dev", "test"] | None = Field(default=None)
    tags: list[str] | None = Field(default=None)
    doc_key: str | None = Field(
        default=None, description="The gold document, when the drafted one is wrong"
    )
    version: str | None = Field(default=None)
    section: str | None = Field(default=None)


for _model in (
    AddedOut,
    AgreementOut,
    AliasOut,
    AliasesOut,
    AnnotationIn,
    AnnotationOut,
    AnswerDetailOut,
    AnswerResultOut,
    AnswerResultsOut,
    AnswerRunIn,
    AnswerRunOut,
    AnswerRunsOut,
    ApprovalOut,
    ArtifactInputOut,
    ArtifactOut,
    Asker,
    AskerIn,
    AuditEventOut,
    AuthorIn,
    AuthorOut,
    AutomationOut,
    BatchIn,
    BatchOut,
    BatchesOut,
    CancelOut,
    CandidateOut,
    CandidatesOut,
    CatalogModelOut,
    CatalogOut,
    CellOut,
    Change,
    ChangeOut,
    ChatIn,
    ChatMessage,
    ChatModelOut,
    ChatModelsOut,
    Check,
    CheckOut,
    ChunkOut,
    ChunkStage,
    CodeFileOut,
    CompareManyOut,
    CompareOut,
    ComparedRunOut,
    ComponentList,
    ComponentOut,
    ComponentRef,
    ConfirmIn,
    ConfirmationOut,
    ConfirmationsOut,
    CorpusDetail,
    CorpusIn,
    CorpusOut,
    Count,
    CreatedTokenOut,
    CreatedWebhookOut,
    CurveOut,
    DeliveriesOut,
    DeliveryOut,
    DenseRetrieverSettings,
    DiffHunkOut,
    DiffOut,
    DocumentDetail,
    DocumentOut,
    EntriesIn,
    EntryIn,
    EntryOut,
    EvalConfiguration,
    EvalExecution,
    EvalInputs,
    EvalItemOut,
    EvalOptions,
    EvalOutput,
    EvalRunIn,
    EvalRunInfo,
    EvalRunManifest,
    EvalRunOut,
    EvalRunSearchIn,
    EvalRunSearchKeys,
    EvalRunSearchOut,
    EvalSample,
    EvalSetImport,
    EvalSetOut,
    EvalSetVersionOut,
    EvalShardRecord,
    EvidenceRef,
    ExperimentIn,
    ExperimentOut,
    ExtractStage,
    FeatureIn,
    FeatureOut,
    FeaturesOut,
    FileTypeOut,
    FileTypesOut,
    Filters,
    FolderImport,
    FunnelOut,
    Fusion,
    GeneratorIn,
    Health,
    HistogramOut,
    HistoryOut,
    Hit,
    ImportOut,
    ImportReport,
    IndexStage,
    IngestOut,
    IngestedOut,
    IntakeSourcesOut,
    Issue,
    IssueOut,
    JobOut,
    LabelIn,
    LabelOut,
    LatencyOut,
    LatencyProfileOut,
    Liveness,
    ManifestComponent,
    ManifestConfiguration,
    ManifestCorpus,
    ManifestCost,
    ManifestDocument,
    ManifestEgress,
    ManifestEnvironment,
    ManifestExecution,
    ManifestGpu,
    ManifestHardware,
    ManifestImport,
    ManifestInputs,
    ManifestModel,
    ManifestOutput,
    ManifestRun,
    ManifestSample,
    MapEdgeOut,
    MapNodeOut,
    MapOut,
    MatrixCellOut,
    Me,
    MetricDelta,
    MetricRowOut,
    ModelComparisonOut,
    ModelPairOut,
    ModelRowOut,
    NamesIn,
    NeighbourOut,
    NeighboursOut,
    OriginOut,
    PageAuditEventOut,
    PageChunkOut,
    PageCorpusOut,
    PageDocumentOut,
    PageEntryOut,
    PageEvalItemOut,
    PageEvalRunOut,
    PageEvalSetOut,
    PageEvalSetVersionOut,
    PageExperimentOut,
    PageJobOut,
    PagePipelineSpecVersion,
    PageProfileOut,
    PageQueryResultOut,
    PageRetrievalConfigVersion,
    PageRunEntryOut,
    PageRunOut,
    PageSuiteOut,
    PageTaskOut,
    PageTestRunOut,
    PageVersionOut,
    ParamDiffOut,
    ParseStage,
    PassageOut,
    PipelineSpec,
    PipelineSpecValidation,
    PipelineSpecVersion,
    PlanOut,
    PlannedTrialOut,
    ProblemDetails,
    ProblemIssue,
    ProfileDiffOut,
    ProfileFromTrialIn,
    ProfileImportIn,
    ProfileOut,
    ProjectionOut,
    ProjectionPoint,
    ProviderUsage,
    PurgeOut,
    QueryDelta,
    QueryDetailOut,
    QueryResultOut,
    QueryRowOut,
    RagPipelineOut,
    RagPipelinesOut,
    RerunIn,
    ResultsOut,
    RetrievalConfig,
    RetrievalConfigValidation,
    RetrievalConfigVersion,
    RetrieverSettings,
    Retrievers,
    RuleCountOut,
    RunCounts,
    RunEntryOut,
    RunIn,
    RunManifest,
    RunOut,
    RunRef,
    S3Import,
    SampleIn,
    ScanStage,
    ScheduleOut,
    SearchIn,
    SearchOut,
    SignInOut,
    SliceOut,
    SlowQueryOut,
    SpanOut,
    SpecDiff,
    SplitIn,
    StageExecution,
    Stages,
    StartIn,
    StartedOut,
    StepProfileOut,
    SuiteIn,
    SuiteOut,
    TaskExecution,
    TaskOut,
    TestRunIn,
    TestRunOut,
    TokenIn,
    TokenOut,
    TokensOut,
    TraceBody,
    TraceEntryOut,
    TraceListOut,
    TraceOut,
    TrackedRunOut,
    TrainedModelOut,
    TrainedModelsOut,
    TrainerIn,
    TrainingPassage,
    TrainingRecord,
    TrainingRecordsOut,
    TrainingRunIn,
    TrainingRunOut,
    TrainingRunsOut,
    TrainingSetIn,
    TrainingSetOptions,
    TrainingSetOut,
    TrainingSetsOut,
    TraitOut,
    TrialOut,
    TrialsOut,
    UrlImport,
    ValidationOut,
    VersionOut,
    WebhookIn,
    WebhookOut,
    WebhooksOut,
    WhereLostOut,
    WorkOut,
    WorkspaceOut,
    stratafiapiroutesprofilesDecisionIn,
    stratafiapiroutesquestionsDecisionIn,
):
    _model.model_rebuild()
