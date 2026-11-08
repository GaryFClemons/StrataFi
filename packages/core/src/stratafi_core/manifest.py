"""The run manifest, ``stratafi.run-manifest.v1``: what every run records.

A run's manifest is written when the run starts and finalized when it ends; ``manifest_hash``
of the final document is the run's fingerprint. It holds every area of the run-record table:

| Area | Field |
| --- | --- |
| Configuration | ``configuration``: the resolved spec and its hash; each component and params |
| Inputs | ``inputs``: corpus version, each document's SHA-256, type and traits, the sample |
| Models | ``models``: alias, served model, revision, dimensions, dtype, prefixes, tokenizer |
| Environment | ``environment``: Stratafi version and commit, Python, libraries, OS, image |
| Hardware | ``hardware``: CPU, cores, RAM, GPUs, the worker's resource classes |
| Execution | ``execution``: per stage and per task: status, attempts, times, cache hits, errors |
| Cost | ``cost``: calls and tokens per provider, and an estimate from a recorded price table |
| Egress | ``egress``: which providers received which documents |
| Outputs | ``outputs``: every artifact's cache key, kind, size and summary |

Everything here is metadata: IDs, hashes, counts and codes, never document text.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from stratafi_core.hashing import content_hash

RUN_MANIFEST_SCHEMA = "stratafi.run-manifest.v1"

RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]
TaskStatus = Literal[
    "pending", "queued", "running", "succeeded", "cached", "skipped", "failed", "cancelled"
]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ManifestRun(_M):
    id: str
    kind: Literal["pipeline"] = "pipeline"
    mode: Literal["interactive", "batch"] = "batch"
    until: str | None = Field(default=None, description="The last stage the run goes to")
    rerun_of: str | None = None
    rerun_from: str | None = Field(default=None, description="The stage a re-run started from")
    created_by: str
    created_at: datetime


class ManifestComponent(_M):
    path: str = Field(description="Where the spec names it, e.g. stages.chunk")
    id: str
    version: str
    params: dict[str, Any]
    params_hash: str


class ManifestConfiguration(_M):
    pipeline: dict[str, Any] = Field(description="The resolved pipeline spec")
    pipeline_hash: str
    pipeline_ref: dict[str, Any] | None = Field(
        default=None, description="The saved spec it came from: name and version"
    )
    retrieval: dict[str, Any] | None = None
    retrieval_hash: str | None = None
    components: list[ManifestComponent] = Field(default_factory=list)


class ManifestDocument(_M):
    entry: str = Field(description="The document's name in the corpus version")
    document_id: str
    sha256: str
    size_bytes: int
    media_type: str
    traits: dict[str, Any] = Field(default_factory=dict)


class ManifestCorpus(_M):
    corpus_id: str
    corpus_version_id: str
    version: int
    manifest_hash: str


class ManifestSample(_M):
    size: int | None = None
    seed: int | None = None
    entries: list[str] | None = None


class ManifestInputs(_M):
    corpus: ManifestCorpus | None = None
    documents: list[ManifestDocument] = Field(default_factory=list)
    sample: ManifestSample | None = None
    eval_set: dict[str, Any] | None = Field(default=None, description="Null for a pipeline run")


class ManifestModel(_M):
    role: str = Field(description="embed, rerank, ...")
    component: str
    alias: str | None = None
    served_model: str | None = None
    revision: str | None = None
    dimensions: int | None = None
    dtype: str | None = None
    prefixes: dict[str, str] = Field(default_factory=dict)
    tokenizer: str | None = None


class ManifestEnvironment(_M):
    stratafi_version: str
    commit: str | None = None
    python: str
    libraries: dict[str, str] = Field(default_factory=dict)
    os: str
    container_image: str | None = None


class ManifestGpu(_M):
    model: str
    memory_bytes: int | None = None
    driver: str | None = None


class ManifestHardware(_M):
    host: str | None = Field(default=None, description="The worker host's name, if recorded")
    cpu_model: str | None = None
    cpu_cores: int | None = None
    ram_bytes: int | None = None
    gpus: list[ManifestGpu] = Field(default_factory=list)
    resource_classes: list[str] = Field(default_factory=list)


class TaskExecution(_M):
    stage: str
    entry: str | None = Field(default=None, description="None for corpus-level tasks")
    component: str
    component_version: str
    cache_key: str | None = None
    status: TaskStatus
    cache_hit: bool = False
    attempts: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = None
    peak_memory_bytes: int | None = None
    error_code: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class StageExecution(_M):
    stage: str
    tasks: int = 0
    succeeded: int = 0
    cached: int = 0
    skipped: int = 0
    failed: int = 0
    cancelled: int = 0
    duration_ms: float = 0.0
    throughput_per_second: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class ManifestExecution(_M):
    status: RunStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = None
    error_code: str | None = None
    stages: list[StageExecution] = Field(default_factory=list)
    tasks: list[TaskExecution] = Field(default_factory=list)


class ProviderUsage(_M):
    calls: int = 0
    tokens: int = 0


class ManifestCost(_M):
    providers: dict[str, ProviderUsage] = Field(default_factory=dict)
    currency: str = "USD"
    estimated: float | None = None
    price_table: dict[str, float] = Field(
        default_factory=dict, description="Price per 1,000 tokens by provider, at run time"
    )


class ManifestEgress(_M):
    provider: str
    documents: list[str] = Field(default_factory=list, description="Document IDs")


class ManifestOutput(_M):
    cache_key: str
    kind: str
    stage: str
    size_bytes: int
    summary: dict[str, Any] = Field(default_factory=dict)


class RunManifest(_M):
    """``stratafi.run-manifest.v1``."""

    schema_: Literal["stratafi.run-manifest.v1"] = Field(
        default=RUN_MANIFEST_SCHEMA, alias="schema"
    )
    run: ManifestRun
    configuration: ManifestConfiguration
    inputs: ManifestInputs
    models: list[ManifestModel] = Field(default_factory=list)
    environment: ManifestEnvironment
    hardware: ManifestHardware
    execution: ManifestExecution
    cost: ManifestCost = Field(default_factory=ManifestCost)
    egress: list[ManifestEgress] = Field(default_factory=list)
    outputs: list[ManifestOutput] = Field(default_factory=list)

    def fingerprint(self) -> str:
        """The manifest's hash: the run's fingerprint."""
        return content_hash(self.model_dump(mode="json", by_alias=True))
