"""Test suites: declarative checks with thresholds and baselines.

``stratafi.test-suite.v1`` names what is tested (a knowledge profile, or a pipeline spec with a
retrieval config), the corpus and the eval set's dev split it's tested on, an optional baseline,
triggers, and checks. A check's type is written as ``type`` or follows from its fields:

| Type | Fields that make it |
| --- | --- |
| ``metric`` | ``metric`` (a retrieval metric: ``recall@5``), ``min`` and/or ``max``, ``where?`` |
| ``budget`` | ``metric`` naming a latency, cost, duration or token measure, and ``max`` |
| ``safety`` | ``metric`` naming a violation count; or ``stage`` with ``assert: no_findings`` |
| ``regression`` | ``metric`` and ``vs_baseline: {max_drop or max_relative_drop, significance?}`` |
| ``stage`` | ``stage`` with ``metric`` (``parse.success_rate``...) or ``assert`` |
| ``golden`` | ``query``, ``asker`` and ``expect`` or ``expect_not`` |
| ``data_change`` | ``type: data_change``, and ``max`` on the counts it may reach |
| ``revision_guard`` | ``allow_change`` |
| ``custom`` | ``python: path:function``, the file in the suite's ``code`` (sandboxed) |

Evaluating a check is a pure function of what the runs recorded (the ``evaluate_*`` functions
here; the worker gathers their inputs, ``stratafi_worker.testruns``). A check ends ``pass``,
``fail``, ``error`` (it couldn't run: an input is missing or failed) or ``skipped`` (a condition
wasn't met, such as a baseline the suite doesn't have). The suite passes when no check fails or
errors.

``stratafi.test-run.v1`` is a test run's record; ``junit_xml`` and ``markdown_summary`` render
it for CI. Records, reports and results hold IDs, codes and numbers: never a document's, a
chunk's or a query's text. A golden check's query is the suite's own text and stays in the suite;
its result names the check, not the query.

A custom check's file travels in the suite's ``code`` (path to source), so the suite's version
and hash cover it. The worker runs it only in its sandbox, and only in a workspace whose owner
turned custom checks on after the sandbox's security review; ``custom_result`` reads
what the sandbox printed as untrusted input.
"""

import json
import math
import re
import unicodedata
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    StringConstraints,
    Tag,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from stratafi_core import cron
from stratafi_core.evalsets import AskerName, DocKey, DocVersion, Principal
from stratafi_core.experiments import METRIC_RE, parse_ref
from stratafi_core.hashing import content_hash, sha256_hex
from stratafi_core.logs import is_text_field, safe_value
from stratafi_core.pii import PII_CATEGORIES, PiiCategory, canonical_categories
from stratafi_core.specs import NAME_PATTERN, SpecInvalid, SpecIssue, validation_issues
from stratafi_core.stats import DEFAULT_RESAMPLES, bootstrap_mean, compare_paired

SUITE_SCHEMA = "stratafi.test-suite.v1"
TEST_RUN_SCHEMA = "stratafi.test-run.v1"
MAX_CHECKS = 200
MAX_TRIGGERS = 20
CHECK_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
# Custom checks' code: files by relative path, kept with the suite's version.
MAX_CODE_FILES = 20
MAX_CODE_BYTES = 64 * 1024
MAX_CODE_TOTAL_BYTES = 256 * 1024
CODE_PATH_PATTERN = r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,199}\.py"
PROFILE_SELECTORS = ("current", "previous", "latest")
PROFILE_REF_RE = re.compile(
    r"(?P<name>[a-z0-9][a-z0-9._-]{0,63})(?:@(?P<version>[1-9][0-9]{0,8}|current|previous|latest))?"
)

Result = Literal["pass", "fail", "error", "skipped"]
CheckType = Literal[
    "metric",
    "budget",
    "safety",
    "regression",
    "stage",
    "golden",
    "data_change",
    "revision_guard",
    "custom",
]
CHECK_TYPES: tuple[str, ...] = CheckType.__args__  # type: ignore[attr-defined]
Stage = Literal["extract", "scan", "parse", "clean", "chunk", "enrich", "embed", "index"]
Rule = Literal["knowledge-hit-v1", "evidence-coverage-v1", "doc-v1"]
Checkpoint = Literal["dense", "keyword", "fused", "candidates", "reranked", "final"]
Outcome = Literal["infected", "encrypted", "malformed", "limit_exceeded", "pii", "error"]
Trigger = Literal["api", "schedule", "corpus_version_frozen", "target_changed", "ci"]

# Measures a ``metric`` check can name besides the retrieval metrics: what an eval run or its
# pipeline run recorded. Budget measures may only have a ``max``.
BUDGET_MEASURES: dict[str, str] = {
    "latency_ms.p50": "Query latency, median, in milliseconds",
    "latency_ms.p95": "Query latency, 95th percentile, in milliseconds",
    "latency_ms.mean": "Query latency, mean, in milliseconds",
    "cost_per_1k_queries_usd": "Estimated query-time cost per 1,000 queries",
    "ingest_cost_usd": "Estimated cost of the pipeline run",
    "eval_cost_usd": "Estimated cost of the eval run",
    "run_duration_s": "The pipeline run's wall-clock seconds",
    "eval_duration_s": "The eval run's wall-clock seconds",
    "ingest_tokens": "Tokens the pipeline run sent to model providers",
    "eval_tokens": "Tokens the eval run sent to model providers",
}
SAFETY_MEASURES: dict[str, str] = {
    "must_not_return_violations": "Queries that returned a must-not-return document",
    "avoid_violations": "Queries where an avoided version ranked above the first hit",
}
QUALITY_MEASURES: dict[str, str] = {
    "findability": "Share of evidence found in the parsed and cleaned text",
}
# Stage metrics: what a pipeline run's tasks recorded, per stage (``evaluate_stage``).
STAGE_METRICS: dict[str, str] = {
    "extract.documents": "Documents extracted",
    "scan.flagged": "Documents a scan check flagged or excluded",
    "scan.excluded": "Documents the scan policy excluded",
    "parse.success_rate": "Share of the parse tasks that ended and succeeded",
    "parse.failed": "Parse tasks that failed",
    "parse.evidence_findability": "Share of the eval set's evidence found in the cleaned text",
    "parse.ocr_pages": "Pages OCR read",
    "clean.redactions": "Spans clean/redact replaced",
    "chunk.count": "Chunks made",
    "chunk.tokens_max": "The largest chunk's tokens",
    "chunk.share_over_embed_limit": "Share of the chunks longer than the embedder's input limit",
    "embed.truncated_inputs": "Chunks the embedder truncated to its input limit",
}
ASSERTIONS = ("all_succeeded", "no_findings", "no_outcome")
DATA_CHANGES = ("added", "removed", "changed", "unparseable", "flagged")

CheckId = Annotated[str, StringConstraints(pattern=CHECK_ID_PATTERN)]
Name = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]
DocumentRef = Annotated[str, StringConstraints(max_length=80)]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def _invalid(code: str, message: str) -> PydanticCustomError:
    """A validation error whose type is ``code``: what a ``SpecIssue`` reports."""
    return PydanticCustomError(code, message)


def _check_ref(value: str) -> str:
    parse_ref(value)
    return value


def parse_profile_ref(text: str) -> tuple[str, int | str]:
    """``name``, ``name@3``, ``name@current``, ``name@previous`` or ``name@latest``: the name and
    a version number or selector. A bare name is its current (latest approved) version."""
    match = PROFILE_REF_RE.fullmatch(text)
    if match is None:
        raise ValueError(
            "a profile reference is name@current, name@previous, name@latest or name@N"
        )
    version = match.group("version") or "current"
    return match.group("name"), int(version) if version.isdigit() else version


def _optional_profile_ref(value: str | None) -> str | None:
    if value is not None:
        parse_profile_ref(value)
    return value


def _optional_ref(value: str | None) -> str | None:
    if value is not None:
        parse_ref(value)
    return value


# --- what is tested -------------------------------------------------------------------------------


class TargetRef(_Spec):
    """What a suite tests: a knowledge profile (its pipeline spec and retrieval config), or a
    pipeline spec with an optional retrieval config. Checks that search need a retrieval
    config."""

    profile: DocumentRef | None = Field(
        default=None, description="name@current (the latest approved), @previous, @latest or @N"
    )
    pipeline: DocumentRef | None = Field(default=None, description="name, name@N or name@latest")
    retrieval: DocumentRef | None = Field(default=None, description="name, name@N or name@latest")

    _profile = field_validator("profile")(_optional_profile_ref)
    _refs = field_validator("pipeline", "retrieval")(_optional_ref)

    @model_validator(mode="after")
    def _one_target(self) -> "TargetRef":
        if self.profile is not None and (self.pipeline or self.retrieval):
            raise _invalid(
                "target_ambiguous", "a target is a profile, or a pipeline with a retrieval config"
            )
        if self.profile is None and self.pipeline is None:
            raise _invalid("target_missing", "a target names a profile or a pipeline")
        return self


class BaselineRef(_Spec):
    """What regressions, data changes and the revision guard compare with: a pinned eval run,
    a profile version (``@previous`` resolves when the run starts), a pipeline spec with a
    retrieval config, or the last passing run of this suite."""

    eval_run: uuid.UUID | None = Field(default=None, description="A pinned, succeeded eval run")
    profile: DocumentRef | None = None
    pipeline: DocumentRef | None = None
    retrieval: DocumentRef | None = None
    last_passing: bool = Field(default=False, description="The last passing run of this suite")

    _profile = field_validator("profile")(_optional_profile_ref)
    _refs = field_validator("pipeline", "retrieval")(_optional_ref)

    @model_validator(mode="after")
    def _one_baseline(self) -> "BaselineRef":
        kinds = [
            self.eval_run is not None,
            self.profile is not None,
            self.pipeline is not None or self.retrieval is not None,
            self.last_passing,
        ]
        if sum(kinds) != 1:
            raise _invalid(
                "baseline_ambiguous",
                "a baseline is one of eval_run, profile, pipeline, last_passing",
            )
        if (self.pipeline is None) != (self.retrieval is None):
            raise _invalid(
                "baseline_incomplete", "a pipeline baseline names its retrieval config too"
            )
        return self

    @property
    def kind(self) -> str:
        if self.eval_run is not None:
            return "eval_run"
        if self.profile is not None:
            return "profile"
        if self.pipeline is not None:
            return "pipeline"
        return "last_passing"


class EvalSetRef(_Spec):
    name: Name
    version: int | Literal["latest"] | None = Field(
        default=None, description="A frozen version; the latest frozen one if none"
    )
    split: Literal["dev"] = Field(
        default="dev",
        description="Suites run on dev: the locked test split only confirms a candidate",
    )

    @field_validator("version")
    @classmethod
    def _positive(cls, value: Any) -> Any:
        if isinstance(value, int) and value < 1:
            raise _invalid("invalid_version", "versions start at 1")
        return value

    @property
    def number(self) -> int | None:
        return self.version if isinstance(self.version, int) else None


class TriggerSpec(_Spec):
    """When a suite runs on its own: an event, or a cron schedule in the suite's time zone."""

    on: Literal["corpus_version_frozen", "target_changed"] | None = None
    schedule: str | None = Field(default=None, max_length=cron.MAX_LENGTH)

    @model_validator(mode="before")
    @classmethod
    def _yaml_on(cls, data: Any) -> Any:
        # YAML 1.1 reads a bare ``on`` key as true, as GitHub Actions' workflows know well.
        if isinstance(data, Mapping) and True in data and "on" not in data:
            return {("on" if key is True else key): value for key, value in data.items()}
        return data

    @field_validator("schedule")
    @classmethod
    def _cron(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                cron.parse(value)
            except cron.CronError as error:
                raise _invalid(error.code, "the schedule can't be read") from None
        return value

    @model_validator(mode="after")
    def _one(self) -> "TriggerSpec":
        if (self.on is None) == (self.schedule is None):
            raise _invalid("trigger_ambiguous", "a trigger is an event (on) or a schedule")
        return self


# --- checks ---------------------------------------------------------------------------------------


class Where(_Spec):
    """Which queries (metric checks) or documents (stage checks) a check covers."""

    tag: str | None = Field(default=None, max_length=64, description="Queries with this tag")
    file_type: str | None = Field(
        default=None,
        max_length=127,
        description="A media type: queries whose gold document has it, or documents of it",
    )

    @model_validator(mode="after")
    def _something(self) -> "Where":
        if self.tag is None and self.file_type is None:
            raise _invalid("where_empty", "where names a tag or a file_type")
        return self


class _Check(_Spec):
    id: CheckId
    description: str = Field(default="", max_length=500)


class _Bounded(_Check):
    min: float | None = None
    max: float | None = None

    def _bounds(self) -> None:
        if self.min is None and self.max is None:
            raise _invalid("threshold_missing", "a threshold needs min or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise _invalid("threshold_inverted", "min is more than max")


def metric_family(metric: str) -> str | None:
    """``metric``, ``budget`` or ``safety`` for a name a metric check may use; None otherwise."""
    if METRIC_RE.fullmatch(metric) or metric in QUALITY_MEASURES:
        return "metric"
    if metric in BUDGET_MEASURES:
        return "budget"
    if metric in SAFETY_MEASURES:
        return "safety"
    return None


class MetricCheck(_Bounded):
    """A retrieval metric or a measure, held to ``min`` and/or ``max``."""

    type: Literal["metric", "budget", "safety"] | None = Field(
        default=None, description="Follows from the metric when left out"
    )
    metric: str = Field(description="recall@5, mrr@10, latency_ms.p95, avoid_violations…")
    rule: Rule = "knowledge-hit-v1"
    checkpoint: Checkpoint = "final"
    where: Where | None = None

    @model_validator(mode="after")
    def _valid(self) -> "MetricCheck":
        family = metric_family(self.metric)
        if family is None:
            raise _invalid("unknown_metric", "unknown metric")
        if self.type is not None and self.type != family:
            raise _invalid("type_mismatch", "the type doesn't match the metric")
        if family == "budget" and self.min is not None:
            raise _invalid("budget_min", "a budget has a max only")
        if self.where is not None and not METRIC_RE.fullmatch(self.metric):
            raise _invalid("where_not_supported", "where applies to retrieval metrics")
        self._bounds()
        return self

    @property
    def kind(self) -> str:
        return metric_family(self.metric) or "metric"


class VsBaseline(_Spec):
    max_drop: float | None = Field(default=None, ge=0, le=1, description="Absolute drop allowed")
    max_relative_drop: float | None = Field(
        default=None, ge=0, le=1, description="Drop allowed as a share of the baseline"
    )
    significance: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Fail only when the paired test's p-value is at most this: noise alone "
        "doesn't fail a build",
    )

    @model_validator(mode="after")
    def _one(self) -> "VsBaseline":
        if (self.max_drop is None) == (self.max_relative_drop is None):
            raise _invalid("drop_missing", "vs_baseline has max_drop or max_relative_drop")
        return self


class RegressionCheck(_Check):
    """A retrieval metric against the baseline, query by query (the paired test)."""

    type: Literal["regression"] | None = None
    metric: str
    rule: Rule = "knowledge-hit-v1"
    checkpoint: Checkpoint = "final"
    where: Where | None = None
    vs_baseline: VsBaseline

    @field_validator("metric")
    @classmethod
    def _retrieval(cls, value: str) -> str:
        if not METRIC_RE.fullmatch(value):
            raise _invalid("not_a_retrieval_metric", "a regression compares a retrieval metric")
        return value

    @property
    def kind(self) -> str:
        return "regression"


class NoOutcome(_Spec):
    """``assert: {no_outcome: [infected, encrypted]}``: the scan outcomes not allowed."""

    no_outcome: list[Outcome] = Field(min_length=1, max_length=6)


class StageCheck(_Bounded):
    """A pipeline stage's metric held to a threshold, or an assertion about the stage."""

    type: Literal["stage", "safety"] | None = None
    stage: Stage
    where: Where | None = None
    metric: str | None = None
    assert_: Literal["all_succeeded", "no_findings", "no_outcome"] | NoOutcome | None = Field(
        default=None,
        alias="assert",
        description="all_succeeded, no_findings, or no_outcome (with outcomes, or as a mapping)",
    )
    outcomes: list[Outcome] = Field(
        default_factory=list, max_length=6, description="no_outcome: scan outcomes not allowed"
    )
    detector: Literal["scan/pii"] | None = Field(
        default=None, description="no_findings: what looks for sensitive data"
    )
    categories: list[PiiCategory] = Field(
        default_factory=list,
        max_length=len(PII_CATEGORIES),
        description="no_findings: the PII categories that count; none means the detector's "
        "defaults (ssn, card_number, iban)",
    )

    @model_validator(mode="before")
    @classmethod
    def _assert_mapping(cls, data: Any) -> Any:
        # ``assert: {no_outcome: [infected, encrypted]}`` is ``assert: no_outcome`` and outcomes.
        if isinstance(data, Mapping) and isinstance(data.get("assert"), Mapping):
            found = dict(data["assert"])
            if list(found) == ["no_outcome"]:
                return {**data, "assert": "no_outcome", "outcomes": found["no_outcome"]}
        return data

    @field_validator("categories")
    @classmethod
    def _canonical_categories(cls, value: list[str]) -> list[str]:
        return canonical_categories(value)

    @model_validator(mode="after")
    def _valid(self) -> "StageCheck":
        if (self.metric is None) == (self.assert_ is None):
            raise _invalid("metric_or_assert", "a stage check has a metric or an assert")
        if self.metric is not None:
            if self.metric not in STAGE_METRICS:
                raise _invalid("unknown_metric", "unknown stage metric")
            if self.metric.split(".")[0] != self.stage:
                raise _invalid("wrong_stage", "the metric belongs to another stage")
            self._bounds()
        else:
            if self.min is not None or self.max is not None:
                raise _invalid("assert_threshold", "an assert has no threshold")
            if self.assert_ == "no_outcome" and (self.stage != "scan" or not self.outcomes):
                raise _invalid("outcomes_missing", "no_outcome is a scan assertion with outcomes")
            if self.assert_ == "no_findings":
                if self.detector is None:
                    raise _invalid("detector_missing", "no_findings names its detector")
                if self.stage not in ("parse", "clean", "chunk", "enrich", "index"):
                    raise _invalid("wrong_stage", "no_findings looks at text: parse to index")
        if self.type == "safety" and self.assert_ != "no_findings":
            raise _invalid("type_mismatch", "a safety stage check asserts no_findings")
        if self.where is not None and self.where.tag is not None:
            raise _invalid("where_not_supported", "stage checks select documents by file_type")
        return self

    @property
    def kind(self) -> str:
        return "safety" if self.assert_ == "no_findings" else "stage"


class AskerSpec(_Spec):
    """Who asks a golden query: a user, roles and groups, or a name from the eval set's
    ``askers`` table (a bare name, or a name without roles or groups)."""

    name: AskerName | None = None
    user: Principal | None = None
    roles: list[Principal] = Field(default_factory=list, max_length=64)
    groups: list[Principal] = Field(default_factory=list, max_length=256)


class Expect(_Spec):
    in_top_k: int = Field(default=10, ge=1, le=100, description="Within the first k passages")
    doc_key: DocKey
    version: DocVersion | None = None
    section: str | None = Field(default=None, max_length=500)
    contains: str | None = Field(
        default=None,
        max_length=500,
        description="A phrase the passage holds (case, accents and spacing ignored)",
    )

    @field_validator("version", mode="before")
    @classmethod
    def _text(cls, value: Any) -> Any:
        return str(value) if isinstance(value, int) and not isinstance(value, bool) else value


class ExpectNot(_Spec):
    in_top_k: int | None = Field(default=None, ge=1, le=100, description="Every passage if none")
    doc_key: DocKey
    version: DocVersion | None = None

    @field_validator("version", mode="before")
    @classmethod
    def _text(cls, value: Any) -> Any:
        return str(value) if isinstance(value, int) and not isinstance(value, bool) else value


class GoldenCheck(_Check):
    """One query's results: a passage it must find, or a document it must not return."""

    type: Literal["golden"] | None = None
    query: str = Field(min_length=1, max_length=2000)
    asker: AskerSpec | AskerName = Field(default_factory=AskerSpec)
    as_of: date | None = Field(default=None, description="The day the query is asked")
    expect: Expect | None = None
    expect_not: ExpectNot | None = None

    @model_validator(mode="after")
    def _something(self) -> "GoldenCheck":
        if self.expect is None and self.expect_not is None:
            raise _invalid("expect_missing", "a golden query has expect or expect_not")
        return self

    @property
    def kind(self) -> str:
        return "golden"

    @property
    def asker_spec(self) -> AskerSpec:
        return self.asker if isinstance(self.asker, AskerSpec) else AskerSpec(name=self.asker)


class DataChangeMax(_Spec):
    added: int | None = Field(default=None, ge=0)
    removed: int | None = Field(default=None, ge=0)
    changed: int | None = Field(default=None, ge=0)
    unparseable: int | None = Field(default=None, ge=0, description="Newly unparseable")
    flagged: int | None = Field(default=None, ge=0, description="Newly flagged by a scan")


class DataChangeCheck(_Check):
    """What changed in the corpus since the baseline; always reported, held to ``max``."""

    type: Literal["data_change"]
    max: DataChangeMax = Field(default_factory=DataChangeMax)

    @property
    def kind(self) -> str:
        return "data_change"


class RevisionGuardCheck(_Check):
    """The embedding model and revision are the baseline's, or a reindex is accepted."""

    type: Literal["revision_guard"] | None = None
    allow_change: bool = Field(
        description="false refuses a new embedding model or revision, which would need every "
        "document embedded again"
    )

    @property
    def kind(self) -> str:
        return "revision_guard"


CUSTOM_RE = re.compile(rf"(?P<path>{CODE_PATH_PATTERN}):(?P<function>[A-Za-z_]\w{{0,63}})")
CODE_PATH_RE = re.compile(CODE_PATH_PATTERN)


def code_path_ok(path: str) -> bool:
    """A relative ``.py`` path without empty, ``.`` or ``..`` segments."""
    return CODE_PATH_RE.fullmatch(path) is not None and all(
        segment not in ("", ".", "..") for segment in path.split("/")
    )


class CustomCheck(_Check):
    """A Python function's verdict, run in the sandbox."""

    type: Literal["custom"] | None = None
    python: str = Field(description="path.py:function; the file is in the suite's code")
    timeout_s: int = Field(default=60, ge=1, le=600)

    @field_validator("python")
    @classmethod
    def _reference(cls, value: str) -> str:
        match = CUSTOM_RE.fullmatch(value)
        if match is None or not code_path_ok(match["path"]):
            raise _invalid("invalid_reference", "python is a relative path.py:function")
        return value

    @property
    def kind(self) -> str:
        return "custom"

    @property
    def path(self) -> str:
        return self.python.rpartition(":")[0]

    @property
    def function(self) -> str:
        return self.python.rpartition(":")[2]


# The fields that say which check a mapping is, when it has no type.
_SHAPES: tuple[tuple[str, str], ...] = (
    ("python", "custom"),
    ("query", "golden"),
    ("vs_baseline", "regression"),
    ("stage", "stage"),
    ("allow_change", "revision_guard"),
    ("metric", "metric"),
)
_TYPE_TAGS = {
    "metric": "metric",
    "budget": "metric",
    "safety": "metric",
    "regression": "regression",
    "stage": "stage",
    "golden": "golden",
    "data_change": "data_change",
    "revision_guard": "revision_guard",
    "custom": "custom",
}


_MODEL_TAGS: dict[type, str] = {
    MetricCheck: "metric",
    RegressionCheck: "regression",
    StageCheck: "stage",
    GoldenCheck: "golden",
    DataChangeCheck: "data_change",
    RevisionGuardCheck: "revision_guard",
    CustomCheck: "custom",
}


def check_tag(data: Any) -> str | None:
    """Which check model a document is: by its ``type``, else by the fields it has. A safety
    check is a metric check or a stage check, by its fields."""
    if isinstance(data, BaseModel):
        return _MODEL_TAGS.get(type(data))
    if not isinstance(data, Mapping):
        return None
    declared = data.get("type")
    if declared == "safety":
        return "stage" if "stage" in data else "metric"
    if isinstance(declared, str):
        return _TYPE_TAGS.get(declared)
    for key, tag in _SHAPES:
        if key in data:
            return tag
    return None


Check = Annotated[
    Annotated[MetricCheck, Tag("metric")]
    | Annotated[RegressionCheck, Tag("regression")]
    | Annotated[StageCheck, Tag("stage")]
    | Annotated[GoldenCheck, Tag("golden")]
    | Annotated[DataChangeCheck, Tag("data_change")]
    | Annotated[RevisionGuardCheck, Tag("revision_guard")]
    | Annotated[CustomCheck, Tag("custom")],
    Discriminator(
        check_tag,
        custom_error_type="unknown_check_type",
        custom_error_message="a check's type, or the fields that make one, wasn't recognized",
    ),
]
AnyCheck = (
    MetricCheck
    | RegressionCheck
    | StageCheck
    | GoldenCheck
    | DataChangeCheck
    | RevisionGuardCheck
    | CustomCheck
)


class TestSuite(_Spec):
    """``stratafi.test-suite.v1``."""

    __test__ = False  # not a pytest class

    schema_: Literal["stratafi.test-suite.v1"] = Field(alias="schema")
    name: Name
    description: str = Field(default="", max_length=2000)
    target: TargetRef
    corpus: DocumentRef = Field(description="A corpus: name, name@version or name@latest")
    eval_set: EvalSetRef | None = Field(
        default=None, description="Needed by checks that score queries"
    )
    baseline: BaselineRef | Literal["last_passing"] | None = None
    time_zone: str = Field(default="UTC", max_length=64, description="Schedules' time zone")
    triggers: list[TriggerSpec] = Field(default_factory=list, max_length=MAX_TRIGGERS)
    checks: list[Check] = Field(min_length=1, max_length=MAX_CHECKS)
    outputs: list[Literal["json", "junit", "markdown"]] = Field(
        default_factory=lambda: ["json", "junit", "markdown"], max_length=3
    )
    code: dict[str, str] = Field(
        default_factory=dict,
        description="Custom checks' Python files by relative path: kept and hashed with the "
        "suite's version. stratafi test run attaches the files the checks name.",
    )

    _corpus = field_validator("corpus")(_check_ref)

    @field_validator("code")
    @classmethod
    def _code(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > MAX_CODE_FILES:
            raise _invalid("too_many_code_files", f"at most {MAX_CODE_FILES} files")
        total = 0
        for path, source in value.items():
            if not code_path_ok(path):
                raise _invalid("code_path_invalid", "a code path is a relative path.py")
            try:
                size = len(source.encode("utf-8"))
            except UnicodeEncodeError:
                raise _invalid("code_not_text", "a file is UTF-8 text") from None
            if "\x00" in source:
                raise _invalid("code_not_text", "a file is UTF-8 text")
            if size > MAX_CODE_BYTES:
                raise _invalid("code_too_large", f"a file holds at most {MAX_CODE_BYTES} bytes")
            total += size
        if total > MAX_CODE_TOTAL_BYTES:
            raise _invalid("code_too_large", f"the files hold at most {MAX_CODE_TOTAL_BYTES} bytes")
        return dict(sorted(value.items()))

    @field_validator("baseline", mode="after")
    @classmethod
    def _baseline_word(cls, value: Any) -> Any:
        return BaselineRef(last_passing=True) if value == "last_passing" else value

    @field_validator("time_zone")
    @classmethod
    def _zone(cls, value: str) -> str:
        try:
            cron.time_zone(value)
        except cron.CronError as error:
            raise _invalid(error.code, "unknown time zone") from None
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "TestSuite":
        ids = [check.id for check in self.checks]
        if len(ids) != len(set(ids)):
            raise _invalid("duplicate_check_id", "check ids must be unique")
        if self.needs_eval_run and self.eval_set is None:
            raise _invalid("eval_set_missing", "these checks score queries: name an eval_set")
        if self.needs_retrieval and self.target.profile is None and self.target.retrieval is None:
            raise _invalid(
                "retrieval_missing", "these checks search: the target names a retrieval config"
            )
        if any(check.path not in self.code for check in self.custom_checks):
            raise _invalid("code_missing", "a custom check's file isn't in the suite's code")
        return self

    @property
    def baseline_ref(self) -> BaselineRef | None:
        assert not isinstance(self.baseline, str)  # noqa: S101 - the validator made it a model
        return self.baseline

    @property
    def needs_eval_run(self) -> bool:
        return any(needs_eval_run(check) for check in self.checks)

    @property
    def needs_retrieval(self) -> bool:
        return self.needs_eval_run

    @property
    def needs_pipeline_run(self) -> bool:
        return bool(self.checks)

    @property
    def needs_baseline(self) -> bool:
        return any(needs_baseline(check) for check in self.checks)

    @property
    def custom_checks(self) -> list[CustomCheck]:
        return [check for check in self.checks if isinstance(check, CustomCheck)]


def code_manifest(suite: TestSuite) -> list[dict[str, Any]]:
    """The suite's code as a run record shows it: each file's path, SHA-256 and size."""
    return [
        {
            "path": path,
            "sha256": sha256_hex(source.encode("utf-8")),
            "bytes": len(source.encode("utf-8")),
        }
        for path, source in sorted(suite.code.items())
    ]


def needs_eval_run(check: AnyCheck) -> bool:
    """Whether a check reads the target's eval run: scores, violations, query latency or cost,
    or the evidence's findability. A golden query searches with the eval run's retrieval, its
    indexes, models and as-of date."""
    if isinstance(check, RegressionCheck | GoldenCheck):
        return True
    if isinstance(check, MetricCheck):
        return check.metric not in ("ingest_cost_usd", "run_duration_s", "ingest_tokens")
    return isinstance(check, StageCheck) and check.metric == "parse.evidence_findability"


def needs_baseline(check: AnyCheck) -> bool:
    return isinstance(check, RegressionCheck | DataChangeCheck | RevisionGuardCheck)


def parse_suite(data: Mapping[str, Any] | TestSuite) -> TestSuite:
    """Validate a suite. Raises ``SpecInvalid`` with paths and codes, never values."""
    if isinstance(data, TestSuite):
        return data
    if not isinstance(data, Mapping):
        raise SpecInvalid([SpecIssue("(root)", "not_an_object")])
    try:
        return TestSuite.model_validate(dict(data))
    except ValidationError as error:
        issues = validation_issues(error, branch_tags=_UNION_TAGS)
    # Pydantic names a check's model after its index (checks.3.golden.asker): drop it.
    cleaned: list[SpecIssue] = []
    for issue in issues:
        parts = issue.path.split(".")
        if len(parts) > 2 and parts[0] == "checks" and parts[2] in _CHECK_TAGS:
            del parts[2]
        found = SpecIssue(".".join(parts), issue.code)
        if found not in cleaned:
            cleaned.append(found)
    raise SpecInvalid(cleaned)


_CHECK_TAGS = frozenset(_TYPE_TAGS.values())
# An asker is an object or a name; validation_issues reports such a union as one field.
_UNION_TAGS = frozenset({"AskerSpec"})


def dump_suite(suite: TestSuite) -> dict[str, Any]:
    """The suite as stored: every default filled in, each check's type written out, and
    ``baseline`` always an object."""
    data = suite.model_dump(mode="json", by_alias=True)
    data["checks"] = [
        {**dumped, "type": check.kind}
        for dumped, check in zip(data["checks"], suite.checks, strict=True)
    ]
    return data


def suite_hash(suite: TestSuite) -> str:
    """The suite's configuration: everything but its name and description."""
    data = dump_suite(suite)
    data.pop("name", None)
    data.pop("description", None)
    return content_hash(data)


def document_hash(suite: TestSuite) -> str:
    return content_hash(dump_suite(suite))


# --- results --------------------------------------------------------------------------------------


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class EvidenceLink(_M):
    """Where a result's evidence is: an eval run, a query in it, a pipeline run, a stage of it,
    a document or a chunk. IDs only."""

    kind: Literal["eval_run", "query", "run", "stage", "document", "chunk", "comparison"]
    id: str = Field(max_length=200)
    run_id: str | None = Field(default=None, description="The run the query, stage or chunk is in")
    label: str | None = Field(default=None, max_length=200, description="A name, never text")


class CheckResult(_M):
    """One check's outcome, with what was measured and what it was held to."""

    id: str
    type: CheckType
    result: Result
    code: str | None = Field(default=None, description="Why it failed, errored or was skipped")
    value: float | None = Field(default=None, description="What was measured")
    threshold: dict[str, Any] = Field(default_factory=dict)
    baseline_value: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict, description="Numbers, codes and counts")
    evidence: list[EvidenceLink] = Field(default_factory=list, max_length=50)
    duration_ms: float | None = None


def _result(
    check: AnyCheck,
    result: Result,
    *,
    code: str | None = None,
    value: float | None = None,
    threshold: Mapping[str, Any] | None = None,
    baseline_value: float | None = None,
    detail: Mapping[str, Any] | None = None,
    evidence: Sequence[EvidenceLink] = (),
) -> CheckResult:
    return CheckResult(
        id=check.id,
        type=check.kind,  # type: ignore[arg-type]
        result=result,
        code=code,
        value=_number(value),
        threshold={key: v for key, v in (threshold or {}).items() if v is not None},
        baseline_value=_number(baseline_value),
        detail=dict(detail or {}),
        evidence=list(evidence)[:50],
    )


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return round(number, 6) if math.isfinite(number) else None


def errored(check: AnyCheck, code: str, **detail: Any) -> CheckResult:
    """A check that couldn't run."""
    return _result(check, "error", code=code, detail=detail)


def skipped(check: AnyCheck, code: str, **detail: Any) -> CheckResult:
    """A check whose condition wasn't met."""
    return _result(check, "skipped", code=code, detail=detail)


def held_to(
    check: AnyCheck,
    value: float | None,
    low: float | None,
    high: float | None,
    *,
    missing: str = "no_value",
    **extra: Any,
) -> CheckResult:
    """``value`` against ``min`` and ``max``: pass, fail (``below_min`` or ``above_max``), or
    error when there's nothing to measure."""
    threshold = {"min": low, "max": high}
    if value is None:
        return _result(check, "error", code=missing, threshold=threshold, **extra)
    if low is not None and value < low - 1e-12:
        return _result(check, "fail", code="below_min", value=value, threshold=threshold, **extra)
    if high is not None and value > high + 1e-12:
        return _result(check, "fail", code="above_max", value=value, threshold=threshold, **extra)
    return _result(check, "pass", value=value, threshold=threshold, **extra)


# --- metric, budget and safety checks ------------------------------------------------------------


@dataclass(frozen=True)
class QueryRow:
    """One query of an eval run, as stored: its tags, its gold documents' file types, and its
    scores (``QueryScore`` as a mapping)."""

    item_id: str
    has_evidence: bool
    tags: Sequence[str]
    file_types: Sequence[str]
    score: Mapping[str, Any]

    def value(self, rule: str, checkpoint: str, name: str) -> float | None:
        found = ((self.score.get("metrics") or {}).get(rule) or {}).get(checkpoint, {}).get(name)
        return float(found) if isinstance(found, int | float) else None

    def covered_by(self, where: Where | None) -> bool:
        if where is None:
            return True
        if where.tag is not None and where.tag not in self.tags:
            return False
        return where.file_type is None or where.file_type in self.file_types


@dataclass
class RunData:
    """What a test run's checks read about one configuration: its eval run's aggregates and
    per-query rows, and its pipeline run's record. IDs, codes and numbers only."""

    eval_run_id: str | None = None
    eval_status: str | None = None
    eval_set_version_id: str | None = None
    metrics: Mapping[str, Any] | None = None
    queries: Sequence[QueryRow] = ()
    eval_manifest: Mapping[str, Any] | None = None
    eval_seconds: float | None = None
    run_id: str | None = None
    run_status: str | None = None
    run_manifest: Mapping[str, Any] | None = None
    run_seconds: float | None = None
    embedding: Mapping[str, Any] | None = None  # component, params and revision
    entries: Sequence["EntryState"] = ()
    tasks: Sequence["TaskState"] = ()
    findings: Mapping[str, Any] = field(default_factory=dict)  # no_findings scans, by check ID


def _tokens(manifest: Mapping[str, Any] | None) -> float | None:
    if not manifest:
        return None
    providers = (manifest.get("cost") or {}).get("providers") or {}
    return float(sum(int((usage or {}).get("tokens") or 0) for usage in providers.values()))


def measure(name: str, data: RunData) -> float | None:
    """A measure (``BUDGET_MEASURES``, ``SAFETY_MEASURES``, ``QUALITY_MEASURES``) from what
    the runs recorded; None if they didn't record it."""
    metrics = data.metrics or {}
    latency = (metrics.get("latency_ms") or {}).get("total_ms") or {}
    violations = metrics.get("violations") or {}
    found: dict[str, Any] = {
        "latency_ms.p50": latency.get("p50"),
        "latency_ms.p95": latency.get("p95"),
        "latency_ms.mean": latency.get("mean"),
        "cost_per_1k_queries_usd": (metrics.get("cost") or {}).get("per_1k_queries"),
        "ingest_cost_usd": ((data.run_manifest or {}).get("cost") or {}).get("estimated"),
        "eval_cost_usd": ((data.eval_manifest or {}).get("cost") or {}).get("estimated"),
        "run_duration_s": data.run_seconds,
        "eval_duration_s": data.eval_seconds,
        "ingest_tokens": _tokens(data.run_manifest),
        "eval_tokens": _tokens(data.eval_manifest),
        "must_not_return_violations": (violations.get("must_not_return") or {}).get("count"),
        "avoid_violations": (violations.get("avoid") or {}).get("count"),
        "findability": (metrics.get("findability") or {}).get("rate"),
    }
    value = found.get(name)
    return float(value) if isinstance(value, int | float) else None


def _query_evidence(data: RunData, rows: Sequence[QueryRow], limit: int = 10) -> list[EvidenceLink]:
    return [
        EvidenceLink(kind="query", id=row.item_id, run_id=data.eval_run_id) for row in rows[:limit]
    ]


def evaluate_metric(
    check: MetricCheck, data: RunData, *, resamples: int = DEFAULT_RESAMPLES, seed: int = 0
) -> CheckResult:
    """A metric, budget or safety threshold on the target's runs."""
    eval_link = [EvidenceLink(kind="eval_run", id=data.eval_run_id)] if data.eval_run_id else []
    if needs_eval_run(check) and data.eval_status != "succeeded":
        return errored(check, f"eval_run_{data.eval_status or 'missing'}")
    if not METRIC_RE.fullmatch(check.metric):
        value = measure(check.metric, data)
        evidence = list(eval_link)
        if check.metric in SAFETY_MEASURES and value:
            flag = (
                "must_not_return_violation"
                if check.metric == "must_not_return_violations"
                else "avoid_violation"
            )
            violating = [row for row in data.queries if row.score.get(flag)]
            evidence += _query_evidence(data, violating)
        if check.metric in ("ingest_cost_usd", "run_duration_s", "ingest_tokens") and data.run_id:
            evidence = [EvidenceLink(kind="run", id=data.run_id)]
        return held_to(check, value, check.min, check.max, evidence=evidence)
    if check.where is None:
        estimate = (
            (((data.metrics or {}).get("metrics") or {}).get(check.rule) or {})
            .get(check.checkpoint, {})
            .get(check.metric)
        )
        if not isinstance(estimate, Mapping):
            return errored(check, "metric_not_recorded")
        detail = {key: estimate.get(key) for key in ("low", "high", "n")}
        return held_to(
            check,
            estimate.get("mean"),
            check.min,
            check.max,
            detail=detail,
            evidence=eval_link + _misses(check, data, data.queries),
        )
    rows = [row for row in data.queries if row.has_evidence and row.covered_by(check.where)]
    values = [row.value(check.rule, check.checkpoint, check.metric) for row in rows]
    kept = [value for value in values if value is not None]
    if not kept:
        return errored(check, "no_queries", queries=0)
    interval = bootstrap_mean(kept, resamples=resamples, seed=seed)
    detail = {"low": interval.low, "high": interval.high, "n": interval.n} if interval else {}
    return held_to(
        check,
        sum(kept) / len(kept),
        check.min,
        check.max,
        detail=detail,
        evidence=eval_link + _misses(check, data, rows),
    )


def _misses(check: MetricCheck, data: RunData, rows: Sequence[QueryRow]) -> list[EvidenceLink]:
    """The queries that scored lowest on the metric: where to look when it fails."""
    scored = [
        (row.value(check.rule, check.checkpoint, check.metric), row.item_id)
        for row in rows
        if row.has_evidence
    ]
    worst = sorted((value, item) for value, item in scored if value is not None and value < 1)
    return [EvidenceLink(kind="query", id=item, run_id=data.eval_run_id) for _, item in worst[:10]]


# --- regression -----------------------------------------------------------------------------------


def evaluate_regression(
    check: RegressionCheck,
    target: RunData,
    baseline: RunData | None,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> CheckResult:
    """The metric against the baseline on the queries both scored (paired). It fails when
    the drop is more than allowed and, with ``significance``, the paired test says it isn't
    noise."""
    if baseline is None:
        return skipped(check, "no_baseline")
    if target.eval_status != "succeeded":
        return errored(check, f"eval_run_{target.eval_status or 'missing'}")
    if baseline.eval_status != "succeeded":
        return errored(check, f"baseline_eval_run_{baseline.eval_status or 'missing'}")
    if target.eval_set_version_id != baseline.eval_set_version_id:
        return errored(check, "baseline_not_paired")

    def values(data: RunData) -> dict[str, float]:
        found = {}
        for row in data.queries:
            if row.has_evidence and row.covered_by(check.where):
                value = row.value(check.rule, check.checkpoint, check.metric)
                if value is not None:
                    found[row.item_id] = value
        return found

    mine, theirs = values(target), values(baseline)
    paired = sorted(set(mine) & set(theirs))
    if not paired:
        return errored(check, "no_paired_queries")
    comparison = compare_paired(
        [theirs[item] for item in paired],
        [mine[item] for item in paired],
        resamples=resamples,
        seed=seed,
    )
    assert comparison is not None  # noqa: S101 - there is at least one pair
    rule = check.vs_baseline
    drop = -comparison.delta
    relative = drop / comparison.mean_a if comparison.mean_a else (0.0 if drop <= 0 else math.inf)
    allowed = rule.max_drop if rule.max_drop is not None else rule.max_relative_drop
    measured = drop if rule.max_drop is not None else relative
    beyond = measured > (allowed or 0.0) + 1e-12
    significant = rule.significance is None or comparison.p_value <= rule.significance
    detail = {
        **comparison.as_dict(),
        "drop": round(drop, 6),
        "relative_drop": round(relative, 6) if math.isfinite(relative) else None,
        "significant": significant,
    }
    threshold = {
        "max_drop": rule.max_drop,
        "max_relative_drop": rule.max_relative_drop,
        "significance": rule.significance,
    }
    lost = sorted((mine[item] - theirs[item], item) for item in paired if mine[item] < theirs[item])
    evidence = [
        EvidenceLink(kind="comparison", id=f"{baseline.eval_run_id}..{target.eval_run_id}"),
        *[EvidenceLink(kind="query", id=item, run_id=target.eval_run_id) for _, item in lost[:10]],
    ]
    if beyond and significant:
        result: Result = "fail"
        code: str | None = "regression"
    else:
        result, code = "pass", ("not_significant" if beyond else None)
    return _result(
        check,
        result,
        code=code,
        value=comparison.mean_b,
        threshold=threshold,
        baseline_value=comparison.mean_a,
        detail=detail,
        evidence=evidence,
    )


# --- stage checks ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryState:
    """A corpus entry in a pipeline run: its name, bytes, detected type and where it stopped."""

    name: str
    sha256: str
    document_id: str
    media_type: str | None
    state: str
    stop_code: str | None = None
    scan_outcome: str | None = None
    scan_decision: str | None = None
    parse_status: str | None = None


@dataclass(frozen=True)
class TaskState:
    """A stage task of a pipeline run: its stage, status, entry and what its artifact says."""

    stage: str
    status: str
    entry: str | None  # the entry's name; None for corpus-level tasks
    summary: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)


DONE = ("succeeded", "cached")
ENDED = ("succeeded", "cached", "failed")


def _covered_entries(check: StageCheck, data: RunData) -> set[str] | None:
    """The entries a stage check covers (by detected media type), None for all of them."""
    if check.where is None or check.where.file_type is None:
        return None
    return {entry.name for entry in data.entries if entry.media_type == check.where.file_type}


def _stage_tasks(check: StageCheck, data: RunData) -> list[TaskState]:
    covered = _covered_entries(check, data)
    return [
        task
        for task in data.tasks
        if task.stage == check.stage
        and (covered is None or task.entry is None or task.entry in covered)
    ]


def _summed(tasks: Sequence[TaskState], key: str, source: str = "summary") -> float:
    return float(
        sum(
            float((getattr(task, source) or {}).get(key) or 0)
            for task in tasks
            if task.status in DONE
        )
    )


def stage_value(check: StageCheck, data: RunData) -> tuple[float | None, dict[str, Any]]:
    """A stage metric's value over the documents the check covers, and its counts."""
    tasks = _stage_tasks(check, data)
    name = check.metric
    if name == "parse.evidence_findability":
        findability = (data.metrics or {}).get("findability") or {}
        if check.where is not None and check.where.file_type is not None:
            findability = (findability.get("by_file_type") or {}).get(check.where.file_type) or {}
        rate = findability.get("rate")
        counts = {"findable": findability.get("findable"), "of": findability.get("of")}
        return (float(rate) if isinstance(rate, int | float) else None), counts
    if name == "extract.documents":
        return float(sum(1 for task in tasks if task.status in DONE)), {}
    if name in ("scan.flagged", "scan.excluded"):
        decisions = [task.summary.get("decision") for task in tasks if task.status in DONE]
        wanted = ("flag", "exclude") if name == "scan.flagged" else ("exclude",)
        return float(sum(1 for decision in decisions if decision in wanted)), {
            "scanned": len(decisions)
        }
    if name == "parse.success_rate":
        ended = [task for task in tasks if task.status in ENDED]
        done = sum(1 for task in ended if task.status in DONE)
        counts = {"succeeded": done, "ended": len(ended)}
        return (done / len(ended) if ended else None), counts
    if name == "parse.failed":
        return float(sum(1 for task in tasks if task.status == "failed")), {}
    if name == "parse.ocr_pages":
        return _summed(tasks, "ocr_pages"), {}
    if name == "clean.redactions":
        return _summed(tasks, "redactions"), {}
    if name == "chunk.count":
        return _summed(tasks, "chunk_count"), {}
    if name == "chunk.tokens_max":
        done = [float(task.summary.get("tokens_max") or 0) for task in tasks if task.status in DONE]
        return (max(done) if done else None), {}
    covered = _covered_entries(check, data)
    embeds = [
        task
        for task in data.tasks
        if task.stage == "embed"
        and task.status in DONE
        and (covered is None or task.entry in covered)
    ]
    truncated = _summed(embeds, "truncated_inputs")
    if name == "embed.truncated_inputs":
        return truncated, {}
    chunks = _summed(embeds, "chunks")
    counts = {"over_limit": int(truncated), "chunks": int(chunks)}
    return (truncated / chunks if chunks else None), counts


def evaluate_stage(check: StageCheck, data: RunData) -> CheckResult:
    """A stage metric against its threshold, or a stage assertion."""
    if check.metric == "parse.evidence_findability":
        if data.eval_status != "succeeded":
            return errored(check, f"eval_run_{data.eval_status or 'missing'}")
    elif data.run_status is None:
        return errored(check, "run_missing")
    elif data.run_status not in ("succeeded", "failed"):
        return errored(check, f"run_{data.run_status}")
    stage_link = [EvidenceLink(kind="stage", id=check.stage, run_id=data.run_id)]
    if check.metric is not None:
        value, counts = stage_value(check, data)
        return held_to(
            check,
            value,
            check.min,
            check.max,
            missing="no_documents",
            detail=counts,
            evidence=stage_link,
        )
    tasks = _stage_tasks(check, data)
    if check.assert_ == "all_succeeded":
        bad = [task for task in tasks if task.status not in (*DONE, "skipped")]
        evidence = stage_link + [
            EvidenceLink(kind="document", id=task.entry, run_id=data.run_id)
            for task in bad[:20]
            if task.entry
        ]
        counts = {"tasks": len(tasks), "not_succeeded": len(bad)}
        if not tasks:
            return errored(check, "no_tasks", **counts)
        result: Result = "fail" if bad else "pass"
        return _result(
            check,
            result,
            code="not_all_succeeded" if bad else None,
            value=len(bad),
            detail=counts,
            evidence=evidence,
        )
    if check.assert_ == "no_outcome":
        covered = _covered_entries(check, data)
        hits = [
            entry
            for entry in data.entries
            if (covered is None or entry.name in covered) and entry.scan_outcome in check.outcomes
        ]
        by_outcome: dict[str, int] = {}
        for entry in hits:
            by_outcome[str(entry.scan_outcome)] = by_outcome.get(str(entry.scan_outcome), 0) + 1
        evidence = stage_link + [
            EvidenceLink(
                kind="document", id=entry.document_id, run_id=data.run_id, label=entry.name
            )
            for entry in hits[:20]
        ]
        return _result(
            check,
            "fail" if hits else "pass",
            code="outcome_found" if hits else None,
            value=len(hits),
            threshold={"outcomes": list(check.outcomes)},
            detail={"by_outcome": by_outcome},
            evidence=evidence,
        )
    found = data.findings.get(check.id)
    if not isinstance(found, Mapping):
        return errored(check, "not_scanned")
    if found.get("error"):
        return errored(check, str(found["error"]))
    total = int(found.get("findings") or 0)
    chunk_ids = [str(chunk) for chunk in found.get("chunks") or []]
    evidence = stage_link + [
        EvidenceLink(kind="chunk", id=chunk, run_id=data.run_id) for chunk in chunk_ids[:20]
    ]
    return _result(
        check,
        "fail" if total else "pass",
        code="findings" if total else None,
        value=total,
        threshold={"categories": list(check.categories)} if check.categories else {},
        detail={
            "scanned": found.get("scanned"),
            "by_category": dict(found.get("by_category") or {}),
        },
        evidence=evidence,
    )


# --- golden queries -------------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenPassage:
    """A passage the golden query returned, in rank order. ``text`` is read to check
    ``contains`` and never leaves this function."""

    chunk_id: str
    doc_key: str | None
    version: str | None
    section: str | None
    text: str


def normalized(text: str) -> str:
    """Text for ``contains``: case folded, accents and spacing ignored."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(plain.casefold().split())


def _matches(passage: GoldenPassage, doc_key: str, version: str | None) -> bool:
    return passage.doc_key == doc_key and (version is None or passage.version == version)


def judge_golden(
    check: GoldenCheck, passages: Sequence[GoldenPassage], *, run_id: str | None = None
) -> CheckResult:
    """``expect``: a passage of the document (and version, section and phrase, if given) within
    the first ``in_top_k``. ``expect_not``: no passage of the document there (every passage when
    it gives no k)."""
    detail: dict[str, Any] = {"returned": len(passages)}
    evidence: list[EvidenceLink] = []
    failures: list[str] = []
    rank: int | None = None
    if check.expect is not None:
        expect = check.expect
        phrase = normalized(expect.contains) if expect.contains else None
        for position, passage in enumerate(passages[: expect.in_top_k], start=1):
            if not _matches(passage, expect.doc_key, expect.version):
                continue
            if expect.section is not None and passage.section != expect.section:
                continue
            if phrase is not None and phrase not in normalized(passage.text):
                continue
            rank = position
            evidence.append(EvidenceLink(kind="chunk", id=passage.chunk_id, run_id=run_id))
            break
        detail["rank"] = rank
        if rank is None:
            failures.append("not_found")
            evidence += [
                EvidenceLink(kind="chunk", id=passage.chunk_id, run_id=run_id)
                for passage in passages[: min(expect.in_top_k, 5)]
            ]
    if check.expect_not is not None:
        expect_not = check.expect_not
        window = passages[: expect_not.in_top_k] if expect_not.in_top_k else passages
        unwanted = [
            (position, passage)
            for position, passage in enumerate(window, start=1)
            if _matches(passage, expect_not.doc_key, expect_not.version)
        ]
        detail["unwanted_rank"] = unwanted[0][0] if unwanted else None
        if unwanted:
            failures.append("returned_unwanted")
            evidence += [
                EvidenceLink(kind="chunk", id=passage.chunk_id, run_id=run_id)
                for _, passage in unwanted[:5]
            ]
    threshold = {
        "in_top_k": check.expect.in_top_k if check.expect else None,
        "not_in_top_k": (check.expect_not.in_top_k or "all") if check.expect_not else None,
    }
    return _result(
        check,
        "fail" if failures else "pass",
        code=failures[0] if failures else None,
        value=rank,
        threshold=threshold,
        detail=detail,
        evidence=evidence,
    )


# --- data change and revision guard ---------------------------------------------------------------


def data_change(target: Sequence[EntryState], baseline: Sequence[EntryState]) -> dict[str, Any]:
    """What changed between the baseline's corpus entries and the target's, by entry name:
    counts, and the names of what changed (synthetic file names, never content)."""
    before = {entry.name: entry for entry in baseline}
    after = {entry.name: entry for entry in target}
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(
        name for name in set(after) & set(before) if after[name].sha256 != before[name].sha256
    )

    def unparseable(entry: EntryState | None) -> bool:
        return entry is not None and entry.parse_status == "failed"

    def flagged(entry: EntryState | None) -> bool:
        return entry is not None and entry.scan_decision in ("flag", "exclude")

    newly_unparseable = sorted(
        name
        for name, entry in after.items()
        if unparseable(entry) and not unparseable(before.get(name))
    )
    newly_flagged = sorted(
        name for name, entry in after.items() if flagged(entry) and not flagged(before.get(name))
    )
    names = {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unparseable": newly_unparseable,
        "flagged": newly_flagged,
    }
    return {
        "counts": {key: len(value) for key, value in names.items()},
        "names": {key: value[:20] for key, value in names.items()},
    }


def evaluate_data_change(
    check: DataChangeCheck,
    target: RunData,
    baseline: RunData | None,
) -> CheckResult:
    if baseline is None:
        return skipped(check, "no_baseline")
    if target.run_status is None or baseline.run_status is None:
        return errored(check, "run_missing")
    change = data_change(target.entries, baseline.entries)
    counts = change["counts"]
    limits = check.max.model_dump(exclude_none=True)
    over = sorted(key for key, limit in limits.items() if counts[key] > limit)
    return _result(
        check,
        "fail" if over else "pass",
        code="over_max" if over else None,
        value=sum(counts.values()),
        threshold=limits,
        detail={**change, "over": over},
        evidence=[
            EvidenceLink(kind="run", id=run_id)
            for run_id in (baseline.run_id, target.run_id)
            if run_id
        ],
    )


def evaluate_revision(
    check: RevisionGuardCheck, target: RunData, baseline: RunData | None
) -> CheckResult:
    """The target's embedding model and revision against the baseline's. A change needs every
    document embedded again: vectors from two models can't share an index."""
    if baseline is None:
        return skipped(check, "no_baseline")
    if not target.embedding or not baseline.embedding:
        return errored(check, "embedding_unknown")
    keys = ("component", "model", "revision")
    mine = {key: target.embedding.get(key) for key in keys}
    theirs = {key: baseline.embedding.get(key) for key in keys}
    changed = sorted(key for key in keys if mine[key] != theirs[key])
    detail = {
        "changed": changed,
        "target": mine,
        "baseline": theirs,
        "reindex_required": bool(changed),
    }
    if changed and not check.allow_change:
        result: Result = "fail"
        code: str | None = "reindex_required"
    else:
        result, code = "pass", ("reindex_planned" if changed else None)
    return _result(
        check,
        result,
        code=code,
        threshold={"allow_change": check.allow_change},
        detail=detail,
        evidence=[
            EvidenceLink(kind="run", id=run_id)
            for run_id in (baseline.run_id, target.run_id)
            if run_id
        ],
    )


# --- custom checks --------------------------------------------------------------------------------

# What the sandbox's entry point reports when a check couldn't give a verdict
# (docker/sandbox/run_check.py), and what the worker's runner adds (stratafi_worker.sandbox).
CUSTOM_ERRORS = frozenset(
    {
        "usage",
        "code_unreadable",
        "function_missing",
        "check_raised",
        "not_a_check_result",
        "timeout",
        "killed",
        "output_too_large",
        "output_invalid",
        "no_result",
        "sandbox_failed",
        "sandbox_unavailable",
    }
)
MAX_CUSTOM_DETAIL_CHARS = 4096
_CUSTOM_CODE_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_CUSTOM_KEY_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,63}")


def _token_detail(value: Any, depth: int = 0) -> Any:
    """``value`` as a result's detail may hold it: numbers, booleans, and token-shaped strings
    (IDs, codes, file names) in lists and mappings of at most 50 items, 5 levels deep. A string
    that could be text becomes None; a key that could hold text is dropped."""
    if depth > 4:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) < 2**53 else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value if safe_value(value) is value else None
    if isinstance(value, Mapping):
        kept: dict[str, Any] = {}
        for key, item in list(value.items())[:50]:
            if isinstance(key, str) and _CUSTOM_KEY_RE.fullmatch(key) and not is_text_field(key):
                kept[key] = _token_detail(item, depth + 1)
        return kept
    if isinstance(value, list):
        return [_token_detail(item, depth + 1) for item in value[:50]]
    return None


def custom_result(
    check: CustomCheck, payload: Any, *, evidence: Sequence[EvidenceLink] = ()
) -> CheckResult:
    """A custom check's result from what the sandbox printed: ``{"result": {passed, value,
    code, detail}}`` or ``{"error": code}``. The payload is the check's own output, so nothing
    in it is trusted: unknown shapes are ``output_invalid``, and the detail keeps only
    metadata, at most 4 KB of it."""
    if not isinstance(payload, Mapping):
        return errored(check, "output_invalid")
    if "error" in payload:
        code = payload.get("error")
        if not isinstance(code, str) or code not in CUSTOM_ERRORS:
            return errored(check, "output_invalid")
        exception = payload.get("exception")
        if code == "check_raised" and isinstance(exception, str):
            name = exception if re.fullmatch(r"[A-Za-z_]\w{0,63}", exception) else None
            return errored(check, code, exception=name)
        return errored(check, code)
    verdict = payload.get("result")
    if not isinstance(verdict, Mapping):
        return errored(check, "output_invalid")
    passed, value, code = verdict.get("passed"), verdict.get("value"), verdict.get("code")
    if passed is not None and not isinstance(passed, bool):
        return errored(check, "output_invalid")
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        return errored(check, "output_invalid")
    if code is not None and not (isinstance(code, str) and _CUSTOM_CODE_RE.fullmatch(code)):
        code = "check_code_invalid"
    raw = verdict.get("detail")
    detail = _token_detail(raw) if isinstance(raw, Mapping) else {}
    if len(json.dumps(detail)) > MAX_CUSTOM_DETAIL_CHARS:
        detail = {"truncated": True}
    if passed is None:
        result: Result = "skipped"
        code = code or "skipped"
    else:
        result = "pass" if passed else "fail"
        code = None if passed else (code or "assertion_failed")
    return _result(
        check,
        result,
        code=code,
        value=float(value) if value is not None else None,
        detail=detail,
        evidence=list(evidence),
    )


# --- the record -----------------------------------------------------------------------------------


class SuiteRef(_M):
    id: str
    name: str
    version: int
    document_hash: str


class TestRunInfo(_M):
    __test__ = False

    id: str
    suite: SuiteRef
    trigger: Trigger
    triggers: list[str] = Field(default_factory=list, description="Triggers combined into it")
    status: str
    error_code: str | None = None
    created_by: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = None


class Summary(_M):
    checks: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0

    @classmethod
    def of(cls, results: Sequence[CheckResult]) -> "Summary":
        counts = {name: 0 for name in ("pass", "fail", "error", "skipped")}
        for result in results:
            counts[result.result] += 1
        return cls(
            checks=len(results),
            passed=counts["pass"],
            failed=counts["fail"],
            errored=counts["error"],
            skipped=counts["skipped"],
        )

    @property
    def outcome(self) -> Literal["passed", "failed"]:
        return "failed" if self.failed or self.errored else "passed"


class TestRunRecord(_M):
    """``stratafi.test-run.v1``."""

    __test__ = False

    schema_: Literal["stratafi.test-run.v1"] = Field(default=TEST_RUN_SCHEMA, alias="schema")
    test_run: TestRunInfo
    resolved: dict[str, Any] = Field(
        default_factory=dict,
        description="Target, corpus version, eval set version and baseline, as fixed IDs",
    )
    runs: dict[str, str | None] = Field(
        default_factory=dict, description="The pipeline and eval runs, the baseline's too"
    )
    summary: Summary = Field(default_factory=Summary)
    outcome: Literal["passed", "failed"] | None = Field(
        default=None, description="failed if a check failed or errored"
    )
    checks: list[CheckResult] = Field(default_factory=list)
    data_change: dict[str, Any] | None = Field(
        default=None, description="What changed in the corpus since the baseline"
    )
    code: list[dict[str, Any]] = Field(
        default_factory=list, description="Custom checks' code: path and SHA-256"
    )

    def fingerprint(self) -> str:
        return content_hash(self.model_dump(mode="json", by_alias=True))


def outcome_of(results: Sequence[CheckResult]) -> Literal["passed", "failed"]:
    return Summary.of(results).outcome


# --- reports --------------------------------------------------------------------------------------


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def threshold_text(result: CheckResult) -> str:
    """A result's threshold as people read it: ``>= 0.85``, ``<= 800``, ``drop <= 0.02``."""
    threshold = result.threshold
    parts = []
    if "min" in threshold:
        parts.append(f">= {_format(threshold['min'])}")
    if "max" in threshold and not isinstance(threshold["max"], Mapping):
        parts.append(f"<= {_format(threshold['max'])}")
    if "max_drop" in threshold:
        parts.append(f"drop <= {_format(threshold['max_drop'])}")
    if "max_relative_drop" in threshold:
        parts.append(f"relative drop <= {_format(threshold['max_relative_drop'])}")
    if "significance" in threshold:
        parts.append(f"p <= {_format(threshold['significance'])}")
    if "in_top_k" in threshold:
        parts.append(f"in top {threshold['in_top_k']}")
    if "not_in_top_k" in threshold:
        parts.append(f"not in top {threshold['not_in_top_k']}")
    if "allow_change" in threshold:
        parts.append("change allowed" if threshold["allow_change"] else "no change")
    if "outcomes" in threshold:
        parts.append("none of " + ", ".join(threshold["outcomes"]))
    if "categories" in threshold:
        parts.append("none of " + ", ".join(threshold["categories"]))
    if not parts and threshold:
        parts.append(", ".join(f"{key} <= {value}" for key, value in sorted(threshold.items())))
    return ", ".join(parts)


def result_message(result: CheckResult) -> str:
    """One line about a result, from its code and numbers only."""
    parts = [f"{result.id}: {result.result}"]
    if result.code:
        parts.append(f"({result.code})")
    if result.value is not None:
        parts.append(f"value {_format(result.value)}")
    threshold = threshold_text(result)
    if threshold:
        parts.append(f"threshold {threshold}")
    if result.baseline_value is not None:
        parts.append(f"baseline {_format(result.baseline_value)}")
    return " ".join(parts)


def junit_xml(record: TestRunRecord) -> bytes:
    """JUnit XML (the Ant format CI systems read): the suite as a ``testsuite``, each check a
    ``testcase`` with a ``failure``, an ``error`` or ``skipped``. A run that couldn't evaluate
    its checks is one ``run`` test case with an ``error``, so CI fails visibly."""
    summary = record.summary
    broken = record.test_run.error_code is not None and not record.checks
    counts = {
        "tests": str(summary.checks + broken),
        "failures": str(summary.failed),
        "errors": str(summary.errored + broken),
        "skipped": str(summary.skipped),
    }
    seconds = (record.test_run.duration_ms or 0.0) / 1000
    root = ET.Element("testsuites", {"name": "stratafi", **counts, "time": f"{seconds:.3f}"})
    started = record.test_run.started_at or record.test_run.created_at
    suite = ET.SubElement(
        root,
        "testsuite",
        {
            "name": record.test_run.suite.name,
            **counts,
            "time": f"{seconds:.3f}",
            "timestamp": started.astimezone(UTC).replace(tzinfo=None, microsecond=0).isoformat(),
            "hostname": "stratafi",
            "id": "0",
            "package": "stratafi",
        },
    )
    properties = ET.SubElement(suite, "properties")
    for name, value in (
        ("stratafi.test_run", record.test_run.id),
        ("stratafi.suite_version", str(record.test_run.suite.version)),
        ("stratafi.suite_hash", record.test_run.suite.document_hash),
        ("stratafi.trigger", record.test_run.trigger),
        ("stratafi.outcome", record.outcome or record.test_run.status),
        *((f"stratafi.run.{key}", value or "") for key, value in sorted(record.runs.items())),
    ):
        ET.SubElement(properties, "property", {"name": name, "value": value})
    for result in record.checks:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "name": result.id,
                "classname": f"stratafi.{record.test_run.suite.name}.{result.type}",
                "time": f"{(result.duration_ms or 0.0) / 1000:.3f}",
            },
        )
        message = result_message(result)
        if result.result == "fail":
            ET.SubElement(case, "failure", {"message": message, "type": result.code or "fail"})
        elif result.result == "error":
            ET.SubElement(case, "error", {"message": message, "type": result.code or "error"})
        elif result.result == "skipped":
            ET.SubElement(case, "skipped", {"message": result.code or "skipped"})
    if broken:
        code = str(record.test_run.error_code)
        case = ET.SubElement(
            suite,
            "testcase",
            {"name": "run", "classname": f"stratafi.{record.test_run.suite.name}", "time": "0.000"},
        )
        ET.SubElement(case, "error", {"message": f"the run couldn't finish ({code})", "type": code})
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


RESULT_WORDS = {"pass": "pass", "fail": "**fail**", "error": "**error**", "skipped": "skipped"}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_summary(record: TestRunRecord, *, link_base: str | None = None) -> str:
    """A Markdown summary for CI pages and pull requests: the outcome, a table of checks, and
    links to the evidence of what failed (``link_base`` is the lab's address)."""
    run = record.test_run
    summary = record.summary
    title = "passed" if record.outcome == "passed" else "failed"
    if record.outcome is None:
        title = run.status
    lines = [
        f"## {run.suite.name}: {title}",
        "",
        f"Suite version {run.suite.version}, test run `{run.id}`, trigger `{run.trigger}`. "
        f"{summary.checks} checks: {summary.passed} passed, {summary.failed} failed, "
        f"{summary.errored} errors, {summary.skipped} skipped.",
    ]
    if run.error_code:
        lines += ["", f"The run couldn't finish: `{run.error_code}`."]
    if record.checks:
        lines += [
            "",
            "| Check | Type | Result | Value | Threshold | Baseline | Code |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for result in record.checks:
            lines.append(
                "| "
                + " | ".join(
                    _cell(text)
                    for text in (
                        result.id,
                        result.type,
                        RESULT_WORDS[result.result],
                        _format(result.value),
                        threshold_text(result),
                        _format(result.baseline_value),
                        result.code or "",
                    )
                )
                + " |"
            )
    problems = [result for result in record.checks if result.result in ("fail", "error")]
    if problems:
        lines += ["", "### Evidence", ""]
        for result in problems:
            links = [_link(link, link_base) for link in result.evidence[:5]]
            lines.append(f"- **{result.id}**: " + (", ".join(links) if links else "no evidence"))
    if record.data_change:
        counts = record.data_change.get("counts") or {}
        lines += [
            "",
            "Corpus changes since the baseline: "
            + ", ".join(f"{counts.get(key, 0)} {key}" for key in DATA_CHANGES)
            + ".",
        ]
    if record.code:
        lines += ["", "Custom check code:", ""]
        lines += [
            f"- `{_cell(str(item.get('path')))}`: SHA-256 `{item.get('sha256')}`, "
            f"{item.get('bytes')} bytes"
            for item in record.code
        ]
    return "\n".join(lines) + "\n"


def evidence_path(link: EvidenceLink) -> str:
    """Where the lab shows a piece of evidence, relative to its address."""
    if link.kind == "eval_run":
        return f"eval-runs/{link.id}"
    if link.kind == "query":
        return f"eval-runs/{link.run_id}?query={link.id}"
    if link.kind == "comparison":
        first, _, second = link.id.partition("..")
        return f"compare?a={first}&b={second}"
    if link.kind in ("run", "stage"):
        run_id = link.id if link.kind == "run" else link.run_id
        return f"runs/{run_id}" + (f"?stage={link.id}" if link.kind == "stage" else "")
    if link.kind == "chunk":
        return f"runs/{link.run_id}?chunk={link.id}"
    return f"runs/{link.run_id}?document={link.id}"


def _link(link: EvidenceLink, base: str | None) -> str:
    # IDs only: a report is shared further than the lab, so it names no document.
    label = f"{link.kind} {link.id}"
    if base is None:
        return f"`{label}`"
    return f"[{_cell(label)}]({base.rstrip('/')}/{evidence_path(link)})"
