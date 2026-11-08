"""Experiments: a space of configurations against one eval set.

``stratafi.experiment.v1`` names a corpus version, an eval set's dev split, a base pipeline spec
and retrieval config, an objective, constraints, factors and a budget. Each factor is a path
into one of the two base documents with the levels it takes:

- a path starting with a pipeline stage (``parse``, ``chunk``, ``embed``…) or with ``stages``
  goes into the pipeline spec: ``chunk.params.max_tokens``, ``parse.route."application/pdf"``
- a path starting with a retrieval config field (``rerank``, ``fusion``, ``top_k``…) goes into
  the retrieval config
- ``pipeline.`` and ``retrieval.`` say which one explicitly

Segments are separated by dots; a segment with dots in it is quoted. A level set on a component
reference (``chunk``, ``rerank``, a parse route) replaces its component and parameters and keeps
the rest (the chunk stage's ``index_text``); a bare component ID means "with its defaults".

Planning, all pure functions:

- ``plan_trials`` expands the factors in order (the first factor varies slowest), drops the
  combinations an ``exclude`` rule matches, applies the strategy (``grid``, or ``random`` with
  ``n`` and a seed; ``successive-halving`` keeps every combination, or ``n`` of them, and
  ``halving_plan`` lays out its rungs), and resolves each trial's pipeline spec and retrieval
  config with the same validator as everything else. A trial that doesn't validate is an
  issue, as is a level that resolves to the same configuration as another.
- ``unique_work`` walks every corpus entry through every distinct pipeline with the run
  planner (``stratafi_core.planner``), so each task gets the cache key a run would give it.
  Shared prefixes have the same keys, so the distinct keys are the work the experiment does;
  the ones whose artifacts exist are cached. Trials that differ only in retrieval share a
  pipeline run.
- ``estimate`` turns the work into time and cost: past task durations per component (median),
  else a default per stage; embedding tokens from the cleaned text's size, else the document's;
  and the price table.

``measures`` reads a finished trial's numbers from its eval run's metrics, and
``check_constraints`` holds them to the constraints. Everything here is configuration, IDs and
numbers: never a document's or a query's text.
"""

import copy
import itertools
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from stratafi_core.adaptive import (
    DEFAULT_ETA,
    DEFAULT_MIN_QUERIES,
    halving_schedule,
    nominal_cost,
)
from stratafi_core.components import COMPONENT_ID_RE, ComponentRegistry
from stratafi_core.evalruns import EvalOptions
from stratafi_core.hashing import canonical_json, content_hash
from stratafi_core.planner import (
    DOCUMENT_STAGES,
    CorpusRef,
    EntryInput,
    StageOutput,
    Stop,
    TaskSpec,
    Wait,
    index_tasks,
    next_task,
)
from stratafi_core.specs import (
    NAME_PATTERN,
    PipelineSpec,
    RetrievalConfig,
    SpecInvalid,
    SpecIssue,
    diff_documents,
    dump_document,
    resolve_pipeline,
    resolve_retrieval,
    spec_hash,
    validation_issues,
)

EXPERIMENT_SCHEMA = "stratafi.experiment.v1"
MAX_FACTORS = 8
MAX_LEVELS = 24
MAX_TRIALS = 1000
GRID_LIMIT = 20_000
MAX_PATH_DEPTH = 12

PIPELINE_STAGES = ("extract", "scan", "parse", "clean", "chunk", "enrich", "embed", "index")
RETRIEVAL_FIELDS = ("retrievers", "fusion", "candidates", "filters", "rerank", "top_k")
METRIC_RE = re.compile(r"(hit|recall|precision|mrr|ndcg)@(\d{1,3}|all)")
REF_RE = re.compile(r"(?P<name>[a-z0-9][a-z0-9._-]{0,63})(?:@(?P<version>[1-9][0-9]{0,8}|latest))?")
# Numbers a trial is measured by besides the retrieval metrics (constraints, Pareto charts).
MEASURES: dict[str, str] = {
    "latency_ms.p50": "Query latency, median, in milliseconds",
    "latency_ms.p95": "Query latency, 95th percentile, in milliseconds",
    "cost_per_1k_queries_usd": "Estimated query-time cost per 1,000 queries",
    "ingest_cost_usd": "Estimated cost of the trial's pipeline run",
    "must_not_return_violations": "Queries that returned a must-not-return document",
    "avoid_violations": "Queries where an avoided version ranked above the first hit",
    "findability": "Share of evidence found in the parsed and cleaned text",
}
# Milliseconds per task when no past task of the component says otherwise.
DEFAULT_TASK_MS: dict[str, float] = {
    "extract": 20.0,
    "scan": 150.0,
    "parse": 400.0,
    "clean": 20.0,
    "chunk": 50.0,
    "enrich": 5.0,
    "embed": 300.0,
    "index": 500.0,
}
DEFAULT_QUERY_MS = 50.0
GATEWAY = "model_gateway"
QUERY_TOKENS = 16  # an embedded query, roughly
CANDIDATE_TOKENS = 200  # a passage the reranker scores, roughly
OVERLAP_FACTOR = 1.15  # chunks overlap and carry their title and heading path

Name = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]
Target = Literal["pipeline", "retrieval"]
Goal = Literal["maximize", "minimize"]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def parse_ref(text: str) -> tuple[str, int | None]:
    """``name``, ``name@3`` or ``name@latest``: the name and the version (None: the latest)."""
    match = REF_RE.fullmatch(text)
    if match is None:
        raise ValueError("a reference is name, name@version or name@latest")
    version = match.group("version")
    return match.group("name"), int(version) if version and version != "latest" else None


def _check_ref(value: str) -> str:
    parse_ref(value)
    return value


DocumentRef = Annotated[str, StringConstraints(max_length=80)]


class EvalSetRef(_Spec):
    name: Name
    version: int | None = Field(default=None, ge=1, description="The latest frozen if none")
    split: Literal["dev"] = Field(
        default="dev",
        description="Experiments tune on dev; the test split only confirms a candidate",
    )


class BaseRef(_Spec):
    pipeline: DocumentRef = Field(description="A saved pipeline spec: name or name@version")
    retrieval: DocumentRef = Field(description="A saved retrieval config: name or name@version")

    _refs = field_validator("pipeline", "retrieval")(_check_ref)


class Objective(_Spec):
    metric: str = Field(description="A retrieval metric: recall@5, mrr@10, ndcg@10…")
    rule: Literal["knowledge-hit-v1", "evidence-coverage-v1", "doc-v1"] = "knowledge-hit-v1"
    checkpoint: Literal["dense", "keyword", "fused", "candidates", "reranked", "final"] = "final"
    goal: Goal = "maximize"

    @field_validator("metric")
    @classmethod
    def _metric(cls, value: str) -> str:
        if not METRIC_RE.fullmatch(value):
            raise ValueError("the objective is a retrieval metric such as recall@5")
        return value


class Constraint(_Spec):
    metric: str = Field(description="A retrieval metric (recall@5) or a measure (latency_ms.p95)")
    min: float | None = None
    max: float | None = None

    @field_validator("metric")
    @classmethod
    def _metric(cls, value: str) -> str:
        if not (METRIC_RE.fullmatch(value) or value in MEASURES):
            raise ValueError("unknown metric")
        return value

    @model_validator(mode="after")
    def _bounded(self) -> "Constraint":
        if self.min is None and self.max is None:
            raise ValueError("a constraint needs min or max")
        return self


class Budget(_Spec):
    max_trials: int = Field(default=60, ge=1, le=MAX_TRIALS, description="Trials evaluated")
    max_cost_usd: float | None = Field(default=None, ge=0, description="Estimated spend")
    max_hours: float | None = Field(default=None, gt=0, le=24 * 14, description="Wall clock")


class Strategy(_Spec):
    method: Literal["grid", "random", "successive-halving"] = "grid"
    n: int | None = Field(
        default=None,
        ge=1,
        le=MAX_TRIALS,
        description="Trials drawn from the grid: random search's, or successive halving's start",
    )
    seed: int = Field(
        default=0, ge=0, le=2**31 - 1, description="Seeds the draw and halving's query order"
    )
    eta: int = Field(
        default=DEFAULT_ETA,
        ge=2,
        le=10,
        description="successive-halving: a rung keeps 1/eta of its trials for eta x the queries",
    )
    min_queries: int = Field(
        default=DEFAULT_MIN_QUERIES,
        ge=5,
        le=10_000,
        description="successive-halving: the first rung's queries",
    )

    @model_validator(mode="after")
    def _options_for_the_method(self) -> "Strategy":
        if self.method == "random" and self.n is None:
            raise ValueError("random search needs n")
        if self.method == "grid" and self.n is not None:
            raise ValueError("a grid runs every combination; n is for random search")
        halving = (self.eta, self.min_queries) != (DEFAULT_ETA, DEFAULT_MIN_QUERIES)
        if self.method != "successive-halving" and halving:
            raise ValueError("eta and min_queries are for successive halving")
        return self


class Execution(_Spec):
    max_parallel_runs: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Pipeline runs at once; with one, every shared prefix runs exactly once",
    )
    max_parallel_evals: int = Field(default=4, ge=1, le=64, description="Eval runs at once")


class ExperimentSpec(_Spec):
    """``stratafi.experiment.v1``."""

    schema_: Literal["stratafi.experiment.v1"] = Field(alias="schema")
    name: Name
    description: str = Field(default="", max_length=2000)
    corpus: DocumentRef = Field(description="A corpus: name, name@version or name@latest")
    eval_set: EvalSetRef
    base: BaseRef
    objective: Objective
    constraints: list[Constraint] = Field(default_factory=list, max_length=20)
    factors: dict[str, list[Any]] = Field(
        min_length=1,
        max_length=MAX_FACTORS,
        description="Paths into the base pipeline spec or retrieval config, each with its levels",
    )
    exclude: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=100,
        description="Combinations to leave out: each maps paths to values, all of which match",
    )
    strategy: Strategy | Literal["grid", "random", "successive-halving"] = Field(
        default_factory=Strategy, description="A method's name stands for it with its defaults"
    )
    budget: Budget = Field(default_factory=Budget)
    execution: Execution = Field(default_factory=Execution)
    eval_options: EvalOptions = Field(default_factory=EvalOptions)

    _corpus = field_validator("corpus")(_check_ref)

    @field_validator("strategy", mode="after")
    @classmethod
    def _strategy_word(cls, value: Any) -> Strategy:
        return Strategy(method=value) if isinstance(value, str) else value

    @field_validator("eval_options", mode="after")
    @classmethod
    def _every_query(cls, value: EvalOptions) -> EvalOptions:
        if value.sample is not None:
            raise ValueError("the strategy decides which queries a trial is scored on")
        return value

    @property
    def search(self) -> Strategy:
        """The strategy, always as an object."""
        assert isinstance(self.strategy, Strategy)  # noqa: S101 - the validator made it one
        return self.strategy


def parse_experiment(data: Mapping[str, Any] | ExperimentSpec) -> ExperimentSpec:
    """Validate an experiment's shape. Raises ``SpecInvalid`` with paths and codes."""
    if isinstance(data, ExperimentSpec):
        return data
    if not isinstance(data, Mapping):
        raise SpecInvalid([SpecIssue("(root)", "not_an_object")])
    try:
        return ExperimentSpec.model_validate(dict(data))
    except ValidationError as error:
        raise SpecInvalid(validation_issues(error)) from None


def dump_experiment(spec: ExperimentSpec) -> dict[str, Any]:
    return spec.model_dump(mode="json", by_alias=True)


# --- factor paths -----------------------------------------------------------------------------

_SEGMENT_RE = re.compile(r'"([^"\x00-\x1f]{1,128})"|([A-Za-z0-9_\-/+\[\],]{1,128})')


@dataclass(frozen=True)
class FactorPath:
    """A path as written, the document it goes into, and its keys from that document's root."""

    text: str
    target: Target
    keys: tuple[str, ...]

    @property
    def stage(self) -> str:
        """The pipeline stage the path changes, or ``query`` for the retrieval config."""
        return self.keys[1] if self.target == "pipeline" and len(self.keys) > 1 else "query"


def _segments(text: str) -> list[str]:
    parts: list[str] = []
    position = 0
    while True:
        match = _SEGMENT_RE.match(text, position)
        if match is None:
            raise ValueError("invalid_path")
        parts.append(match.group(1) if match.group(1) is not None else match.group(2))
        position = match.end()
        if position == len(text):
            return parts
        if text[position] != ".":
            raise ValueError("invalid_path")
        position += 1


def parse_path(text: str) -> FactorPath:
    """A factor or exclusion path. Raises ``ValueError`` with a code: ``invalid_path`` or
    ``unknown_path``."""
    if not isinstance(text, str) or not text:
        raise ValueError("invalid_path")
    parts = _segments(text)
    if len(parts) > MAX_PATH_DEPTH:
        raise ValueError("invalid_path")
    first, rest = parts[0], parts[1:]
    if first == "pipeline" and rest[:1] == ["stages"] and len(rest) > 1:
        return FactorPath(text, "pipeline", tuple(rest))
    if first == "retrieval" and rest and rest[0] in RETRIEVAL_FIELDS:
        return FactorPath(text, "retrieval", tuple(rest))
    if first == "stages" and rest:
        if rest[0] not in PIPELINE_STAGES:
            raise ValueError("unknown_path")
        return FactorPath(text, "pipeline", tuple(parts))
    if first in PIPELINE_STAGES:
        return FactorPath(text, "pipeline", ("stages", *parts))
    if first in RETRIEVAL_FIELDS:
        return FactorPath(text, "retrieval", tuple(parts))
    raise ValueError("unknown_path")


def _index(node: list[Any], key: str) -> int:
    if not key.isdigit() or int(key) >= len(node):
        raise ValueError("no_such_path")
    return int(key)


def read_path(document: Any, keys: Sequence[str]) -> Any:
    """The value at ``keys``, or None when the path doesn't exist."""
    node = document
    for key in keys:
        if isinstance(node, list):
            if not key.isdigit() or int(key) >= len(node):
                return None
            node = node[int(key)]
        elif isinstance(node, Mapping):
            node = node.get(key)
        else:
            return None
    return node


def _is_ref(value: Any) -> bool:
    return isinstance(value, Mapping) and "component" in value


_REPLACED = ("component", "params")


def _merge_level(current: Any, level: Any) -> Any:
    """A level as it lands on ``current``. On a component reference, a component (an ID or a
    reference) replaces the component and its parameters, and a mapping without one replaces
    the keys it names; either way the reference's other keys stay."""
    if _is_ref(current):
        if isinstance(level, str) and COMPONENT_ID_RE.fullmatch(level):
            level = {"component": level}
        if _is_ref(level):
            kept = {key: value for key, value in current.items() if key not in _REPLACED}
            return {**kept, "params": {}, **copy.deepcopy(dict(level))}
        if isinstance(level, Mapping):
            return {**copy.deepcopy(dict(current)), **copy.deepcopy(dict(level))}
    return copy.deepcopy(level)


def set_path(document: dict[str, Any], keys: Sequence[str], level: Any) -> None:
    """Set a level at ``keys``, making mappings on the way. Raises ``ValueError("no_such_path")``
    for a list index that doesn't exist or a path through a value."""
    node: Any = document
    for key in keys[:-1]:
        if isinstance(node, list):
            node = node[_index(node, key)]
            continue
        if not isinstance(node, dict):
            raise ValueError("no_such_path")
        if node.get(key) is None:
            node[key] = {}
        node = node[key]
    last = keys[-1]
    if isinstance(node, list):
        position = _index(node, last)
        node[position] = _merge_level(node[position], level)
    elif isinstance(node, dict):
        node[last] = _merge_level(node.get(last), level)
    else:
        raise ValueError("no_such_path")


def matches(actual: Any, wanted: Any) -> bool:
    """Whether a document's value matches an exclusion's: a bare component ID matches a
    reference to it, a mapping matches the keys it names, anything else must be equal."""
    if isinstance(wanted, str) and _is_ref(actual):
        return actual["component"] == wanted
    if isinstance(wanted, Mapping) and isinstance(actual, Mapping):
        return all(key in actual and matches(actual[key], value) for key, value in wanted.items())
    if isinstance(wanted, bool) or isinstance(actual, bool):
        return actual is wanted
    if isinstance(wanted, int | float) and isinstance(actual, int | float):
        return math.isclose(float(actual), float(wanted), rel_tol=1e-12, abs_tol=1e-12)
    return actual == wanted


def level_label(level: Any, limit: int = 80) -> str:
    """A level as people read it: ``chunk/recursive size=512 overlap=64``, ``embed-local``."""
    if isinstance(level, str):
        text = level
    elif isinstance(level, bool) or level is None:
        text = canonical_json(level)
    elif isinstance(level, int | float):
        text = f"{level:g}" if isinstance(level, float) else str(level)
    elif _is_ref(level):
        params = level.get("params") or {}
        extra = {key: value for key, value in level.items() if key not in ("component", "params")}
        parts = [str(level["component"])]
        parts += [f"{key}={_short(value)}" for key, value in sorted(params.items())]
        parts += [f"{key}={_short(value)}" for key, value in sorted(extra.items())]
        text = " ".join(parts)
    else:
        text = canonical_json(level)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _short(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return f"{value:g}"
    return canonical_json(value)


# --- expanding and resolving trials -------------------------------------------------------------


@dataclass(frozen=True)
class Factor:
    path: FactorPath
    levels: tuple[Any, ...]
    labels: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.text,
            "target": self.path.target,
            "stage": self.path.stage,
            "levels": list(self.levels),
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class PlannedTrial:
    """One configuration: its place in the grid, its level per factor, and its resolved
    documents with their hashes."""

    ordinal: int
    grid_index: int
    levels: tuple[int, ...]
    pipeline: PipelineSpec
    retrieval: RetrievalConfig
    pipeline_hash: str
    retrieval_hash: str

    def values(self, factors: Sequence[Factor]) -> dict[str, str]:
        return {
            factor.path.text: factor.labels[level]
            for factor, level in zip(factors, self.levels, strict=True)
        }


@dataclass(frozen=True)
class TrialPlan:
    factors: list[Factor]
    trials: list[PlannedTrial]
    grid: int  # combinations before exclusions
    excluded: int  # left out by an exclude rule
    issues: list[SpecIssue] = field(default_factory=list)

    @property
    def pipelines(self) -> dict[str, PipelineSpec]:
        """The distinct pipeline specs, by hash, in trial order."""
        found: dict[str, PipelineSpec] = {}
        for trial in self.trials:
            found.setdefault(trial.pipeline_hash, trial.pipeline)
        return found


def parse_factors(spec: ExperimentSpec) -> tuple[list[Factor], list[SpecIssue]]:
    factors: list[Factor] = []
    issues: list[SpecIssue] = []
    for text, levels in spec.factors.items():
        try:
            path = parse_path(text)
        except ValueError as error:
            issues.append(SpecIssue("factors.*", str(error)))
            continue
        where = f"factors.{text}"
        if not 1 <= len(levels) <= MAX_LEVELS:
            issues.append(SpecIssue(where, "levels_count"))
            continue
        seen: set[str] = set()
        for index, level in enumerate(levels):
            key = canonical_json(level)
            if key in seen:
                issues.append(SpecIssue(f"{where}.{index}", "duplicate_level"))
            seen.add(key)
        factors.append(Factor(path, tuple(levels), tuple(level_label(level) for level in levels)))
    for first, second in itertools.combinations(factors, 2):
        a, b = first.path, second.path
        shorter = min(len(a.keys), len(b.keys))
        if a.target == b.target and a.keys[:shorter] == b.keys[:shorter]:
            issues.append(SpecIssue(f"factors.{second.path.text}", "overlapping_factor"))
    return factors, issues


def _exclusions(spec: ExperimentSpec) -> tuple[list[list[tuple[FactorPath, Any]]], list[SpecIssue]]:
    rules: list[list[tuple[FactorPath, Any]]] = []
    issues: list[SpecIssue] = []
    for index, rule in enumerate(spec.exclude):
        parsed = []
        for text, value in rule.items():
            try:
                parsed.append((parse_path(text), value))
            except ValueError as error:
                issues.append(SpecIssue(f"exclude.{index}", str(error)))
        if not rule:
            issues.append(SpecIssue(f"exclude.{index}", "empty_rule"))
        rules.append(parsed)
    return rules, issues


def _excludes(rule: Sequence[tuple[FactorPath, Any]], documents: Mapping[str, Any]) -> bool:
    return all(matches(read_path(documents[path.target], path.keys), value) for path, value in rule)


def plan_trials(
    spec: ExperimentSpec,
    base_pipeline: PipelineSpec,
    base_retrieval: RetrievalConfig,
    registry: ComponentRegistry,
) -> TrialPlan:
    """Expand, exclude, select and resolve the experiment's trials (the module docstring says
    how). Issues don't raise: the caller decides whether a plan with issues may be saved."""
    factors, issues = parse_factors(spec)
    rules, more = _exclusions(spec)
    issues += more
    grid = math.prod(len(factor.levels) for factor in factors) if factors else 0
    if issues:
        return TrialPlan(factors, [], grid, 0, issues)
    if grid > GRID_LIMIT:
        return TrialPlan(factors, [], grid, 0, [SpecIssue("factors", "too_many_combinations")])
    base = {"pipeline": dump_document(base_pipeline), "retrieval": dump_document(base_retrieval)}
    candidates: list[tuple[int, tuple[int, ...], dict[str, Any]]] = []
    excluded = 0
    for grid_index, combo in enumerate(
        itertools.product(*(range(len(factor.levels)) for factor in factors))
    ):
        documents = copy.deepcopy(base)
        for factor, level in zip(factors, combo, strict=True):
            try:
                set_path(documents[factor.path.target], factor.path.keys, factor.levels[level])
            except ValueError as error:
                issue = SpecIssue(f"factors.{factor.path.text}.{level}", str(error))
                return TrialPlan(factors, [], grid, 0, [issue])
        if any(rule and _excludes(rule, documents) for rule in rules):
            excluded += 1
            continue
        candidates.append((grid_index, combo, documents))
    if spec.search.n is not None and spec.search.n < len(candidates):
        chosen = random.Random(spec.search.seed).sample(  # noqa: S311 - a seeded draw
            range(len(candidates)), spec.search.n
        )
        candidates = [candidates[index] for index in sorted(chosen)]
    if len(candidates) > MAX_TRIALS:
        return TrialPlan(factors, [], grid, excluded, [SpecIssue("factors", "too_many_trials")])
    resolved_pipelines: dict[str, PipelineSpec | SpecInvalid] = {}
    resolved_retrievals: dict[str, RetrievalConfig | SpecInvalid] = {}
    trials: list[PlannedTrial] = []
    seen: dict[tuple[str, str], int] = {}
    for grid_index, combo, documents in candidates:
        pipeline = _resolve(documents["pipeline"], resolved_pipelines, resolve_pipeline, registry)
        retrieval = _resolve(
            documents["retrieval"], resolved_retrievals, resolve_retrieval, registry
        )
        problems = [
            SpecIssue(f"trials.{grid_index}.{target}.{issue.path}", issue.code)
            for target, result in (("pipeline", pipeline), ("retrieval", retrieval))
            if isinstance(result, SpecInvalid)
            for issue in result.issues
        ]
        if problems or not (
            isinstance(pipeline, PipelineSpec) and isinstance(retrieval, RetrievalConfig)
        ):
            issues += problems
            continue
        hashes = (spec_hash(pipeline), spec_hash(retrieval))
        if hashes in seen:
            issues.append(SpecIssue(f"trials.{grid_index}", "duplicate_trial"))
            continue
        seen[hashes] = grid_index
        trials.append(
            PlannedTrial(
                ordinal=len(trials),
                grid_index=grid_index,
                levels=combo,
                pipeline=pipeline,
                retrieval=retrieval,
                pipeline_hash=hashes[0],
                retrieval_hash=hashes[1],
            )
        )
    if not trials and not issues:
        issues.append(SpecIssue("exclude", "every_trial_excluded"))
    return TrialPlan(factors, trials, grid, excluded, issues)


def _resolve(
    document: Mapping[str, Any],
    cache: dict[str, Any],
    resolver: Callable[[Mapping[str, Any], ComponentRegistry], Any],
    registry: ComponentRegistry,
) -> Any:
    """A document resolved once however many trials share it: the result or its error."""
    key = canonical_json(document)
    if key not in cache:
        try:
            cache[key] = resolver(document, registry)
        except SpecInvalid as error:
            cache[key] = error
    return cache[key]


def variant_name(base: str, configuration_hash: str) -> str:
    """The name a trial's variant of a base document is saved under: the base's name and a
    digest of the variant's configuration, within the 64 characters a name may have."""
    suffix = f".x{configuration_hash[:10]}"
    return base[: 64 - len(suffix)] + suffix


def variant_description(base: str, version: int, before: Any, after: Any) -> str:
    """Names the base and the paths a variant changes; the same variant always gets the same
    description, so saving it again adds no version."""
    paths = sorted({change.path for change in diff_documents(before, after)} - {"name"})
    changed = ", ".join(paths) if paths else "nothing"
    text = f"An experiment's variant of {base} v{version}: {changed}"
    return text if len(text) <= 2000 else text[:1999] + "…"


# --- the work an experiment does ----------------------------------------------------------------


@dataclass(frozen=True)
class PlanEntry:
    """A corpus entry as planning sees it: what the run planner needs, and what intake
    detected, which routing reads until the extract stage has run."""

    entry: EntryInput
    media_type: str | None = None
    traits: Mapping[str, Any] = field(default_factory=dict)
    size_bytes: int = 0
    refused: bool = False
    # How new it is, for corpus-wide clean rules: ``stratafi_store.runs.version_rank``'s.
    rank: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskRef:
    stage: str
    component_id: str
    component_version: str
    resource_class: str
    entry: str | None  # None for corpus-level tasks


@dataclass
class PipelineWork:
    """One distinct pipeline over the corpus: each task's key, in entry order then index."""

    pipeline_hash: str
    keys: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    stopped: Counter[str] = field(default_factory=Counter)

    @property
    def tasks(self) -> int:
        return sum(len(keys) for keys in self.keys.values())


Lookup = Callable[[Sequence[str]], Mapping[str, Mapping[str, Any]]]


def _assumed(task: TaskSpec, item: PlanEntry) -> dict[str, Any]:
    """What a task's summary will hold, as far as planning needs it."""
    if task.stage == "extract":
        return {"media_type": item.media_type or "", "traits": dict(item.traits)}
    if task.stage == "scan":
        return {"decision": "continue"}
    return {}


def walk_pipeline(
    pipeline_hash: str,
    spec: PipelineSpec,
    registry: ComponentRegistry,
    entries: Sequence[PlanEntry],
    *,
    model_env: Mapping[str, Mapping[str, Any]],
    known: Mapping[str, Mapping[str, Any]],
    tasks: dict[str, TaskRef],
    until: str | None = None,
) -> PipelineWork:
    """Plan every entry through ``spec`` as a run would, from what's ``known`` of finished
    artifacts (their summaries by key) and assumptions for the rest: extracted types are
    intake's, scans continue. A corpus-wide clean rule makes entries wait at Clean until every
    entry is parsed, as in a run; then each one's clean task reads every parse key and rank.
    Records each task in ``tasks``."""
    work = PipelineWork(pipeline_hash)
    embedded: list[tuple[str, str]] = []

    def walk(
        item: PlanEntry, outputs: dict[str, StageOutput], corpus: list[CorpusRef] | None
    ) -> bool:
        """Plan the entry on from ``outputs``; False when it waits for the corpus."""
        while True:
            step = next_task(
                spec,
                registry,
                item.entry,
                outputs,
                until=until,
                model_env=model_env,
                corpus=corpus,
            )
            if isinstance(step, Wait):
                return False
            if isinstance(step, Stop):
                if step.reason != "done":
                    work.stopped[step.reason] += 1
                elif "embed" in outputs:
                    embedded.append((outputs["enrich"].cache_key, outputs["embed"].cache_key))
                return True
            work.keys[step.stage].append(step.cache_key)
            tasks.setdefault(
                step.cache_key,
                TaskRef(
                    step.stage,
                    step.component_id,
                    step.component_version,
                    step.resource_class,
                    item.entry.name,
                ),
            )
            summary = known.get(step.cache_key)
            outputs[step.stage] = StageOutput(
                step.cache_key, summary if summary is not None else _assumed(step, item)
            )

    waiting: list[tuple[PlanEntry, dict[str, StageOutput]]] = []
    for item in entries:
        if item.refused:
            work.stopped["refused"] += 1
            continue
        outputs: dict[str, StageOutput] = {}
        if not walk(item, outputs, None):
            waiting.append((item, outputs))
    if waiting:
        corpus: list[CorpusRef] = [
            (outputs["parse"].cache_key, list(item.rank or ("", "", item.entry.name)))
            for item, outputs in waiting
        ]
        for item, outputs in waiting:
            walk(item, outputs, corpus)
    if embedded and until is None:
        for step in index_tasks(spec, registry, embedded, model_env=model_env):
            work.keys["index"].append(step.cache_key)
            tasks.setdefault(
                step.cache_key,
                TaskRef(
                    "index", step.component_id, step.component_version, step.resource_class, None
                ),
            )
    return work


@dataclass(frozen=True)
class StageWork:
    """One stage's work: ``tasks`` if every trial ran alone, ``unique`` across the experiment,
    ``cached`` of those already made, and so ``to_run``."""

    tasks: int
    unique: int
    cached: int

    @property
    def to_run(self) -> int:
        return self.unique - self.cached

    def as_dict(self) -> dict[str, int]:
        return {
            "tasks": self.tasks,
            "unique": self.unique,
            "cached": self.cached,
            "to_run": self.to_run,
        }


@dataclass
class WorkPlan:
    pipelines: dict[str, PipelineWork]
    tasks: dict[str, TaskRef]  # every distinct task, by key
    known: dict[str, dict[str, Any]]  # summaries of the artifacts that exist, by key
    stages: dict[str, StageWork]

    @property
    def to_run(self) -> list[str]:
        return [key for key in self.tasks if key not in self.known]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipelines": len(self.pipelines),
            "stages": {stage: work.as_dict() for stage, work in self.stages.items()},
            "stopped": dict(sum((work.stopped for work in self.pipelines.values()), Counter())),
        }


def unique_work(
    trials: Sequence[PlannedTrial],
    registry: ComponentRegistry,
    entries: Sequence[PlanEntry],
    *,
    model_envs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    lookup: Lookup,
) -> WorkPlan:
    """The distinct stage tasks the trials need, with the cached ones (the module docstring).
    ``model_envs`` gives each pipeline's environment by hash (the embedding revision), and
    ``lookup`` the summaries of the artifacts that exist among some keys."""
    pipelines: dict[str, PipelineSpec] = {}
    for trial in trials:
        pipelines.setdefault(trial.pipeline_hash, trial.pipeline)
    scratch: dict[str, TaskRef] = {}
    early: set[str] = set()
    for pipeline_hash, spec in pipelines.items():
        walked = walk_pipeline(
            pipeline_hash,
            spec,
            registry,
            entries,
            model_env=model_envs.get(pipeline_hash, {}),
            known={},
            tasks=scratch,
            until="scan",
        )
        early.update(walked.keys["extract"], walked.keys["scan"])
    known = {key: dict(summary) for key, summary in lookup(sorted(early)).items()}
    tasks: dict[str, TaskRef] = {}
    works = {
        pipeline_hash: walk_pipeline(
            pipeline_hash,
            spec,
            registry,
            entries,
            model_env=model_envs.get(pipeline_hash, {}),
            known=known,
            tasks=tasks,
        )
        for pipeline_hash, spec in pipelines.items()
    }
    rest = sorted(key for key in tasks if key not in known)
    known.update({key: dict(summary) for key, summary in lookup(rest).items()})
    per_trial: Counter[str] = Counter()
    for trial in trials:
        for stage, keys in works[trial.pipeline_hash].keys.items():
            per_trial[stage] += len(keys)
    unique: Counter[str] = Counter(ref.stage for ref in tasks.values())
    cached: Counter[str] = Counter(ref.stage for key, ref in tasks.items() if key in known)
    stages = {
        stage: StageWork(per_trial[stage], unique[stage], cached[stage])
        for stage in (*DOCUMENT_STAGES, "index")
        if per_trial[stage] or unique[stage]
    }
    return WorkPlan(works, tasks, known, stages)


# --- estimates ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class History:
    """What past work says: median task milliseconds by (component ID, version), median
    milliseconds per query of past eval runs, and past query-time cost per 1,000 queries by
    reranker (component ID and parameters' hash)."""

    task_ms: Mapping[tuple[str, str], float] = field(default_factory=dict)
    query_ms: float | None = None
    rerank_cost_per_1k: Mapping[str, float] = field(default_factory=dict)


def rerank_key(component_id: str, params: Mapping[str, Any]) -> str:
    return f"{component_id}:{content_hash(dict(params))[:16]}"


def _embed_tokens(
    work: WorkPlan, key: str, entries: Mapping[str, PlanEntry], cleaned: Mapping[str, int]
) -> int:
    """An embed task's tokens: its document's cleaned characters if a cleaned artifact of it is
    known (the largest, as an upper bound), else its stored bytes, over four, with overlap."""
    ref = work.tasks[key]
    name = ref.entry or ""
    chars = cleaned.get(name)
    if chars is None:
        entry = entries.get(name)
        chars = entry.size_bytes if entry is not None else 0
    return int(chars / 4 * OVERLAP_FACTOR)


def estimate(
    work: WorkPlan,
    trials: Sequence[PlannedTrial],
    registry: ComponentRegistry,
    entries: Sequence[PlanEntry],
    *,
    queries: int,
    shard_size: int,
    history: History,
    prices: Mapping[str, float],
    slots: Mapping[str, int],
    evaluated: Collection[int] = (),
    query_share: float = 1.0,
) -> dict[str, Any]:
    """Time and cost of the work still to run, and of evaluating the trials, but those in
    ``evaluated`` (ordinals with an identical finished eval run already). ``query_share`` is
    the part of the grid's queries the strategy scores (successive halving's nominal share)."""
    by_name = {item.entry.name: item for item in entries}
    cleaned: dict[str, int] = {}
    for key, ref in work.tasks.items():
        if ref.stage == "clean" and key in work.known and ref.entry is not None:
            chars = int(work.known[key].get("chars", 0) or 0)
            cleaned[ref.entry] = max(cleaned.get(ref.entry, 0), chars)
    price = float(prices.get(GATEWAY, 0.0))
    ms_by_class: Counter[str] = Counter()
    basis: dict[str, str] = {}
    tokens = 0
    for key in work.to_run:
        ref = work.tasks[key]
        past = history.task_ms.get((ref.component_id, ref.component_version))
        ms_by_class[ref.resource_class] += past if past is not None else DEFAULT_TASK_MS[ref.stage]
        if basis.get(ref.stage) != "default":
            basis[ref.stage] = "history" if past is not None else "default"
        if ref.stage == "embed" and registry.get(ref.component_id).info.egress == GATEWAY:
            tokens += _embed_tokens(work, key, by_name, cleaned)
    ingest_cost = tokens / 1000 * price
    pending = [trial for trial in trials if trial.ordinal not in evaluated]
    to_evaluate = len(pending)
    query_ms = history.query_ms if history.query_ms is not None else DEFAULT_QUERY_MS
    eval_cost = 0.0
    for trial in pending:
        rerank = trial.retrieval.rerank
        key = rerank_key(rerank.component, rerank.params)
        if key in history.rerank_cost_per_1k:
            per_query = history.rerank_cost_per_1k[key] / 1000
        else:
            paid = 0
            if registry.get(trial.pipeline.stages.embed.component).info.egress == GATEWAY:
                paid += QUERY_TOKENS
            if registry.get(rerank.component).info.egress == GATEWAY:
                paid += trial.retrieval.candidates * CANDIDATE_TOKENS
            per_query = paid / 1000 * price
        eval_cost += per_query * queries * query_share
    eval_ms = to_evaluate * queries * query_ms * query_share
    ms_by_class["io"] += eval_ms
    wall = sum(ms / max(1, int(slots.get(cls, 1) or 1)) for cls, ms in ms_by_class.items())
    shards = math.ceil(queries / shard_size) if queries else 0
    return {
        "cpu_seconds": round(sum(ms_by_class.values()) / 1000, 3),
        "wall_seconds": round(wall / 1000, 3),
        "embed_tokens": tokens,
        "cost_usd": {
            "ingest": round(ingest_cost, 6),
            "eval": round(eval_cost, 6),
            "total": round(ingest_cost + eval_cost, 6),
        },
        "eval": {
            "trials": len(trials),
            "to_run": to_evaluate,
            "queries": queries,
            "shards": math.ceil(shards * to_evaluate * query_share),
            "query_share": query_share,
        },
        "basis": {
            **basis,
            "eval": "history" if history.query_ms is not None else "default",
        },
    }


def halving_plan(spec: ExperimentSpec, trials: int, queries: int) -> dict[str, Any] | None:
    """Successive halving's rungs for the trials the budget admits over the split's queries,
    and what they nominally score (``stratafi_core.adaptive``); None for another strategy."""
    search = spec.search
    if search.method != "successive-halving":
        return None
    admitted = min(trials, spec.budget.max_trials)
    schedule = halving_schedule(admitted, queries, eta=search.eta, min_queries=search.min_queries)
    return {
        "eta": search.eta,
        "min_queries": search.min_queries,
        "seed": search.seed,
        "rungs": [rung.as_dict() for rung in schedule],
        **nominal_cost(schedule, admitted),
    }


def budget_check(spec: ExperimentSpec, trials: int, estimated: Mapping[str, Any]) -> dict[str, Any]:
    """What the plan would spend against the budget. Running stops at the budget anyway."""
    budget = spec.budget
    cost = float(estimated["cost_usd"]["total"])
    hours = float(estimated["wall_seconds"]) / 3600
    return {
        "trials": {"planned": trials, "max": budget.max_trials, "over": trials > budget.max_trials},
        "cost_usd": {
            "estimated": round(cost, 6),
            "max": budget.max_cost_usd,
            "over": budget.max_cost_usd is not None and cost > budget.max_cost_usd,
        },
        "hours": {
            "estimated": round(hours, 4),
            "max": budget.max_hours,
            "over": budget.max_hours is not None and hours > budget.max_hours,
        },
    }


# --- a trial's numbers ------------------------------------------------------------------------


def metric_estimate(
    metrics: Mapping[str, Any] | None, rule: str, checkpoint: str, name: str
) -> dict[str, Any] | None:
    """A retrieval metric's estimate (mean, low, high, n) from an eval run's metrics."""
    value = (((metrics or {}).get("metrics") or {}).get(rule) or {}).get(checkpoint, {})
    found = value.get(name) if isinstance(value, Mapping) else None
    return dict(found) if isinstance(found, Mapping) else None


def measures(
    metrics: Mapping[str, Any] | None,
    *,
    objective: Objective,
    ingest_cost: float | None,
    names: Iterable[str] = (),
) -> dict[str, float | None]:
    """A finished trial's measures (``MEASURES``), and the retrieval metrics in ``names`` at the
    objective's rule and checkpoint (their means)."""
    metrics = metrics or {}
    latency = (metrics.get("latency_ms") or {}).get("total_ms") or {}
    violations = metrics.get("violations") or {}
    cost = metrics.get("cost") or {}
    found: dict[str, float | None] = {
        "latency_ms.p50": latency.get("p50"),
        "latency_ms.p95": latency.get("p95"),
        "cost_per_1k_queries_usd": cost.get("per_1k_queries"),
        "ingest_cost_usd": ingest_cost,
        "must_not_return_violations": (violations.get("must_not_return") or {}).get("count"),
        "avoid_violations": (violations.get("avoid") or {}).get("count"),
        "findability": (metrics.get("findability") or {}).get("rate"),
    }
    for name in {objective.metric, *names}:
        if METRIC_RE.fullmatch(name):
            estimate_ = metric_estimate(metrics, objective.rule, objective.checkpoint, name)
            found[name] = estimate_["mean"] if estimate_ else None
    return found


def check_constraints(
    constraints: Sequence[Constraint], found: Mapping[str, float | None]
) -> list[dict[str, Any]]:
    """Each constraint with its value and whether it passed; a missing value fails."""
    results = []
    for constraint in constraints:
        value = found.get(constraint.metric)
        passed = value is not None
        if value is not None and constraint.min is not None:
            passed = passed and value >= constraint.min
        if value is not None and constraint.max is not None:
            passed = passed and value <= constraint.max
        results.append(
            {
                "metric": constraint.metric,
                "value": value,
                "min": constraint.min,
                "max": constraint.max,
                "passed": passed,
            }
        )
    return results
