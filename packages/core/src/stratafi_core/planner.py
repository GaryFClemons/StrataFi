"""Planning stage tasks.

``next_task`` is the planner's core, and a pure function: given a resolved pipeline spec, one
corpus entry and the outputs of the stages it has finished, it returns the entry's next stage
task, with its cache key, or why the entry stops:

- ``done``: it reached the run's last stage (``until``)
- ``excluded``: its scan report's decision is ``exclude``, or Clean dropped it as a near
  duplicate of a newer version, so it gets no downstream tasks
- ``unsupported_type``: no parser on its route accepts its type and traits

or ``Wait``: the clean stage has a corpus-wide rule (``clean/near-duplicates``), and the entry
waits until every entry of the run has been parsed. The orchestrator then plans each waiting
entry's clean task with ``corpus``: every parsed entry's parse key and rank (newer is higher),
which its parameters and inputs carry, so the key changes when any document does.

Each document stage reads the previous stage's artifact, so a cache key depends on everything
upstream, and the same work always gets the same key:

| Stage | Inputs | Parameters hashed | Environment |
| --- | --- | --- | --- |
| extract | the document's SHA-256 | the component's | the entry's extension |
| scan | the raw artifact | each check's ID, version, params; the policy | is a duplicate |
| parse | the raw artifact | the routed parser's | a library version |
| clean | the parsed artifact | each rule's ID, version and params | |
| chunk | the cleaned artifact | the chunker's, the index text template, the label | |
| enrich | the chunks | the enricher's | |
| embed | the enriched chunks | the embedder's | the model's revision |

Scan and clean run several components, so their tasks name a composite component
(``scan/checks``, ``clean/rules``) whose parameters list each member with its version: bumping
any member's version changes the key. ``index_tasks`` plans the corpus-level index.

A re-run from a stage passes ``rerun=(stage, nonce)``: that stage and every later one get the
nonce in their keys, so they run again and make new artifacts, while earlier stages still reuse
the cache.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from stratafi_core.components import ComponentRegistry
from stratafi_core.hashing import cache_key
from stratafi_core.specs import ComponentRef, PipelineSpec, accepts, route_for

DOCUMENT_STAGES = ("extract", "scan", "parse", "clean", "chunk", "enrich", "embed")
CORPUS_STAGES = ("index",)
ALL_STAGES = (*DOCUMENT_STAGES, *CORPUS_STAGES)
SCAN_COMPONENT = "scan/checks"
CLEAN_COMPONENT = "clean/rules"
COMPOSITE_VERSION = "1.0.0"

StopReason = Literal["done", "excluded", "unsupported_type"]
CorpusRef = tuple[str, list[str]]  # (parse key, rank)


@dataclass(frozen=True)
class EntryInput:
    """One corpus entry, as the planner sees it."""

    name: str
    document_id: str
    sha256: str
    extension: str
    duplicate: bool = False  # an earlier entry of the corpus version has the same bytes
    label: str | None = None  # names a document without a title: its key, or its file's stem


@dataclass(frozen=True)
class StageOutput:
    """A finished stage: its artifact's key and its summary (metadata only)."""

    cache_key: str
    summary: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    stage: str
    component_id: str
    component_version: str
    params: dict[str, Any]
    inputs: tuple[str, ...]
    env: dict[str, Any]
    cache_key: str
    resource_class: str = "cpu"
    route: str | None = None  # parse: the route key that chose the parser


@dataclass(frozen=True)
class Stop:
    reason: StopReason


@dataclass(frozen=True)
class Wait:
    """The entry can't go on until the rest of the run catches up."""

    reason: Literal["corpus"]


def needs_corpus(spec: PipelineSpec, registry: ComponentRegistry) -> bool:
    """Whether a clean rule compares documents across the corpus version."""
    return any(
        getattr(registry.get(ref.component), "corpus_scope", False) for ref in spec.stages.clean
    )


def stages_until(until: str | None) -> tuple[str, ...]:
    """The stages a run goes through, ending at ``until`` (every stage when None)."""
    if until is None:
        return ALL_STAGES
    if until not in ALL_STAGES:
        raise ValueError("unknown stage")
    return ALL_STAGES[: ALL_STAGES.index(until) + 1]


def _member(registry: ComponentRegistry, ref: ComponentRef) -> dict[str, Any]:
    component = registry.get(ref.component)
    return {"component": ref.component, "version": component.info.version, "params": ref.params}


Rerun = tuple[str, str] | None  # (from stage, nonce)


def _nonce(stage: str, rerun: Rerun) -> str | None:
    if rerun is None:
        return None
    start, nonce = rerun
    return nonce if ALL_STAGES.index(stage) >= ALL_STAGES.index(start) else None


def _task(
    stage: str,
    registry: ComponentRegistry,
    ref: ComponentRef,
    inputs: Sequence[str],
    env: Mapping[str, Any] | None = None,
    *,
    extra_params: Mapping[str, Any] | None = None,
    route: str | None = None,
    rerun: Rerun = None,
) -> TaskSpec:
    component = registry.get(ref.component)
    params = {**ref.params, **(extra_params or {})}
    env = {**component.static_env(component.resolve_params(ref.params)), **(env or {})}
    return TaskSpec(
        stage=stage,
        component_id=ref.component,
        component_version=component.info.version,
        params=params,
        inputs=tuple(inputs),
        env=env,
        cache_key=cache_key(
            stage=stage,
            component_id=ref.component,
            component_version=component.info.version,
            params=params,
            inputs=inputs,
            env=env,
            nonce=_nonce(stage, rerun),
        ),
        resource_class=component.info.resource_class,
        route=route,
    )


def _composite(
    stage: str,
    component_id: str,
    params: dict[str, Any],
    inputs: Sequence[str],
    env: Mapping[str, Any],
    resource_class: str,
    rerun: Rerun = None,
) -> TaskSpec:
    return TaskSpec(
        stage=stage,
        component_id=component_id,
        component_version=COMPOSITE_VERSION,
        params=params,
        inputs=tuple(inputs),
        env=dict(env),
        cache_key=cache_key(
            stage=stage,
            component_id=component_id,
            component_version=COMPOSITE_VERSION,
            params=params,
            inputs=inputs,
            env=env,
            nonce=_nonce(stage, rerun),
        ),
        resource_class=resource_class,
    )


def next_task(
    spec: PipelineSpec,
    registry: ComponentRegistry,
    entry: EntryInput,
    outputs: Mapping[str, StageOutput],
    *,
    until: str | None = None,
    model_env: Mapping[str, Mapping[str, Any]] | None = None,
    rerun: Rerun = None,
    corpus: Sequence[CorpusRef] | None = None,
) -> TaskSpec | Stop | Wait:
    """The entry's next task after ``outputs`` (stage → output), why it stops, or that it
    waits. ``model_env`` adds environment resolved when the run starts, by stage: the
    embedding model's revision. ``rerun`` forces a stage and the ones after it to run again.
    ``corpus`` is every parsed entry's (parse key, rank), for corpus-wide clean rules."""
    stages = spec.stages
    planned = [stage for stage in stages_until(until) if stage in DOCUMENT_STAGES]
    remaining = [stage for stage in planned if stage not in outputs]
    if "scan" in outputs and outputs["scan"].summary.get("decision") == "exclude":
        return Stop("excluded")
    if "clean" in outputs and outputs["clean"].summary.get("excluded"):
        return Stop("excluded")
    if not remaining:
        return Stop("done")
    stage = remaining[0]
    extra_env = dict((model_env or {}).get(stage, {}))
    if stage == "extract":
        return _task(
            "extract",
            registry,
            stages.extract.component,
            [entry.sha256],
            {"extension": entry.extension, **extra_env},
            rerun=rerun,
        )
    raw = outputs["extract"]
    if stage == "scan":
        members = [_member(registry, ref) for ref in stages.scan.checks]
        io = any(
            registry.get(ref.component).info.resource_class == "io" for ref in stages.scan.checks
        )
        return _composite(
            "scan",
            SCAN_COMPONENT,
            {"checks": members, "on_outcome": dict(stages.scan.on_outcome)},
            [raw.cache_key],
            {"duplicate": entry.duplicate, **extra_env},
            "io" if io else "cpu",
            rerun,
        )
    if stage == "parse":
        media_type = str(raw.summary.get("media_type", ""))
        traits = frozenset(
            name for name, value in dict(raw.summary.get("traits", {})).items() if value is True
        )
        key, ref = route_for(stages.parse.route, media_type, traits)
        parser = registry.get(ref.component)
        if not accepts(parser.info.accepts, media_type, traits):
            return Stop("unsupported_type")
        return _task("parse", registry, ref, [raw.cache_key], extra_env, route=key, rerun=rerun)
    if stage == "clean":
        members = [_member(registry, ref) for ref in stages.clean]
        params: dict[str, Any] = {"rules": members}
        own = outputs["parse"].cache_key
        inputs = [own]
        if needs_corpus(spec, registry):
            if corpus is None:
                return Wait("corpus")
            ordered = sorted(corpus)
            params["corpus"] = [{"key": key, "rank": list(rank)} for key, rank in ordered]
            inputs += [key for key, _ in ordered if key != own]
        io = any(registry.get(ref.component).info.resource_class == "io" for ref in stages.clean)
        return _composite(
            "clean", CLEAN_COMPONENT, params, inputs, extra_env, "io" if io else "cpu", rerun
        )
    if stage == "chunk":
        ref = ComponentRef(component=stages.chunk.component, params=stages.chunk.params)
        return _task(
            "chunk",
            registry,
            ref,
            [outputs["clean"].cache_key],
            extra_env,
            extra_params={"index_text": stages.chunk.index_text, "label": entry.label},
            rerun=rerun,
        )
    if stage == "enrich":
        return _task(
            "enrich", registry, stages.enrich, [outputs["chunk"].cache_key], extra_env, rerun=rerun
        )
    return _task(
        "embed", registry, stages.embed, [outputs["enrich"].cache_key], extra_env, rerun=rerun
    )


def index_tasks(
    spec: PipelineSpec,
    registry: ComponentRegistry,
    embedded: Sequence[tuple[str, str]],
    *,
    model_env: Mapping[str, Mapping[str, Any]] | None = None,
    rerun: Rerun = None,
) -> list[TaskSpec]:
    """The corpus-level index tasks over every embedded entry: ``(chunks key, vectors key)``
    pairs, in entry order. The dense index reads both; the keyword index reads the chunks."""
    extra_env = dict((model_env or {}).get("index", {}))
    tasks = []
    chunks = [pair[0] for pair in embedded]
    if spec.stages.index.dense is not None:
        inputs = [key for pair in embedded for key in pair]
        tasks.append(
            _task("index", registry, spec.stages.index.dense, inputs, extra_env, rerun=rerun)
        )
    if spec.stages.index.keyword is not None:
        tasks.append(
            _task("index", registry, spec.stages.index.keyword, chunks, extra_env, rerun=rerun)
        )
    return tasks
