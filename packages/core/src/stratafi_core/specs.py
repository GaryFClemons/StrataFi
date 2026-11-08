"""Pipeline specs and retrieval configs.

``stratafi.pipeline.v1`` says how documents become an index: which component runs each stage,
with which parameters. ``stratafi.retrieval.v1`` says how a query searches that index. The UI's
step screens, its YAML view, the SDK and the experiment planner all edit these two documents,
and all of them validate through ``resolve_pipeline`` and ``resolve_retrieval``:

- the document's shape (Pydantic), then
- every component exists and belongs to the stage that names it
- every parameter set validates against its component's ``Params``
- the parse route has a ``default``, and every routed type has a parser that accepts it
- the scan stage keeps the security floor: ``scan/content-safety`` runs (``scan/clamav`` too,
  recommended wherever a clamd runs), and infected files are excluded

Errors are ``SpecIssue``\\ s: a dotted path (``stages.chunk.params.max_tokens``) and a code,
never the offending value. Resolution fills in every default, so a resolved spec is complete
without the YAML it came from.

Hashes: ``spec_hash`` covers the configuration (the schema and the stages, or for a retrieval
config everything but its name and description), so identical configurations are recognized
whatever they're called. ``document_hash`` covers the whole document; saving a document whose
hash matches the latest version creates no new version.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from stratafi_core.artifacts import Decision
from stratafi_core.components import ComponentRegistry, UnknownComponent
from stratafi_core.hashing import content_hash

PIPELINE_SCHEMA = "stratafi.pipeline.v1"
RETRIEVAL_SCHEMA = "stratafi.retrieval.v1"
DEFAULT_INDEX_TEXT = "{title}\n{heading_path}\n{text}"
INDEX_TEXT_FIELDS = frozenset({"title", "heading_path", "text", "section", "context"})

NON_CLEAN_OUTCOMES: tuple[str, ...] = (
    "infected",
    "encrypted",
    "malformed",
    "limit_exceeded",
    "pii",
    "error",
    "duplicate",
)
# The in-process safety check runs in every pipeline. ClamAV (scan/clamav) is recommended
# wherever a clamd runs (compose's clamav service), but a laptop may have none.
REQUIRED_SCAN_CHECKS = ("scan/content-safety",)

NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
COMPONENT_ID_PATTERN = r"^[a-z][a-z0-9-]{0,31}/[a-z0-9][a-z0-9.-]{0,63}$"
# A parse route key: "default", a media type, or a short type, either with traits in brackets:
# "application/pdf", "pdf[scanned]", "text/markdown".
ROUTE_KEY_RE = re.compile(
    r"default"
    r"|[a-z0-9][a-z0-9.+-]{0,63}(/[a-z0-9][a-z0-9.+-]{0,126})?"
    r"(\[[a-z_]{1,32}(,[a-z_]{1,32}){0,7}\])?"
)
SHORT_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "txt": "text/plain",
    "text": "text/plain",
    "html": "text/html",
    "htm": "text/html",
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "json": "application/json",
    "xml": "application/xml",
    "eml": "message/rfc822",
    "msg": "application/vnd.ms-outlook",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "doc": "application/msword",
    "xls": "application/vnd.ms-excel",
    "ppt": "application/vnd.ms-powerpoint",
    "zip": "application/zip",
    "gz": "application/gzip",
}

Name = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]
ComponentId = Annotated[str, StringConstraints(pattern=COMPONENT_ID_PATTERN)]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ComponentRef(_Spec):
    """A component and its parameters. In a document, a bare ID means "with defaults"."""

    component: ComponentId
    params: dict[str, Any] = Field(default_factory=dict)


def _as_ref(value: "ComponentRef | str") -> "ComponentRef":
    return ComponentRef(component=value) if isinstance(value, str) else value


# A component reference in a document: an object, or a bare component ID.
Ref = Annotated[ComponentRef | ComponentId, AfterValidator(_as_ref)]


def _ref(component: str, **params: Any) -> ComponentRef:
    return ComponentRef(component=component, params=params)


# --- the pipeline spec -----------------------------------------------------------------------


class ExtractStage(_Spec):
    component: Ref = Field(
        default_factory=lambda: _ref("extract/upload"),
        description="Reads each document's stored bytes, checks its hash and detects its type",
    )
    unpack: list[Ref] = Field(
        default_factory=lambda: [_ref("unpack/zip")],
        description="Unpackers applied when documents arrive",
    )


def _default_policy() -> dict[str, str]:
    return dict.fromkeys(NON_CLEAN_OUTCOMES, "exclude")


class ScanStage(_Spec):
    checks: list[Ref] = Field(
        default_factory=lambda: [
            _ref("scan/content-safety"),
            _ref("scan/dedupe"),
        ],
        description="Checks run in order on each raw document, before any parser opens it",
    )
    on_outcome: dict[
        Literal[
            "infected",
            "encrypted",
            "malformed",
            "limit_exceeded",
            "pii",
            "error",
            "duplicate",
        ],
        Decision,
    ] = Field(
        default_factory=_default_policy,
        description="What each outcome does to the document: continue, flag or exclude",
    )

    @model_validator(mode="before")
    @classmethod
    def _complete_policy(cls, data: Any) -> Any:
        # Outcomes the document doesn't name are excluded: an incomplete policy fails closed.
        if isinstance(data, Mapping) and isinstance(data.get("on_outcome"), Mapping):
            return {**data, "on_outcome": {**_default_policy(), **data["on_outcome"]}}
        return data


# The default parse route: Markdown and plain text by their structure, PDFs by their text layer,
# and office and web formats through markitdown; anything else, markitdown too.
DEFAULT_PARSE_ROUTE: dict[str, str] = {
    "default": "parse/markitdown",
    "docx": "parse/markitdown",
    "html": "parse/markitdown",
    "md": "parse/markdown",
    "pdf": "parse/pypdf",
    "pptx": "parse/markitdown",
    "txt": "parse/markdown",
    "xlsx": "parse/markitdown",
}


class ParseStage(_Spec):
    route: dict[str, Ref] = Field(
        default_factory=lambda: {key: _ref(value) for key, value in DEFAULT_PARSE_ROUTE.items()},
        description='File type → parser. Keys are "default", a media type, or a short type with '
        'traits such as "pdf[scanned]"',
    )


class ChunkStage(ComponentRef):
    component: ComponentId = "chunk/sections"
    index_text: str = Field(
        default=DEFAULT_INDEX_TEXT,
        max_length=500,
        description="What each chunk's index text holds: {title}, {heading_path}, {section}, "
        "{context} and {text}, one per line; lines that come out empty are dropped",
    )


class IndexStage(_Spec):
    dense: Ref | None = Field(default_factory=lambda: _ref("index/exact"))
    keyword: Ref | None = Field(default_factory=lambda: _ref("index/bm25"))

    @model_validator(mode="after")
    def _one_at_least(self) -> "IndexStage":
        if self.dense is None and self.keyword is None:
            raise ValueError("an index needs a dense or a keyword part")
        return self


class Stages(_Spec):
    extract: ExtractStage = Field(default_factory=ExtractStage)
    scan: ScanStage = Field(default_factory=ScanStage)
    parse: ParseStage = Field(default_factory=ParseStage)
    clean: list[Ref] = Field(
        default_factory=lambda: [_ref("clean/whitespace")],
        description="Cleaning rules, applied in order",
    )
    chunk: ChunkStage = Field(default_factory=ChunkStage)
    enrich: Ref = Field(default_factory=lambda: _ref("enrich/none"))
    embed: Ref = Field(default_factory=lambda: _ref("embed/gateway"))
    index: IndexStage = Field(default_factory=IndexStage)


class PipelineSpec(_Spec):
    """``stratafi.pipeline.v1``."""

    schema_: Literal["stratafi.pipeline.v1"] = Field(alias="schema")
    name: Name
    description: str = Field(default="", max_length=2000)
    stages: Stages = Field(default_factory=Stages)


# --- the retrieval config --------------------------------------------------------------------


class RetrieverSettings(_Spec):
    top_n: int = Field(default=50, ge=1, le=1000, description="Results each retriever returns")


MODEL_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


class DenseRetrieverSettings(RetrieverSettings):
    # Added after v1's first configs: it's left out of a dump when unset, which keeps their
    # hashes, and so their cached eval work, unchanged.
    adapter: str | None = Field(
        default=None,
        pattern=MODEL_ID_PATTERN,
        exclude_if=lambda value: value is None,
        description="A trained query adapter's model ID, applied to query vectors",
    )


class Retrievers(_Spec):
    dense: DenseRetrieverSettings | None = Field(default_factory=DenseRetrieverSettings)
    keyword: RetrieverSettings | None = Field(default_factory=RetrieverSettings)

    @model_validator(mode="after")
    def _one_at_least(self) -> "Retrievers":
        if self.dense is None and self.keyword is None:
            raise ValueError("a retrieval config needs a dense or a keyword retriever")
        return self


class Fusion(_Spec):
    method: Literal["rrf", "weighted", "none"] = "rrf"
    k: int = Field(default=60, ge=1, le=10_000, description="RRF's k")
    alpha: float = Field(default=0.5, ge=0, le=1, description="Weighted fusion: the dense weight")


class Filters(_Spec):
    collections: list[str] | None = Field(default=None, max_length=100)
    current_only: bool = Field(
        default=True, description="Only versions in effect on the as-of date"
    )
    statuses: list[str] | None = Field(default=None, max_length=20)


class RetrievalConfig(_Spec):
    """``stratafi.retrieval.v1``."""

    schema_: Literal["stratafi.retrieval.v1"] = Field(alias="schema")
    name: Name
    description: str = Field(default="", max_length=2000)
    retrievers: Retrievers = Field(default_factory=Retrievers)
    fusion: Fusion = Field(default_factory=Fusion)
    candidates: int = Field(
        default=50, ge=1, le=1000, description="Fused results the reranker sees"
    )
    filters: Filters = Field(default_factory=Filters)
    rerank: Ref = Field(default_factory=lambda: _ref("rerank/none"))
    top_k: int = Field(default=8, ge=1, le=100, description="Passages returned")

    @model_validator(mode="after")
    def _consistent(self) -> "RetrievalConfig":
        if self.top_k > self.candidates:
            raise ValueError("top_k can't be more than candidates")
        both = self.retrievers.dense is not None and self.retrievers.keyword is not None
        if both and self.fusion.method == "none":
            raise ValueError("two retrievers need a fusion method")
        return self


# --- validation and resolution ----------------------------------------------------------------


@dataclass(frozen=True)
class SpecIssue:
    """One problem with a document: where (a dotted path) and what (a code). Never the value."""

    path: str
    code: str


class SpecInvalid(ValueError):
    def __init__(self, issues: Sequence[SpecIssue]) -> None:
        super().__init__(f"{len(issues)} problem(s)")
        self.issues = list(issues)

    @property
    def paths(self) -> list[str]:
        return [issue.path for issue in self.issues]


_SAFE_PART_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+/\[\],-]{0,127}")
_IDENTIFIER_RE = re.compile(r"[a-z_][a-z0-9_]{0,63}")


def _join(*parts: object) -> str:
    return ".".join(str(part) for part in parts if part != "")


# A component reference is an object or a bare ID, so Pydantic reports both union branches.
_BRANCH_TAGS = frozenset({"ComponentRef", "constrained-str", "str"})
_TYPE_MISMATCHES = frozenset({"model_type", "model_attributes_type", "dict_type", "string_type"})


def _issues_from(
    error: ValidationError, prefix: str = "", branch_tags: frozenset[str] = frozenset()
) -> list[SpecIssue]:
    tags = _BRANCH_TAGS | branch_tags
    raw: list[tuple[list[Any], int | None, str]] = []
    branches: dict[tuple[Any, ...], list[str]] = {}
    for item in error.errors(include_input=False, include_url=False):
        loc = list(item.get("loc", ()))
        branch_at = next((i for i, part in enumerate(loc) if part in tags), None)
        error_type = str(item.get("type", "invalid"))
        raw.append((loc, branch_at, error_type))
        if branch_at is not None:
            branches.setdefault(tuple(loc[:branch_at]), []).append(error_type)
    issues: list[SpecIssue] = []
    for loc, branch_at, error_type in raw:
        if branch_at is not None:
            if all(kind in _TYPE_MISMATCHES for kind in branches[tuple(loc[:branch_at])]):
                # Neither an object nor a string: one issue for the field.
                loc, error_type = loc[:branch_at], "invalid_type"
            elif error_type in _TYPE_MISMATCHES:
                continue  # the branch the input wasn't meant for
            else:
                loc = loc[:branch_at] + loc[branch_at + 1 :]
        parts: list[str] = []
        for index, part in enumerate(loc):
            text = str(part)
            unexpected = error_type == "extra_forbidden" and index == len(loc) - 1
            if (
                isinstance(part, int)
                or (not unexpected and _SAFE_PART_RE.fullmatch(text))
                or (unexpected and _IDENTIFIER_RE.fullmatch(text))
            ):
                parts.append(text)
            else:
                parts.append("*")
        issue = SpecIssue(_join(prefix, *parts) or "(root)", error_type)
        if issue not in issues:
            issues.append(issue)
    return issues


def validation_issues(
    error: ValidationError, prefix: str = "", *, branch_tags: frozenset[str] = frozenset()
) -> list[SpecIssue]:
    """A Pydantic error as ``SpecIssue``\\ s: paths and codes, never values. For other documents
    that report problems the way specs do (eval sets, corpus manifests). ``branch_tags`` names
    the models in an object-or-string union, as ``ComponentRef`` is in a spec."""
    return _issues_from(error, prefix, branch_tags)


def route_key_type(key: str) -> tuple[str, frozenset[str]]:
    """A route key's media type and traits: ``pdf[scanned]`` → (``application/pdf``,
    {``scanned``})."""
    base, _, rest = key.partition("[")
    traits = frozenset(rest.rstrip("]").split(",")) if rest else frozenset()
    return SHORT_TYPES.get(base, base), traits


def accepts(patterns: Sequence[str], media_type: str, traits: frozenset[str] = frozenset()) -> bool:
    """Whether a component's ``accepts`` list covers a media type with traits. A pattern is
    ``*``, a media type, ``type/*``, or any of these with required traits in brackets."""
    for pattern in patterns:
        pattern_type, needed = route_key_type(pattern)
        if not needed <= traits and needed:
            continue
        if pattern_type in ("*", media_type):
            return True
        if pattern_type.endswith("/*") and media_type.startswith(pattern_type[:-1]):
            return True
    return False


class _Resolver:
    def __init__(self, registry: ComponentRegistry) -> None:
        self.registry = registry
        self.issues: list[SpecIssue] = []

    def ref(self, ref: ComponentRef, path: str, stages: Sequence[str]) -> ComponentRef:
        try:
            component = self.registry.get(ref.component)
        except UnknownComponent:
            self.issues.append(SpecIssue(_join(path, "component"), "unknown_component"))
            return ref
        if component.info.stage not in stages:
            self.issues.append(SpecIssue(_join(path, "component"), "wrong_stage"))
            return ref
        try:
            params = component.resolve_params(ref.params)
        except ValidationError as exc:
            self.issues += _issues_from(exc, _join(path, "params"))
            return ref
        return ref.model_copy(update={"params": component.dump_params(params)})


def _parse(model: type[BaseModel], data: Mapping[str, Any] | BaseModel) -> Any:
    if isinstance(data, model):
        return data
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json", by_alias=True)
    if not isinstance(data, Mapping):
        raise SpecInvalid([SpecIssue("(root)", "not_an_object")])
    try:
        return model.model_validate(dict(data))
    except ValidationError as exc:
        raise SpecInvalid(_issues_from(exc)) from None


def resolve_pipeline(
    data: Mapping[str, Any] | PipelineSpec, registry: ComponentRegistry
) -> PipelineSpec:
    """Validate a pipeline spec and fill in every default. Raises ``SpecInvalid``."""
    spec: PipelineSpec = _parse(PipelineSpec, data)
    resolver = _Resolver(registry)
    stages = spec.stages
    base = "stages"

    extract = stages.extract.model_copy(
        update={
            "component": resolver.ref(
                stages.extract.component, _join(base, "extract", "component"), ["extract"]
            ),
            "unpack": [
                resolver.ref(ref, _join(base, "extract", "unpack", index), ["extract"])
                for index, ref in enumerate(stages.extract.unpack)
            ],
        }
    )

    checks = [
        resolver.ref(ref, _join(base, "scan", "checks", index), ["scan"])
        for index, ref in enumerate(stages.scan.checks)
    ]
    present = {ref.component for ref in stages.scan.checks}
    for required in REQUIRED_SCAN_CHECKS:
        if required not in present:
            resolver.issues.append(
                SpecIssue(_join(base, "scan", "checks"), "required_check_missing")
            )
            break
    if stages.scan.on_outcome.get("infected") != "exclude":
        resolver.issues.append(
            SpecIssue(_join(base, "scan", "on_outcome", "infected"), "must_exclude")
        )
    scan = stages.scan.model_copy(update={"checks": checks})

    route: dict[str, ComponentRef] = {}
    if "default" not in stages.parse.route:
        resolver.issues.append(SpecIssue(_join(base, "parse", "route"), "default_route_missing"))
    for key, ref in stages.parse.route.items():
        path = _join(base, "parse", "route", key if ROUTE_KEY_RE.fullmatch(key) else "*")
        if not ROUTE_KEY_RE.fullmatch(key):
            resolver.issues.append(SpecIssue(path, "invalid_route_key"))
            continue
        resolved = resolver.ref(ref, path, ["parse"])
        route[key] = resolved
        if key != "default" and ref.component in registry:
            media_type, traits = route_key_type(key)
            component = registry.get(ref.component)
            if component.info.stage == "parse" and not accepts(
                component.info.accepts, media_type, traits
            ):
                resolver.issues.append(
                    SpecIssue(_join(path, "component"), "parser_does_not_accept")
                )
    parse = stages.parse.model_copy(update={"route": dict(sorted(route.items()))})

    clean = [
        resolver.ref(ref, _join(base, "clean", index), ["clean"])
        for index, ref in enumerate(stages.clean)
    ]
    chunk_ref = resolver.ref(
        ComponentRef(component=stages.chunk.component, params=stages.chunk.params),
        _join(base, "chunk"),
        ["chunk"],
    )
    for field_name in re.findall(r"\{([^{}]*)\}", stages.chunk.index_text):
        if field_name not in INDEX_TEXT_FIELDS:
            resolver.issues.append(SpecIssue(_join(base, "chunk", "index_text"), "unknown_field"))
            break
    chunk = stages.chunk.model_copy(update={"params": chunk_ref.params})
    enrich = resolver.ref(stages.enrich, _join(base, "enrich"), ["enrich"])
    embed = resolver.ref(stages.embed, _join(base, "embed"), ["embed"])
    index = stages.index.model_copy(
        update={
            "dense": resolver.ref(stages.index.dense, _join(base, "index", "dense"), ["index"])
            if stages.index.dense is not None
            else None,
            "keyword": resolver.ref(
                stages.index.keyword, _join(base, "index", "keyword"), ["index"]
            )
            if stages.index.keyword is not None
            else None,
        }
    )
    if resolver.issues:
        raise SpecInvalid(resolver.issues)
    return spec.model_copy(
        update={
            "stages": Stages(
                extract=extract,
                scan=scan,
                parse=parse,
                clean=clean,
                chunk=chunk,
                enrich=enrich,
                embed=embed,
                index=index,
            )
        }
    )


def resolve_retrieval(
    data: Mapping[str, Any] | RetrievalConfig, registry: ComponentRegistry
) -> RetrievalConfig:
    """Validate a retrieval config and fill in every default. Raises ``SpecInvalid``."""
    config: RetrievalConfig = _parse(RetrievalConfig, data)
    resolver = _Resolver(registry)
    rerank = resolver.ref(config.rerank, "rerank", ["rerank"])
    if resolver.issues:
        raise SpecInvalid(resolver.issues)
    return config.model_copy(update={"rerank": rerank})


def dump_document(document: PipelineSpec | RetrievalConfig) -> dict[str, Any]:
    return document.model_dump(mode="json", by_alias=True)


def spec_hash(document: PipelineSpec | RetrievalConfig) -> str:
    """The configuration's hash: everything but the name and the description."""
    data = dump_document(document)
    data.pop("name", None)
    data.pop("description", None)
    return content_hash(data)


def document_hash(document: PipelineSpec | RetrievalConfig) -> str:
    return content_hash(dump_document(document))


def component_refs(spec: PipelineSpec) -> list[tuple[str, ComponentRef]]:
    """Every component a resolved spec uses, with its path."""
    stages = spec.stages
    refs: list[tuple[str, ComponentRef]] = [("stages.extract.component", stages.extract.component)]
    refs += [(f"stages.extract.unpack.{i}", ref) for i, ref in enumerate(stages.extract.unpack)]
    refs += [(f"stages.scan.checks.{i}", ref) for i, ref in enumerate(stages.scan.checks)]
    refs += [(f"stages.parse.route.{key}", ref) for key, ref in stages.parse.route.items()]
    refs += [(f"stages.clean.{i}", ref) for i, ref in enumerate(stages.clean)]
    refs.append(
        ("stages.chunk", ComponentRef(component=stages.chunk.component, params=stages.chunk.params))
    )
    refs.append(("stages.enrich", stages.enrich))
    refs.append(("stages.embed", stages.embed))
    if stages.index.dense is not None:
        refs.append(("stages.index.dense", stages.index.dense))
    if stages.index.keyword is not None:
        refs.append(("stages.index.keyword", stages.index.keyword))
    return refs


# --- diffs ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    path: str
    op: Literal["added", "removed", "changed"]
    before: Any = None
    after: Any = None


def diff_documents(before: Any, after: Any, path: str = "") -> list[Change]:
    """What changed between two JSON documents, as paths: mappings by key, lists by position."""
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        changes: list[Change] = []
        for key in sorted(set(before) | set(after)):
            child = _join(path, key)
            if key not in after:
                changes.append(Change(child, "removed", before=before[key]))
            elif key not in before:
                changes.append(Change(child, "added", after=after[key]))
            else:
                changes += diff_documents(before[key], after[key], child)
        return changes
    if isinstance(before, list) and isinstance(after, list):
        changes = []
        for index in range(max(len(before), len(after))):
            child = _join(path, index)
            if index >= len(after):
                changes.append(Change(child, "removed", before=before[index]))
            elif index >= len(before):
                changes.append(Change(child, "added", after=after[index]))
            else:
                changes += diff_documents(before[index], after[index], child)
        return changes
    if before != after or type(before) is not type(after):
        return [Change(path or "(root)", "changed", before=before, after=after)]
    return []


def route_for(
    route: Mapping[str, ComponentRef], media_type: str, traits: frozenset[str]
) -> tuple[str, ComponentRef]:
    """The parse route for a document: the matching key with the most traits, else
    ``default``."""
    best: tuple[int, str] | None = None
    for key in route:
        if key == "default":
            continue
        key_type, needed = route_key_type(key)
        if key_type == media_type and needed <= traits:
            rank = (len(needed), key)
            if best is None or rank > best:
                best = rank
    chosen = best[1] if best is not None else "default"
    return chosen, route[chosen]


Resolver = Callable[[Mapping[str, Any], ComponentRegistry], PipelineSpec | RetrievalConfig]
SCHEMA_RESOLVERS: dict[str, Resolver] = {
    PIPELINE_SCHEMA: resolve_pipeline,
    RETRIEVAL_SCHEMA: resolve_retrieval,
}
