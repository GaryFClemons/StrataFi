"""The client's resources: corpora, pipelines, retrieval configs, runs, eval sets, evals,
experiments, knowledge profiles and test suites.

References name things as people do:

- a corpus version: ``halcyon@latest``, ``halcyon@3``, or its ID
- a pipeline or retrieval config: ``baseline-v1`` (the latest version),
  ``baseline-v1@2``, a saved version's ID, a ``Pipeline``, or a document (saved first)
- an eval set version: ``halcyon-retrieval@1``, ``halcyon-retrieval@latest``, or its ID
- a run, an eval run or an experiment: its ID or the object

``Pipeline.with_params`` makes a variant: a draft saved, when it's used, under a name derived
from its base and its changes (``baseline-v1.chunk.max_tokens-256``), so the base's
own versions stay as they were, and running a sweep again finds the same specs.
"""

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

import yaml
from pydantic import BaseModel

from stratafi.errors import ApiError, ConfigError
from stratafi.models import (
    AliasOut,
    AnnotationOut,
    CheckOut,
    CompareManyOut,
    CompareOut,
    CorpusDetail,
    CorpusOut,
    EvalRunOut,
    EvalSetOut,
    EvalSetVersionOut,
    ExperimentOut,
    HistoryOut,
    ImportOut,
    IngestOut,
    LatencyProfileOut,
    MapOut,
    ModelComparisonOut,
    PipelineSpecValidation,
    PipelineSpecVersion,
    PlanOut,
    ProfileOut,
    QueryResultOut,
    RagPipelineOut,
    RetrievalConfigValidation,
    RetrievalConfigVersion,
    RunOut,
    StartedOut,
    SuiteOut,
    TestRunOut,
    TraceOut,
    TrackedRunOut,
    TrainedModelOut,
    TrainingRunOut,
    TrainingSetOut,
    TrialOut,
    ValidationOut,
    VersionOut,
    WorkOut,
)
from stratafi.results import DEFAULT_CHECKPOINT, DEFAULT_RULE, Comparison, Metrics, Records, plain

if TYPE_CHECKING:
    from stratafi.client import Client

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
MAX_NAME = 64
MAX_DESCRIPTION = 2000
_REF_RE = re.compile(r"^(?P<name>[a-z0-9][a-z0-9._-]{0,63})(?:@(?P<version>latest|[1-9][0-9]*))?$")
_COMPONENT_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}/[a-z0-9][a-z0-9.-]{0,63}$")
_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
EVAL_SET_FORMATS = {".yaml": "yaml", ".yml": "yaml", ".jsonl": "jsonl", ".csv": "csv"}


def as_id(value: Any) -> str | None:
    """``value`` as an ID (a UUID in its usual form), or None."""
    text = str(value).strip().lower()
    return text if _ID_RE.fullmatch(text) else None


def parse_ref(ref: Any, what: str) -> tuple[str, int | None]:
    """``name``, ``name@latest`` or ``name@N``: the name, and the version (None: the latest)."""
    match = _REF_RE.fullmatch(ref.strip()) if isinstance(ref, str) else None
    if match is None:
        raise ConfigError(f"name a {what} as name, name@latest or name@N, or by its ID")
    version = match["version"]
    return match["name"], int(version) if version and version != "latest" else None


def _id_of(value: Any, what: str) -> str:
    """The ID of a run, an eval run or a record."""
    found = as_id(getattr(value, "id", value))
    if found is None:
        raise ConfigError(f"name the {what} by its ID")
    return found


def _not_found(error: ApiError, message: str) -> Exception:
    return ConfigError(message) if error.status == 404 else error


def load_document(source: Any) -> dict[str, Any]:
    """A spec document from a mapping or a YAML or JSON file."""
    if isinstance(source, Mapping):
        document: Any = copy.deepcopy(dict(source))
    else:
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            raise ConfigError(f"can't read {path}") from None
        try:
            document = yaml.safe_load(text)
        except yaml.YAMLError:
            raise ConfigError(f"{path} isn't YAML or JSON") from None
    if not isinstance(document, dict):
        raise ConfigError("a spec document is a mapping")
    # YAML reads dates as dates; the API reads them as strings.
    return json.loads(json.dumps(document, default=str))


# --- specs: pipelines and retrieval configs -----------------------------------------------


def _slots(node: Any, keys: set[str] | None = None) -> list[tuple[Any, Any]]:
    """Where components are named below ``node`` (only under ``keys`` at its top): pairs of a
    container and a key whose value is a ``{"component": ...}`` mapping or a bare component
    ID, which a document may use to mean "with defaults"."""
    if isinstance(node, dict):
        items = [(key, value) for key, value in node.items() if keys is None or key in keys]
    elif isinstance(node, list):
        items = list(enumerate(node))
    else:
        return []
    found: list[tuple[Any, Any]] = []
    for key, value in items:
        if isinstance(node, dict) and key == "params":
            continue
        if isinstance(value, dict | list):
            if isinstance(value, dict) and isinstance(value.get("component"), str):
                found.append((node, key))
            found += _slots(value)
        elif isinstance(value, str) and key != "component" and _COMPONENT_RE.fullmatch(value):
            found.append((node, key))
    return found


def _component(value: Any) -> str:
    return value["component"] if isinstance(value, dict) else value


def _slug(value: Any) -> str | None:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return re.sub(r"[^0-9a-z.-]", "-", str(value).lower())
    if isinstance(value, str):
        return re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-") or None
    return None


def variant_name(base: str, changes: Mapping[str, Mapping[str, Any]]) -> str:
    """A variant's name: its base and its changes when they fit (``base.chunk.max_tokens-256``),
    or its base and a digest of its changes."""
    parts: list[str] | None = []
    for target, params in changes.items():
        parts.append(target.rsplit("/", 1)[-1])
        for key, value in params.items():
            slug = _slug(value)
            if slug is None:
                parts = None
                break
            parts.append(f"{re.sub(r'[^a-z0-9_]+', '-', key.lower())}-{slug}")
        if parts is None:
            break
    if parts is not None:
        name = ".".join([base, *parts])
        if len(name) <= MAX_NAME and NAME_RE.fullmatch(name):
            return name
    digest = hashlib.sha256(
        json.dumps(changes, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"{base[: MAX_NAME - len(digest) - 1]}.{digest}"


def _describe(base: str, version: int | None, changes: Mapping[str, Mapping[str, Any]]) -> str:
    listed = "; ".join(
        f"{target}: " + ", ".join(f"{key}={value}" for key, value in params.items())
        for target, params in changes.items()
    )
    origin = f"{base} v{version}" if version else base
    return f"A variant of {origin}: {listed}"[:MAX_DESCRIPTION]


class Spec:
    """A pipeline spec or a retrieval config: a saved version (``saved``) or a draft that's
    saved when it's pushed or used. ``document`` is the spec itself."""

    kind: ClassVar[str] = "spec"

    def __init__(
        self,
        document: dict[str, Any],
        *,
        record: BaseModel | None = None,
        created: bool | None = None,
        base: tuple[str, int | None] | None = None,
        changes: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.document = document
        self.record = record
        self.created = created
        self._base = base
        self._changes = {key: dict(value) for key, value in (changes or {}).items()}

    @property
    def name(self) -> str:
        return str(self.document.get("name", ""))

    @property
    def id(self) -> str | None:
        return str(self.record.id) if self.record is not None else None  # type: ignore[attr-defined]

    @property
    def version(self) -> int | None:
        return self.record.version if self.record is not None else None  # type: ignore[attr-defined]

    @property
    def saved(self) -> bool:
        return self.record is not None

    def __repr__(self) -> str:
        where = f"v{self.version}" if self.saved else "draft"
        return f"{type(self).__name__}({self.name!r}, {where})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "document": self.document,
        }

    def with_params(self, target: str, *, name: str | None = None, **params: Any) -> Self:
        """A variant with ``params`` merged into a component's: ``target`` is a stage
        (``"chunk"``, ``"rerank"``) that holds one component, or a component ID
        (``"clean/whitespace"``), changed wherever it's used. ``name`` overrides the derived
        name."""
        document = copy.deepcopy(self.document)
        if "/" in target:
            slots = [
                (container, key)
                for container, key in _slots(document)
                if _component(container[key]) == target
            ]
        else:
            stages = document.get("stages")
            slots = _slots(stages if isinstance(stages, dict) else document, {target})
            if len(slots) > 1:
                first = _component(slots[0][0][slots[0][1]])
                raise ConfigError(
                    f"{target} holds {len(slots)} components: name one by its ID, such as {first}"
                )
        if not slots:
            raise ConfigError(f"{self.name} has no component at {target}")
        for container, key in slots:
            value = container[key]
            if isinstance(value, str):
                container[key] = {"component": value, "params": dict(params)}
            else:
                value["params"] = {**(value.get("params") or {}), **params}
        changes = {key: dict(value) for key, value in self._changes.items()}
        changes.setdefault(target, {}).update(params)
        base_name, base_version = self._base or (self.name, self.version)
        document["name"] = name or variant_name(base_name, changes)
        if not NAME_RE.fullmatch(document["name"]):
            raise ConfigError("a spec's name is lowercase letters, digits, '.', '_' and '-'")
        document["description"] = _describe(base_name, base_version, changes)
        return type(self)(document, base=(base_name, base_version), changes=changes)


class Pipeline(Spec):
    kind = "pipeline spec"


class RetrievalConfig(Spec):
    kind = "retrieval config"


class _Specs[S: Spec]:
    prefix: ClassVar[str]
    model: ClassVar[type[BaseModel]]
    validation: ClassVar[type[BaseModel]]
    wrapper: type[S]

    def __init__(self, client: "Client") -> None:
        self._client = client

    def _wrap(self, body: dict[str, Any], created: bool | None = None) -> S:
        record = self.model.model_validate(body)
        return self.wrapper(copy.deepcopy(body["document"]), record=record, created=created)

    def get(self, ref: str, version: int | None = None) -> S:
        """A saved version: ``name`` (the latest), ``name@N``, or ``name`` and ``version``."""
        name, pinned = parse_ref(ref, self.wrapper.kind)
        number = version if version is not None else pinned
        path = f"{self.prefix}/{name}" + (f"/versions/{number}" if number else "")
        try:
            return self._wrap(self._client.get(path))
        except ApiError as error:
            raise _not_found(error, f"no {self.wrapper.kind} {ref}") from None

    def list(self) -> Records[S]:
        """The latest version of each, by name."""
        return Records(self._wrap(item) for item in self._client.pages(self.prefix))

    def validate(self, source: Any) -> Any:
        """The API's verdict on a document (a mapping, a file or a ``Spec``): ``valid``, and
        ``issues`` as paths and codes."""
        document = source.document if isinstance(source, Spec) else load_document(source)
        return self.validation.model_validate(
            self._client.post(f"{self.prefix}/validate", document)
        )

    def push(self, source: Any) -> S:
        """Save a document (a mapping, a file or a draft) as its name's next version;
        ``created`` is False when it matched the latest version."""
        document = source.document if isinstance(source, Spec) else load_document(source)
        status, body = self._client.send("POST", self.prefix, json=document)
        return self._wrap(body, created=status == 201)

    def resolve(self, value: Any) -> str:
        """A saved version's ID; drafts and documents are pushed first."""
        if isinstance(value, Spec):
            return value.id if value.id is not None else str(self.push(value).id)
        if isinstance(value, Mapping):
            return str(self.push(value).id)
        found = as_id(value)
        if found is not None:
            return found
        return str(self.get(value).id)


class Pipelines(_Specs[Pipeline]):
    prefix = "/pipeline-specs"
    model = PipelineSpecVersion
    validation = PipelineSpecValidation
    wrapper = Pipeline


class RetrievalConfigs(_Specs[RetrievalConfig]):
    prefix = "/retrieval-configs"
    model = RetrievalConfigVersion
    validation = RetrievalConfigValidation
    wrapper = RetrievalConfig


# --- corpora ------------------------------------------------------------------------------


def _manifest_paths(manifest: Path) -> list[str]:
    try:
        document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        raise ConfigError(f"can't read the manifest {manifest}") from None
    documents = document.get("documents") if isinstance(document, dict) else None
    if not isinstance(documents, list) or not all(
        isinstance(entry, dict) and isinstance(entry.get("path"), str) for entry in documents
    ):
        raise ConfigError("a manifest lists documents, each with a path")
    return [entry["path"] for entry in documents]


def upload_files(paths: Sequence[Any], manifest: Path | None) -> list[tuple[str, Path]]:
    """The files to upload and their names: a file by its own name, each file below a folder
    by its path in the folder (hidden ones skipped), and with only a manifest, the files it
    lists, by their paths beside it."""
    files: list[tuple[str, Path]] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for found in sorted(path.rglob("*")):
                relative = found.relative_to(path)
                if found.is_file() and not any(part.startswith(".") for part in relative.parts):
                    files.append((relative.as_posix(), found))
        elif path.is_file():
            files.append((path.name, path))
        else:
            raise ConfigError(f"{path} isn't a file or a folder")
    if manifest is not None and not paths:
        for listed in _manifest_paths(manifest):
            path = manifest.parent / listed
            if not path.is_file():
                raise ConfigError(f"the manifest lists {listed}, which isn't beside it")
            files.append((listed, path))
    names = [name for name, _ in files]
    if len(set(names)) != len(names):
        raise ConfigError("two files would have the same name")
    if not files:
        raise ConfigError("there are no files to upload")
    return files


class Corpora:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def list(self) -> Records[CorpusOut]:
        return Records(CorpusOut.model_validate(item) for item in self._client.pages("/corpora"))

    def find(self, name: str) -> CorpusOut:
        """The corpus called ``name``."""
        for item in self._client.pages("/corpora"):
            if item["name"] == name:
                return CorpusOut.model_validate(item)
        raise ConfigError(f"no corpus is called {name}")

    def get(self, name: str) -> CorpusDetail:
        """A corpus with its working set's size and its latest frozen version."""
        corpus_id = as_id(name) or str(self.find(name).id)
        try:
            return CorpusDetail.model_validate(self._client.get(f"/corpora/{corpus_id}"))
        except ApiError as error:
            raise _not_found(error, f"no corpus {name}") from None

    def create(self, name: str, *, description: str = "") -> CorpusOut:
        return CorpusOut.model_validate(
            self._client.post("/corpora", {"name": name, "description": description})
        )

    def upload(
        self,
        name: str,
        paths: Sequence[str | Path] = (),
        *,
        manifest: str | Path | None = None,
        batch_size: int = 100,
    ) -> Records[IngestOut]:
        """Upload files into a corpus's working set (``upload_files`` names them). With a
        manifest (``manifest.yaml`` in the corpus manifest format), its entries get
        their metadata, and all files go in one request, as the API needs."""
        corpus_id = as_id(name) or str(self.find(name).id)
        manifest_path = Path(manifest) if manifest is not None else None
        files = upload_files(paths, manifest_path)
        manifest_bytes = None
        if manifest_path is not None:
            try:
                manifest_bytes = manifest_path.read_bytes()
            except OSError:
                raise ConfigError(f"can't read the manifest {manifest_path}") from None
        size = len(files) if manifest_bytes is not None else max(batch_size, 1)
        results = []
        for start in range(0, len(files), size):
            with ExitStack() as stack:
                parts: list[tuple[str, tuple[str, Any, str]]] = []
                for file_name, path in files[start : start + size]:
                    handle = stack.enter_context(path.open("rb"))
                    parts.append(("file", (file_name, handle, "application/octet-stream")))
                if manifest_bytes is not None:
                    parts.append(
                        ("manifest", ("manifest.yaml", manifest_bytes, "application/yaml"))
                    )
                _, body = self._client.send(
                    "POST", "/documents", params={"corpus_id": corpus_id}, files=parts
                )
            results.append(IngestOut.model_validate(body))
        return Records(results)

    def freeze(self, name: str) -> VersionOut:
        """Freeze the working set as the next version (or the latest, when it's unchanged)."""
        corpus_id = as_id(name) or str(self.find(name).id)
        return VersionOut.model_validate(self._client.post(f"/corpora/{corpus_id}/versions"))

    def version(self, ref: str) -> VersionOut:
        """A frozen version: ``name@latest`` (or ``name``) or ``name@N``."""
        name, number = parse_ref(ref, "corpus version")
        corpus = self.find(name)
        if number is None:
            latest = self.get(str(corpus.id)).latest_version
            if latest is None:
                raise ConfigError(f"the corpus {name} has no frozen version: freeze it first")
            return latest
        try:
            return VersionOut.model_validate(
                self._client.get(f"/corpora/{corpus.id}/versions/{number}")
            )
        except ApiError as error:
            raise _not_found(error, f"the corpus {name} has no version {number}") from None

    def resolve(self, value: Any) -> str:
        if isinstance(value, VersionOut):
            return str(value.id)
        return as_id(value) or str(self.version(value).id)


# --- runs and eval runs -------------------------------------------------------------------


class _Tracked[R: BaseModel]:
    """A run or an eval run: its record, which ``refresh`` reads again and ``wait`` follows."""

    kind: ClassVar[str]
    path: ClassVar[str]
    model: ClassVar[type[BaseModel]]
    finished: ClassVar[frozenset[str]] = frozenset({"succeeded", "stopped", "failed", "cancelled"})

    def __init__(self, client: "Client", record: R) -> None:
        self._client = client
        self.record = record

    @property
    def id(self) -> str:
        return str(self.record.id)  # type: ignore[attr-defined]

    @property
    def status(self) -> str:
        return str(self.record.status)  # type: ignore[attr-defined]

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id!r}, {self.status})"

    def to_dict(self) -> dict[str, Any]:
        return plain(self.record)

    def refresh(self) -> Self:
        self.record = self.model.model_validate(self._client.get(f"{self.path}/{self.id}"))  # type: ignore[assignment]
        return self

    def wait(
        self,
        *,
        timeout: float | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> Self:
        """Follow its events until it finishes (``succeeded``, ``failed`` or ``cancelled``).
        ``on_update`` sees each event's data: status and counts."""
        self._client.follow(
            self.kind,
            self.id,
            self.refresh,
            timeout=timeout,
            on_update=on_update,
            finished=self.finished,
        )
        return self

    def cancel(self) -> Self:
        self.record = self.model.model_validate(  # type: ignore[assignment]
            self._client.post(f"{self.path}/{self.id}/cancel")
        )
        return self


class _Annotated:
    """A run, an eval run or an experiment's title, tags, Markdown note and archive."""

    # Given by _Tracked, which every annotated resource also is.
    _client: "Client"
    path: ClassVar[str]
    id: str

    def annotation(self) -> AnnotationOut:
        return AnnotationOut.model_validate(self._client.get(f"{self.path}/{self.id}/annotation"))

    def annotate(
        self,
        *,
        title: str | None = None,
        tags: Mapping[str, str] | None = None,
        delete_tags: Sequence[str] = (),
        note: str | None = None,
        archived: bool | None = None,
    ) -> AnnotationOut:
        """Change what's given: a title, tags to add or replace, tags to remove, a Markdown
        note (empty clears it), or archive (hidden from lists and searches; nothing is
        deleted)."""
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        if tags:
            body["set_tags"] = dict(tags)
        if delete_tags:
            body["delete_tags"] = list(delete_tags)
        if note is not None:
            body["note"] = note
        if archived is not None:
            body["archived"] = archived
        return AnnotationOut.model_validate(
            self._client.patch(f"{self.path}/{self.id}/annotation", body)
        )

    def archive(self) -> AnnotationOut:
        return self.annotate(archived=True)

    def restore(self) -> AnnotationOut:
        return self.annotate(archived=False)


class Run(_Annotated, _Tracked[RunOut]):
    """A pipeline run."""

    kind = "run"
    path = "/runs"
    model = RunOut

    def map(self) -> MapOut:
        """Its pipeline's map with each stage's tasks, cache hits, time and bytes."""
        return MapOut.model_validate(self._client.get(f"{self.path}/{self.id}/map"))


class EvalRun(_Annotated, _Tracked[EvalRunOut]):
    """An eval run: a pipeline run's indexes searched with a retrieval config for an eval set's
    queries, and scored."""

    kind = "eval_run"
    path = "/eval-runs"
    model = EvalRunOut

    @property
    def metrics(self) -> Metrics:
        return Metrics(self.record.metrics)

    def queries(
        self, *, outcome: str | None = None, lost_at: str | None = None, tag: str | None = None
    ) -> Records[QueryResultOut]:
        """One row per query, in the eval set's order; filter by outcome (hit, miss,
        abstain), where a miss lost its evidence, or tag."""
        return Records(
            QueryResultOut.model_validate(item)
            for item in self._client.pages(
                f"{self.path}/{self.id}/queries", outcome=outcome, lost_at=lost_at, tag=tag
            )
        )

    def trace(self, item_id: str, *, rule: str = DEFAULT_RULE) -> TraceOut:
        """One query's retrieval as spans on a timeline, each with its lists."""
        return TraceOut.model_validate(
            self._client.get(f"{self.path}/{self.id}/queries/{item_id}/trace", rule=rule)
        )

    def latency(self, *, bins: int = 20) -> LatencyProfileOut:
        """Each step's latency percentiles and histogram, and the slowest queries."""
        return LatencyProfileOut.model_validate(
            self._client.get(f"{self.path}/{self.id}/latency", bins=bins)
        )

    def map(self) -> MapOut:
        """Its RAG pipeline's map: the pipeline run's stages and its query steps."""
        return MapOut.model_validate(self._client.get(f"{self.path}/{self.id}/map"))


def _sample(sample: Any) -> dict[str, Any] | None:
    if sample is None:
        return None
    if isinstance(sample, bool) or not isinstance(sample, int | Mapping):
        raise ConfigError("a sample is a number of entries or a mapping (size, seed, entries)")
    return {"size": sample} if isinstance(sample, int) else dict(sample)


class Runs:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def start(
        self,
        pipeline: Any,
        *,
        corpus: Any,
        until: str | None = None,
        sample: int | Mapping[str, Any] | None = None,
        mode: str = "batch",
        wait: bool = False,
        timeout: float | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> Run:
        """Run a pipeline (a reference, a ``Pipeline`` or a document) over a corpus version;
        ``until`` stops after a stage, ``sample`` takes some entries. With ``wait``, returns
        once the run finishes."""
        body = {
            "pipeline_spec_id": self._client.pipelines.resolve(pipeline),
            "corpus_version_id": self._client.corpora.resolve(corpus),
            "until": until,
            "sample": _sample(sample),
            "mode": mode,
        }
        run = Run(self._client, RunOut.model_validate(self._client.post("/runs", body)))
        return run.wait(timeout=timeout, on_update=on_update) if wait else run

    def get(self, run: Any) -> Run:
        run_id = _id_of(run, "run")
        try:
            return Run(self._client, RunOut.model_validate(self._client.get(f"/runs/{run_id}")))
        except ApiError as error:
            raise _not_found(error, f"no run {run_id}") from None

    def list(self, *, status: str | None = None) -> Records[RunOut]:
        return Records(
            RunOut.model_validate(item) for item in self._client.pages("/runs", status=status)
        )


# --- eval sets and evals ------------------------------------------------------------------


class EvalSets:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def import_file(self, path: str | Path, *, name: str, **options: Any) -> ImportOut:
        """Import an eval set file (YAML, JSONL or CSV, by its extension unless ``format``
        says) as a new version of ``name``; ``import_text`` lists the options."""
        source = Path(path)
        if "format" not in options:
            options["format"] = EVAL_SET_FORMATS.get(source.suffix.lower())
            if options["format"] is None:
                raise ConfigError("an eval set file is .yaml, .jsonl or .csv; or give its format")
        try:
            content = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            raise ConfigError(f"can't read {source}") from None
        return self.import_text(content, name=name, **options)

    def import_text(
        self,
        content: str,
        *,
        name: str,
        format: str = "yaml",
        description: str | None = None,
        as_of: date | str | None = None,
        k: int | None = None,
        test_fraction: float = 0.3,
        seed: int = 0,
        resplit: bool = False,
        freeze: bool = True,
    ) -> ImportOut:
        """``test_fraction`` of the items go to the test split (seeded, stratified by tags);
        ``as_of`` is the day queries are asked; ``k`` overrides the set's cutoff; ``resplit``
        splits again items that already have one; ``freeze`` freezes the version."""
        body = {
            "name": name,
            "description": description,
            "format": format,
            "content": content,
            "as_of": str(as_of) if as_of is not None else None,
            "k": k,
            "test_fraction": test_fraction,
            "seed": seed,
            "resplit": resplit,
            "freeze": freeze,
        }
        return ImportOut.model_validate(self._client.post("/eval-sets/import", body))

    def list(self) -> Records[EvalSetOut]:
        return Records(EvalSetOut.model_validate(item) for item in self._client.pages("/eval-sets"))

    def find(self, name: str) -> EvalSetOut:
        for item in self._client.pages("/eval-sets"):
            if item["name"] == name:
                return EvalSetOut.model_validate(item)
        raise ConfigError(f"no eval set is called {name}")

    def version(self, ref: str) -> EvalSetVersionOut:
        """A version: ``name@N``, or ``name@latest`` (or ``name``)."""
        name, number = parse_ref(ref, "eval set version")
        found = self.find(name)
        if number is None:
            if found.latest_version is None:
                raise ConfigError(f"the eval set {name} has no version")
            return found.latest_version
        try:
            return EvalSetVersionOut.model_validate(
                self._client.get(f"/eval-sets/{found.id}/versions/{number}")
            )
        except ApiError as error:
            raise _not_found(error, f"the eval set {name} has no version {number}") from None

    def resolve(self, value: Any) -> str:
        if isinstance(value, EvalSetVersionOut):
            return str(value.id)
        return as_id(value) or str(self.version(value).id)


class Evals:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def run(
        self,
        run: Any,
        *,
        retrieval: Any,
        eval_set: Any,
        split: str = "dev",
        confirm_test_read: bool = False,
        options: Mapping[str, Any] | None = None,
        wait: bool = False,
        timeout: float | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> EvalRun:
        """Evaluate a pipeline run's indexes with a retrieval config (a reference, a
        ``RetrievalConfig`` or a document) on an eval set version's split. Reading the test
        split needs ``confirm_test_read``. ``options``: ``shard_size``,
        ``coverage_threshold``, ``resamples``, ``seed``."""
        body = {
            "run_id": _id_of(run, "run"),
            "retrieval_config_id": self._client.retrieval.resolve(retrieval),
            "eval_set_version_id": self._client.eval_sets.resolve(eval_set),
            "split": split,
            "confirm_test_read": confirm_test_read,
            "options": dict(options or {}),
        }
        started = EvalRun(
            self._client, EvalRunOut.model_validate(self._client.post("/eval-runs", body))
        )
        return started.wait(timeout=timeout, on_update=on_update) if wait else started

    def get(self, eval_run: Any) -> EvalRun:
        eval_run_id = _id_of(eval_run, "eval run")
        try:
            record = EvalRunOut.model_validate(self._client.get(f"/eval-runs/{eval_run_id}"))
        except ApiError as error:
            raise _not_found(error, f"no eval run {eval_run_id}") from None
        return EvalRun(self._client, record)

    def list(
        self, *, run: Any = None, eval_set: Any = None, status: str | None = None
    ) -> Records[EvalRunOut]:
        params = {
            "run_id": _id_of(run, "run") if run is not None else None,
            "eval_set_version_id": (
                self._client.eval_sets.resolve(eval_set) if eval_set is not None else None
            ),
            "status": status,
        }
        return Records(
            EvalRunOut.model_validate(item) for item in self._client.pages("/eval-runs", **params)
        )

    def compare(
        self,
        a: Any,
        b: Any,
        *,
        rule: str = DEFAULT_RULE,
        checkpoint: str = DEFAULT_CHECKPOINT,
        metrics: Sequence[str] | None = None,
        resamples: int | None = None,
        seed: int | None = None,
    ) -> Comparison:
        """B against A, query by query (both eval runs of one eval set version)."""
        params = {
            "a": _id_of(a, "eval run"),
            "b": _id_of(b, "eval run"),
            "rule": rule,
            "checkpoint": checkpoint,
            "metric": list(metrics) if metrics else None,
            "resamples": resamples,
            "seed": seed,
        }
        return Comparison(
            CompareOut.model_validate(self._client.get("/eval-comparisons", **params))
        )

    def compare_many(
        self,
        runs: Sequence[Any],
        *,
        baseline: Any = None,
        rule: str = DEFAULT_RULE,
        checkpoint: str = DEFAULT_CHECKPOINT,
        metrics: Sequence[str] | None = None,
        resamples: int | None = None,
        seed: int | None = None,
    ) -> CompareManyOut:
        """Two to six eval runs of one eval set version, paired against a baseline (the
        first unless named): intervals, deltas with Holm's adjustment, groups, curves, the
        checkpoint funnel, latency, slices, the parameters that differ and the queries they
        disagree on."""
        params = {
            "ids": [_id_of(run, "eval run") for run in runs],
            "baseline": _id_of(baseline, "eval run") if baseline is not None else None,
            "rule": rule,
            "checkpoint": checkpoint,
            "metric": list(metrics) if metrics else None,
            "resamples": resamples,
            "seed": seed,
        }
        return CompareManyOut.model_validate(self._client.get("/eval-comparisons/many", **params))

    def search(
        self,
        filter: str = "",
        *,
        order_by: Sequence[str] = (),
        columns: Sequence[str] = (),
        include_archived: bool = False,
        max_results: int | None = None,
    ) -> Records[TrackedRunOut]:
        """Eval runs the search language matches, in order: ``filter`` such as
        ``"metrics.recall@5 > 0.8 and params.rerank.component = 'rerank/none'"``,
        ``order_by`` such as ``["metrics.recall@5 DESC"]``, and ``columns`` whose values each
        result carries."""
        found: list[TrackedRunOut] = []
        cursor = None
        while True:
            page = self._client.post(
                "/eval-runs/search",
                {
                    "filter": filter,
                    "order_by": list(order_by),
                    "columns": list(columns),
                    "include_archived": include_archived,
                    "limit": 500,
                    "cursor": cursor,
                },
            )
            found += [TrackedRunOut.model_validate(item) for item in page["items"]]
            cursor = page.get("next_cursor")
            if not cursor or (max_results is not None and len(found) >= max_results):
                break
        return Records(found[:max_results] if max_results is not None else found)


# --- experiments --------------------------------------------------------------------------


class Experiment(_Annotated, _Tracked[ExperimentOut]):
    """An experiment: trials over a space of configurations, against one eval set's dev split.
    It ends ``succeeded``, ``stopped`` (a budget), ``failed`` or ``cancelled``."""

    kind = "experiment"
    path = "/experiments"
    model = ExperimentOut

    @property
    def plan(self) -> dict[str, Any]:
        return dict(self.record.plan)

    def trials(self) -> Records[TrialOut]:
        """Every trial in plan order, with its results so far."""
        found = self._client.get(f"{self.path}/{self.id}/trials")
        return Records(TrialOut.model_validate(item) for item in found["items"])

    def work(self) -> WorkOut:
        """The plan's unique work beside what the experiment's runs did."""
        return WorkOut.model_validate(self._client.get(f"{self.path}/{self.id}/work"))

    def start(
        self,
        *,
        wait: bool = False,
        timeout: float | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> Self:
        """Confirm the plan and start it; with ``wait``, return once it ends."""
        self.record = ExperimentOut.model_validate(
            self._client.post(f"{self.path}/{self.id}/start", {"confirm": True})
        )
        return self.wait(timeout=timeout, on_update=on_update) if wait else self


def _experiment_body(source: Any) -> dict[str, Any]:
    """An experiment as the API takes it: YAML text stays text, so the server reads it."""
    if isinstance(source, str) and "\n" in source:
        return {"content": source}
    return {"document": load_document(source)}


class Experiments:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def plan(self, source: Any) -> PlanOut:
        """Check an experiment (a mapping, a YAML or JSON file, or YAML text) and return its
        plan, saving nothing: trials, unique work per stage, estimate and budget check."""
        return PlanOut.model_validate(
            self._client.post("/experiments/plan", _experiment_body(source))
        )

    def create(self, source: Any) -> Experiment:
        """Save an experiment with its plan. Nothing runs until ``Experiment.start``."""
        return Experiment(
            self._client,
            ExperimentOut.model_validate(
                self._client.post("/experiments", _experiment_body(source))
            ),
        )

    def get(self, experiment: Any) -> Experiment:
        experiment_id = _id_of(experiment, "experiment")
        try:
            record = ExperimentOut.model_validate(self._client.get(f"/experiments/{experiment_id}"))
        except ApiError as error:
            raise _not_found(error, f"no experiment {experiment_id}") from None
        return Experiment(self._client, record)

    def list(self, *, status: str | None = None) -> Records[ExperimentOut]:
        return Records(
            ExperimentOut.model_validate(item)
            for item in self._client.pages("/experiments", status=status)
        )


# --- knowledge profiles -------------------------------------------------------------------


class Profiles:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def export(
        self,
        trial: Any,
        *,
        experiment: Any,
        name: str,
        description: str = "",
    ) -> ProfileOut:
        """Export a trial confirmed on the test split as a version of profile ``name``: its
        pipeline and retrieval config, with the evidence."""
        body = {
            "experiment_id": _id_of(experiment, "experiment"),
            "trial_id": _id_of(trial, "trial"),
            "name": name,
            "description": description,
        }
        return ProfileOut.model_validate(self._client.post("/profiles", body))

    def import_file(self, path: str | Path) -> ProfileOut:
        """Import a stratafi.knowledge-profile.v1 file (YAML or JSON)."""
        try:
            content = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            raise ConfigError(f"can't read {path}") from None
        return ProfileOut.model_validate(
            self._client.post("/profiles/import", {"content": content})
        )

    def get(self, profile: Any) -> ProfileOut:
        profile_id = _id_of(profile, "profile")
        try:
            return ProfileOut.model_validate(self._client.get(f"/profiles/{profile_id}"))
        except ApiError as error:
            raise _not_found(error, f"no profile {profile_id}") from None

    def list(self, *, name: str | None = None) -> Records[ProfileOut]:
        return Records(
            ProfileOut.model_validate(item) for item in self._client.pages("/profiles", name=name)
        )

    def download(self, profile: Any, *, format: str = "yaml") -> str:
        """The profile version as YAML or JSON text, with its review."""
        profile_id = _id_of(profile, "profile")
        return self._client.get_text(f"/profiles/{profile_id}/export", format=format)

    def set_alias(self, profile: Any, alias: str) -> AliasOut:
        """Name this version with ``alias`` (owners), moving it from another version."""
        profile_id = _id_of(profile, "profile")
        return AliasOut.model_validate(self._client.put(f"/profiles/{profile_id}/aliases/{alias}"))

    def remove_alias(self, profile: Any, alias: str) -> None:
        self._client.delete(f"/profiles/{_id_of(profile, 'profile')}/aliases/{alias}")

    def by_alias(self, name: str, alias: str) -> ProfileOut:
        """The version ``name@alias`` names."""
        try:
            found = AliasOut.model_validate(self._client.get(f"/profiles/by-alias/{name}/{alias}"))
        except ApiError as error:
            raise _not_found(error, f"no profile {name}@{alias}") from None
        return self.get(found.target_id)


# --- test suites --------------------------------------------------------------


def _check_files(document: Any, base: Path) -> dict[str, str]:
    """The files a suite's custom checks name (``python: checks/tables.py:function``),
    read from beside the suite file. A file the suite already carries in ``code`` may be
    missing here."""
    if not isinstance(document, Mapping):
        return {}
    inline = document.get("code") if isinstance(document.get("code"), Mapping) else {}
    root = base.resolve()
    files: dict[str, str] = {}
    for check in document.get("checks") or []:
        reference = check.get("python") if isinstance(check, Mapping) else None
        if not isinstance(reference, str):
            continue
        relative = reference.rpartition(":")[0]
        if not relative or relative in files:
            continue
        path = (root / relative).resolve()
        if relative.startswith("/") or not path.is_relative_to(root):
            raise ConfigError(f"{relative}: a check's file is a path beside the suite")
        if not path.is_file() and relative in inline:
            continue
        try:
            files[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            raise ConfigError(f"can't read {relative}, which a custom check names") from None
    return files


def _suite_body(source: Any) -> dict[str, Any]:
    """A suite as the API takes it: a file or YAML text goes as text, so the server reads it
    (a bare ``on:`` key included); a mapping goes as a document. A file's custom checks bring
    the files they name, from beside it, as ``code``."""
    if isinstance(source, Mapping):
        return {"document": load_document(source)}
    if isinstance(source, str) and "\n" in source:
        return {"content": source}
    path = Path(source)
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise ConfigError(f"can't read {path}") from None
    try:
        document = yaml.safe_load(content)
    except yaml.YAMLError:
        document = None  # the server says what's wrong with it
    code = _check_files(document, path.parent)
    return {"content": content, **({"code": code} if code else {})}


class TestRun(_Tracked[TestRunOut]):
    """A test run. It ends ``passed``, ``failed`` (a check failed or errored), ``error`` (it
    couldn't evaluate: ``record.error_code`` says why) or ``cancelled``."""

    __test__ = False
    kind = "test_run"
    path = "/test-runs"
    model = TestRunOut
    finished = frozenset({"passed", "failed", "error", "cancelled"})

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def checks(self) -> Records[CheckOut]:
        """Each check's result, once the run has finished."""
        self.refresh()
        return Records(CheckOut.model_validate(plain(check)) for check in self.record.checks or [])

    def report(self, format: str = "json", *, link_base: str | None = None) -> str:
        """The finished run's report: ``json`` (``stratafi.test-run.v1``), ``junit`` or
        ``markdown``, as text."""
        return self._client.get_text(
            f"{self.path}/{self.id}/report", format=format, link_base=link_base
        )


class Tests:
    """Test suites and their runs."""

    __test__ = False

    def __init__(self, client: "Client") -> None:
        self._client = client

    def validate(self, source: Any) -> ValidationOut:
        """Check a suite (a mapping, a YAML or JSON file, or YAML text) without saving it."""
        return ValidationOut.model_validate(
            self._client.post("/test-suites/validate", _suite_body(source))
        )

    def save(self, source: Any) -> SuiteOut:
        """Save a suite as its name's next version (the same document adds none)."""
        return SuiteOut.model_validate(self._client.post("/test-suites", _suite_body(source)))

    def run(
        self,
        suite: Any = None,
        *,
        source: Any = None,
        trigger: str = "api",
        wait: bool = False,
        timeout: float | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> TestRun:
        """Run a saved suite (``name`` or ``name@N``), or ``source``, a suite saved first. With
        ``wait``, return once it ends. ``trigger`` is ``api`` or ``ci``."""
        if (suite is None) == (source is None):
            raise ConfigError("run a saved suite by name, or a suite's source")
        body: dict[str, Any] = {"suite": str(suite)} if suite is not None else _suite_body(source)
        started = StartedOut.model_validate(
            self._client.post("/test-runs", {**body, "trigger": trigger})
        )
        run = TestRun(self._client, TestRunOut.model_validate(plain(started)))
        return run.wait(timeout=timeout, on_update=on_update) if wait else run

    def get(self, test_run: Any) -> TestRun:
        test_run_id = _id_of(test_run, "test run")
        try:
            record = TestRunOut.model_validate(self._client.get(f"/test-runs/{test_run_id}"))
        except ApiError as error:
            raise _not_found(error, f"no test run {test_run_id}") from None
        return TestRun(self._client, record)

    def list(self, *, suite: str | None = None, status: str | None = None) -> Records[TestRunOut]:
        return Records(
            TestRunOut.model_validate(item)
            for item in self._client.pages("/test-runs", suite=suite, status=status)
        )

    def history(self, name: str, *, limit: int = 50) -> HistoryOut:
        """A suite's finished runs with each check's result and value: its trends."""
        return HistoryOut.model_validate(
            self._client.get(f"/test-suites/{name}/history", limit=limit)
        )


# --- post-training --------------------------------------------------------------------------


class TrainingSet(_Tracked[TrainingSetOut]):
    """A training set mined from a dev eval run: records with each item's query,
    positives and hard negatives."""

    kind = "training_set"
    path = "/training-sets"
    model = TrainingSetOut
    finished = frozenset({"succeeded", "failed", "cancelled"})

    def records(self, *, split: str = "all") -> list[dict[str, Any]]:
        """Every record (``stratafi.training-set.v1``), with its texts."""
        found: list[dict[str, Any]] = []
        while True:
            page = self._client.get(
                f"{self.path}/{self.id}/records", split=split, offset=len(found), limit=200
            )
            found += page["items"]
            if len(found) >= page["total"] or not page["items"]:
                return found

    def export(self, format: str = "sentence-transformers", *, split: str = "all") -> str:
        """JSON Lines: ``sentence-transformers``, ``cross-encoder`` or ``chat``."""
        return self._client.get_text(f"{self.path}/{self.id}/export", format=format, split=split)

    def cancel(self) -> Self:
        raise ConfigError("mining a training set can't be cancelled; it's one short job")


class TrainingRun(_Tracked[TrainingRunOut]):
    """A training run: a trainer over a training set, from a base model."""

    kind = "training_run"
    path = "/training-runs"
    model = TrainingRunOut
    finished = frozenset({"succeeded", "failed", "cancelled"})

    @property
    def model_id(self) -> str | None:
        found = self.record.model_id
        return str(found) if found is not None else None

    def manifest(self) -> dict[str, Any]:
        """Its record, ``stratafi.training-run.v1``, once it has succeeded."""
        return dict(self._client.get(f"{self.path}/{self.id}/manifest"))


class Training:
    """Training sets and training runs (lab only)."""

    def __init__(self, client: "Client") -> None:
        self._client = client

    def mine(
        self,
        eval_run: Any,
        *,
        name: str | None = None,
        options: Mapping[str, Any] | None = None,
        wait: bool = False,
        timeout: float | None = None,
    ) -> TrainingSet:
        """Mine a training set from a succeeded eval run on the dev split. ``options``:
        ``rule``, ``max_positives``, ``max_negatives``, ``checkpoints``, ``depth``,
        ``negative_max_grade``, ``abstentions``, ``validation_share``,
        ``duplicate_threshold``, ``seed``."""
        body: dict[str, Any] = {"eval_run_id": _id_of(eval_run, "eval run")}
        if name is not None:
            body["name"] = name
        if options:
            body["options"] = dict(options)
        started = TrainingSet(
            self._client, TrainingSetOut.model_validate(self._client.post("/training-sets", body))
        )
        return started.wait(timeout=timeout) if wait else started

    def get_set(self, training_set: Any) -> TrainingSet:
        set_id = _id_of(training_set, "training set")
        try:
            record = TrainingSetOut.model_validate(self._client.get(f"/training-sets/{set_id}"))
        except ApiError as error:
            raise _not_found(error, f"no training set {set_id}") from None
        return TrainingSet(self._client, record)

    def sets(self, *, eval_run: Any = None) -> Records[TrainingSetOut]:
        params = {"limit": 200}
        if eval_run is not None:
            params["eval_run_id"] = _id_of(eval_run, "eval run")
        page = self._client.get("/training-sets", **params)
        return Records(TrainingSetOut.model_validate(item) for item in page["items"])

    def train(
        self,
        training_set: Any,
        *,
        name: str,
        trainer: str = "train/query-adapter",
        params: Mapping[str, Any] | None = None,
        wait: bool = False,
        timeout: float | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> TrainingRun:
        """Train a model named ``name`` (each run is a new version) with a ``train/``
        component on a training set."""
        if not NAME_RE.match(name):
            raise ConfigError("a model's name is lowercase letters, digits, '.', '_' and '-'")
        if not _COMPONENT_RE.match(trainer):
            raise ConfigError("a trainer is a component ID, such as train/query-adapter")
        body = {
            "training_set_id": _id_of(training_set, "training set"),
            "name": name,
            "trainer": {"component": trainer, "params": dict(params or {})},
        }
        started = TrainingRun(
            self._client, TrainingRunOut.model_validate(self._client.post("/training-runs", body))
        )
        return started.wait(timeout=timeout, on_update=on_update) if wait else started

    def get_run(self, training_run: Any) -> TrainingRun:
        run_id = _id_of(training_run, "training run")
        try:
            record = TrainingRunOut.model_validate(self._client.get(f"/training-runs/{run_id}"))
        except ApiError as error:
            raise _not_found(error, f"no training run {run_id}") from None
        return TrainingRun(self._client, record)

    def runs(self, *, training_set: Any = None) -> Records[TrainingRunOut]:
        params: dict[str, Any] = {"limit": 200}
        if training_set is not None:
            params["training_set_id"] = _id_of(training_set, "training set")
        page = self._client.get("/training-runs", **params)
        return Records(TrainingRunOut.model_validate(item) for item in page["items"])


class Models:
    """The trained-model registry: cards, withdrawal and downloads."""

    def __init__(self, client: "Client") -> None:
        self._client = client

    def list(
        self, *, kind: str | None = None, name: str | None = None, status: str | None = None
    ) -> Records[TrainedModelOut]:
        params = {
            key: value
            for key, value in (("kind", kind), ("name", name), ("status", status))
            if value is not None
        }
        page = self._client.get("/trained-models", limit=200, **params)
        return Records(TrainedModelOut.model_validate(item) for item in page["items"])

    def get(self, model: Any) -> TrainedModelOut:
        model_id = _id_of(model, "trained model")
        try:
            return TrainedModelOut.model_validate(self._client.get(f"/trained-models/{model_id}"))
        except ApiError as error:
            raise _not_found(error, f"no trained model {model_id}") from None

    def withdraw(self, model: Any) -> TrainedModelOut:
        model_id = _id_of(model, "trained model")
        return TrainedModelOut.model_validate(
            self._client.post(f"/trained-models/{model_id}/withdraw")
        )

    def download(self, model: Any, path: str | Path) -> Path:
        """Write the model's zip (its files, manifest and card) to ``path``."""
        model_id = _id_of(model, "trained model")
        target = Path(path)
        target.write_bytes(self._client.get_bytes(f"/trained-models/{model_id}/download"))
        return target

    def set_alias(self, model: Any, alias: str) -> AliasOut:
        """Name this version with ``alias``, such as ``champion``, moving it from another
        version of the same name."""
        model_id = _id_of(model, "trained model")
        return AliasOut.model_validate(
            self._client.put(f"/trained-models/{model_id}/aliases/{alias}")
        )

    def remove_alias(self, model: Any, alias: str) -> None:
        self._client.delete(f"/trained-models/{_id_of(model, 'trained model')}/aliases/{alias}")

    def by_alias(self, name: str, alias: str) -> TrainedModelOut:
        """The version ``name@alias`` names."""
        try:
            found = AliasOut.model_validate(
                self._client.get(f"/trained-models/by-alias/{name}/{alias}")
            )
        except ApiError as error:
            raise _not_found(error, f"no trained model {name}@{alias}") from None
        return self.get(found.target_id)

    def compare(
        self,
        eval_set: Any,
        *,
        role: str = "embed",
        metric: str = "recall@5",
        rule: str = DEFAULT_RULE,
        checkpoint: str = DEFAULT_CHECKPOINT,
        split: str | None = None,
    ) -> ModelComparisonOut:
        """Embedders, rerankers or adapters across an eval set version's eval runs: each
        model's best run, and controlled pairs that differ only in the model."""
        return ModelComparisonOut.model_validate(
            self._client.get(
                "/model-comparisons",
                eval_set_version_id=self._client.eval_sets.resolve(eval_set),
                role=role,
                metric=metric,
                rule=rule,
                checkpoint=checkpoint,
                split=split,
            )
        )


class Maps:
    """Pipeline maps: each RAG pipeline's stages with the datastores and services they touch
    ."""

    def __init__(self, client: "Client") -> None:
        self._client = client

    def get(self, *, pipeline: Any = None, retrieval: Any = None) -> MapOut:
        """The map of a saved pipeline spec, a retrieval config, or both (references as
        ``Pipelines.resolve`` and ``RetrievalConfigs.resolve`` take them)."""
        params = {
            "pipeline_spec_id": self._client.pipelines.resolve(pipeline) if pipeline else None,
            "retrieval_config_id": self._client.retrieval.resolve(retrieval) if retrieval else None,
        }
        return MapOut.model_validate(self._client.get("/pipeline-maps", **params))

    def of_run(self, run: Any) -> MapOut:
        return MapOut.model_validate(self._client.get(f"/runs/{_id_of(run, 'run')}/map"))

    def of_eval_run(self, eval_run: Any) -> MapOut:
        return MapOut.model_validate(
            self._client.get(f"/eval-runs/{_id_of(eval_run, 'eval run')}/map")
        )

    def rag_pipelines(self) -> Records[RagPipelineOut]:
        """Every pipeline spec and retrieval config pair the workspace has evaluated."""
        page = self._client.get("/rag-pipelines")
        return Records(RagPipelineOut.model_validate(item) for item in page["items"])
