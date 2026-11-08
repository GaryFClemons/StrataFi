"""Components and their registry.

A component is one implementation of one stage: ``parse/markdown``,
``chunk/fixed-tokens``. Each declares the fields of the components table in its ``info``, and
its parameters as a Pydantic model, ``Params``. That model's JSON Schema drives the UI's forms,
sweep editors and spec validation, so a new component needs no form code.

Parameters are always resolved before use: validated, with every default filled in. Resolved
parameters are what a cache key hashes and what a run manifest records, so ``{}`` and the
explicit defaults mean the same work.

The registry holds the components a process can run. ``stratafi_components.default_registry``
builds the built-in one; the API syncs it to the ``components`` table at startup, so old runs
can still name versions the code no longer has.

Each stage's components share a base class with the stage's ``run`` signature; those live with
the stage implementations in ``stratafi_components``. The worker calls them with a
``StageContext``.
"""

import importlib.util
import logging
import re
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from stratafi_core.hashing import content_hash

# The ingest graph's stages, in order, then the query-time component kinds.
STAGES = ("extract", "scan", "parse", "clean", "chunk", "enrich", "embed", "index")
QUERY_STAGES = ("rerank", "generate")
# Trainers post-train retrieval models; they never run in a pipeline.
TRAIN_STAGES = ("train",)
Stage = Literal["extract", "scan", "parse", "clean", "chunk", "enrich", "embed", "index"]
ComponentStage = Literal[
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
ResourceClass = Literal["cpu", "gpu", "io"]
Platform = Literal["linux", "windows"]
# The stores and services a component uses besides the artifact store and the database, for
# pipeline maps. Egress already names the model gateway and hosted providers.
Service = Literal["clamd", "lab-index", "model-files", "s3-source", "web", "server-folder"]
SERVICES: tuple[str, ...] = (
    "clamd",
    "lab-index",
    "model-files",
    "s3-source",
    "web",
    "server-folder",
)

COMPONENT_ID_RE = re.compile(r"[a-z][a-z0-9-]{0,31}/[a-z0-9][a-z0-9.-]{0,63}")
VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]{1,32})?")
EGRESS_RE = re.compile(r"none|model_gateway|provider:[a-z][a-z0-9-]{0,62}")


class ComponentError(Exception):
    """A component failed on one input. ``code`` is short and text-free, safe to log and store;
    ``retryable`` says whether trying again could help (a timeout, an unreachable service)."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class UnknownComponent(LookupError):
    def __init__(self, component_id: str) -> None:
        super().__init__("unknown component")
        self.component_id = component_id


class NoParams(BaseModel):
    """For components without parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentInfo(BaseModel):
    """The fields every component declares (the components table in the pipeline chapter)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Kind and name, for example parse/docling")
    stage: ComponentStage
    version: str = Field(description="Bumped whenever the component's output could change")
    title: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=600)
    accepts: tuple[str, ...] = Field(
        default=("*",), description="Media types or traits it can process; * for any"
    )
    resource_class: ResourceClass = "cpu"
    deterministic: bool = True
    license: str = Field(min_length=1, max_length=64)
    platforms: tuple[Platform, ...] = ("linux", "windows")
    egress: str = Field(default="none", description="none, model_gateway or provider:<name>")
    services: tuple[Service, ...] = Field(
        default=(),
        description="The stores and services it uses besides the artifact store and the "
        "database: clamd, lab-index, model-files, s3-source, web or server-folder (pipeline "
        "maps)",
    )
    lab_only: bool = Field(
        default=False,
        description="Never in production: a fake, an in-process model or a provider the model "
        "gateway doesn't serve",
    )
    baseline: bool = Field(default=False, description="Part of a baseline pipeline")

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not COMPONENT_ID_RE.fullmatch(value):
            raise ValueError("a component ID is kind/name in lowercase")
        return value

    @field_validator("version")
    @classmethod
    def _version_shape(cls, value: str) -> str:
        if not VERSION_RE.fullmatch(value):
            raise ValueError("a component version is semantic: 1.2.3")
        return value

    @field_validator("egress")
    @classmethod
    def _egress_shape(cls, value: str) -> str:
        if not EGRESS_RE.fullmatch(value):
            raise ValueError("egress is none, model_gateway or provider:<name>")
        return value


class Component:
    """The base of every component. Subclasses set ``info`` and ``Params``; each stage's base
    class adds the method the worker calls."""

    info: ClassVar[ComponentInfo]
    Params: ClassVar[type[BaseModel]] = NoParams
    # Optional libraries the component imports, and the extra of stratafi-components that
    # installs them. Without them the component stays registered but can't run.
    requires: ClassVar[tuple[str, ...]] = ()
    extra: ClassVar[str | None] = None
    # A license that needs the owner's review, such as AGPL-3.0: the component refuses
    # to run until the configuration allows it (``licenses.allowed``).
    license_gate: ClassVar[str | None] = None

    @classmethod
    def unavailable(cls) -> str | None:
        """Why this installation can't run the component (a missing extra), or None."""
        missing = [name for name in cls.requires if importlib.util.find_spec(name) is None]
        if not missing:
            return None
        return f"needs the {cls.extra or missing[0]} extra (uv sync --extra {cls.extra})"

    @classmethod
    def params_schema(cls) -> dict[str, Any]:
        return cls.Params.model_json_schema()

    @classmethod
    def resolve_params(cls, raw: Mapping[str, Any] | BaseModel | None = None) -> BaseModel:
        """Validated parameters with every default filled in. Raises ``ValidationError``."""
        if isinstance(raw, cls.Params):
            return raw
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(mode="json")
        return cls.Params.model_validate(dict(raw or {}))

    @classmethod
    def dump_params(cls, params: BaseModel) -> dict[str, Any]:
        return params.model_dump(mode="json")

    @classmethod
    def params_hash(cls, params: BaseModel) -> str:
        return content_hash(cls.dump_params(params))

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        """Output-relevant environment known before anything runs, such as a library's
        version: part of every cache key. Most components have none. A model's revision is
        resolved when a run starts instead (``stratafi_core.planner``)."""
        return {}


@dataclass(frozen=True)
class ComponentDescription:
    info: ComponentInfo
    params_schema: dict[str, Any]
    unavailable: str | None = None  # why this installation can't run it


class ComponentRegistry:
    """The components one process can run, by ID."""

    def __init__(self, components: Iterable[type[Component]] = ()) -> None:
        self._components: dict[str, type[Component]] = {}
        for component in components:
            self.register(component)

    def register(self, component: type[Component]) -> type[Component]:
        info = getattr(component, "info", None)
        if not isinstance(info, ComponentInfo):
            raise TypeError("a component needs an info")
        if not (isinstance(component.Params, type) and issubclass(component.Params, BaseModel)):
            raise TypeError("a component's Params is a Pydantic model")
        if info.id in self._components:
            raise ValueError(f"{info.id} is registered twice")
        self._components[info.id] = component
        return component

    def get(self, component_id: str) -> type[Component]:
        try:
            return self._components[component_id]
        except KeyError:
            raise UnknownComponent(component_id) from None

    def __contains__(self, component_id: object) -> bool:
        return component_id in self._components

    def __iter__(self) -> Iterator[type[Component]]:
        return iter(self._components[key] for key in sorted(self._components))

    def __len__(self) -> int:
        return len(self._components)

    def for_stage(self, stage: str) -> list[type[Component]]:
        return [component for component in self if component.info.stage == stage]

    def describe(self, component_id: str) -> ComponentDescription:
        component = self.get(component_id)
        return ComponentDescription(
            component.info, component.params_schema(), component.unavailable()
        )

    def descriptions(self) -> list[ComponentDescription]:
        return [ComponentDescription(c.info, c.params_schema(), c.unavailable()) for c in self]


def validation_paths(error: ValidationError, prefix: str = "") -> list[str]:
    """The dotted paths of a Pydantic error's failures, under ``prefix``. Never the values."""
    paths: list[str] = []
    for item in error.errors(include_input=False, include_url=False):
        parts = [str(part) for part in item.get("loc", ())]
        path = ".".join(part for part in [prefix, *parts] if part)
        if path not in paths:
            paths.append(path or "(root)")
    return paths


# --- running a component ---------------------------------------------------------------------


class Cancelled(Exception):
    """Raised by ``StageContext.checkpoint`` when the work must stop."""


@dataclass
class StageContext:
    """What a component gets besides its inputs and parameters.

    - ``checkpoint()`` raises ``Cancelled`` once the job is cancelled or the worker stops
    - ``metric(name, value)`` records a number for the run record; names and values only
    - ``scratch_dir()`` is a private directory, deleted after the task
    - ``provider(name, tokens=...)`` waits for a remote provider's rate and concurrency limits
      (the worker sets ``rate_limit``; elsewhere it doesn't wait)
    - ``services`` holds clients a component may use: the model provider, the clamd address,
      the redaction key; components look up what they need and fail with a code if it's
      missing
    - ``logger`` is for metadata only: IDs, counts and codes, never text
    """

    workspace_id: str = ""
    services: Mapping[str, Any] = field(default_factory=dict)
    limits: Any = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("stratafi.stage"))
    cancelled: threading.Event = field(default_factory=threading.Event)
    metrics: dict[str, float] = field(default_factory=dict)
    rate_limit: Callable[[str, int], AbstractContextManager[Any]] | None = None
    _scratch: tempfile.TemporaryDirectory[str] | None = None

    def provider(self, name: str, *, tokens: int = 0) -> AbstractContextManager[Any]:
        return self.rate_limit(name, tokens) if self.rate_limit else nullcontext()

    def checkpoint(self) -> None:
        if self.cancelled.is_set():
            raise Cancelled()

    def metric(self, name: str, value: float, *, add: bool = False) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_.]{0,63}", name):
            raise ValueError("metric names are lowercase identifiers")
        self.metrics[name] = self.metrics.get(name, 0) + value if add else value

    def service(self, name: str) -> Any:
        try:
            return self.services[name]
        except KeyError:
            raise ComponentError(f"{name}_not_configured") from None

    def scratch_dir(self) -> Path:
        if self._scratch is None:
            self._scratch = tempfile.TemporaryDirectory(prefix="stratafi-stage-")
        return Path(self._scratch.name)

    def close(self) -> None:
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None
