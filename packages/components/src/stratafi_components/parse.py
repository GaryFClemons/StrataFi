"""The parse stage.

A parser turns a raw document into ``stratafi.parsed-document.v1``. Every parse runs in a
constrained subprocess (``stratafi_components.sandbox``) with the limits in ``limits``:
``parse_timeout_seconds``, ``parse_memory_bytes`` and ``parse_confinement``. A parser
implements ``parse_local``, which runs inside that subprocess; the worker calls ``parse``.

The parsers themselves are in ``stratafi_components.parsers``; the default route sends Markdown
and plain text to ``parse/markdown``, PDFs to ``parse/pypdf`` and office and web formats to
``parse/markitdown`` (``stratafi_core.specs.ParseStage``).
"""

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from stratafi_components.sandbox import DEFAULT_MEMORY_BYTES, run_isolated
from stratafi_core.artifacts import ParsedDocument, RawDocument
from stratafi_core.components import Component, ComponentError, StageContext

DEFAULT_TIMEOUT_SECONDS = 120.0
SANDBOX_TARGET = "stratafi_components.parse:parse_in_sandbox"


class ParseComponent(Component):
    """A parser: raw document → parsed document, in a sandbox. A parser that calls a service
    (Tika, a hosted API) sets ``isolated`` False and implements ``parse_service``: the service
    is the isolation, and the call runs in the worker."""

    isolated: ClassVar[bool] = True
    # The address space a parser needs when it's more than the configured limit: model runtimes
    # such as onnxruntime reserve far more virtual memory than they use.
    memory_bytes: ClassVar[int | None] = None
    # Numbers a parse records while it runs (OCR pages); they travel back from the sandbox.
    local_metrics: dict[str, float]

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        """Parse in this process. Runs inside the sandbox; call ``parse`` instead."""
        raise NotImplementedError

    def parse_service(
        self, path: Path, raw: RawDocument, params: BaseModel, context: StageContext
    ) -> ParsedDocument:
        raise NotImplementedError

    def parse(
        self, path: Path, raw: RawDocument, params: BaseModel, context: StageContext
    ) -> ParsedDocument:
        if self.unavailable() is not None:
            raise ComponentError("extra_missing")
        if self.license_gate and self.license_gate not in context.services.get("licenses", ()):
            raise ComponentError("license_not_allowed")
        if not self.isolated:
            context.checkpoint()
            document = self.parse_service(path, raw, params, context)
            record_parse_metrics(context, document)
            return document
        limits = context.limits
        timeout = float(getattr(limits, "parse_timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        memory = int(getattr(limits, "parse_memory_bytes", DEFAULT_MEMORY_BYTES))
        confinement = getattr(limits, "parse_confinement", "best_effort")
        context.checkpoint()
        stats: dict[str, Any] = {}
        document_path = str(path.resolve())
        result = run_isolated(
            SANDBOX_TARGET,
            {
                "component": self.info.id,
                "path": document_path,
                "raw": raw.model_dump(mode="json", by_alias=True),
                "params": self.dump_params(params),
            },
            timeout=timeout,
            memory_bytes=memory,
            # Model runtimes reserve address space they don't use; their writable
            # memory stays under the configured limit.
            address_space_bytes=self.memory_bytes,
            stats=stats,
            confinement=confinement,
            read_files=(document_path,),
        )
        document = ParsedDocument.model_validate(result["document"])
        for name, number in result.get("metrics", {}).items():
            context.metric(name, number)
        if "peak_memory_bytes" in stats:
            context.metric("peak_memory_bytes", stats["peak_memory_bytes"])
        record_confinement(context, stats.get("confinement"))
        record_parse_metrics(context, document)
        return document


def record_confinement(context: StageContext, confinement: Any) -> None:
    """How the parse was confined, as numbers in the task's record."""
    if not isinstance(confinement, dict):
        return
    context.metric("sandbox_seccomp", 1 if confinement.get("seccomp") else 0)
    context.metric(
        "sandbox_network_namespace", 1 if confinement.get("network") == "namespace" else 0
    )
    context.metric("sandbox_landlock_abi", int(confinement.get("landlock") or 0))


def record_parse_metrics(context: StageContext, document: ParsedDocument) -> None:
    from stratafi_components.parse_quality import parse_quality

    context.metric("elements", len(document.elements))
    context.metric("pages", len(document.pages))
    context.metric("chars", len(document.text))
    for name, number in parse_quality(document).items():
        if number is not None:
            context.metric(name, number)


def parse_in_sandbox(payload: dict[str, Any]) -> dict[str, Any]:
    """The sandbox's entry point: build the parser and run it here."""
    from stratafi_components import default_registry

    component = default_registry().get(payload["component"])()
    if not isinstance(component, ParseComponent):
        raise ComponentError("not_a_parser")
    if component.unavailable() is not None:
        raise ComponentError("extra_missing")
    params = component.resolve_params(payload["params"])
    raw = RawDocument.model_validate(payload["raw"])
    component.local_metrics = {}
    document = component.parse_local(Path(payload["path"]), raw, params)
    return {
        "document": document.model_dump(mode="json", by_alias=True),
        "metrics": dict(component.local_metrics),
    }


COMPONENTS: list[type[Component]] = []
