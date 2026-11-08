"""A registry of stub components for spec tests: the baseline components' IDs, stages and
parameter shapes, with no behavior. Spec validation, hashing and the spec routes are tested
against it, so those tests don't change as real components arrive."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from stratafi_core.components import Component, ComponentInfo, ComponentRegistry
from stratafi_core.pii import DEFAULT_CATEGORIES, PiiCategory


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClamavParams(_Params):
    max_signature_age_hours: int = Field(default=72, ge=1)


class PiiParams(_Params):
    categories: list[PiiCategory] = Field(default_factory=lambda: list(DEFAULT_CATEGORIES))


class ChunkParams(_Params):
    max_tokens: int = Field(default=400, ge=1, le=100_000)
    min_tokens: int = Field(default=200, ge=0)
    overlap: int = Field(default=0, ge=0)
    token_estimator: Literal["chars/4"] = "chars/4"  # noqa: S105 - not a secret


class FixedParams(_Params):
    size: int = Field(default=512, ge=1)
    overlap: int = Field(default=64, ge=0)


class EmbedParams(_Params):
    alias: str = "embed-default"
    batch_size: int = Field(default=32, ge=1, le=2048)


class RerankParams(_Params):
    alias: str = "rerank-default"


# What parse/markitdown reads: the default route's office and web formats among them.
MARKITDOWN_TYPES = (
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/html",
    "text/csv",
    "application/json",
    "application/xml",
    "text/plain",
    "text/markdown",
)


def stub(
    component_id: str,
    stage: str,
    params: type[BaseModel] | None = None,
    *,
    accepts: tuple[str, ...] = ("*",),
    version: str = "1.0.0",
    **attributes_: object,
) -> type[Component]:
    info = ComponentInfo(
        id=component_id,
        stage=stage,  # type: ignore[arg-type]
        version=version,
        title=component_id,
        description=f"A stub of {component_id} for tests.",
        accepts=accepts,
        license="MIT",
    )
    attributes: dict[str, object] = {"info": info, **attributes_}
    if params is not None:
        attributes["Params"] = params
    name = "Stub" + "".join(part.title() for part in component_id.replace("/", "-").split("-"))
    return type(name, (Component,), attributes)


def stub_registry() -> ComponentRegistry:
    return ComponentRegistry(
        [
            stub("extract/upload", "extract"),
            stub("unpack/zip", "extract"),
            stub("scan/content-safety", "scan"),
            stub("scan/clamav", "scan", ClamavParams),
            stub("scan/pii", "scan", PiiParams),
            stub("scan/dedupe", "scan"),
            stub("parse/markdown", "parse", accepts=("text/markdown", "text/plain")),
            stub("parse/pypdf", "parse", accepts=("application/pdf",)),
            stub("parse/markitdown", "parse", accepts=MARKITDOWN_TYPES),
            stub("parse/stub-ocr", "parse", accepts=("pdf[scanned]",)),
            stub("clean/whitespace", "clean"),
            stub("clean/unicode", "clean"),
            stub("clean/near-duplicates", "clean", corpus_scope=True),
            stub("chunk/sections", "chunk", ChunkParams),
            stub("chunk/fixed-tokens", "chunk", FixedParams),
            stub("enrich/none", "enrich"),
            stub("embed/gateway", "embed", EmbedParams),
            stub("index/exact", "index"),
            stub("index/bm25", "index"),
            stub("rerank/none", "rerank"),
            stub("rerank/gateway", "rerank", RerankParams),
        ]
    )
