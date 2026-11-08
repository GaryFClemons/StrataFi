"""Artifact payload schemas.

- ``RawDocument`` is ``stratafi.raw-document.v1``: the extract stage's record of a document's
  stored bytes and what they are (type, traits).
- ``ScanReport`` is ``stratafi.scan-report.v1``: what each scan check found (outcomes and
  finding counts, never values) and what the policy decided.
- ``ParsedDocument`` is ``stratafi.parsed-document.v1``: what a parser makes of one document.
  Its ``text`` is the canonical concatenation that every element's ``char_start`` and
  ``char_end`` point into: element texts joined by ``ELEMENT_SEPARATOR``. A cleaned document is
  the same schema plus ``edits``, which record positions only; the before and after text is
  rebuilt on demand.
- ``Chunk`` is ``stratafi.chunk.v1``, one per line of a chunks artifact (JSONL).
- ``VectorsSidecar`` is ``stratafi.vectors.v1``: the JSON beside a vectors array (``.npy``,
  float32 ``[n, d]``), naming its chunks, model and revision.
- ``IndexManifest`` is ``stratafi.index.v1``: a corpus-level index's description, beside its
  payload (an exact index's matrix, a keyword index's term statistics).

The JSON Schemas in ``contracts/schemas`` are exported from these models
(``scripts/export_contracts.py``), and the fixtures in ``contracts/fixtures`` must validate
against both.

Payloads are keyed by content: two corpus entries with the same bytes share every
artifact, so a payload holds nothing specific to one entry (its name, its source). That lives
in the document and corpus records.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

RAW_DOCUMENT_SCHEMA = "stratafi.raw-document.v1"
SCAN_REPORT_SCHEMA = "stratafi.scan-report.v1"
PARSED_DOCUMENT_SCHEMA = "stratafi.parsed-document.v1"
CHUNK_SCHEMA = "stratafi.chunk.v1"
VECTORS_SCHEMA = "stratafi.vectors.v1"
INDEX_SCHEMA = "stratafi.index.v1"
# Element texts are joined with a blank line, so the canonical text reads as paragraphs.
ELEMENT_SEPARATOR = "\n\n"

ElementType = Literal[
    "title",
    "heading",
    "paragraph",
    "list_item",
    "table",
    "figure",
    "caption",
    "code",
    "page_header",
    "page_footer",
    "page_number",
    "other",
]
ReplacementKind = Literal[
    "delete", "space", "newline", "nfkc", "join", "mask", "placeholder", "hash"
]
ScanOutcome = Literal[
    "clean",
    "infected",
    "encrypted",
    "malformed",
    "limit_exceeded",
    "pii",
    "error",
    "duplicate",
]
CheckOutcome = Literal[
    "clean",
    "infected",
    "encrypted",
    "malformed",
    "limit_exceeded",
    "pii",
    "error",
    "duplicate",
    "skipped",
]
Decision = Literal["continue", "flag", "exclude"]
# The order a report's summary outcome is chosen in, when checks disagree.
OUTCOME_PRIORITY: tuple[str, ...] = (
    "infected",
    "pii",
    "encrypted",
    "malformed",
    "limit_exceeded",
    "error",
    "duplicate",
)
DECISION_RANK = {"continue": 0, "flag": 1, "exclude": 2}
Hex64 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ElementId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,64}$")]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class RawDocument(_Model):
    """``stratafi.raw-document.v1``: a document's stored bytes, and what they are."""

    schema_: Literal["stratafi.raw-document.v1"] = Field(
        default=RAW_DOCUMENT_SCHEMA, alias="schema"
    )
    document_id: str = Field(min_length=1, max_length=64)
    sha256: Hex64
    size_bytes: int = Field(ge=0)
    extension: str = Field(max_length=32, description="From the entry's name, e.g. .pdf")
    media_type: str = Field(min_length=1, max_length=255, description="What the bytes are")
    kind: str = Field(min_length=1, max_length=32, description="The sniffed kind, e.g. pdf")
    declared_type: str | None = Field(default=None, description="What the extension says")
    mismatch: bool = Field(description="The bytes disagree with the extension")
    known: bool = Field(description="The extension is one Stratafi lists")
    traits: dict[str, bool | int | None] = Field(default_factory=dict)

    @property
    def trait_names(self) -> frozenset[str]:
        """The traits that are true, for parse routes such as ``pdf[scanned]``."""
        return frozenset(name for name, value in self.traits.items() if value is True)


class ScanCheck(_Model):
    """One check's result. ``findings`` are counts by category; values never leave the scanner."""

    component: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    outcome: CheckOutcome
    reason: str | None = Field(default=None, max_length=64, description="A short code")
    findings: dict[str, int] = Field(default_factory=dict)
    details: dict[str, str | int | bool | None] = Field(
        default_factory=dict, description="Metadata: engine version, signature revision"
    )


class ScanReport(_Model):
    """``stratafi.scan-report.v1``."""

    schema_: Literal["stratafi.scan-report.v1"] = Field(default=SCAN_REPORT_SCHEMA, alias="schema")
    document_id: str = Field(min_length=1, max_length=64)
    sha256: Hex64
    outcome: ScanOutcome = Field(description="The most serious outcome any check reached")
    decision: Decision = Field(description="What the policy does with the document")
    checks: list[ScanCheck]
    findings: dict[str, int] = Field(default_factory=dict, description="Totals by category")

    @property
    def excluded(self) -> bool:
        return self.decision == "exclude"


class ParserRef(_Model):
    id: str = Field(min_length=1, max_length=128, description="The parse component's ID")
    version: str = Field(min_length=1, max_length=64)
    params_hash: Hex64 = Field(description="The hash of its resolved parameters")


class Page(_Model):
    number: int = Field(ge=1)
    width: float | None = Field(default=None, ge=0, description="Points")
    height: float | None = Field(default=None, ge=0, description="Points")


class TableCell(_Model):
    r: int = Field(ge=0)
    c: int = Field(ge=0)
    text: str
    header: bool = False


class TableData(_Model):
    rows: int = Field(ge=0)
    cols: int = Field(ge=0)
    cells: list[TableCell] = Field(default_factory=list)


class Element(_Model):
    id: ElementId
    type: ElementType
    text: str
    level: int | None = Field(default=None, ge=0, le=9, description="Headings: 1 is the top")
    page: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="x0, y0, x1, y1 in points, from the page's bottom left"
    )
    table: TableData | None = None
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    parent_id: ElementId | None = None


class Edit(_Model):
    """One cleaning edit, as a position in the input document's text. A redaction's
    ``replacement_kind`` is ``mask`` (one character per character), ``placeholder`` or ``hash``;
    the last two keep their replacement token in ``label`` (``[SSN]``, ``[SSN:1a2b3c4d]``),
    since it can't be rebuilt without the value. ``label`` never holds document text."""

    rule: str = Field(min_length=1, max_length=128, description="The clean component's ID")
    element_id: ElementId
    start: int = Field(ge=0, description="Offset into the input document's text")
    end: int = Field(ge=0)
    replacement_kind: ReplacementKind
    label: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^\[[A-Z0-9_]{1,32}(:[0-9a-f]{8,16})?\]$",
        description="A redaction's token: [CATEGORY] or [CATEGORY:keyed hash]",
    )


class ParsedDocument(_Model):
    schema_: Literal["stratafi.parsed-document.v1"] = Field(
        default=PARSED_DOCUMENT_SCHEMA, alias="schema"
    )
    document_id: str = Field(min_length=1, max_length=64)
    parser: ParserRef
    title: str | None = None
    media_type: str = Field(min_length=1, max_length=255)
    pages: list[Page] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)
    text: str = Field(description="The canonical concatenation that element offsets point into")
    warnings: list[str] = Field(default_factory=list, description="Codes, never text")
    language: str | None = Field(
        default=None,
        pattern=r"^[a-z]{2,3}$",
        description="The document's main language (ISO 639-1), when clean/language tagged it",
    )
    edits: list[Edit] | None = Field(
        default=None, description="Cleaned documents only: the edits that made them"
    )

    @model_validator(mode="after")
    def _offsets_point_into_the_text(self) -> "ParsedDocument":
        seen: set[str] = set()
        for index, element in enumerate(self.elements):
            if element.id in seen:
                raise ValueError(f"elements.{index}.id is a duplicate")
            seen.add(element.id)
            if not 0 <= element.char_start <= element.char_end <= len(self.text):
                raise ValueError(f"elements.{index}: offsets are outside the text")
            if self.text[element.char_start : element.char_end] != element.text:
                raise ValueError(f"elements.{index}: offsets don't match the element's text")
        return self


def build_text(texts: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join element texts into the canonical text; returns it and each element's offsets."""
    offsets: list[tuple[int, int]] = []
    position = 0
    for index, text in enumerate(texts):
        if index:
            position += len(ELEMENT_SEPARATOR)
        offsets.append((position, position + len(text)))
        position += len(text)
    return ELEMENT_SEPARATOR.join(texts), offsets


class Chunk(_Model):
    """One chunk: a line of a chunks artifact."""

    chunk_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    document_id: str = Field(min_length=1, max_length=64)
    ordinal: int = Field(ge=0)
    text: str
    index_text: str = Field(description="Exactly what is embedded and keyword-indexed")
    element_ids: list[ElementId] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    char_start: int = Field(ge=0, description="Offset into the cleaned document's text")
    char_end: int = Field(ge=0)
    heading_path: list[str] = Field(default_factory=list)
    tokens: dict[str, int] = Field(
        default_factory=dict, description="Token counts by estimator, e.g. chars/4"
    )
    context: str | None = Field(default=None, description="Added by an enrich component")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ranges(self) -> "Chunk":
        if self.char_end < self.char_start:
            raise ValueError("char_end is before char_start")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end is before page_start")
        return self


class VectorsSidecar(_Model):
    """``stratafi.vectors.v1``: what a vectors array holds, row by row."""

    schema_: Literal["stratafi.vectors.v1"] = Field(default=VECTORS_SCHEMA, alias="schema")
    chunk_ids: list[str] = Field(description="Row i is chunk_ids[i]'s vector")
    component: str = Field(description="The embed component")
    model: str = Field(description="The model alias, or the fake's name")
    served_model: str | None = Field(default=None, description="What the gateway says it ran")
    revision: str = Field(min_length=1, max_length=128, description="X-Model-Revision")
    dims: int = Field(ge=1, description="Dimensions after any truncation")
    normalized: bool
    dtype: Literal["float32", "float16", "int8", "binary"] = Field(
        default="float32", description="As stored: int8 is round(x * 127); binary packs sign bits"
    )
    full_dims: int | None = Field(
        default=None, ge=1, description="What the model returned, before truncation"
    )
    prefix: str = Field(default="", description="Prepended to each index text before embedding")
    tokens: int = Field(default=0, ge=0, description="Tokens the provider reported")
    truncated_inputs: int = Field(default=0, ge=0, description="Texts over the model's input limit")

    @model_validator(mode="after")
    def _distinct(self) -> "VectorsSidecar":
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk_ids repeat")
        return self


class IndexManifest(_Model):
    """``stratafi.index.v1``: a corpus-level index, as built."""

    schema_: Literal["stratafi.index.v1"] = Field(default=INDEX_SCHEMA, alias="schema")
    kind: Literal["dense", "keyword"]
    component: str
    version: str
    params: dict[str, Any] = Field(default_factory=dict)
    chunk_ids: list[str] = Field(description="Row or posting i is chunk_ids[i]")
    document_ids: list[str] = Field(description="Each chunk's document, row by row")
    chunks: list[Hex64] = Field(description="The chunks artifacts it was built from, in order")
    model: str | None = Field(default=None, description="Dense: the embedding model")
    revision: str | None = Field(default=None, description="Dense: the one embedding revision")
    dims: int | None = Field(default=None, ge=1)
    metric: Literal["cosine", "dot"] | None = None
    dtype: str | None = Field(default=None, description="Dense: how the vectors were stored")
    backend: Literal["memory", "postgres"] = Field(
        default="memory", description="postgres: the index lives in a table"
    )
    table: str | None = Field(default=None, max_length=63, description="postgres: its table")
    stats: dict[str, float] = Field(
        default_factory=dict, description="Build numbers, such as ANN recall@10"
    )

    @model_validator(mode="after")
    def _aligned(self) -> "IndexManifest":
        if len(self.document_ids) != len(self.chunk_ids):
            raise ValueError("document_ids and chunk_ids differ in length")
        if self.kind == "dense" and (self.revision is None or self.dims is None):
            raise ValueError("a dense index names its revision and dimensions")
        return self
