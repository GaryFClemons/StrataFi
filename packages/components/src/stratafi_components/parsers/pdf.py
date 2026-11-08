"""Text-layer PDF parsers, the reference points for every other PDF parser.

- ``parse/pypdf`` reads each page's text layer with pypdf (BSD-3-Clause), in its plain or
  layout mode, as one paragraph per block of text between blank lines (or one per page). It
  sees no structure: no headings, no tables, no boxes.
- ``parse/pdfplumber`` (MIT, the ``pdfplumber`` extra) reads words with their positions: lines
  are grouped into paragraphs by the gaps between them, lines in a larger font become headings,
  and tables found by ruling lines become table elements with their cells. Every element has a
  box, so the Parse screen can draw it over the page.

Neither does OCR: a scanned PDF comes out empty, with the ``no_text`` warning.
"""

import io
import re
import statistics
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import pypdf
from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.detect import PDF
from stratafi_components.parse import ParseComponent
from stratafi_components.parsers.common import Builder
from stratafi_core.artifacts import ParsedDocument, RawDocument
from stratafi_core.components import ComponentError, ComponentInfo

_BLANK_LINES = re.compile(r"\n\s*\n")


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _reader(data: bytes) -> pypdf.PdfReader:
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ComponentError("encrypted")
        return reader
    except ComponentError:
        raise
    except Exception:  # untrusted input: any failure is malformed
        raise ComponentError("malformed") from None


def page_sizes(reader: pypdf.PdfReader) -> list[dict[str, Any]]:
    return [
        {"number": n, "width": float(page.mediabox.width), "height": float(page.mediabox.height)}
        for n, page in enumerate(reader.pages, start=1)
    ]


class PypdfParams(_Params):
    mode: Literal["plain", "layout"] = Field(
        default="plain",
        description="pypdf's extraction mode: plain text, or laid out as on the page",
    )
    paragraphs: Literal["blank_lines", "page"] = Field(
        default="blank_lines", description="One element per block between blank lines, or per page"
    )


class PypdfParser(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/pypdf",
        stage="parse",
        version="1.0.0",
        title="pypdf text layer",
        description="Each page's text layer with pypdf, as paragraphs between blank lines or one "
        "per page: a reference point with no structure and no OCR.",
        accepts=(PDF,),
        license="BSD-3-Clause",
    )
    Params: ClassVar[type[BaseModel]] = PypdfParams

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        return {"pypdf": pypdf.__version__}

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        params = cast(PypdfParams, params)
        reader = _reader(path.read_bytes())
        builder = Builder(raw, self, params)
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text(extraction_mode=params.mode) or ""
            except Exception:  # one unreadable page doesn't lose the rest
                builder.warn("page_unreadable")
                continue
            blocks = [text] if params.paragraphs == "page" else _BLANK_LINES.split(text)
            for block in blocks:
                builder.add("paragraph", block.strip(), page=number)
        metadata = reader.metadata
        title = str(metadata.title).strip() if metadata and metadata.title else None
        return builder.document(title=title or None, pages=page_sizes(reader))


class PdfplumberParams(_Params):
    tables: bool = Field(default=True, description="Find ruled tables and keep their cells")
    headings: bool = Field(default=True, description="Lines in a larger font become headings")
    heading_ratio: float = Field(
        default=1.15, ge=1.0, le=3.0, description="Font size over the page's body size"
    )
    paragraph_gap: float = Field(
        default=1.4, ge=0.5, le=5.0, description="A gap this many line heights starts a paragraph"
    )


def _line_size(line: dict[str, Any]) -> float:
    sizes = [float(char.get("size", 0)) for char in line.get("chars", []) if char.get("size")]
    return statistics.median(sizes) if sizes else float(line["bottom"] - line["top"])


def _inside(line: dict[str, Any], boxes: list[tuple[float, float, float, float]]) -> bool:
    middle_x = (line["x0"] + line["x1"]) / 2
    middle_y = (line["top"] + line["bottom"]) / 2
    return any(x0 <= middle_x <= x1 and top <= middle_y <= bottom for x0, top, x1, bottom in boxes)


class PdfplumberParser(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/pdfplumber",
        stage="parse",
        version="1.0.0",
        title="pdfplumber",
        description="Words with positions: paragraphs from line gaps, headings from font size, "
        "ruled tables with their cells, and a box for every element. No OCR.",
        accepts=(PDF,),
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = PdfplumberParams
    requires: ClassVar[tuple[str, ...]] = ("pdfplumber",)
    extra: ClassVar[str | None] = "pdfplumber"

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        try:
            import pdfplumber

            return {"pdfplumber": pdfplumber.__version__}
        except ImportError:
            return {}

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        import pdfplumber

        params = cast(PdfplumberParams, params)
        builder = Builder(raw, self, params)
        pages = []
        try:
            pdf = pdfplumber.open(str(path))
        except Exception:
            raise ComponentError("malformed") from None
        with pdf:
            for number, page in enumerate(pdf.pages, start=1):
                height = float(page.height)
                pages.append({"number": number, "width": float(page.width), "height": height})

                def box(x0, top, x1, bottom, height=height):
                    return (x0, height - bottom, x1, height - top)

                tables = page.find_tables() if params.tables else []
                boxes = [table.bbox for table in tables]
                lines = [
                    line
                    for line in page.extract_text_lines(return_chars=True)
                    if line["text"].strip() and not _inside(line, boxes)
                ]
                body = statistics.median([_line_size(line) for line in lines]) if lines else 0
                placed: list[tuple[float, str, Any]] = []
                for table in tables:
                    placed.append((table.bbox[1], "table", table))
                block: list[dict[str, Any]] = []
                blocks: list[list[dict[str, Any]]] = []
                for line in lines:
                    gap_limit = (line["bottom"] - line["top"]) * params.paragraph_gap
                    heading = (
                        params.headings and body and _line_size(line) > body * params.heading_ratio
                    )
                    if block and (
                        line["top"] - block[-1]["bottom"] > gap_limit
                        or heading != block[-1]["_heading"]
                    ):
                        blocks.append(block)
                        block = []
                    block.append({**line, "_heading": bool(heading)})
                if block:
                    blocks.append(block)
                for lines_of_block in blocks:
                    placed.append((lines_of_block[0]["top"], "text", lines_of_block))
                for _, kind, item in sorted(placed, key=lambda entry: entry[0]):
                    if kind == "table":
                        rows = item.extract() or []
                        builder.add_table(rows, page=number, bbox=box(*item.bbox))
                        continue
                    text = "\n".join(line["text"].strip() for line in item)
                    x0 = min(line["x0"] for line in item)
                    x1 = max(line["x1"] for line in item)
                    bbox = box(x0, item[0]["top"], x1, item[-1]["bottom"])
                    if item[0]["_heading"]:
                        builder.add(
                            "heading", " ".join(text.split()), level=1, page=number, bbox=bbox
                        )
                    else:
                        builder.add("paragraph", text, page=number, bbox=bbox)
            title = (pdf.metadata or {}).get("Title")
        return builder.document(title=str(title).strip() if title else None, pages=pages)


COMPONENTS = [PypdfParser, PdfplumberParser]
