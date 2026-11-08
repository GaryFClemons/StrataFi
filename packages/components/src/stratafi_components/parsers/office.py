"""Office and web formats, read with defusedxml and the standard library: no
document library, nothing fetched, and the same on every platform.

- ``parse/spreadsheet``: XLSX workbooks and CSV or TSV files as **rows with their headers**:
  each sheet gets a heading, and each row becomes ``Header: value; Header: value``, so a chunk
  of rows still says what each value is. ``layout: table`` keeps a sheet as one table element
  with its cells instead. The first non-empty row is the header row. Formulas show their last
  computed value; dates show as Excel stores them.
- ``parse/pptx``: a **section per slide**: a level-1 heading "Slide n: title", then the slide's
  text frames as paragraphs and list items, its tables, and optionally its notes. Each element's
  page is its slide number.
- ``parse/html``: HTML structure: the title, headings by level, paragraphs, list items, tables
  (header cells from ``th`` or ``thead``), code. Scripts, styles and embedded objects are
  skipped and nothing they point at is fetched; ``nav``, a page-level ``header`` and ``footer``
  are typed ``page_header`` and ``page_footer``, so ``clean/boilerplate`` can drop them.
"""

import csv
import io
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal, cast
from xml.etree.ElementTree import Element as XmlElement

from defusedxml import ElementTree
from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.detect import PPTX, XLSX
from stratafi_components.parse import ParseComponent
from stratafi_components.parsers.common import Builder
from stratafi_core.artifacts import ParsedDocument, RawDocument
from stratafi_core.components import ComponentError, ComponentInfo

MAX_PART_BYTES = 64 * 1024 * 1024
NS = {
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}
_CELL_REF = re.compile(r"([A-Z]{1,3})(\d+)")


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _part(archive: zipfile.ZipFile, name: str) -> XmlElement:
    info = archive.getinfo(name)
    if info.file_size > MAX_PART_BYTES:
        raise ComponentError("limit_exceeded")
    return ElementTree.fromstring(archive.read(name))


def _zip(path: Path) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise ComponentError("malformed") from None


def _relationships(archive: zipfile.ZipFile, part: str) -> dict[str, str]:
    """A part's relationships: ID → target part name."""
    folder = PurePosixPath(part).parent
    rels = folder / "_rels" / f"{PurePosixPath(part).name}.rels"
    if str(rels) not in archive.namelist():
        return {}
    targets = {}
    for rel in _part(archive, str(rels)).findall("rel:Relationship", NS):
        target = rel.get("Target", "")
        resolved = PurePosixPath(target.lstrip("/")) if target.startswith("/") else folder / target
        parts: list[str] = []
        for piece in resolved.parts:
            if piece == "..":
                if parts:
                    parts.pop()
            elif piece != ".":
                parts.append(piece)
        targets[rel.get("Id", "")] = "/".join(parts)
    return targets


# --- parse/spreadsheet -----------------------------------------------------------------------


class SpreadsheetParams(_Params):
    layout: Literal["rows", "table"] = Field(
        default="rows",
        description="rows: one element per row with its headers; table: one table per sheet",
    )
    max_rows: int = Field(default=5000, ge=1, le=1_000_000, description="Rows per sheet")


def _column(letters: str) -> int:
    number = 0
    for letter in letters:
        number = number * 26 + (ord(letter) - 64)
    return number - 1


def xlsx_sheets(path: Path, max_rows: int) -> list[tuple[str, list[list[str]], bool]]:
    """Each sheet's name, rows of cell texts, and whether it was cut at ``max_rows``."""
    with _zip(path) as archive:
        names = set(archive.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            for item in _part(archive, "xl/sharedStrings.xml").findall("s:si", NS):
                shared.append("".join(text.text or "" for text in item.iter(f"{{{NS['s']}}}t")))
        workbook = _part(archive, "xl/workbook.xml")
        targets = _relationships(archive, "xl/workbook.xml")
        sheets = []
        for sheet in workbook.findall("s:sheets/s:sheet", NS):
            target = targets.get(sheet.get(f"{{{NS['r']}}}id", ""), "")
            if target not in names:
                continue
            grid: dict[int, dict[int, str]] = {}
            cut = False
            for row in _part(archive, target).iter(f"{{{NS['s']}}}row"):
                for cell in row.findall("s:c", NS):
                    match = _CELL_REF.fullmatch(cell.get("r", ""))
                    if match is None:
                        continue
                    r, c = int(match.group(2)) - 1, _column(match.group(1))
                    if r >= max_rows:
                        cut = True
                        continue
                    kind = cell.get("t")
                    if kind == "inlineStr":
                        text = "".join(t.text or "" for t in cell.iter(f"{{{NS['s']}}}t"))
                    else:
                        value = cell.find("s:v", NS)
                        text = value.text or "" if value is not None else ""
                        if kind == "s" and text.isdigit() and int(text) < len(shared):
                            text = shared[int(text)]
                        elif kind == "b":
                            text = "TRUE" if text == "1" else "FALSE"
                    if text.strip():
                        grid.setdefault(r, {})[c] = text.strip()
            width = max((max(cols) + 1 for cols in grid.values()), default=0)
            rows = [[grid[r].get(c, "") for c in range(width)] for r in sorted(grid)]
            sheets.append((sheet.get("name", "Sheet"), rows, cut))
        return sheets


def delimited_rows(path: Path, delimiter: str, max_rows: int) -> tuple[list[list[str]], bool]:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    rows: list[list[str]] = []
    for row in csv.reader(io.StringIO(text), delimiter=delimiter):
        if len(rows) >= max_rows:
            return rows, True
        if any(cell.strip() for cell in row):
            rows.append([cell.strip() for cell in row])
    return rows, False


def add_sheet(builder: Builder, name: str | None, rows: list[list[str]], layout: str) -> None:
    if name:
        builder.add("heading", name, level=1)
    if not rows:
        return
    if layout == "table":
        builder.add_table(rows)
        return
    header, body = rows[0], rows[1:]
    if not body:
        builder.add("paragraph", "; ".join(cell for cell in header if cell))
        return
    for row in body:
        pairs = [
            f"{header[i] if i < len(header) and header[i] else f'Column {i + 1}'}: {value}"
            for i, value in enumerate(row)
            if value
        ]
        builder.add("list_item", "; ".join(pairs))


class SpreadsheetParser(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/spreadsheet",
        stage="parse",
        version="1.0.0",
        title="Spreadsheets (rows with headers)",
        description="XLSX, CSV and TSV: a heading per sheet, then each row with its headers "
        "(Header: value; …), or one table per sheet.",
        accepts=(XLSX, "text/csv", "text/tab-separated-values"),
        license="MIT (defusedxml: PSF-2.0)",
    )
    Params: ClassVar[type[BaseModel]] = SpreadsheetParams

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        params = cast(SpreadsheetParams, params)
        builder = Builder(raw, self, params)
        if raw.media_type == XLSX:
            try:
                sheets = xlsx_sheets(path, params.max_rows)
            except (KeyError, ElementTree.ParseError, ValueError):
                raise ComponentError("malformed") from None
        else:
            delimiter = "\t" if raw.media_type == "text/tab-separated-values" else ","
            rows, cut = delimited_rows(path, delimiter, params.max_rows)
            sheets = [("", rows, cut)]
        for name, rows, cut in sheets:
            add_sheet(builder, name or None, rows, params.layout)
            if cut:
                builder.warn("rows_truncated")
        return builder.document(title=None)


# --- parse/pptx ------------------------------------------------------------------------------


class PptxParams(_Params):
    notes: bool = Field(default=False, description="Add each slide's speaker notes")


def _paragraphs(body: XmlElement) -> list[tuple[str, bool]]:
    """A text body's paragraphs: (text, is a bullet)."""
    found = []
    for paragraph in body.findall("a:p", NS):
        text = "".join(run.text or "" for run in paragraph.iter(f"{{{NS['a']}}}t")).strip()
        if not text:
            continue
        props = paragraph.find("a:pPr", NS)
        bullet = props is not None and (
            props.get("lvl") is not None
            or props.find("a:buChar", NS) is not None
            or props.find("a:buAutoNum", NS) is not None
        )
        found.append((text, bullet))
    return found


class PptxParser(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/pptx",
        stage="parse",
        version="1.0.0",
        title="PowerPoint (a section per slide)",
        description="A section per slide: a heading with the slide's number and title, then its "
        "text, bullets and tables, and optionally its notes; each slide is a page.",
        accepts=(PPTX,),
        license="MIT (defusedxml: PSF-2.0)",
    )
    Params: ClassVar[type[BaseModel]] = PptxParams

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        params = cast(PptxParams, params)
        builder = Builder(raw, self, params)
        try:
            with _zip(path) as archive:
                presentation = _part(archive, "ppt/presentation.xml")
                targets = _relationships(archive, "ppt/presentation.xml")
                slides = [
                    targets.get(slide.get(f"{{{NS['r']}}}id", ""), "")
                    for slide in presentation.findall("p:sldIdLst/p:sldId", NS)
                ]
                pages = []
                for number, slide_part in enumerate(slides, start=1):
                    if slide_part not in archive.namelist():
                        continue
                    pages.append({"number": number})
                    self._slide(archive, slide_part, number, builder, params)
        except (KeyError, ElementTree.ParseError, ValueError):
            raise ComponentError("malformed") from None
        return builder.document(title=None, pages=pages)

    def _slide(
        self,
        archive: zipfile.ZipFile,
        part: str,
        number: int,
        builder: Builder,
        params: PptxParams,
    ) -> None:
        slide = _part(archive, part)
        title = ""
        blocks: list[tuple[str, Any]] = []
        for shape in slide.iter(f"{{{NS['p']}}}sp"):
            placeholder = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
            body = shape.find("p:txBody", NS)
            if body is None:
                continue
            paragraphs = _paragraphs(body)
            if placeholder is not None and placeholder.get("type") in ("title", "ctrTitle"):
                title = " ".join(text for text, _ in paragraphs)
                continue
            blocks += [("text", paragraph) for paragraph in paragraphs]
        for table in slide.iter(f"{{{NS['a']}}}tbl"):
            rows = [
                [
                    "".join(t.text or "" for t in cell.iter(f"{{{NS['a']}}}t")).strip()
                    for cell in row.findall("a:tc", NS)
                ]
                for row in table.findall("a:tr", NS)
            ]
            blocks.append(("table", rows))
        heading = f"Slide {number}: {title}" if title else f"Slide {number}"
        builder.add("heading", heading, level=1, page=number)
        for kind, block in blocks:
            if kind == "table":
                builder.add_table(block, page=number)
            else:
                text, bullet = block
                builder.add("list_item" if bullet else "paragraph", text, page=number)
        if params.notes:
            for target in _relationships(archive, part).values():
                if "notesSlide" in target and target in archive.namelist():
                    notes = _part(archive, target)
                    for body in notes.iter(f"{{{NS['p']}}}txBody"):
                        for text, _ in _paragraphs(body):
                            if not text.isdigit():
                                builder.add("paragraph", text, page=number)


# --- parse/html ------------------------------------------------------------------------------

SKIPPED_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "iframe", "object", "embed", "canvas"}
)
BLOCK_TAGS = {
    "p": "paragraph",
    "li": "list_item",
    "pre": "code",
    "blockquote": "paragraph",
    "caption": "caption",
    "figcaption": "caption",
    "dt": "paragraph",
    "dd": "paragraph",
    "div": "paragraph",
    "address": "paragraph",
    "section": "paragraph",
    "article": "paragraph",
    "main": "paragraph",
}
HEADINGS = {f"h{level}": level for level in range(1, 7)}
FURNITURE = {"nav": "page_header", "header": "page_header", "footer": "page_footer"}


class HtmlStructure(HTMLParser):
    """Reads HTML into blocks, in order: (type, text, level) and tables as rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, Any, int | None]] = []
        self.title = ""
        self._skip = 0
        self._stack: list[str] = []
        self._text: list[str] = []
        self._in_title = False
        self._furniture: list[str] = []
        self._table: list[list[tuple[str, bool]]] | None = None
        self._head_rows = 0
        self._cell: list[str] | None = None
        self._cell_header = False
        self._in_thead = False

    def _flush(self, kind: str = "paragraph", level: int | None = None) -> None:
        text = " ".join("".join(self._text).split())
        if self._stack and self._stack[-1] == "pre":
            text = "".join(self._text).strip("\n")
        self._text = []
        if text:
            if self._furniture and kind not in ("heading",):
                kind = FURNITURE[self._furniture[-1]]
            self.blocks.append((kind, text, level))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIPPED_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in FURNITURE and not ({"article", "section", "main"} & set(self._stack)):
            self._flush()
            self._furniture.append(tag)
        if tag == "table":
            self._flush()
            self._table, self._head_rows = [], 0
        elif tag == "thead":
            self._in_thead = True
        elif tag == "tr" and self._table is not None:
            self._table.append([])
        elif tag in ("td", "th") and self._table is not None:
            self._cell, self._cell_header = [], tag == "th"
        elif tag == "br":
            self._text.append("\n" if self._stack and self._stack[-1] == "pre" else " ")
        elif tag in HEADINGS or tag in BLOCK_TAGS:
            if self._cell is None:
                self._flush()
        self._stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in ("td", "th") and self._cell is not None and self._table:
            self._table[-1].append((" ".join("".join(self._cell).split()), self._cell_header))
            self._cell = None
        elif tag == "thead":
            self._in_thead = False
        elif tag == "tr" and self._table is not None and self._in_thead:
            self._head_rows += 1
        elif tag == "table" and self._table is not None:
            rows = [row for row in self._table if row]
            header_rows = self._head_rows or sum(
                1 for row in rows[:1] if row and all(header for _, header in row)
            )
            self.blocks.append(("table", [[text for text, _ in row] for row in rows], header_rows))
            self._table = None
        elif self._cell is None and tag in HEADINGS:
            self._flush("heading", HEADINGS[tag])
        elif self._cell is None and tag in BLOCK_TAGS:
            self._flush(BLOCK_TAGS[tag])
        if tag in FURNITURE and self._furniture and self._furniture[-1] == tag:
            self._flush()
            self._furniture.pop()
        while self._stack:
            if self._stack.pop() == tag:
                break

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title += data
        elif self._cell is not None:
            self._cell.append(data)
        else:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def html_elements(builder: Builder, markup: str) -> str | None:
    """Read HTML into the builder's elements; returns the page's title."""
    reader = HtmlStructure()
    reader.feed(markup)
    reader.close()
    title = " ".join(reader.title.split()) or None
    for kind, content, extra in reader.blocks:
        if kind == "table":
            builder.add_table(content, header_rows=int(extra or 0))
        elif kind == "heading":
            builder.add("heading", content, level=extra)
        else:
            builder.add(kind, content)
    return title


class HtmlParser(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/html",
        stage="parse",
        version="1.0.0",
        title="HTML structure",
        description="The title, headings by level, paragraphs, list items, tables and code; "
        "scripts and styles skipped, nothing fetched; navigation, header and footer typed as "
        "page furniture for clean/boilerplate.",
        accepts=("text/html",),
        license="MIT",
    )

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        data = path.read_bytes()
        try:
            markup = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            markup = data.decode("latin-1")
        builder = Builder(raw, self, params)
        title = html_elements(builder, markup)
        return builder.document(title=title)


COMPONENTS = [SpreadsheetParser, PptxParser, HtmlParser]
