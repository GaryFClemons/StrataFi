#!/usr/bin/env python3
"""Render a Markdown corpus as other file types, keeping its documents' keys.

    uv run python scripts/corpus_variants.py fixtures/demo-corpus/corpus data/variants/halcyon
    uv run python scripts/corpus_variants.py SOURCE OUT --variants docx,pdf --seed 7
    uv run python scripts/corpus_variants.py SOURCE OUT --evidence retrieval.yaml
    uv run python scripts/corpus_variants.py SOURCE OUT --import-as halcyon

SOURCE holds a ``manifest.yaml`` in the corpus manifest format (``fixtures/demo-corpus``'s) that
lists Markdown documents. Each variant goes to ``OUT/<variant>/``: every document rendered at
its path with the variant's extension, a ``manifest.yaml`` whose entries keep their doc_key,
version, dates, ACL and the rest, and ``variant.json``. Each variant then imports as its own
corpus whose documents share their keys with the source:

- ``docx``: Word, with heading styles, numbered and bulleted lists and tables, written directly
  (a DOCX file is zipped XML)
- ``pdf``: a digital PDF with a text layer (reportlab); headings larger and bold
- ``scanned-pdf``: the digital PDF's pages rasterized (pypdfium2) at ``--dpi``, each skewed,
  speckled and blurred a little from the seed, saved as image-only pages (Pillow)
- ``html``: an HTML page per document
- ``pdf-table-images``: the digital PDF with each table replaced by a picture of itself
  (rasterized at ``--dpi``), so its text is gone

The same source and seed give the same bytes: timestamps and IDs are fixed, and a scanned
page's noise comes from the seed and the document's path. The libraries are permissive and have
wheels for Linux and Windows. Only the block structure the corpus uses is rendered:
headings, paragraphs, lists and pipe tables; inline Markdown stays as written.

``--evidence`` checks, with a reference parser per type (``parse/markitdown`` for DOCX,
``parse/pypdf`` for PDF, the standard library's HTML parser for HTML), which of an eval set's
evidence phrases each variant still holds. ``--import-as NAME`` imports each variant through the
API with the ``stratafi`` SDK (``STRATAFI_URL`` and a token, as ``Client.from_env`` reads
them): the corpus ``NAME-<variant>``, uploaded with its manifest, then frozen.
"""

import argparse
import copy
import hashlib
import html
import io
import json
import random
import re
import shutil
import sys
import time
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath
from typing import Any
from xml.sax.saxutils import escape

import yaml

VARIANTS = ("docx", "pdf", "scanned-pdf", "html", "pdf-table-images")
DIGITAL = ("docx", "pdf", "html")  # every word of the source stays in their text
EXTENSIONS = {
    "docx": ".docx",
    "pdf": ".pdf",
    "scanned-pdf": ".pdf",
    "html": ".html",
    "pdf-table-images": ".pdf",
}
LIBRARIES = {
    "docx": (),
    "pdf": ("reportlab",),
    "scanned-pdf": ("reportlab", "pypdfium2", "pillow"),
    "html": (),
    "pdf-table-images": ("reportlab", "pypdfium2", "pillow"),
}
AUTHOR = "Stratafi corpus variants"
FIXED_ISO = "2026-01-01T00:00:00Z"
FIXED_TIME = time.gmtime(1767225600)  # 2026-01-01T00:00:00Z
ZIP_TIME = (2026, 1, 1, 0, 0, 0)
MARKER = "variant.json"

# The scanned look: up to this skew in degrees, this share of pixels speckled, this blur.
MAX_SKEW_DEGREES = 1.5
SPECKLES = 0.002
BLUR_RADIUS = 0.6
SCAN_QUALITY = 60  # JPEG

# Page geometry, in points: US Letter with one-inch margins.
PAGE_WIDTH, PAGE_HEIGHT, MARGIN = 612.0, 792.0, 72.0
TEXT_WIDTH = PAGE_WIDTH - 2 * MARGIN
FRAME_HEIGHT = PAGE_HEIGHT - 2 * MARGIN - 12  # less the frame's own padding
TABLE_ROWS_PER_IMAGE = 20


# --- the Markdown ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Heading:
    level: int
    text: str


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Items:
    ordered: bool
    items: tuple[str, ...]
    start: int = 1


@dataclass(frozen=True)
class Table:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @property
    def width(self) -> int:
        return max([len(self.header), *(len(row) for row in self.rows)])

    def padded(self) -> list[tuple[str, ...]]:
        """The header and the rows, each as wide as the widest."""
        return [(*row, *[""] * (self.width - len(row))) for row in (self.header, *self.rows)]


Block = Heading | Text | Items | Table


@dataclass(frozen=True)
class Document:
    title: str | None
    blocks: tuple[Block, ...]


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+(.*)$")
_NUMBER_RE = re.compile(r"^[ \t]*(\d{1,9})[.)][ \t]+(.*)$")
_RULE_RE = re.compile(r"^\|?[ \t]*:?-{3,}:?[ \t]*(\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$")


def _cells(line: str) -> tuple[str, ...]:
    body = line.strip()
    body = body.removeprefix("|")
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    return tuple(cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", body))


def parse_markdown(text: str) -> Document:
    """The blocks a corpus document uses: ATX headings, paragraphs, bulleted and numbered
    lists, and pipe tables. The first heading is the title when it's level 1."""
    lines = text.replace("\r\n", "\n").split("\n")
    blocks: list[Block] = []
    paragraph: list[str] = []
    items: list[str] = []
    ordered, start = False, 1

    def flush() -> None:
        if paragraph:
            blocks.append(Text(" ".join(part.strip() for part in paragraph)))
            paragraph.clear()
        if items:
            blocks.append(Items(ordered, tuple(items), start))
            items.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        heading = _HEADING_RE.match(line)
        bullet, number = _BULLET_RE.match(line), _NUMBER_RE.match(line)
        if not line.strip():
            flush()
        elif heading:
            flush()
            blocks.append(Heading(len(heading.group(1)), heading.group(2).strip()))
        elif (
            line.lstrip().startswith("|")
            and index + 1 < len(lines)
            and _RULE_RE.match(lines[index + 1].strip())
        ):
            flush()
            header, rows = _cells(line), []
            index += 2
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                rows.append(_cells(lines[index]))
                index += 1
            blocks.append(Table(header, tuple(rows)))
            continue
        elif bullet or number:
            if paragraph or (items and (number is not None) != ordered):
                flush()
            if not items:
                ordered, start = number is not None, int(number.group(1)) if number else 1
            items.append((number.group(2) if number else bullet.group(1)).strip())  # type: ignore[union-attr]
        elif items and line[:1] in (" ", "\t"):
            items[-1] = f"{items[-1]} {line.strip()}"  # a list item goes on
        else:
            if items:
                flush()
            paragraph.append(line)
        index += 1
    flush()
    title = None
    first = next((block for block in blocks if isinstance(block, Heading)), None)
    if first is not None and first.level == 1:
        title = first.text
        blocks.remove(first)
    return Document(title, tuple(blocks))


def _rng(seed: int, key: str) -> random.Random:
    """A document's own random numbers: from the seed and its path, not the order of work."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))  # noqa: S311 - seeded noise, not a secret


# --- HTML -----------------------------------------------------------------------------------


def render_html(document: Document) -> bytes:
    def text(value: str) -> str:
        return html.escape(value, quote=False)

    title = document.title or ""
    out = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{text(title)}</title>",
        "</head>",
        "<body>",
    ]
    if document.title:
        out.append(f"<h1>{text(document.title)}</h1>")
    for block in document.blocks:
        if isinstance(block, Heading):
            out.append(f"<h{block.level}>{text(block.text)}</h{block.level}>")
        elif isinstance(block, Text):
            out.append(f"<p>{text(block.text)}</p>")
        elif isinstance(block, Items):
            tag = "ol" if block.ordered else "ul"
            start = f' start="{block.start}"' if block.ordered and block.start != 1 else ""
            out.append(f"<{tag}{start}>")
            out += [f"<li>{text(item)}</li>" for item in block.items]
            out.append(f"</{tag}>")
        else:
            header, *rows = block.padded()
            out.append("<table>")
            out.append(
                "<thead><tr>" + "".join(f"<th>{text(c)}</th>" for c in header) + "</tr></thead>"
            )
            out.append("<tbody>")
            out += ["<tr>" + "".join(f"<td>{text(c)}</td>" for c in row) + "</tr>" for row in rows]
            out.append("</tbody>")
            out.append("</table>")
    out += ["</body>", "</html>", ""]
    return "\n".join(out).encode("utf-8")


# --- DOCX -----------------------------------------------------------------------------------

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

_CONTENT_TYPES = (
    _XML + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" '
    'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/'
    'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/'
    'vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    '<Override PartName="/word/numbering.xml" ContentType="application/'
    'vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
    '<Override PartName="/docProps/core.xml" '
    'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
    '<Override PartName="/docProps/app.xml" ContentType="application/'
    'vnd.openxmlformats-officedocument.extended-properties+xml"/>'
    "</Types>"
)
_PACKAGE_RELS = (
    _XML + f'<Relationships xmlns="{_REL_NS}">'
    f'<Relationship Id="rId1" Type="{_OFFICE_REL}/officeDocument" Target="word/document.xml"/>'
    '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/'
    'relationships/metadata/core-properties" Target="docProps/core.xml"/>'
    f'<Relationship Id="rId3" Type="{_OFFICE_REL}/extended-properties" '
    'Target="docProps/app.xml"/>'
    "</Relationships>"
)
_DOCUMENT_RELS = (
    _XML + f'<Relationships xmlns="{_REL_NS}">'
    f'<Relationship Id="rId1" Type="{_OFFICE_REL}/styles" Target="styles.xml"/>'
    f'<Relationship Id="rId2" Type="{_OFFICE_REL}/numbering" Target="numbering.xml"/>'
    "</Relationships>"
)
_APP = (
    _XML + '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/'
    f'extended-properties"><Application>{AUTHOR}</Application></Properties>'
)
_BORDER = 'w:val="single" w:sz="4" w:space="0" w:color="auto"'


def _heading_style(level: int) -> str:
    size = {1: 32, 2: 28, 3: 24}.get(level, 22)
    return (
        f'<w:style w:type="paragraph" w:styleId="Heading{level}">'
        f'<w:name w:val="heading {level}"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>'
        f'<w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/>'
        f'<w:outlineLvl w:val="{level - 1}"/></w:pPr>'
        f'<w:rPr><w:b/><w:sz w:val="{size}"/></w:rPr></w:style>'
    )


_STYLES = (
    _XML + f'<w:styles xmlns:w="{_W_NS}">'
    '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" '
    'w:cs="Calibri"/><w:sz w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>'
    '<w:spacing w:after="120"/></w:pPr></w:pPrDefault></w:docDefaults>'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>'
    "</w:style>"
    '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
    '<w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:spacing w:after="240"/>'
    '</w:pPr><w:rPr><w:b/><w:sz w:val="40"/></w:rPr></w:style>'
    + "".join(_heading_style(level) for level in range(1, 7))
    + '<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/>'
    '<w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720"/></w:pPr></w:style>'
    '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr>'
    f"<w:tblBorders><w:top {_BORDER}/><w:left {_BORDER}/><w:bottom {_BORDER}/>"
    f"<w:right {_BORDER}/><w:insideH {_BORDER}/><w:insideV {_BORDER}/></w:tblBorders>"
    "</w:tblPr></w:style>"
    "</w:styles>"
)


def _level(numbering: str) -> str:
    text = "•" if numbering == "bullet" else "%1."
    return (
        f'<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="{numbering}"/>'
        f'<w:lvlText w:val="{text}"/><w:lvlJc w:val="left"/>'
        '<w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl>'
    )


def _numbering(ordered_lists: Sequence[int]) -> str:
    """Bullets are list 1; each numbered list is its own list, so each starts again."""
    nums = ['<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>']
    for number, start in enumerate(ordered_lists, start=2):
        nums.append(
            f'<w:num w:numId="{number}"><w:abstractNumId w:val="1"/>'
            f'<w:lvlOverride w:ilvl="0"><w:startOverride w:val="{start}"/></w:lvlOverride>'
            "</w:num>"
        )
    return (
        _XML
        + f'<w:numbering xmlns:w="{_W_NS}">'
        + f'<w:abstractNum w:abstractNumId="0">{_level("bullet")}</w:abstractNum>'
        + f'<w:abstractNum w:abstractNumId="1">{_level("decimal")}</w:abstractNum>'
        + "".join(nums)
        + "</w:numbering>"
    )


def _run(text: str, *, bold: bool = False) -> str:
    properties = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f'<w:r>{properties}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _paragraph(text: str, *, style: str | None = None, numbering: int | None = None) -> str:
    properties = ""
    if style or numbering:
        inner = f'<w:pStyle w:val="{style}"/>' if style else ""
        if numbering:
            inner += f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{numbering}"/></w:numPr>'
        properties = f"<w:pPr>{inner}</w:pPr>"
    return f"<w:p>{properties}{_run(text)}</w:p>"


def _column_shares(rows: Sequence[Sequence[str]]) -> list[float]:
    """Each column's share of a table's width, by its longest cell, none under a tenth."""
    longest = [max(len(row[column]) for row in rows) or 1 for column in range(len(rows[0]))]
    raw = [max(length / sum(longest), 0.1) for length in longest]
    return [share / sum(raw) for share in raw]


def _docx_table(table: Table) -> str:
    rows = table.padded()
    widths = [round(9360 * share) for share in _column_shares(rows)]  # 6.5 inches, in twips
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in widths)
    out = [
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
        f"</w:tblPr><w:tblGrid>{grid}</w:tblGrid>"
    ]
    for index, row in enumerate(rows):
        header = index == 0
        out.append("<w:tr><w:trPr><w:tblHeader/></w:trPr>" if header else "<w:tr>")
        for cell, width in zip(row, widths, strict=True):
            out.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>'
                f"<w:p>{_run(cell, bold=header)}</w:p></w:tc>"
            )
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def render_docx(document: Document, *, compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    """A DOCX of the document. ``ZIP_STORED`` writes the same bytes wherever zlib differs."""
    body: list[str] = []
    ordered_lists: list[int] = []
    if document.title:
        body.append(_paragraph(document.title, style="Title"))
    for block in document.blocks:
        if isinstance(block, Heading):
            body.append(_paragraph(block.text, style=f"Heading{block.level}"))
        elif isinstance(block, Text):
            body.append(_paragraph(block.text))
        elif isinstance(block, Items):
            numbering = 1
            if block.ordered:
                ordered_lists.append(block.start)
                numbering = len(ordered_lists) + 1
            body += [
                _paragraph(item, style="ListParagraph", numbering=numbering) for item in block.items
            ]
        else:
            body.append(_docx_table(block))
            body.append("<w:p/>")  # Word wants a paragraph between a table and what follows
    body.append(
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" '
        'w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
    )
    main = _XML + f'<w:document xmlns:w="{_W_NS}"><w:body>{"".join(body)}</w:body></w:document>'
    core = (
        _XML + '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/'
        'metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{escape(document.title or '')}</dc:title><dc:creator>{AUTHOR}</dc:creator>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{FIXED_ISO}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{FIXED_ISO}</dcterms:modified>'
        "</cp:coreProperties>"
    )
    parts = [
        ("[Content_Types].xml", _CONTENT_TYPES),
        ("_rels/.rels", _PACKAGE_RELS),
        ("docProps/core.xml", core),
        ("docProps/app.xml", _APP),
        ("word/_rels/document.xml.rels", _DOCUMENT_RELS),
        ("word/document.xml", main),
        ("word/styles.xml", _STYLES),
        ("word/numbering.xml", _numbering(ordered_lists)),
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in parts:
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.compress_type = compression
            info.external_attr = 0o644 << 16
            info.create_system = 0  # its default is the OS's, and it's part of the bytes
            archive.writestr(info, content.encode("utf-8"))
    return buffer.getvalue()


# --- PDF ------------------------------------------------------------------------------------


def _pdf_styles() -> dict[Any, Any]:
    from reportlab.lib.styles import ParagraphStyle

    body = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=13.5, spaceAfter=6)
    styles: dict[Any, Any] = {
        "body": body,
        "title": ParagraphStyle(
            "title", fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=12
        ),
        "cell": ParagraphStyle("cell", parent=body, spaceAfter=0),
        "head": ParagraphStyle("head", parent=body, fontName="Helvetica-Bold", spaceAfter=0),
    }
    for level, size in ((1, 17.0), (2, 14.0), (3, 12.0)):
        styles[level] = ParagraphStyle(
            f"h{level}",
            fontName="Helvetica-Bold",
            fontSize=size,
            leading=size * 1.25,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=1,
        )
    styles[4] = ParagraphStyle(
        "h4", parent=styles[3], fontSize=11, leading=14
    )  # and below: bold at nearly body size
    return styles


def _pdf_table(rows: Sequence[Sequence[str]], widths: Sequence[float], styles: dict) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, TableStyle
    from reportlab.platypus import Table as PdfTable

    data = [
        [Paragraph(escape(cell), styles["head" if index == 0 else "cell"]) for cell in row]
        for index, row in enumerate(rows)
    ]
    table = PdfTable(data, colWidths=list(widths), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e1e1e1")),
            ]
        )
    )
    return table


def _table_images(table: Table, widths: Sequence[float], styles: dict, dpi: int) -> list[Any]:
    """A table as pictures of itself: the digital table, a header and up to
    ``TABLE_ROWS_PER_IMAGE`` rows at a time, drawn on its own page, rasterized at ``dpi``
    (pypdfium2) and placed at its own size, or smaller when it's taller than a page."""
    import pypdfium2
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Image

    header, *rows = table.padded()
    images = []
    for start in range(0, max(len(rows), 1), TABLE_ROWS_PER_IMAGE):
        flowable = _pdf_table([header, *rows[start : start + TABLE_ROWS_PER_IMAGE]], widths, styles)
        width, height = flowable.wrap(TEXT_WIDTH, PAGE_HEIGHT)
        size = (width + 2, height + 2)
        page = io.BytesIO()
        canvas = Canvas(page, pagesize=size, invariant=1)
        flowable.drawOn(canvas, 1, 1)
        canvas.showPage()
        canvas.save()
        rendered = pypdfium2.PdfDocument(page.getvalue())
        try:
            picture = rendered[0].render(scale=dpi / 72, grayscale=True).to_pil().convert("L")
        finally:
            rendered.close()
        encoded = io.BytesIO()
        picture.save(encoded, "PNG")
        encoded.seek(0)
        fit = min(1.0, FRAME_HEIGHT / size[1])
        images.append(Image(encoded, width=size[0] * fit, height=size[1] * fit, hAlign="LEFT"))
    return images


def render_pdf(
    document: Document, *, table_images: bool = False, dpi: int = 150, compress: bool = True
) -> bytes:
    """A digital PDF; with ``table_images``, each table is an image instead of text. Without
    ``compress``, page contents aren't deflated, so the bytes are the same wherever zlib
    differs."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

    styles = _pdf_styles()
    story: list[Any] = []
    if document.title:
        story.append(Paragraph(escape(document.title), styles["title"]))
    for block in document.blocks:
        if isinstance(block, Heading):
            story.append(Paragraph(escape(block.text), styles[min(block.level, 4)]))
        elif isinstance(block, Text):
            story.append(Paragraph(escape(block.text), styles["body"]))
        elif isinstance(block, Items):
            entries = [ListItem(Paragraph(escape(item), styles["body"])) for item in block.items]
            options: dict[str, Any] = (
                {"bulletType": "1", "start": str(block.start)}
                if block.ordered
                else {"bulletType": "bullet", "start": "•"}
            )
            story.append(ListFlowable(entries, leftIndent=18, bulletFontSize=10, **options))
        else:
            rows = block.padded()
            widths = [TEXT_WIDTH * share for share in _column_shares(rows)]
            if table_images:
                story += _table_images(block, widths, styles, dpi)
            else:
                story.append(_pdf_table(rows, widths, styles))
            story.append(Spacer(1, 8))
    buffer = io.BytesIO()
    template = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=document.title or "",
        author=AUTHOR,
        creator=AUTHOR,
        invariant=1,
        pageCompression=1 if compress else 0,
    )
    template.build(story)
    return buffer.getvalue()


def render_scanned(pdf: bytes, *, rng: random.Random, dpi: int, title: str | None) -> bytes:
    """``pdf``'s pages as a scanner might return them: rasterized, skewed, speckled and
    softened, in an image-only PDF."""
    import pypdfium2
    from PIL import Image, ImageDraw, ImageFilter

    pages = []
    source = pypdfium2.PdfDocument(pdf)
    try:
        for index in range(len(source)):
            page = source[index]
            image = page.render(scale=dpi / 72, grayscale=True).to_pil().convert("L")
            page.close()
            image = image.rotate(
                rng.uniform(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES),
                resample=Image.Resampling.BICUBIC,
                fillcolor=255,
            )
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for _ in range(int(width * height * SPECKLES)):
                draw.point((rng.randrange(width), rng.randrange(height)), fill=rng.randrange(160))
            pages.append(image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)))
    finally:
        source.close()
    buffer = io.BytesIO()
    pages[0].save(
        buffer,
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=float(dpi),
        quality=SCAN_QUALITY,
        title=title or "",
        author=AUTHOR,
        creator=AUTHOR,
        producer=AUTHOR,
        creationDate=FIXED_TIME,
        modDate=FIXED_TIME,
    )
    return buffer.getvalue()


# --- a corpus -------------------------------------------------------------------------------


def load_manifest(folder: Path) -> dict[str, Any]:
    manifest = yaml.safe_load((folder / "manifest.yaml").read_text(encoding="utf-8"))
    documents = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(documents, list) or not documents:
        raise ValueError("the manifest lists no documents")
    return manifest


def variant_path(path: str, variant: str) -> str:
    return str(PurePosixPath(path).with_suffix(EXTENSIONS[variant]))


def variant_manifest(manifest: dict[str, Any], variant: str) -> dict[str, Any]:
    """The source's manifest with each document at its variant's path; everything else, from
    doc_key and version to ACLs, as it was."""
    copied = copy.deepcopy(manifest)
    for entry in copied["documents"]:
        entry["path"] = variant_path(entry["path"], variant)
    return copied


def _versions(libraries: Iterable[str]) -> dict[str, str]:
    return {name: package_version(name) for name in libraries}


def render_corpus(
    source: Path,
    out: Path,
    *,
    variants: Sequence[str] = VARIANTS,
    seed: int = 0,
    dpi: int = 150,
) -> dict[str, Path]:
    """Render every document in ``source``'s manifest as each variant, under ``out``; each
    variant's folder."""
    manifest = load_manifest(source)
    source_hash = hashlib.sha256((source / "manifest.yaml").read_bytes()).hexdigest()
    documents: dict[str, Document] = {}
    for entry in manifest["documents"]:
        path = str(entry["path"])
        if PurePosixPath(path).suffix.lower() not in (".md", ".markdown"):
            raise ValueError("every source document must be Markdown")
        documents[path] = parse_markdown((source / path).read_text(encoding="utf-8"))
    digital: dict[str, bytes] = {}
    folders: dict[str, Path] = {}
    for variant in variants:
        folder = out / variant
        if folder.exists():
            if not (folder / MARKER).is_file():
                raise ValueError(f"{folder} exists and this script didn't write it")
            shutil.rmtree(folder)
        for path, document in documents.items():
            if variant == "html":
                data = render_html(document)
            elif variant == "docx":
                data = render_docx(document)
            elif variant == "pdf-table-images":
                data = render_pdf(document, table_images=True, dpi=dpi)
            else:
                if path not in digital:
                    digital[path] = render_pdf(document)
                data = digital[path]
                if variant == "scanned-pdf":
                    data = render_scanned(data, rng=_rng(seed, path), dpi=dpi, title=document.title)
            target = folder / variant_path(path, variant)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        header = (
            f"# A Markdown corpus as {variant}, by scripts/corpus_variants.py (seed {seed}).\n"
            "# Generated: render it again rather than editing it.\n"
        )
        text = yaml.safe_dump(variant_manifest(manifest, variant), sort_keys=False, width=1000)
        (folder / "manifest.yaml").write_text(header + text, encoding="utf-8")
        record = {
            "variant": variant,
            "source_manifest": source_hash,
            "seed": seed,
            "dpi": dpi if variant in ("scanned-pdf", "pdf-table-images") else None,
            "documents": len(documents),
            "libraries": _versions(LIBRARIES[variant]),
        }
        (folder / MARKER).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        folders[variant] = folder
    return folders


# --- the reference parsers ------------------------------------------------------------------


class _HtmlText(HTMLParser):
    """An HTML page's text, a block per line and table cells joined by ``|``."""

    BLOCKS = frozenset({"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "br", "table"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0
        self.in_row = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("head", "script", "style"):
            self.hidden += 1
        elif tag in ("td", "th"):
            self.parts.append(" | " if self.in_row else "")
            self.in_row = True
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("head", "script", "style"):
            self.hidden = max(self.hidden - 1, 0)
        elif tag == "tr":
            self.in_row = False
            self.parts.append("\n")
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


REFERENCE_PARSERS = {
    ".md": "parse/markdown",
    ".markdown": "parse/markdown",
    ".txt": "parse/markdown",
    ".docx": "parse/markitdown",
    ".pdf": "parse/pypdf",
}


def reference_text(path: Path) -> str:
    """A rendered document's text by a reference parser: the standard library's HTML parser
    for HTML, else the lab's own parser for its type (``REFERENCE_PARSERS``), in process."""
    data = path.read_bytes()
    if path.suffix.lower() in (".html", ".htm"):
        parser = _HtmlText()
        parser.feed(data.decode("utf-8"))
        parser.close()
        lines = (" ".join(line.split()) for line in "".join(parser.parts).split("\n"))
        return "\n".join(line for line in lines if line)
    from stratafi_components import default_registry
    from stratafi_components.detect import detect
    from stratafi_components.extract import raw_document

    component = default_registry().get(REFERENCE_PARSERS[path.suffix.lower()])()
    digest = hashlib.sha256(data).hexdigest()
    raw = raw_document("reference", digest, len(data), detect(path, path.name))
    document = component.parse_local(path, raw, component.resolve_params({}))  # type: ignore[attr-defined]
    return "\n".join([document.title or "", document.text])


def evidence_findability(folder: Path, eval_set: dict[str, Any]) -> list[tuple[str, int, bool]]:
    """For each evidence item of ``eval_set`` (the retrieval YAML format) whose document is in
    the variant at ``folder``: its query, its position, and whether its text survives."""
    from stratafi_core.evalsets import EvidenceRef
    from stratafi_core.relevance import findability

    manifest = load_manifest(folder)
    texts = {
        (str(entry["doc_key"]), str(entry["version"])): reference_text(folder / entry["path"])
        for entry in manifest["documents"]
    }
    found = []
    for item in eval_set.get("queries", []):
        for position, expected in enumerate(item.get("expected") or []):
            ref = EvidenceRef.model_validate(expected)
            text = texts.get((ref.doc_key, ref.version or ""))
            if text is not None:
                found.append((str(item["id"]), position, findability(ref, text).findable))
    return found


# --- importing ------------------------------------------------------------------------------


def import_variants(folders: dict[str, Path], name: str) -> list[str]:
    """Import each variant as the corpus ``<name>-<variant>``: created (or reused), uploaded
    with its manifest, frozen. A line per variant."""
    from stratafi import ApiError, Client

    lines = []
    with Client.from_env() as client:
        for variant, folder in folders.items():
            corpus = f"{name}-{variant}"
            try:
                client.corpora.create(corpus, description=f"{name}, rendered as {variant}")
            except ApiError as error:
                if error.code != "name_taken":
                    raise
            client.corpora.upload(corpus, manifest=folder / "manifest.yaml")
            frozen = client.corpora.freeze(corpus)
            lines.append(f"{corpus} version {frozen.version}: {frozen.document_count} documents")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a Markdown corpus as other file types.")
    parser.add_argument("source", type=Path, help="a folder with manifest.yaml")
    parser.add_argument("out", type=Path, help="each variant goes to OUT/<variant>/")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"comma-separated, from {', '.join(VARIANTS)} (default: all)",
    )
    parser.add_argument("--seed", type=int, default=0, help="the scanned pages' noise")
    parser.add_argument("--dpi", type=int, default=150, help="scanned pages and table images")
    parser.add_argument("--evidence", type=Path, help="an eval set: report evidence findability")
    parser.add_argument("--import-as", metavar="NAME", help="import each as corpus NAME-<variant>")
    args = parser.parse_args(argv)
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown or not variants:
        parser.error(f"unknown variants: {', '.join(unknown)}; choose from {', '.join(VARIANTS)}")
    if not 36 <= args.dpi <= 600:
        parser.error("--dpi is from 36 to 600")
    try:
        folders = render_corpus(
            args.source, args.out, variants=variants, seed=args.seed, dpi=args.dpi
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"corpus_variants: {error}", file=sys.stderr)
        return 2
    for variant, folder in folders.items():
        print(f"{variant}: {folder}")
    if args.evidence is not None:
        eval_set = yaml.safe_load(args.evidence.read_text(encoding="utf-8"))
        for variant, folder in folders.items():
            found = evidence_findability(folder, eval_set)
            print(f"{variant}: {sum(flag for *_, flag in found)} of {len(found)} evidence found")
    if args.import_as:
        from stratafi import StratafiError

        try:
            for line in import_variants(folders, args.import_as):
                print(line)
        except StratafiError as error:
            print(f"corpus_variants: {error}", file=sys.stderr)
            return error.exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
