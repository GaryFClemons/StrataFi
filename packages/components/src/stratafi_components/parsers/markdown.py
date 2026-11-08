"""``parse/markdown``: Markdown and plain text as structure.

The parser reads the block structure of CommonMark and GitHub's pipe tables, line by line, and
leaves inline Markdown (emphasis, links, code spans) as written:

- **Headings** keep their levels: ATX (``## Title``, closing ``#``\\ s dropped) and Setext (a
  paragraph underlined with ``===`` for level 1 or ``---`` for level 2). The first level-1
  heading is the document's title, a ``title`` element; later ones are level-1 headings. A
  document without one has no title, and chunkers title it by its file name.
- **Paragraphs**: consecutive lines, joined with single spaces. A block quote is a paragraph
  without its ``>`` markers.
- **List items**: ``-``, ``*``, ``+``, ``1.`` and ``1)`` items, one element each without its
  marker, with their continuation lines. As in CommonMark, a numbered item interrupts a
  paragraph only when it's numbered 1, so a line that starts with a year stays in its sentence.
- **Tables**: a header row and a delimiter row (``| --- | :-: |``), then each following line
  with a pipe in it, as one ``table`` element: a line per row, cells joined by `` | ``, and its
  cells, the header row marked. Rows are padded or cut to the header's width.
- **Code**: fenced blocks (```` ``` ```` or ``~~~``), verbatim, blank lines included.
- Thematic breaks (``***``, ``- - -``) separate blocks and make no element.

Plain text gets the same reading, so a text file without any of these is paragraphs separated
by blank lines. The bytes are UTF-8 (a byte-order mark is dropped); anything else is read as
Windows-1252, with the warning ``not_utf8``.
"""

import re
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from stratafi_components.parse import ParseComponent
from stratafi_components.parsers.common import Builder, table_data
from stratafi_core.artifacts import ParsedDocument, RawDocument
from stratafi_core.components import ComponentError, ComponentInfo

MEDIA_TYPES = ("text/markdown", "text/plain")

_ATX_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_BREAK_RE = re.compile(r"^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$")
_FENCE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+(\S.*)$")
_ORDERED_RE = re.compile(r"^[ \t]*(\d{1,9})[.)][ \t]+(\S.*)$")
_QUOTE_RE = re.compile(r"^ {0,3}> ?(.*)$")
_DELIMITER_CELL_RE = re.compile(r"^:?-+:?$")


def decode(data: bytes) -> tuple[str, bool]:
    """The text, and whether it was UTF-8."""
    try:
        return data.decode("utf-8-sig"), True
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace"), False


def split_cells(line: str) -> list[str]:
    """A pipe-table row's cells: outer pipes dropped, split at pipes that aren't escaped."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", body)]


def is_delimiter_row(line: str, columns: int) -> bool:
    cells = split_cells(line)
    return len(cells) == columns and all(_DELIMITER_CELL_RE.match(cell) for cell in cells)


class _Reader:
    """Markdown's blocks, line by line, into a builder's elements."""

    def __init__(self, builder: Builder) -> None:
        self.builder = builder
        self.titled = False
        self.paragraph: list[str] = []
        self.quoted = False  # the open paragraph is a block quote's
        self.item: list[str] | None = None  # the open list item's lines

    def flush(self) -> None:
        if self.paragraph:
            self.builder.add("paragraph", " ".join(line.strip() for line in self.paragraph))
            self.paragraph.clear()
        if self.item is not None:
            self.builder.add("list_item", " ".join(line.strip() for line in self.item))
            self.item = None
        self.quoted = False

    def heading(self, text: str, level: int) -> None:
        self.flush()
        text = text.strip()
        if level == 1 and not self.titled and text:
            self.titled = True
            self.builder.add("title", text)
        else:
            self.builder.add("heading", text, level=level)

    def read(self, text: str) -> None:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        index = 0
        while index < len(lines):
            index = self.block(lines, index)
        self.flush()

    def block(self, lines: list[str], index: int) -> int:
        """Read what starts at ``lines[index]``; the index of the line after it."""
        line = lines[index]
        if not line.strip():
            self.flush()
            return index + 1
        fence = _FENCE_RE.match(line)
        if fence and not (fence.group(2)[0] == "`" and "`" in fence.group(3)):
            return self.code(lines, index + 1, len(fence.group(1)), fence.group(2))
        if heading := _ATX_RE.match(line):
            self.heading(heading.group(2) or "", len(heading.group(1)))
            return index + 1
        underline = _SETEXT_RE.match(line)
        if underline and self.paragraph and not self.quoted:
            text = " ".join(part.strip() for part in self.paragraph)
            self.paragraph.clear()
            self.heading(text, 1 if underline.group(1)[0] == "=" else 2)
            return index + 1
        if _BREAK_RE.match(line):
            self.flush()
            return index + 1
        if "|" in line and index + 1 < len(lines):
            header = split_cells(line)
            if is_delimiter_row(lines[index + 1], len(header)):
                return self.table(lines, index + 2, header)
        if quote := _QUOTE_RE.match(line):
            if not self.quoted:
                self.flush()
            if quote.group(1).strip():
                self.paragraph.append(quote.group(1))
                self.quoted = True
            else:
                self.flush()
            return index + 1
        bullet, ordered = _BULLET_RE.match(line), _ORDERED_RE.match(line)
        if bullet or (
            ordered and (not self.paragraph or self.item is not None or ordered.group(1) == "1")
        ):
            self.flush()
            self.item = [bullet.group(1) if bullet else ordered.group(2)]  # type: ignore[union-attr]
            return index + 1
        # A paragraph's next line, or a list item's continuation (indented or lazy).
        (self.item if self.item is not None else self.paragraph).append(line)
        return index + 1

    def table(self, lines: list[str], index: int, header: list[str]) -> int:
        self.flush()
        rows = [header]
        while index < len(lines) and lines[index].strip() and "|" in lines[index]:
            rows.append(split_cells(lines[index]))
            index += 1
        width = len(header)
        cells = [[*row[:width], *[""] * (width - len(row))] for row in rows]
        text = "\n".join(" | ".join(row).rstrip() for row in cells if any(row))
        self.builder.add("table", text, table=table_data(cells, 1))
        return index

    def code(self, lines: list[str], index: int, indent: int, fence: str) -> int:
        self.flush()
        closing = re.compile(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
        body: list[str] = []
        while index < len(lines) and not closing.match(lines[index]):
            line = lines[index]
            body.append(line[min(indent, len(line) - len(line.lstrip(" "))) :])
            index += 1
        self.builder.add("code", "\n".join(body))
        return index + 1


def markdown_document(builder: Builder, text: str) -> str | None:
    """Read ``text`` into ``builder``'s elements; the title, if it has one."""
    _Reader(builder).read(text)
    return next((e["text"] for e in builder.elements if e["type"] == "title"), None)


class MarkdownParser(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/markdown",
        stage="parse",
        version="1.0.0",
        title="Markdown and plain text",
        description="Markdown and plain text as structure: ATX and Setext headings with their "
        "levels, paragraphs, list items, pipe tables and fenced code; the first level-1 "
        "heading is the title.",
        accepts=MEDIA_TYPES,
        license="MIT",
        baseline=True,
    )

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        if raw.media_type not in MEDIA_TYPES:
            raise ComponentError("unsupported_type")
        text, utf8 = decode(path.read_bytes())
        builder = Builder(raw, self, params)
        if not utf8:
            builder.warn("not_utf8")
        title = markdown_document(builder, text)
        return builder.document(title=title)


COMPONENTS = [MarkdownParser]
