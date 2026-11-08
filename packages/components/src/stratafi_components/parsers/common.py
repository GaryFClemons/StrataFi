"""What parsers share: building a ``stratafi.parsed-document.v1`` from elements, flattening
tables, and reading Markdown into elements (markitdown and docling export Markdown).

``Builder`` collects elements in reading order and makes the document: element IDs ``e0``,
``e1``…, the canonical text (element texts joined by a blank line) and every element's offsets.
Empty elements are left out. A table element's text is its rows, one per line, cells joined by
`` | ``, and its ``table`` holds the cells.
"""

import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from stratafi_core.artifacts import ParsedDocument, RawDocument, build_text
from stratafi_core.hashing import content_hash

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def table_text(rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows if any(row))


def table_data(rows: Sequence[Sequence[str]], header_rows: int = 1) -> dict[str, Any]:
    cols = max((len(row) for row in rows), default=0)
    return {
        "rows": len(rows),
        "cols": cols,
        "cells": [
            {"r": r, "c": c, "text": str(cell or "").strip(), "header": r < header_rows}
            for r, row in enumerate(rows)
            for c, cell in enumerate(row)
        ],
    }


class Builder:
    """Elements in reading order → a parsed document."""

    def __init__(self, raw: RawDocument, component: Any, params: BaseModel) -> None:
        self.raw = raw
        self.component = component
        self.params = params
        self.elements: list[dict[str, Any]] = []
        self.warnings: list[str] = []

    def add(
        self,
        kind: str,
        text: str,
        *,
        level: int | None = None,
        page: int | None = None,
        bbox: Sequence[float] | None = None,
        table: dict[str, Any] | None = None,
        parent: int | None = None,
    ) -> int | None:
        """Add an element; returns its index, or None when its text is empty."""
        text = text.strip("\n")
        if not text.strip():
            return None
        element: dict[str, Any] = {"type": kind, "text": text}
        if level is not None:
            element["level"] = max(1, min(int(level), 9))
        if page is not None:
            element["page"] = int(page)
        if bbox is not None:
            element["bbox"] = [round(float(value), 2) for value in bbox]
        if table is not None:
            element["table"] = table
        if parent is not None:
            element["parent_id"] = f"e{parent}"
        self.elements.append(element)
        return len(self.elements) - 1

    def add_table(
        self, rows: Sequence[Sequence[str]], *, page: int | None = None, bbox=None, header_rows=1
    ) -> int | None:
        rows = [[str(cell or "") for cell in row] for row in rows]
        return self.add(
            "table", table_text(rows), page=page, bbox=bbox, table=table_data(rows, header_rows)
        )

    def warn(self, code: str) -> None:
        if code not in self.warnings:
            self.warnings.append(code)

    def document(
        self, *, title: str | None = None, pages: list[dict[str, Any]] | None = None
    ) -> ParsedDocument:
        texts = [element["text"] for element in self.elements]
        text, offsets = build_text(texts)
        for index, (element, (start, end)) in enumerate(zip(self.elements, offsets, strict=True)):
            element.update(id=f"e{index}", char_start=start, char_end=end)
        if not self.elements:
            self.warn("no_text")
        if title is None:
            title = next((e["text"] for e in self.elements if e["type"] == "title"), None)
        return ParsedDocument.model_validate(
            {
                "schema": "stratafi.parsed-document.v1",
                "document_id": self.raw.document_id,
                "parser": {
                    "id": self.component.info.id,
                    "version": self.component.info.version,
                    "params_hash": content_hash(self.component.dump_params(self.params)),
                },
                "title": title,
                "media_type": self.raw.media_type,
                "pages": pages or [],
                "elements": self.elements,
                "text": text,
                "warnings": self.warnings,
            }
        )


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def markdown_elements(builder: Builder, markdown: str, *, page: int | None = None) -> None:
    """Read Markdown blocks into elements: ATX headings, list items, pipe tables, fenced code
    and paragraphs (consecutive lines joined). The first level-1 heading is the title when the
    builder has no element yet."""
    lines = markdown.replace("\r\n", "\n").split("\n")
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            builder.add("paragraph", " ".join(part.strip() for part in paragraph), page=page)
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        if FENCE_RE.match(line):
            flush()
            fence = FENCE_RE.match(line).group(1)  # type: ignore[union-attr]
            body = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence):
                body.append(lines[index])
                index += 1
            builder.add("code", "\n".join(body), page=page)
        elif heading := HEADING_RE.match(line):
            flush()
            level = len(heading.group(1))
            kind = "title" if level == 1 and not builder.elements else "heading"
            builder.add(kind, heading.group(2), level=None if kind == "title" else level, page=page)
        elif (
            TABLE_ROW_RE.match(line)
            and index + 1 < len(lines)
            and TABLE_RULE_RE.match(lines[index + 1])
        ):
            flush()
            rows = [_cells(line)]
            index += 2
            while index < len(lines) and TABLE_ROW_RE.match(lines[index]):
                rows.append(_cells(lines[index]))
                index += 1
            builder.add_table(rows, page=page)
            continue
        elif item := LIST_RE.match(line):
            flush()
            builder.add("list_item", item.group(1), page=page)
        elif not line.strip():
            flush()
        else:
            paragraph.append(line)
        index += 1
    flush()
