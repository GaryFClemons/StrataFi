"""Parse-quality metrics.

What a parsed document says about its parser, without labels: pages, characters per page (the
median), the share of empty pages (pages a text-layer parser read nothing from: OCR is
needed), headings and tables found, and the garbage-character ratio: the share of characters
that are replacement characters, private-use or unassigned code points, or controls other than
line breaks and tabs (what a broken font encoding or a bad OCR leaves). Evidence findability,
the one metric that needs an eval set, is computed by evaluation (``stratafi_core.relevance``).

The run records them per document, so they aggregate per run, file type and parser.
"""

import statistics
import unicodedata
from typing import Any

from stratafi_core.artifacts import ParsedDocument

GARBAGE_CATEGORIES = frozenset({"Co", "Cn", "Cs"})


def _garbage(char: str) -> bool:
    category = unicodedata.category(char)
    if char == "\ufffd" or category in GARBAGE_CATEGORIES:
        return True
    return category == "Cc" and char not in "\n\t\r"


def garbage_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if _garbage(char)) / len(text)


def parse_quality(document: ParsedDocument) -> dict[str, Any]:
    """The document's parse-quality metrics; None where they don't apply."""
    per_page: dict[int, int] = {page.number: 0 for page in document.pages}
    for element in document.elements:
        if element.page is not None:
            per_page[element.page] = per_page.get(element.page, 0) + len(element.text)
    pages = len(per_page)
    return {
        "chars_per_page_p50": float(statistics.median(per_page.values())) if pages else None,
        "empty_page_rate": (sum(1 for n in per_page.values() if n == 0) / pages) if pages else None,
        "headings": sum(1 for e in document.elements if e.type in ("heading", "title")),
        "tables": sum(1 for e in document.elements if e.type == "table"),
        "garbage_char_ratio": round(garbage_ratio(document.text), 6),
    }
