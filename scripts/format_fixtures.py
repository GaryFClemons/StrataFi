#!/usr/bin/env python3
"""Write the format fixtures to fixtures/formats/: three documents of the Halcyon Coffee demo
corpus, each as Markdown, plain text, DOCX and PDF, for the parsers' tests and goldens.

    uv run python scripts/format_fixtures.py            # write them
    uv run python scripts/format_fixtures.py --check    # exit 1 if any differs

The documents come from ``fixtures/demo-corpus/corpus`` (CC0-1.0, everything invented) and were
picked for what they exercise: the Tip Policy has level-3 subsections, the Wholesale Pricing
and Ordering Guide a five-column table, and the Food Safety and Allergen Standard runs past a
page with two tables and lists. Each becomes ``<name>.md``, ``.txt``, ``.docx`` and ``.pdf``:

- ``.md``: the source, byte for byte.
- ``.txt``: plain text as people write it: Setext headings (the title over ``===``, sections
  over ``---``), deeper headings as lines of their own, paragraphs wrapped at 78 columns,
  lists and pipe tables as in the source; with a byte-order mark and CRLF line endings.
- ``.docx`` and ``.pdf``: rendered by ``scripts/corpus_variants.py`` (heading styles, lists and
  tables), the DOCX unzipped and the PDF's pages uncompressed, so the bytes don't depend on
  the machine's zlib.

The files are byte-for-byte reproducible; ``--check`` proves they still are.
"""

import argparse
import importlib.util
import sys
import textwrap
import zipfile
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "fixtures" / "demo-corpus" / "corpus"
OUTPUT = ROOT / "fixtures" / "formats"
DOCUMENTS = {
    "tips": "handbook/tips-v3.md",
    "wholesale": "operations/wholesale-v4.md",
    "food-safety": "operations/food-safety-v4.md",
}
EXTENSIONS = (".md", ".txt", ".docx", ".pdf")
WIDTH = 78


@cache
def _variants():
    """``scripts/corpus_variants.py``, loaded once."""
    path = Path(__file__).resolve().parent / "corpus_variants.py"
    spec = importlib.util.spec_from_file_location("format_fixtures_corpus_variants", path)
    if spec is None or spec.loader is None:
        raise ImportError("can't load scripts/corpus_variants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plain_text(document) -> bytes:
    """A document as a plain-text file: Setext headings, wrapped paragraphs, CRLF and a BOM."""
    variants = _variants()
    lines: list[str] = []

    def block(*block_lines: str) -> None:
        if lines:
            lines.append("")
        lines.extend(block_lines)

    if document.title:
        block(document.title, "=" * len(document.title))
    for item in document.blocks:
        if isinstance(item, variants.Heading):
            underline = "-" * len(item.text) if item.level <= 2 else None
            block(item.text, *([underline] if underline else []))
        elif isinstance(item, variants.Text):
            block(*textwrap.wrap(item.text, WIDTH, break_long_words=False, break_on_hyphens=False))
        elif isinstance(item, variants.Items):
            block(
                *(
                    f"{number}. {text}" if item.ordered else f"- {text}"
                    for number, text in enumerate(item.items, start=item.start)
                )
            )
        else:
            header, *rows = item.padded()
            block(
                f"| {' | '.join(header)} |",
                "|" + "|".join(" --- " for _ in header) + "|",
                *(f"| {' | '.join(row)} |" for row in rows),
            )
    return ("﻿" + "\r\n".join(lines) + "\r\n").encode("utf-8")


def render() -> dict[str, bytes]:
    """Every fixture's bytes, by file name."""
    variants = _variants()
    files: dict[str, bytes] = {}
    for name, source in DOCUMENTS.items():
        markdown = (SOURCE / source).read_bytes()
        document = variants.parse_markdown(markdown.decode("utf-8"))
        files[f"{name}.md"] = markdown
        files[f"{name}.txt"] = plain_text(document)
        files[f"{name}.docx"] = variants.render_docx(document, compression=zipfile.ZIP_STORED)
        files[f"{name}.pdf"] = variants.render_pdf(document, compress=False)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the format fixtures.")
    parser.add_argument("--check", action="store_true", help="exit 1 if a fixture differs")
    args = parser.parse_args(argv)
    files = render()
    if args.check:
        stale = sorted(
            name
            for name, data in files.items()
            if not (OUTPUT / name).is_file() or (OUTPUT / name).read_bytes() != data
        )
        extra = sorted(
            path.name for path in OUTPUT.glob("*") if path.is_file() and path.name not in files
        )
        if stale or extra:
            print(f"stale fixtures: {', '.join(stale + extra)}", file=sys.stderr)
            return 1
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (OUTPUT / name).write_bytes(data)
        print(f"wrote fixtures/formats/{name} ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
