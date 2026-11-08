"""Write (or check) every parser's golden output on the format fixtures.

For each parser this installation can run, and each fixture it accepts, the golden is a compact
record of what it made: each element's type, level, page, whether it has a box, and its text;
the pages; the warnings. ``packages/components/tests/test_parsers.py`` compares the parsers
against them, so a change in a parser's output, or in its library, shows up as a diff and needs
a version bump.

    uv run python scripts/parser_goldens.py            # write
    uv run python scripts/parser_goldens.py --check    # exit 1 when a golden differs

``fixtures/formats`` (``scripts/format_fixtures.py``) holds three documents as Markdown, plain
text, DOCX and PDF. Parsers whose extra isn't installed are skipped, and say so.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "formats"
GOLDENS = ROOT / "fixtures" / "parsers" / "goldens"
PARSERS = (
    "parse/markdown",
    "parse/pypdf",
    "parse/pdfplumber",
    "parse/markitdown",
    "parse/docling",
    "parse/unstructured",
)


def golden_record(document: Any) -> dict[str, Any]:
    return {
        "parser": document.parser.id,
        "version": document.parser.version,
        "title": document.title,
        "pages": len(document.pages),
        "warnings": list(document.warnings),
        "elements": [
            [element.type, element.level, element.page, element.bbox is not None, element.text]
            for element in document.elements
        ],
    }


def golden_path(parser: str, fixture: str) -> Path:
    return GOLDENS / parser.split("/", 1)[1] / f"{fixture}.json"


def parse_fixture(parser_id: str, fixture: Path) -> Any | None:
    from stratafi_components import default_registry
    from stratafi_components.detect import detect
    from stratafi_components.extract import raw_document
    from stratafi_core.components import StageContext
    from stratafi_core.specs import accepts

    parser = default_registry().get(parser_id)()
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / fixture.name
        shutil.copy(fixture, path)
        raw = raw_document("golden", "0" * 64, path.stat().st_size, detect(path, fixture.name))
        if not accepts(parser.info.accepts, raw.media_type, raw.trait_names):
            return None
        return parser.parse(path, raw, parser.resolve_params({}), StageContext())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from stratafi_components import default_registry

    registry = default_registry()
    stale = 0
    for parser_id in PARSERS:
        reason = registry.get(parser_id).unavailable()
        if reason:
            print(f"skip {parser_id}: {reason}")
            continue
        for fixture in sorted(FIXTURES.iterdir()):
            if not fixture.is_file():
                continue
            document = parse_fixture(parser_id, fixture)
            if document is None:
                continue
            record = json.dumps(golden_record(document), indent=1, ensure_ascii=False) + "\n"
            target = golden_path(parser_id, fixture.name)
            if args.check:
                if not target.is_file() or target.read_text(encoding="utf-8") != record:
                    print(f"stale {target.relative_to(ROOT)}")
                    stale += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(record, encoding="utf-8")
            print(f"wrote {target.relative_to(ROOT)}")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
