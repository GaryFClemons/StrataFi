"""Parsers A and routing: each parser emits valid parsed documents for its
accepted types and matches its goldens on the format fixtures; ``parse/markdown`` reads
Markdown's and plain text's structure; routing picks the configured parser; a missing optional
extra skips with a clear reason."""

import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

from stratafi_components import default_registry
from stratafi_components.detect import DOCX, detect
from stratafi_components.extract import raw_document
from stratafi_components.parse import ParseComponent
from stratafi_components.parsers.common import Builder, markdown_elements
from stratafi_components.parsers.layout import DoclingParser, UnstructuredParser
from stratafi_components.parsers.markdown import MarkdownParser
from stratafi_core.artifacts import ParsedDocument, RawDocument
from stratafi_core.components import ComponentError, ComponentInfo, StageContext
from stratafi_core.planner import EntryInput, StageOutput, TaskSpec, next_task
from stratafi_core.specs import accepts, resolve_pipeline

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "fixtures" / "formats"
sys.path.insert(0, str(ROOT / "scripts"))
from parser_goldens import GOLDENS, golden_path, golden_record  # noqa: E402

REGISTRY = default_registry()
PARSERS = [component.info.id for component in REGISTRY.for_stage("parse")]


def raw_for(path: Path) -> RawDocument:
    return raw_document("doc-1", "0" * 64, path.stat().st_size, detect(path, path.name))


def parse(parser_id: str, fixture: Path, tmp_path: Path, **services: Any) -> ParsedDocument:
    path = tmp_path / fixture.name
    shutil.copy(fixture, path)
    parser = REGISTRY.get(parser_id)()
    return parser.parse(
        path, raw_for(path), parser.resolve_params({}), StageContext(services=services)
    )


def golden_cases() -> list[tuple[str, str]]:
    cases = []
    for folder in sorted(GOLDENS.iterdir()):
        for golden in sorted(folder.glob("*.json")):
            cases.append((f"parse/{folder.name}", golden.name.removesuffix(".json")))
    return cases


def test_every_parser_has_a_license_entry_and_says_what_it_accepts() -> None:
    assert {
        "parse/markdown",
        "parse/pypdf",
        "parse/pdfplumber",
        "parse/markitdown",
        "parse/tika",
    } <= set(PARSERS)
    assert {"parse/docling", "parse/unstructured"} <= set(PARSERS)
    for parser_id in PARSERS:
        info = REGISTRY.get(parser_id).info
        assert info.license, parser_id
        assert info.license != "unknown", parser_id
        assert info.accepts, parser_id
        assert "*" not in info.accepts, parser_id


@pytest.mark.parametrize(("parser_id", "fixture"), golden_cases())
def test_parsers_match_their_goldens(parser_id: str, fixture: str, tmp_path: Path) -> None:
    """Each parser emits valid parsed documents for its accepted types, as recorded."""
    reason = REGISTRY.get(parser_id).unavailable()
    if reason:
        pytest.skip(f"{parser_id} {reason}")
    document = parse(parser_id, FIXTURES / fixture, tmp_path)
    expected = json.loads(golden_path(parser_id, fixture).read_text(encoding="utf-8"))
    assert golden_record(document) == expected
    assert document.text, "a text-layer fixture has text"


@pytest.mark.parametrize(
    "parser_id", ["parse/markdown", "parse/pypdf", "parse/pdfplumber", "parse/markitdown"]
)
def test_parsers_cover_every_fixture_they_accept(parser_id: str, tmp_path: Path) -> None:
    reason = REGISTRY.get(parser_id).unavailable()
    if reason:
        pytest.skip(f"{parser_id} {reason}")
    info = REGISTRY.get(parser_id).info
    for fixture in sorted(FIXTURES.glob("*.*")):
        raw = raw_for(fixture)
        if accepts(info.accepts, raw.media_type, raw.trait_names):
            assert golden_path(parser_id, fixture.name).is_file(), (parser_id, fixture.name)


def test_pdfplumber_finds_headings_and_boxes_from_the_bottom_left(tmp_path: Path) -> None:
    if reason := REGISTRY.get("parse/pdfplumber").unavailable():
        pytest.skip(f"parse/pdfplumber {reason}")
    document = parse("parse/pdfplumber", FIXTURES / "food-safety.pdf", tmp_path)
    headings = [e for e in document.elements if e.type == "heading"]
    assert headings
    page = document.pages[0]
    for element in document.elements:
        x0, y0, x1, y1 = element.bbox
        assert 0 <= x0 < x1 <= page.width
        assert 0 <= y0 < y1 <= page.height
    # Reading order runs down the page: the first element is the highest.
    assert document.elements[0].bbox[3] >= document.elements[1].bbox[3]


def test_a_scanned_pdf_parses_empty_with_a_warning(tmp_path: Path) -> None:
    from stratafi_testing.documents import blank_pdf_bytes

    path = tmp_path / "scan.pdf"
    path.write_bytes(blank_pdf_bytes(2))
    parser = REGISTRY.get("parse/pypdf")()
    document = parser.parse(path, raw_for(path), parser.resolve_params({}), StageContext())
    assert document.elements == []
    assert document.warnings == ["no_text"]
    assert len(document.pages) == 2


def test_markdown_is_read_into_elements() -> None:
    raw = RawDocument(
        document_id="d",
        sha256="0" * 64,
        size_bytes=1,
        extension=".md",
        media_type="text/markdown",
        kind="text",
        mismatch=False,
        known=True,
    )
    builder = Builder(
        raw, REGISTRY.get("parse/markitdown"), REGISTRY.get("parse/markitdown").resolve_params({})
    )
    markdown_elements(
        builder,
        "# Fees\n\nIntro line one\nline two.\n\n## Wires\n\n- Domestic\n- International\n\n"
        "| Fee | Amount |\n| --- | --- |\n| Wire | 25 |\n\n```\ncode\n```\n",
    )
    document = builder.document()
    assert [(e.type, e.text) for e in document.elements] == [
        ("title", "Fees"),
        ("paragraph", "Intro line one line two."),
        ("heading", "Wires"),
        ("list_item", "Domestic"),
        ("list_item", "International"),
        ("table", "Fee | Amount\nWire | 25"),
        ("code", "code"),
    ]
    assert document.title == "Fees"
    table = document.elements[5].table
    assert (table.rows, table.cols) == (2, 2)
    assert [cell.header for cell in table.cells] == [True, True, False, False]


# --- docling and unstructured through stand-ins for their libraries ---------------------------


def _install(monkeypatch: pytest.MonkeyPatch, modules: dict[str, types.ModuleType]) -> None:
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_docling_items_map_onto_parsed_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ns = types.SimpleNamespace

    def prov(page: int, left: float, t: float, r: float, b: float) -> list[Any]:
        return [ns(page_no=page, bbox=ns(l=left, t=t, r=r, b=b, coord_origin="BOTTOMLEFT"))]

    cell = lambda r, c, text: ns(start_row_offset_idx=r, start_col_offset_idx=c, text=text)  # noqa: E731
    items = [
        (ns(label=ns(value="title"), text="Wire Policy", prov=prov(1, 72, 760, 300, 740)), 0),
        (
            ns(
                label=ns(value="section_header"),
                text="Callbacks",
                level=2,
                prov=prov(1, 72, 700, 200, 690),
            ),
            1,
        ),
        (
            ns(
                label=ns(value="text"),
                text="Call the number on file.",
                prov=prov(1, 72, 680, 500, 660),
            ),
            1,
        ),
        (
            ns(
                label=ns(value="table"),
                prov=prov(2, 72, 600, 400, 500),
                data=ns(
                    num_rows=2,
                    num_cols=2,
                    table_cells=[
                        cell(0, 0, "Fee"),
                        cell(0, 1, "Amount"),
                        cell(1, 0, "Wire"),
                        cell(1, 1, "25"),
                    ],
                ),
            ),
            1,
        ),
        (ns(label=ns(value="page_footer"), text="Page 1", prov=prov(1, 72, 40, 120, 30)), 0),
    ]
    document = ns(
        pages={
            1: ns(size=ns(width=612.0, height=792.0)),
            2: ns(size=ns(width=612.0, height=792.0)),
        },
        iterate_items=lambda: iter(items),
    )

    class Converter:
        def __init__(self, format_options: Any) -> None:
            self.options = format_options

        def convert(self, source: str) -> Any:
            assert source.endswith(".pdf")
            return ns(document=document)

    docling = types.ModuleType("docling")
    _install(
        monkeypatch,
        {
            "docling": docling,
            "docling.datamodel": types.ModuleType("docling.datamodel"),
            "docling.datamodel.base_models": ns(InputFormat=ns(PDF="pdf")),  # type: ignore[dict-item]
            "docling.datamodel.pipeline_options": ns(PdfPipelineOptions=lambda **kw: kw),  # type: ignore[dict-item]
            "docling.document_converter": ns(  # type: ignore[dict-item]
                DocumentConverter=Converter,
                PdfFormatOption=lambda pipeline_options: pipeline_options,
            ),
        },
    )
    monkeypatch.setattr(DoclingParser, "unavailable", classmethod(lambda cls: None))
    path = tmp_path / "policy.pdf"
    shutil.copy(FIXTURES / "wholesale.pdf", path)
    parser = DoclingParser()
    parsed = parser.parse_local(path, raw_for(path), parser.resolve_params({"ocr": True}))
    assert [(e.type, e.level, e.page) for e in parsed.elements] == [
        ("title", None, 1),
        ("heading", 2, 1),
        ("paragraph", None, 1),
        ("table", None, 2),
        ("page_footer", None, 1),
    ]
    assert parsed.title == "Wire Policy"
    assert parsed.elements[3].text == "Fee | Amount\nWire | 25"
    assert parsed.elements[2].bbox == (72.0, 660.0, 500.0, 680.0)
    assert [page.number for page in parsed.pages] == [1, 2]


def test_unstructured_elements_map_onto_parsed_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ns = types.SimpleNamespace
    system = ns(width=1000.0, height=1400.0)

    def element(category: str, text: str, page: int = 1, depth: int | None = None) -> Any:
        coordinates = ns(points=[(100, 200), (100, 260), (900, 260), (900, 200)], system=system)
        return ns(
            category=category,
            text=text,
            metadata=ns(page_number=page, coordinates=coordinates, category_depth=depth),
        )

    seen = {}

    def partition(filename: str, strategy: str) -> list[Any]:
        seen["strategy"] = strategy
        return [
            element("Title", "Wire Policy", depth=0),
            element("NarrativeText", "Call the number on file."),
            element("ListItem", "Domestic wires"),
            element("Table", "Fee\tAmount\nWire\t25", page=2),
            element("PageNumber", "1"),
        ]

    _install(
        monkeypatch,
        {
            "unstructured": types.ModuleType("unstructured"),
            "unstructured.partition": types.ModuleType("unstructured.partition"),
            "unstructured.partition.auto": ns(partition=partition),  # type: ignore[dict-item]
        },
    )
    path = tmp_path / "policy.pdf"
    shutil.copy(FIXTURES / "wholesale.pdf", path)
    parser = UnstructuredParser()
    parsed = parser.parse_local(path, raw_for(path), parser.resolve_params({"strategy": "hi_res"}))
    assert seen["strategy"] == "hi_res"
    assert [(e.type, e.level, e.page) for e in parsed.elements] == [
        ("heading", 1, 1),
        ("paragraph", None, 1),
        ("list_item", None, 1),
        ("table", None, 2),
        ("page_number", None, 1),
    ]
    assert parsed.elements[3].text == "Fee | Amount\nWire | 25"
    # Pixel space flips to points from the bottom left.
    assert parsed.elements[0].bbox == (100.0, 1140.0, 900.0, 1200.0)


# --- parse/markdown ---------------------------------------------------------------------------


def markdown(text: str | bytes, tmp_path: Path, name: str = "doc.md") -> ParsedDocument:
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    parser = MarkdownParser()
    return parser.parse_local(path, raw_for(path), parser.resolve_params({}))


def shape(document: ParsedDocument) -> list[tuple[str, int | None, str]]:
    return [(e.type, e.level, e.text) for e in document.elements]


def test_markdown_headings_keep_their_levels_and_the_first_h1_is_the_title(
    tmp_path: Path,
) -> None:
    document = markdown(
        "Preamble before any heading.\n\n# Tip Policy #\n\n## 1 Who shares\n\nBaristas do.\n\n"
        "Setext section\n--------------\n\n### 1.1 Porters\n\nThey share too.\n\n"
        "Second top heading\n==================\n\n####### not a heading\n#no space\n",
        tmp_path,
    )
    assert document.title == "Tip Policy"
    assert shape(document) == [
        ("paragraph", None, "Preamble before any heading."),
        ("title", None, "Tip Policy"),
        ("heading", 2, "1 Who shares"),
        ("paragraph", None, "Baristas do."),
        ("heading", 2, "Setext section"),
        ("heading", 3, "1.1 Porters"),
        ("paragraph", None, "They share too."),
        ("heading", 1, "Second top heading"),
        ("paragraph", None, "####### not a heading #no space"),
    ]
    for element in document.elements:
        assert document.text[element.char_start : element.char_end] == element.text


def test_markdown_lists_tables_quotes_and_code(tmp_path: Path) -> None:
    document = markdown(
        "# Opening\n\nSteps, in order:\n1. Disarm the alarm\n   within 60 seconds.\n"
        "2. Turn on the machines.\n\n- Bullets work\n* too\n+ all three\n\n"
        "Effective since\n2026. A year starts this line.\n\n"
        "| Drink | Dose | Time |\n| --- | :---: | ---: |\n| Decaf | 18 g | 28\u201333 s |\n"
        "| Pour-over | 22 g |\n| A \\| B | 1 | 2 | extra |\n\n> Quoted advice\n> on two lines.\n\n"
        "```text\nkeep   spacing\n\nand blank lines\n```\n\n***\n\nThe end.\n",
        tmp_path,
    )
    assert shape(document) == [
        ("title", None, "Opening"),
        ("paragraph", None, "Steps, in order:"),
        ("list_item", None, "Disarm the alarm within 60 seconds."),
        ("list_item", None, "Turn on the machines."),
        ("list_item", None, "Bullets work"),
        ("list_item", None, "too"),
        ("list_item", None, "all three"),
        ("paragraph", None, "Effective since 2026. A year starts this line."),
        (
            "table",
            None,
            "Drink | Dose | Time\nDecaf | 18 g | 28\u201333 s\nPour-over | 22 g |\nA | B | 1 | 2",
        ),
        ("paragraph", None, "Quoted advice on two lines."),
        ("code", None, "keep   spacing\n\nand blank lines"),
        ("paragraph", None, "The end."),
    ]
    table = document.elements[8].table
    assert (table.rows, table.cols) == (4, 3)
    assert [cell.text for cell in table.cells if cell.header] == ["Drink", "Dose", "Time"]
    assert [cell.text for cell in table.cells if cell.r == 3] == ["A | B", "1", "2"]


def test_plain_text_is_paragraphs_and_setext_headings(tmp_path: Path) -> None:
    text = (
        "\ufeffTip Policy\r\n==========\r\n\r\n"
        "Lines that wrap\r\nat the margin.\r\n\r\n3.1 A line\r\n"
    )
    document = markdown(text, tmp_path, "tips.txt")
    assert document.media_type == "text/plain"
    assert shape(document) == [
        ("title", None, "Tip Policy"),
        ("paragraph", None, "Lines that wrap at the margin."),
        ("paragraph", None, "3.1 A line"),
    ]
    assert document.warnings == []
    no_headings = markdown("One paragraph.\n\nAnother.\n", tmp_path, "notes.txt")
    assert no_headings.title is None
    assert shape(no_headings) == [
        ("paragraph", None, "One paragraph."),
        ("paragraph", None, "Another."),
    ]


def test_markdown_reads_other_encodings_and_refuses_other_types(tmp_path: Path) -> None:
    document = markdown("Caf\xe9 hours.\n".encode("cp1252"), tmp_path, "cafe.txt")
    assert document.text == "Café hours."
    assert document.warnings == ["not_utf8"]
    empty = markdown("\n\n", tmp_path, "empty.md")
    assert empty.elements == []
    assert empty.warnings == ["no_text"]
    path = tmp_path / "page.html"
    path.write_text("<p>Hi</p>", encoding="utf-8")
    parser = MarkdownParser()
    with pytest.raises(ComponentError) as error:
        parser.parse_local(path, raw_for(path), parser.resolve_params({}))
    assert error.value.code == "unsupported_type"


def test_markdown_and_plain_text_renderings_read_the_same(tmp_path: Path) -> None:
    """The format fixtures' Markdown and plain text hold the same blocks: the text's Setext
    headings are the Markdown's top two levels, and its deeper headings are lines of their
    own."""
    for name in ("tips", "wholesale", "food-safety"):
        source = parse("parse/markdown", FIXTURES / f"{name}.md", tmp_path)
        text = parse("parse/markdown", FIXTURES / f"{name}.txt", tmp_path)
        assert source.title == text.title
        expected = [
            ("paragraph", None, e.text)
            if e.type == "heading" and (e.level or 0) > 2
            else (e.type, e.level, e.text)
            for e in source.elements
        ]
        assert shape(text) == expected, name


# --- routing ----------------------------------------------------------------------------------


def test_routing_picks_the_configured_parser() -> None:
    """Routing picks the configured parser by media type and trait."""
    pipeline = resolve_pipeline(
        {
            "schema": "stratafi.pipeline.v1",
            "name": "routes",
            "stages": {
                "parse": {
                    "route": {
                        "default": "parse/markdown",
                        "pdf": "parse/pdfplumber",
                        "pdf[scanned]": {"component": "parse/pypdf", "params": {"mode": "layout"}},
                        "docx": "parse/markitdown",
                        "text/markdown": "parse/markitdown",
                    }
                }
            },
        },
        REGISTRY,
    )
    entry = EntryInput("a", "d-1", "0" * 64, ".pdf")
    cases = [
        (
            {"media_type": "application/pdf", "traits": {"scanned": False}},
            "parse/pdfplumber",
            "pdf",
        ),
        (
            {"media_type": "application/pdf", "traits": {"scanned": True}},
            "parse/pypdf",
            "pdf[scanned]",
        ),
        (
            {
                "media_type": DOCX,
                "traits": {},
            },
            "parse/markitdown",
            "docx",
        ),
        ({"media_type": "text/markdown", "traits": {}}, "parse/markitdown", "text/markdown"),
        ({"media_type": "text/plain", "traits": {}}, "parse/markdown", "default"),
    ]
    for raw, expected, route in cases:
        outputs = {
            "extract": StageOutput("1" * 64, raw),
            "scan": StageOutput("2" * 64, {"decision": "continue"}),
        }
        task = next_task(pipeline, REGISTRY, entry, outputs)
        assert isinstance(task, TaskSpec)
        assert (task.component_id, task.route) == (expected, route), raw
    with pytest.raises(Exception, match="problem"):
        resolve_pipeline(
            {
                "schema": "stratafi.pipeline.v1",
                "name": "bad",
                "stages": {
                    "parse": {"route": {"default": "parse/markdown", "pptx": "parse/pypdf"}}
                },
            },
            REGISTRY,
        )


# --- missing extras -------------------------------------------------------------------------


class NeedsAnExtra(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/needs-extra",
        stage="parse",
        version="1.0.0",
        title="Needs an extra",
        description="A parser whose library isn't installed.",
        accepts=("text/plain",),
        license="MIT",
    )
    requires: ClassVar[tuple[str, ...]] = ("stratafi_no_such_library",)
    extra: ClassVar[str | None] = "imaginary"


def test_a_missing_extra_says_which_and_refuses_to_run(tmp_path: Path) -> None:
    """A missing optional extra skips with a clear reason."""
    reason = NeedsAnExtra.unavailable()
    assert reason == "needs the imaginary extra (uv sync --extra imaginary)"
    path = tmp_path / "notes.txt"
    path.write_text("notes")
    with pytest.raises(ComponentError) as error:
        NeedsAnExtra().parse(path, raw_for(path), NeedsAnExtra.resolve_params({}), StageContext())
    assert error.value.code == "extra_missing"
    from stratafi_core.components import ComponentRegistry

    [description] = ComponentRegistry([NeedsAnExtra]).descriptions()
    assert description.unavailable == reason
