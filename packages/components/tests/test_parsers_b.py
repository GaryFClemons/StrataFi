"""Parsers B: OCR reaches the evidence-findability threshold on the scanned-PDF
variant; spreadsheets as rows with headers, a section per slide, HTML structure, Tika's XHTML;
hosted adapters refuse unless allow-listed; pymupdf4llm waits for its license; parse quality."""

import base64
import json
import shutil
from pathlib import Path

import httpx2
import pytest
import yaml

from stratafi_components import default_registry
from stratafi_components.detect import PPTX, XLSX, detect
from stratafi_components.extract import raw_document
from stratafi_components.parse_quality import garbage_ratio, parse_quality
from stratafi_components.parsers.hosted import ProviderAccess
from stratafi_components.parsers.tika import TikaClient, tika_pages
from stratafi_core.artifacts import ParsedDocument
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.evalsets import EvidenceRef
from stratafi_core.relevance import findability
from stratafi_testing.documents import zip_bytes

ROOT = Path(__file__).resolve().parents[3]
MULTI = ROOT / "fixtures" / "multi-format"
REGISTRY = default_registry()
# Set from the first measurement (2026-10-31, Tesseract 5.3.4, 200 dpi): 5 of 5 evidence
# phrases found, mean grade 0.93. The threshold leaves room for one miss.
OCR_FINDABILITY_THRESHOLD = 0.8


def parse_file(parser_id: str, path: Path, params: dict | None = None, **services):
    parser = REGISTRY.get(parser_id)()
    raw = raw_document("doc-1", "0" * 64, path.stat().st_size, detect(path, path.name))
    context = StageContext(services=services)
    document = parser.parse(path, raw, parser.resolve_params(params or {}), context)
    return document, context


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --- OCR ----------------------------------------------------------------------------------------


def test_ocr_reaches_the_findability_threshold_on_the_scanned_variant(tmp_path: Path) -> None:
    """With OCR, the scanned-PDF variant's evidence is findable at the recorded threshold."""
    if shutil.which("tesseract") is None:
        pytest.skip(
            "parse/ocr's default engine needs the tesseract binary (apt install tesseract-ocr)"
        )
    if reason := REGISTRY.get("parse/ocr").unavailable():
        pytest.skip(f"parse/ocr {reason}")
    manifest = yaml.safe_load((MULTI / "variants" / "scanned-pdf" / "manifest.yaml").read_text())
    texts = {}
    for document in manifest["documents"]:
        path = tmp_path / Path(document["path"]).name
        shutil.copy(MULTI / "variants" / "scanned-pdf" / document["path"], path)
        parsed, context = parse_file("parse/ocr", path)
        assert parsed.warnings == ["ocr"]
        assert context.metrics["ocr_pages"] == len(parsed.pages)
        assert all(element.bbox is not None for element in parsed.elements)
        texts[(document["doc_key"], str(document["version"]))] = parsed.text
    queries = yaml.safe_load((MULTI / "retrieval.yaml").read_text())["queries"]
    results = [
        findability(
            EvidenceRef(**{k: v for k, v in item.items() if k != "source_uri"}),
            texts[(item["doc_key"], str(item["version"]))],
        )
        for query in queries
        for item in query["expected"]
        if (item["doc_key"], str(item["version"])) in texts
    ]
    assert results
    rate = sum(result.findable for result in results) / len(results)
    assert rate >= OCR_FINDABILITY_THRESHOLD, [result.grade for result in results]


def test_a_text_layer_page_keeps_its_text_without_ocr(tmp_path: Path) -> None:
    if reason := REGISTRY.get("parse/ocr").unavailable():
        pytest.skip(f"parse/ocr {reason}")
    path = tmp_path / "wire.pdf"
    shutil.copy(ROOT / "fixtures" / "formats" / "wholesale.pdf", path)
    parsed, context = parse_file("parse/ocr", path)
    assert "wholesale portal" in parsed.text.lower()
    assert context.metrics["ocr_pages"] == 0
    assert "ocr" not in parsed.warnings


# --- spreadsheets ---------------------------------------------------------------------------

SHEET = (
    b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
    b'<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
    b'<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>25</v></c></row>'
    b'<row r="3"><c r="A3" t="inlineStr"><is><t>Overdraft</t></is></c><c r="B3"><v>30</v></c></row>'
    b"</sheetData></worksheet>"
)
WORKBOOK = zip_bytes(
    {
        "[Content_Types].xml": b"<Types/>",
        "xl/workbook.xml": (
            b'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            b'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            b'<sheets><sheet name="Fees" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            b'<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>'
        ),
        "xl/worksheets/sheet1.xml": SHEET,
        "xl/sharedStrings.xml": (
            b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b"<si><t>Fee</t></si><si><t>Amount</t></si><si><t>Wire</t></si></sst>"
        ),
    }
)


def test_a_workbook_parses_as_rows_with_headers(tmp_path: Path) -> None:
    path = write(tmp_path, "fees.xlsx", WORKBOOK)
    assert detect(path, "fees.xlsx").media_type == XLSX
    parsed, _ = parse_file("parse/spreadsheet", path)
    assert [(e.type, e.text) for e in parsed.elements] == [
        ("heading", "Fees"),
        ("list_item", "Fee: Wire; Amount: 25"),
        ("list_item", "Fee: Overdraft; Amount: 30"),
    ]
    table, _ = parse_file("parse/spreadsheet", path, {"layout": "table"})
    assert table.elements[1].type == "table"
    assert table.elements[1].table.rows == 3


def test_csv_and_tsv_parse_as_rows_with_headers(tmp_path: Path) -> None:
    path = write(tmp_path, "fees.csv", b"Fee,Amount\nWire,25\n,\nCheck,\n")
    parsed, _ = parse_file("parse/spreadsheet", path)
    assert [e.text for e in parsed.elements] == ["Fee: Wire; Amount: 25", "Fee: Check"]
    tsv = write(tmp_path, "fees.tsv", b"Fee\tAmount\nWire\t25\n")
    parsed, _ = parse_file("parse/spreadsheet", tsv)
    assert [e.text for e in parsed.elements] == ["Fee: Wire; Amount: 25"]
    cut, _ = parse_file("parse/spreadsheet", path, {"max_rows": 2})
    assert cut.warnings == ["rows_truncated"]


# --- slides ---------------------------------------------------------------------------------

P_NS = (
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
)


def run(text: str) -> str:
    return f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p>"


def cell(text: str) -> str:
    return f"<a:tc><a:txBody>{run(text)}</a:txBody></a:tc>"


def slide(title: str, body: str, table: bool = False) -> bytes:
    shapes = (
        '<p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>'
        f"<p:txBody>{run(title)}</p:txBody></p:sp>"
        f"<p:sp><p:nvSpPr><p:nvPr/></p:nvSpPr><p:txBody>{run(body)}"
        '<a:p><a:pPr lvl="1"/><a:r><a:t>A bullet</a:t></a:r></a:p></p:txBody></p:sp>'
    )
    if table:
        row = f"<a:tr>{cell('Fee')}{cell('25')}</a:tr>"
        shapes += f"<p:graphicFrame><a:tbl>{row}</a:tbl></p:graphicFrame>"
    return f"<p:sld {P_NS}><p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:sld>".encode()


DECK = zip_bytes(
    {
        "ppt/presentation.xml": (
            b'<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            b'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            b'<p:sldIdLst><p:sldId id="256" r:id="rId2"/>'
            b'<p:sldId id="257" r:id="rId3"/></p:sldIdLst>'
            b"</p:presentation>"
        ),
        "ppt/_rels/presentation.xml.rels": (
            b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            b'<Relationship Id="rId2" Target="slides/slide1.xml"/>'
            b'<Relationship Id="rId3" Target="slides/slide2.xml"/></Relationships>'
        ),
        "ppt/slides/slide1.xml": slide("Wires", "Call back first."),
        "ppt/slides/slide2.xml": slide("Fees", "What wires cost.", table=True),
    }
)


def test_a_deck_parses_as_a_section_per_slide(tmp_path: Path) -> None:
    path = write(tmp_path, "deck.pptx", DECK)
    assert detect(path, "deck.pptx").media_type == PPTX
    parsed, _ = parse_file("parse/pptx", path)
    assert [(e.type, e.text, e.page) for e in parsed.elements] == [
        ("heading", "Slide 1: Wires", 1),
        ("paragraph", "Call back first.", 1),
        ("list_item", "A bullet", 1),
        ("heading", "Slide 2: Fees", 2),
        ("paragraph", "What wires cost.", 2),
        ("list_item", "A bullet", 2),
        ("table", "Fee | 25", 2),
    ]
    assert [page.number for page in parsed.pages] == [1, 2]


# --- HTML -----------------------------------------------------------------------------------

PAGE = b"""<!doctype html><html><head><title>Wire Policy</title><style>p{color:red}</style>
<script>alert('x')</script></head><body>
<nav><a href="/">Home</a> | <a href="/loans">Loans</a></nav>
<main><h1>Wires</h1><p>Call the number <b>on file</b>.</p>
<h2>Fees</h2><ul><li>Domestic</li><li>International</li></ul>
<table><thead><tr><th>Fee</th><th>Amount</th></tr></thead><tbody><tr><td>Wire</td><td>25</td></tr></tbody></table>
<pre>code  block</pre><img src="https://tracker.example/pixel.gif"></main>
<footer>&copy; 2026 Harbor Point</footer></body></html>"""


def test_html_parses_into_its_structure(tmp_path: Path) -> None:
    path = write(tmp_path, "page.html", PAGE)
    parsed, _ = parse_file("parse/html", path)
    assert parsed.title == "Wire Policy"
    assert [(e.type, e.text, e.level) for e in parsed.elements] == [
        ("page_header", "Home | Loans", None),
        ("heading", "Wires", 1),
        ("paragraph", "Call the number on file.", None),
        ("heading", "Fees", 2),
        ("list_item", "Domestic", None),
        ("list_item", "International", None),
        ("table", "Fee | Amount\nWire | 25", None),
        ("code", "code  block", None),
        ("page_footer", "© 2026 Harbor Point", None),
    ]
    assert "alert" not in parsed.text
    assert "tracker" not in parsed.text
    assert [cell.header for cell in parsed.elements[6].table.cells] == [True, True, False, False]


def test_tika_xhtml_reads_pages_and_structure(tmp_path: Path) -> None:
    xhtml = (
        '<html><body><div class="page"><h1>Wires</h1><p>Call back.</p></div>'
        '<div class="page"><p>Page two.</p><table><tr><td>a</td><td>b</td></tr></table></div>'
        "</body></html>"
    )
    seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, text=xhtml)

    client = TikaClient(
        "http://tika",
        client=httpx2.Client(base_url="http://tika", transport=httpx2.MockTransport(handler)),
    )
    path = tmp_path / "wire.pdf"
    shutil.copy(ROOT / "fixtures" / "formats" / "wholesale.pdf", path)
    parsed, _ = parse_file(
        "parse/tika", path, {"ocr": True, "ocr_language": "eng+spa"}, tika=client
    )
    assert [(e.type, e.text, e.page) for e in parsed.elements] == [
        ("heading", "Wires", 1),
        ("paragraph", "Call back.", 1),
        ("paragraph", "Page two.", 2),
        ("table", "a | b", 2),
    ]
    request = seen[0]
    assert request.headers["accept"] == "text/html"
    assert "x-tika-ocrskipocr" not in request.headers
    assert request.headers["x-tika-ocrlanguage"] == "eng+spa"
    assert tika_pages("<p>no pages</p>") == []


# --- hosted adapters and the license gate ---------------------------------------------------

HOSTED = [
    "parse/azure-document-intelligence",
    "parse/aws-textract",
    "parse/google-document-ai",
    "parse/llamaparse",
    "parse/mistral-ocr",
]


@pytest.mark.parametrize("parser_id", HOSTED)
def test_hosted_adapters_refuse_unless_allow_listed(parser_id: str, tmp_path: Path) -> None:
    """Hosted adapters refuse to run when not allow-listed, before any byte leaves."""
    requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={})

    access = ProviderAccess(
        "https://provider.example",
        "key",
        client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    component = REGISTRY.get(parser_id)
    provider = component.provider
    path = tmp_path / "wire.pdf"
    shutil.copy(ROOT / "fixtures" / "formats" / "wholesale.pdf", path)
    for services in (
        {},
        {"providers_allowed": frozenset({"someone-else"}), f"provider:{provider}": access},
    ):
        with pytest.raises(ComponentError) as error:
            parse_file(parser_id, path, **services)
        assert error.value.code == "provider_not_allowed"
    assert requests == []
    assert component.info.egress == f"provider:{provider}"


def test_mistral_ocr_maps_pages_of_markdown(tmp_path: Path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        assert body["document"]["document_url"].startswith("data:application/pdf;base64,")
        assert request.headers["authorization"] == "Bearer key"
        return httpx2.Response(
            200,
            json={
                "pages": [
                    {"index": 0, "markdown": "# Wires\n\nCall back."},
                    {"index": 1, "markdown": "Page two."},
                ]
            },
        )

    access = ProviderAccess(
        None, "key", client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    path = tmp_path / "wire.pdf"
    shutil.copy(ROOT / "fixtures" / "formats" / "wholesale.pdf", path)
    parsed, context = parse_file(
        "parse/mistral-ocr",
        path,
        providers_allowed=frozenset({"mistral"}),
        **{"provider:mistral": access},
    )
    assert [(e.type, e.text, e.page) for e in parsed.elements] == [
        ("title", "Wires", 1),
        ("paragraph", "Call back.", 1),
        ("paragraph", "Page two.", 2),
    ]
    assert context.metrics["mistral.calls"] == 1


def test_azure_document_intelligence_polls_for_its_markdown(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request.method)
        if request.method == "POST":
            assert request.headers["ocp-apim-subscription-key"] == "key"
            return httpx2.Response(202, headers={"operation-location": "https://az.example/op/1"})
        if len(calls) < 3:
            return httpx2.Response(200, json={"status": "running"})
        return httpx2.Response(
            200,
            json={
                "status": "succeeded",
                "analyzeResult": {
                    "content": "## Fees\n\n| Fee | Amount |\n| --- | --- |\n| Wire | 25 |"
                },
            },
        )

    access = ProviderAccess(
        "https://az.example",
        "key",
        client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        sleep=lambda seconds: None,
    )
    path = tmp_path / "wire.pdf"
    shutil.copy(ROOT / "fixtures" / "formats" / "wholesale.pdf", path)
    services = {
        "providers_allowed": frozenset({"azure-document-intelligence"}),
        "provider:azure-document-intelligence": access,
    }
    parsed, _ = parse_file("parse/azure-document-intelligence", path, **services)
    assert calls == ["POST", "GET", "GET"]
    assert [e.type for e in parsed.elements] == ["heading", "table"]


def test_google_document_ai_maps_paragraph_anchors(tmp_path: Path) -> None:
    text = "Wires\nCall back.\n"

    def handler(request: httpx2.Request) -> httpx2.Response:
        content = json.loads(request.content)["rawDocument"]["content"]
        assert base64.b64decode(content)[:5] == b"%PDF-"
        paragraphs = [
            {"layout": {"textAnchor": {"textSegments": [{"startIndex": 0, "endIndex": 5}]}}},
            {"layout": {"textAnchor": {"textSegments": [{"startIndex": 6, "endIndex": 16}]}}},
        ]
        return httpx2.Response(
            200, json={"document": {"text": text, "pages": [{"paragraphs": paragraphs}]}}
        )

    access = ProviderAccess(
        "https://docai.example/process",
        "token",
        client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    path = tmp_path / "wire.pdf"
    shutil.copy(ROOT / "fixtures" / "formats" / "wholesale.pdf", path)
    services = {
        "providers_allowed": frozenset({"google-document-ai"}),
        "provider:google-document-ai": access,
    }
    parsed, _ = parse_file("parse/google-document-ai", path, **services)
    assert [(e.text, e.page) for e in parsed.elements] == [("Wires", 1), ("Call back.", 1)]


def test_pymupdf4llm_waits_for_its_license(tmp_path: Path) -> None:
    component = REGISTRY.get("parse/pymupdf4llm")
    assert component.info.license == "AGPL-3.0"
    assert component.license_gate == "AGPL-3.0"
    path = tmp_path / "wire.pdf"
    shutil.copy(ROOT / "fixtures" / "formats" / "wholesale.pdf", path)
    with pytest.raises(ComponentError) as error:
        parse_file("parse/pymupdf4llm", path)
    assert error.value.code in ("extra_missing", "license_not_allowed")
    if component.unavailable() is None:
        with pytest.raises(ComponentError) as error:
            parse_file("parse/pymupdf4llm", path)
        assert error.value.code == "license_not_allowed"


# --- parse quality ---------------------------------------------------------------------------


def test_parse_quality_metrics() -> None:
    document = ParsedDocument.model_validate(
        {
            "document_id": "d",
            "parser": {"id": "parse/test", "version": "1.0.0", "params_hash": "0" * 64},
            "media_type": "application/pdf",
            "pages": [{"number": 1}, {"number": 2}, {"number": 3}],
            "elements": [
                {
                    "id": "e0",
                    "type": "heading",
                    "text": "Fees",
                    "page": 1,
                    "char_start": 0,
                    "char_end": 4,
                },
                {
                    "id": "e1",
                    "type": "table",
                    "text": "a | b�",
                    "page": 1,
                    "char_start": 6,
                    "char_end": 12,
                },
                {
                    "id": "e2",
                    "type": "paragraph",
                    "text": "Body",
                    "page": 3,
                    "char_start": 14,
                    "char_end": 18,
                },
            ],
            "text": "Fees\n\na | b�\n\nBody",
        }
    )
    quality = parse_quality(document)
    assert quality == {
        "chars_per_page_p50": 4.0,
        "empty_page_rate": pytest.approx(1 / 3),
        "headings": 1,
        "tables": 1,
        "garbage_char_ratio": round(1 / 18, 6),
    }
    assert garbage_ratio("clean text\n\t") == 0.0
    assert garbage_ratio("\x00") == 1.0


def test_parse_metrics_reach_the_run_record(tmp_path: Path) -> None:
    path = tmp_path / "fees.csv"
    path.write_bytes(b"Fee,Amount\nWire,25\n")
    _, context = parse_file("parse/spreadsheet", path)
    assert {"elements", "chars", "headings", "tables", "garbage_char_ratio"} <= set(context.metrics)
