"""``parse/tika`` against a real Tika 3.2 server (integration tier): each format
fixture's text arrives with its structure, PDFs page by page, and OCR stays off unless asked.

The server is compose's ``tika`` service, or ``STRATAFI_TEST_TIKA_URL``.
"""

import shutil
from pathlib import Path

import pytest

from stratafi_components.detect import detect
from stratafi_components.extract import raw_document
from stratafi_components.parsers.tika import TikaClient, TikaXhtml
from stratafi_core.components import StageContext
from stratafi_testing.documents import blank_pdf_bytes

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "formats"
PHRASES = {
    "tips.pdf": "count the tip jar together",
    "food-safety.pdf": "Food Safety and Allergen Standard",
    "wholesale.docx": "Tier 3: Preferred",
    "tips.txt": "Tip Policy",
    "food-safety.md": "Food Safety and Allergen Standard",
}


@pytest.fixture
def tika(compose_services) -> TikaClient:
    client = TikaClient(compose_services("tika").tika_url())
    yield client
    client.close()


def parse(tika: TikaClient, path: Path, **params: object):
    raw = raw_document("doc-1", "0" * 64, path.stat().st_size, detect(path, path.name))
    parser = TikaXhtml()
    return raw, parser.parse(
        path, raw, parser.resolve_params(params), StageContext(services={"tika": tika})
    )


def test_the_server_is_tika_3_2(tika: TikaClient) -> None:
    assert tika.version().startswith("Apache Tika 3.2")


@pytest.mark.parametrize("name", sorted(PHRASES))
def test_each_format_arrives_with_its_text_and_structure(
    tika: TikaClient, name: str, tmp_path: Path
) -> None:
    """Tika's XHTML read as structure: the fixture's words, in elements whose offsets point
    into the text, a PDF's elements on their pages."""
    path = tmp_path / name
    shutil.copy(FIXTURES / name, path)
    raw, document = parse(tika, path)
    assert raw.media_type in TikaXhtml.info.accepts
    assert PHRASES[name].lower() in " ".join(document.text.split()).lower()
    assert len(document.elements) > 1
    for element in document.elements:
        assert document.text[element.char_start : element.char_end] == element.text
    if name.endswith(".pdf"):
        assert document.pages
        assert all(element.page is not None for element in document.elements)


def test_ocr_stays_off(tika: TikaClient, tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(blank_pdf_bytes(2))
    _, document = parse(tika, path)
    assert document.text == ""
    assert document.warnings == ["no_text"]
