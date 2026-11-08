"""Type detection: magic bytes against the extension, unknown types, traits."""

from pathlib import Path

import pytest

from stratafi_components.detect import (
    DOCX,
    PDF,
    XLSX,
    detect_bytes,
    duplicate_groups,
    sniff,
)
from stratafi_testing.documents import (
    Block,
    blank_pdf_bytes,
    docx_bytes,
    encrypted_pdf_bytes,
    pdf_bytes,
    zip_bytes,
)

BLOCKS = [Block("heading", "1 Purpose", 1), Block("paragraph", "Keep records for seven years.")]
PDF_DATA = pdf_bytes("Retention", BLOCKS)
DOCX_DATA = docx_bytes("Fees", BLOCKS)
XLSX_DATA = zip_bytes({"[Content_Types].xml": b"<Types/>", "xl/workbook.xml": b"<workbook/>"})
PLAIN_ZIP = zip_bytes({"a.md": b"# A\n"})
PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 32
OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 64


# name, bytes → media type, kind, mismatch, known
TABLE = [
    ("policy.pdf", PDF_DATA, PDF, "pdf", False, True),
    ("fees.docx", DOCX_DATA, DOCX, "docx", False, True),
    ("sheet.xlsx", XLSX_DATA, XLSX, "xlsx", False, True),
    ("bundle.zip", PLAIN_ZIP, "application/zip", "zip", False, True),
    ("notes.md", b"# Notes\n\nPlain text.", "text/markdown", "text", False, True),
    ("notes.txt", b"Plain text.\n", "text/plain", "text", False, True),
    ("data.csv", b"a,b\n1,2\n", "text/csv", "text", False, True),
    ("page.html", b"<!doctype html><p>Hi</p>", "text/html", "html", False, True),
    ("notes.md", b"<html><body>markup is text</body></html>", "text/markdown", "html", False, True),
    ("logo.png", PNG, "image/png", "png", False, True),
    ("old.doc", OLE, "application/msword", "ole", False, True),
    # Mismatches: the bytes say one thing, the extension another.
    ("policy.pdf", b"Not a PDF at all.", "text/plain", "text", True, True),
    ("fees.docx", PDF_DATA, PDF, "pdf", True, True),
    ("fees.docx", PLAIN_ZIP, "application/zip", "zip", True, True),
    ("bundle.zip", DOCX_DATA, DOCX, "docx", True, True),
    ("notes.txt", PDF_DATA, PDF, "pdf", True, True),
    ("notes.md", b"\x00\x01\x02binary", "application/octet-stream", "binary", True, True),
    ("logo.png", b"GIF89a....", "image/gif", "gif", True, True),
    # Unknown extensions: typed by their bytes, flagged unknown.
    ("archive.rar", b"Rar!\x1a\x07\x00", "application/octet-stream", "binary", False, False),
    (
        "program.exe",
        b"MZ\x90\x00\x03\x00\x00\x00\x00\x00",
        "application/octet-stream",
        "binary",
        False,
        False,
    ),
    ("README", b"Just words.", "text/plain", "text", False, False),
    ("scan.PDF", PDF_DATA, PDF, "pdf", False, True),
]


@pytest.mark.parametrize(
    ("name", "data", "media_type", "kind", "mismatch", "known"),
    TABLE,
    ids=[f"{i}-{row[0]}" for i, row in enumerate(TABLE)],
)
def test_the_type_detection_table(
    tmp_path: Path, name: str, data: bytes, media_type: str, kind: str, mismatch: bool, known: bool
) -> None:
    detection = detect_bytes(data, name, tmp_path)
    assert (detection.media_type, detection.kind) == (media_type, kind)
    assert (detection.mismatch, detection.known) == (mismatch, known)
    expected_reason = "type_mismatch" if mismatch else (None if known else "unknown_type")
    assert detection.reason == expected_reason


def test_pdf_headers_count_anywhere_in_the_first_kilobyte() -> None:
    assert sniff(b"\n" * 500 + b"%PDF-1.7") == "pdf"
    assert sniff(b"\n" * 1100 + b"%PDF-1.7") == "text"


def test_text_with_many_control_characters_is_binary() -> None:
    assert sniff(b"\x01\x02\x03\x04 words") == "binary"
    assert sniff("Ünïcode text, with a BOM".encode("utf-8-sig")) == "text"
    assert sniff("utf-16 text".encode("utf-16")) == "text"
    assert sniff(b'<?xml version="1.0"?><svg xmlns="x"></svg>') == "svg"


def test_a_pdf_with_text_has_a_text_layer_and_pages(tmp_path: Path) -> None:
    long = [Block("paragraph", "Text. " * 60) for _ in range(30)]
    detection = detect_bytes(pdf_bytes("Long", long), "long.pdf", tmp_path)
    assert detection.traits["scanned"] is False
    assert detection.traits["pages"] > 3
    assert "scanned" not in detection.trait_names


def test_a_pdf_without_text_is_scanned(tmp_path: Path) -> None:
    detection = detect_bytes(blank_pdf_bytes(2), "scan.pdf", tmp_path)
    assert detection.traits == {"pages": 2, "scanned": True, "partly_scanned": False}
    assert detection.trait_names == {"scanned"}


def test_office_files_report_their_shape(tmp_path: Path) -> None:
    from stratafi_testing.documents import docx_table

    plain = detect_bytes(DOCX_DATA, "fees.docx", tmp_path)
    assert plain.traits == {"images": False, "tables": False}
    with_table = docx_bytes("Fees", BLOCKS, extra_body_xml=docx_table([["Fee", "Amount"]]))
    assert detect_bytes(with_table, "fees.docx", tmp_path).traits["tables"] is True
    workbook = zip_bytes(
        {
            "xl/workbook.xml": b"<workbook/>",
            "xl/worksheets/sheet1.xml": b"<worksheet/>",
            "xl/worksheets/sheet2.xml": b"<worksheet/>",
            "xl/media/image1.png": PNG,
        }
    )
    assert detect_bytes(workbook, "book.xlsx", tmp_path).traits == {
        "images": True,
        "sheets": 2,
        "tables": True,
    }
    deck = zip_bytes(
        {
            "ppt/presentation.xml": b"<presentation/>",
            "ppt/slides/slide1.xml": b"<p:sld><a:tbl></a:tbl></p:sld>",
            "ppt/slides/slide2.xml": b"<p:sld/>",
        }
    )
    assert detect_bytes(deck, "deck.pptx", tmp_path).traits == {
        "images": False,
        "slides": 2,
        "tables": True,
    }


def test_markup_and_images_report_their_shape(tmp_path: Path) -> None:
    import struct

    table_md = b"# Fees\n\n| Fee | Amount |\n| --- | ---: |\n| Wire | 25 |\n"
    assert detect_bytes(table_md, "fees.md", tmp_path).traits == {"tables": True}
    assert detect_bytes(b"# Fees\n\n- one\n", "fees.md", tmp_path).traits == {"tables": False}
    html = b"<!doctype html><table><tr><td>a</td></tr></table>"
    assert detect_bytes(html, "page.html", tmp_path).traits == {"tables": True}
    assert detect_bytes(b"a,b\n1,2\n", "data.csv", tmp_path).traits == {"tables": True}
    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 640, 480)
    assert detect_bytes(png + b"\0" * 16, "scan.png", tmp_path).traits == {
        "width": 640,
        "height": 480,
    }
    gif = b"GIF89a" + struct.pack("<HH", 32, 16) + b"\0" * 8
    assert detect_bytes(gif, "icon.gif", tmp_path).traits == {"width": 32, "height": 16}


def test_the_taxonomy_names_families_and_route_keys() -> None:
    from stratafi_components.taxonomy import FILE_TYPES, family_of, route_key, short_name

    assert family_of(PDF) == "document"
    assert family_of(XLSX) == "spreadsheet"
    assert family_of("message/rfc822") == "email"
    assert family_of("application/x-unknown") == "other"
    assert short_name(DOCX) == "docx"
    assert route_key(PDF, {"pages": 3, "scanned": True, "encrypted": False}) == "pdf[scanned]"
    assert route_key("text/markdown", {}) == "md"
    # Every short name a route can use resolves back to its media type.
    from stratafi_core.specs import route_key_type

    for item in FILE_TYPES:
        if item.family != "other":
            assert route_key_type(item.short)[0] == item.media_type, item.short


def test_an_encrypted_pdf_says_so(tmp_path: Path) -> None:
    detection = detect_bytes(encrypted_pdf_bytes(), "locked.pdf", tmp_path)
    assert detection.traits["encrypted"] is True
    assert detection.media_type == PDF


def test_a_broken_pdf_is_unreadable_not_an_error(tmp_path: Path) -> None:
    detection = detect_bytes(b"%PDF-1.4\nthis is not a real pdf", "broken.pdf", tmp_path)
    assert detection.kind == "pdf"
    assert detection.traits.get("unreadable") is True


def test_an_encrypted_office_file_is_ole_with_an_encrypted_package(tmp_path: Path) -> None:
    data = OLE + "EncryptedPackage".encode("utf-16-le") + b"\0" * 16
    detection = detect_bytes(data, "secret.docx", tmp_path)
    assert detection.kind == "ole"
    assert detection.mismatch is False
    assert detection.traits == {"encrypted": True}


def test_duplicates_are_found() -> None:
    entries = [
        ("b/copy.md", "1" * 64),
        ("a/original.md", "1" * 64),
        ("unique.pdf", "2" * 64),
        ("z.txt", "3" * 64),
        ("y.txt", "3" * 64),
        ("x.txt", "3" * 64),
    ]
    assert duplicate_groups(entries) == [
        ["a/original.md", "b/copy.md"],
        ["x.txt", "y.txt", "z.txt"],
    ]
    assert duplicate_groups([("only.md", "1" * 64)]) == []
