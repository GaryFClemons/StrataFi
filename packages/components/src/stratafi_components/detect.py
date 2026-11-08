"""Type detection: what a file is, judged from its bytes, and whether that agrees with its name
(``extract/upload``).

- ``sniff`` names what a file's first bytes are: a signature table for ZIP, images, OLE and
  gzip, a PDF header anywhere in the first kilobyte, then text or binary by whether the first
  8 KiB decode to printable text, and markup (XML, SVG, HTML) by how that text starts.
- A ZIP is refined by its central directory: DOCX, XLSX and PPTX are ZIPs with known parts.
- The extension must agree with the bytes, by the table of what each declared type may be
  (``ALLOWED``): a ``.pdf`` must be a PDF, a ``.md`` any text. A file whose bytes and extension
  disagree is a mismatch; an extension Stratafi doesn't list is unknown. Neither is an error
  here: the scan stage decides what they mean.
- Traits are cheap probes (``stratafi_components.taxonomy`` lists them): a PDF's page count,
  ``scanned`` when none of its first pages has a text layer, ``partly_scanned`` when some
  don't, and encryption; an Office file's pages, sheets or slides and whether it holds
  tables or images; an image's size. (Trait names avoid words like "text", which the metadata
  policy reserves for content.) They read a few pages or parts at most, and a file that can't
  be probed is ``unreadable``, never an exception.

Nothing here needs a network or a model, and nothing reads more than it must: the probes take a
path, so a 200 MB file is never loaded into memory.
"""

import codecs
import logging
import re
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger("stratafi.components.detect")

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
EML = "message/rfc822"
MSG = "application/vnd.ms-outlook"
ZIP = "application/zip"
OCTET_STREAM = "application/octet-stream"
HEAD_BYTES = 8192
TEXT_LAYER_PAGES = 3
# The most of an Office part a trait probe reads.
PART_BYTES = 8 * 1024 * 1024

# Extension → the media type it declares.
EXTENSIONS: dict[str, str] = {
    ".pdf": PDF,
    ".docx": DOCX,
    ".xlsx": XLSX,
    ".pptx": PPTX,
    ".doc": "application/msword",
    ".xls": "application/vnd.ms-excel",
    ".ppt": "application/vnd.ms-powerpoint",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".xml": "application/xml",
    ".eml": EML,
    ".msg": MSG,
    ".zip": ZIP,
    ".gz": "application/gzip",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}
_MARKUP = frozenset({"text", "html", "xml", "svg"})
# Declared type → the sniffed kinds its bytes may be. Text formats may be any text or markup;
# OOXML may also be OLE: an encrypted Office file is a compound file holding an
# ``EncryptedPackage``.
ALLOWED: dict[str, frozenset[str]] = {
    PDF: frozenset({"pdf"}),
    DOCX: frozenset({"docx", "ole"}),
    XLSX: frozenset({"xlsx", "ole"}),
    PPTX: frozenset({"pptx", "ole"}),
    "application/msword": frozenset({"ole"}),
    "application/vnd.ms-excel": frozenset({"ole"}),
    "application/vnd.ms-powerpoint": frozenset({"ole"}),
    "text/markdown": _MARKUP,
    "text/plain": _MARKUP,
    "text/csv": _MARKUP,
    "text/tab-separated-values": _MARKUP,
    "application/json": _MARKUP,
    "text/html": _MARKUP,
    "application/xml": _MARKUP,
    EML: _MARKUP,
    MSG: frozenset({"ole"}),
    ZIP: frozenset({"zip"}),
    "application/gzip": frozenset({"gzip"}),
    "image/png": frozenset({"png"}),
    "image/jpeg": frozenset({"jpeg"}),
    "image/gif": frozenset({"gif"}),
    "image/webp": frozenset({"webp"}),
    "image/svg+xml": frozenset({"svg", "xml"}),
}
# Sniffed kind → the media type the bytes are, when the name says nothing usable.
KIND_TYPES: dict[str, str] = {
    "pdf": PDF,
    "docx": DOCX,
    "xlsx": XLSX,
    "pptx": PPTX,
    "zip": ZIP,
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "gzip": "application/gzip",
    "html": "text/html",
    "svg": "image/svg+xml",
    "xml": "application/xml",
    "text": "text/plain",
    "ole": "application/x-ole-storage",
    "binary": OCTET_STREAM,
}
_OOXML_PARTS = (
    ("word/document.xml", "docx"),
    ("xl/workbook.xml", "xlsx"),
    ("ppt/presentation.xml", "pptx"),
)
# File signatures: a kind and the bytes it starts with, as (offset, bytes) pairs that must all
# match. A WebP is a RIFF container whose form type, at byte 8, is WEBP.
_SIGNATURES: tuple[tuple[str, tuple[tuple[int, bytes], ...]], ...] = (
    ("zip", ((0, b"PK\x03\x04"),)),  # a local file header: almost every ZIP
    ("zip", ((0, b"PK\x05\x06"),)),  # an empty archive: only its end of central directory
    ("zip", ((0, b"PK\x07\x08"),)),  # a spanned archive's marker
    ("png", ((0, b"\x89PNG\r\n\x1a\n"),)),
    ("jpeg", ((0, b"\xff\xd8\xff"),)),
    ("gif", ((0, b"GIF87a"),)),
    ("gif", ((0, b"GIF89a"),)),
    ("webp", ((0, b"RIFF"), (8, b"WEBP"))),
    ("ole", ((0, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"),)),
    ("gzip", ((0, b"\x1f\x8b"),)),
)
# PDF readers find the header anywhere in the first kilobyte, so a sniffer has to as well.
PDF_HEADER_WINDOW = 1024
# Text holds at most one unprintable character in twenty; more means binary.
_UNPRINTABLE_SHARE = 0.05
_TEXT_CONTROLS = frozenset("\t\n\r\f\v\x1b")
_BYTE_ORDER_MARKS = (
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)
_LEADING_COMMENTS = re.compile(r"(?:\s|<!--.*?-->)*", re.DOTALL)
_SVG_TAG = re.compile(r"<svg[\s/>]")
# Tags that start an HTML document, or make one active wherever they appear first.
_HTML_TAG = re.compile(
    r"<(?:!doctype\s+html|html|head|body|title|meta|link|base|style|script|iframe|frameset"
    r"|object|embed)[\s/>]"
)
_ENCRYPTED_PACKAGE = "EncryptedPackage".encode("utf-16-le")


def _signature(head: bytes) -> str | None:
    for kind, parts in _SIGNATURES:
        if all(head[offset : offset + len(expected)] == expected for offset, expected in parts):
            return kind
    return None


def _as_text(head: bytes) -> str | None:
    """``head`` decoded, or None when it isn't text: a NUL byte outside UTF-16, or too many
    characters that aren't printable or ordinary whitespace."""
    for mark, codec in _BYTE_ORDER_MARKS:
        if head.startswith(mark):
            return head[len(mark) :].decode(codec, errors="replace")
    if b"\x00" in head:
        return None
    try:
        # Not final: the head may stop in the middle of a character.
        text = codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
    except UnicodeDecodeError:
        text = head.decode("cp1252", errors="replace")
    unprintable = sum(1 for char in text if not char.isprintable() and char not in _TEXT_CONTROLS)
    return None if unprintable > _UNPRINTABLE_SHARE * len(text) else text


def _markup(text: str) -> str:
    """Which markup ``text`` is, by how it starts after whitespace and comments: xml, svg,
    html, or plain text."""
    skipped = _LEADING_COMMENTS.match(text)
    start = text[skipped.end() if skipped else 0 :].lower()
    if start.startswith("<?xml"):
        return "svg" if _SVG_TAG.search(start, 0, 2048) else "xml"
    if start.startswith("<!doctype svg") or _SVG_TAG.match(start):
        return "svg"
    if _HTML_TAG.match(start):
        return "html"
    return "text"


def sniff(head: bytes) -> str:
    """What a file is, from its first bytes: a signature's kind, ``pdf``, ``binary``, or for
    text, ``xml``, ``svg``, ``html`` or ``text``."""
    kind = _signature(head)
    if kind is not None:
        return kind
    head = head[:HEAD_BYTES]
    if b"%PDF-" in head[:PDF_HEADER_WINDOW]:
        return "pdf"
    text = _as_text(head)
    return "binary" if text is None else _markup(text)


def extension_of(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).suffix.lower()


@dataclass(frozen=True)
class Detection:
    """What a file is. ``media_type`` is what its bytes are; ``declared_type`` what its name
    says. ``mismatch``: they disagree. ``known``: the extension is one Stratafi lists."""

    media_type: str
    kind: str
    extension: str
    declared_type: str | None
    mismatch: bool
    known: bool
    traits: dict[str, Any] = field(default_factory=dict)

    @property
    def reason(self) -> str | None:
        if self.mismatch:
            return "type_mismatch"
        if not self.known:
            return "unknown_type"
        return None

    @property
    def trait_names(self) -> frozenset[str]:
        """The traits that are true, for parse routes such as ``pdf[scanned]``."""
        return frozenset(name for name, value in self.traits.items() if value is True)

    @property
    def is_archive(self) -> bool:
        return self.kind == "zip" and not self.mismatch

    @property
    def is_message(self) -> bool:
        """An email message whose bytes agree with its name: ``unpack/email`` opens it."""
        return self.media_type in (EML, MSG) and not self.mismatch

    def as_dict(self) -> dict[str, Any]:
        return {
            "media_type": self.media_type,
            "kind": self.kind,
            "extension": self.extension,
            "declared_type": self.declared_type,
            "mismatch": self.mismatch,
            "known": self.known,
            "traits": dict(self.traits),
        }


def _read_part(archive: zipfile.ZipFile, name: str) -> bytes:
    with archive.open(name) as handle:
        return handle.read(PART_BYTES)


def _ooxml_traits(archive: zipfile.ZipFile, kind: str, names: set[str]) -> dict[str, Any]:
    """An Office file's shape, from its parts: pages, sheets or slides; tables; images."""
    traits: dict[str, Any] = {
        "images": any(name.startswith(("word/media/", "xl/media/", "ppt/media/")) for name in names)
    }
    if kind == "docx":
        body = _read_part(archive, "word/document.xml")
        traits["tables"] = b"<w:tbl>" in body or b"<w:tbl " in body
        app = _read_part(archive, "docProps/app.xml") if "docProps/app.xml" in names else b""
        if match := re.search(rb"<Pages>(\d{1,6})</Pages>", app):
            traits["pages"] = int(match.group(1))
    elif kind == "xlsx":
        traits["sheets"] = sum(
            1 for name in names if re.fullmatch(r"xl/worksheets/[^/]+\.xml", name)
        )
        traits["tables"] = True
    elif kind == "pptx":
        traits["slides"] = sum(1 for name in names if re.fullmatch(r"ppt/slides/[^/]+\.xml", name))
        traits["tables"] = any(
            b"<a:tbl>" in _read_part(archive, name)
            for name in sorted(names)
            if re.fullmatch(r"ppt/slides/[^/]+\.xml", name)
        )
    return traits


def _zip_kind(path: Path) -> tuple[str, dict[str, Any]]:
    """DOCX, XLSX or PPTX by their parts, else a plain ZIP. Reads the central directory, and for
    Office files the parts their traits need."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            for part, kind in _OOXML_PARTS:
                if part in names:
                    try:
                        return kind, _ooxml_traits(archive, kind, names)
                    except (KeyError, zipfile.BadZipFile, OSError, ValueError, EOFError):
                        return kind, {"unreadable": True}
            return "zip", {"entries": len(names)}
    except (zipfile.BadZipFile, OSError, ValueError):
        return "zip", {"unreadable": True}


def _pdf_traits(path: Path) -> dict[str, Any]:
    import pypdf

    traits: dict[str, Any] = {}
    try:
        reader = pypdf.PdfReader(str(path))
        if reader.is_encrypted:
            traits["encrypted"] = True
            # An empty user password still opens many "encrypted" PDFs; count pages if so.
            if not reader.decrypt(""):
                return traits
        pages = reader.pages
        traits["pages"] = len(pages)
        # A PDF without a text layer in its first pages is a scan that needs OCR; one where
        # only some of them lack it is partly scanned.
        layers = [bool(page.extract_text().strip()) for page in list(pages)[:TEXT_LAYER_PAGES]]
        traits["scanned"] = bool(layers) and not any(layers)
        traits["partly_scanned"] = any(layers) and not all(layers)
    except Exception as exc:  # untrusted input: any failure is a trait, not an error
        logger.info("pdf probe failed", extra={"error_type": type(exc).__name__})
        traits["unreadable"] = True
    return traits


def _image_traits(path: Path, kind: str) -> dict[str, Any]:
    """An image's width and height in pixels, from its header."""
    with path.open("rb") as handle:
        head = handle.read(64 * 1024)
    try:
        if kind == "png" and head[12:16] == b"IHDR":
            width, height = struct.unpack(">II", head[16:24])
            return {"width": width, "height": height}
        if kind == "gif":
            width, height = struct.unpack("<HH", head[6:10])
            return {"width": width, "height": height}
        if kind == "jpeg":
            position = 2
            while position + 9 < len(head):
                if head[position] != 0xFF:
                    break
                marker = head[position + 1]
                length = struct.unpack(">H", head[position + 2 : position + 4])[0]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    height, width = struct.unpack(">HH", head[position + 5 : position + 9])
                    return {"width": width, "height": height}
                position += 2 + length
    except struct.error:
        pass
    return {}


_MARKDOWN_RULE = re.compile(rb"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)


def _markup_traits(path: Path, media_type: str) -> dict[str, Any]:
    """Whether markup or text holds tables: HTML tables, Markdown table rules, or delimited
    rows."""
    if media_type in ("text/csv", "text/tab-separated-values"):
        return {"tables": True}
    with path.open("rb") as handle:
        head = handle.read(PART_BYTES)
    if media_type == "text/html":
        return {"tables": b"<table" in head.lower()}
    if media_type == "text/markdown":
        return {"tables": _MARKDOWN_RULE.search(head) is not None}
    return {}


def _ole_traits(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        head = handle.read(64 * 1024)
    return {"encrypted": True} if _ENCRYPTED_PACKAGE in head else {}


def detect(path: Path, name: str) -> Detection:
    """Detect the file at ``path``, which arrived as ``name``."""
    with path.open("rb") as handle:
        head = handle.read(HEAD_BYTES)
    kind = sniff(head)
    traits: dict[str, Any] = {}
    if kind == "zip":
        kind, traits = _zip_kind(path)
    elif kind == "pdf":
        traits = _pdf_traits(path)
    elif kind == "ole":
        traits = _ole_traits(path)
    elif kind in ("png", "jpeg", "gif"):
        traits = _image_traits(path, kind)
    extension = extension_of(name)
    declared = EXTENSIONS.get(extension)
    if declared is None:
        media_type, mismatch, known = KIND_TYPES[kind], False, False
    else:
        mismatch = kind not in ALLOWED[declared]
        media_type, known = (KIND_TYPES[kind] if mismatch else declared), True
    if kind in _MARKUP:
        traits = _markup_traits(path, media_type)
    return Detection(media_type, kind, extension, declared, mismatch, known, traits)


def detect_bytes(data: bytes, name: str, scratch: Path) -> Detection:
    """``detect`` for bytes in memory (tests and small files): written to ``scratch`` first."""
    path = scratch / "detect.bin"
    path.write_bytes(data)
    try:
        return detect(path, name)
    finally:
        path.unlink(missing_ok=True)


def duplicate_groups(entries: list[tuple[str, str]]) -> list[list[str]]:
    """Names that share content, from ``(name, sha256)`` pairs: one group per SHA-256 with more
    than one name, each group sorted, groups in order of their first name. The first name of a
    group is the one ``scan/dedupe`` keeps."""
    by_hash: dict[str, list[str]] = {}
    for name, sha256 in entries:
        by_hash.setdefault(sha256, []).append(name)
    groups = [sorted(names) for names in by_hash.values() if len(names) > 1]
    return sorted(groups, key=lambda group: group[0])
