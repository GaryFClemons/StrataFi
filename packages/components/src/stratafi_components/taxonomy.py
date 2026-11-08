"""The file-type taxonomy and its traits.

Detection (``stratafi_components.detect``) says what a file's bytes are: a media type and a
few traits. The taxonomy groups media types into families, the way people think about a
corpus (documents, spreadsheets, slides, email…), gives each a short name that parse routes
use (``pdf``, ``docx``), and describes every trait detection can set. The API serves it
(``GET /file-types``), so the UI explains types and traits without its own copy.

A trait that is ``true`` can select a parser: ``pdf[scanned]`` routes scanned PDFs to OCR
(``stratafi_core.specs.route_for``). Numeric traits (pages, sheets) describe a file; they never
route it.
"""

from dataclasses import dataclass
from typing import Literal

from stratafi_components.detect import DOCX, EML, MSG, PDF, PPTX, XLSX, ZIP

Family = Literal[
    "document",
    "spreadsheet",
    "presentation",
    "text",
    "web",
    "email",
    "image",
    "archive",
    "other",
]


@dataclass(frozen=True)
class FileType:
    media_type: str
    short: str  # the name parse routes use: pdf, docx, md
    family: Family
    title: str
    extensions: tuple[str, ...]


@dataclass(frozen=True)
class Trait:
    name: str
    kind: Literal["flag", "count"]
    applies_to: tuple[str, ...]  # short type names
    description: str


FILE_TYPES: tuple[FileType, ...] = (
    FileType(PDF, "pdf", "document", "PDF", (".pdf",)),
    FileType(DOCX, "docx", "document", "Word document", (".docx",)),
    FileType("application/msword", "doc", "document", "Word 97-2003 document", (".doc",)),
    FileType(XLSX, "xlsx", "spreadsheet", "Excel workbook", (".xlsx",)),
    FileType("application/vnd.ms-excel", "xls", "spreadsheet", "Excel 97-2003", (".xls",)),
    FileType("text/csv", "csv", "spreadsheet", "Comma-separated values", (".csv",)),
    FileType("text/tab-separated-values", "tsv", "spreadsheet", "Tab-separated", (".tsv",)),
    FileType(PPTX, "pptx", "presentation", "PowerPoint deck", (".pptx",)),
    FileType(
        "application/vnd.ms-powerpoint", "ppt", "presentation", "PowerPoint 97-2003", (".ppt",)
    ),
    FileType("text/markdown", "md", "text", "Markdown", (".md", ".markdown")),
    FileType("text/plain", "txt", "text", "Plain text", (".txt",)),
    FileType("application/json", "json", "text", "JSON", (".json",)),
    FileType("application/xml", "xml", "text", "XML", (".xml",)),
    FileType("text/html", "html", "web", "HTML page", (".html", ".htm")),
    FileType(EML, "eml", "email", "Email message", (".eml",)),
    FileType(MSG, "msg", "email", "Outlook message", (".msg",)),
    FileType("image/png", "png", "image", "PNG image", (".png",)),
    FileType("image/jpeg", "jpeg", "image", "JPEG image", (".jpg", ".jpeg")),
    FileType("image/gif", "gif", "image", "GIF image", (".gif",)),
    FileType("image/webp", "webp", "image", "WebP image", (".webp",)),
    FileType("image/svg+xml", "svg", "image", "SVG drawing", (".svg",)),
    FileType(ZIP, "zip", "archive", "ZIP archive", (".zip",)),
    FileType("application/gzip", "gz", "archive", "gzip file", (".gz",)),
    FileType("application/x-ole-storage", "ole", "other", "OLE compound file", ()),
    FileType("application/octet-stream", "binary", "other", "Unrecognized bytes", ()),
)

TRAITS: tuple[Trait, ...] = (
    Trait("pages", "count", ("pdf", "docx"), "Pages, as the file records them"),
    Trait(
        "scanned",
        "flag",
        ("pdf",),
        "None of the first pages has a text layer: the PDF is images and needs OCR",
    ),
    Trait(
        "partly_scanned",
        "flag",
        ("pdf",),
        "Some of the first pages have no text layer: parts need OCR",
    ),
    Trait("encrypted", "flag", ("pdf", "docx", "xlsx", "pptx", "doc"), "Password protected"),
    Trait(
        "unreadable", "flag", ("pdf", "docx", "xlsx", "pptx", "zip"), "The probe couldn't read it"
    ),
    Trait(
        "tables",
        "flag",
        ("docx", "xlsx", "pptx", "csv", "tsv", "html", "md"),
        "Holds tables (always, for spreadsheets)",
    ),
    Trait("images", "flag", ("docx", "xlsx", "pptx"), "Holds embedded images"),
    Trait("sheets", "count", ("xlsx",), "Worksheets"),
    Trait("slides", "count", ("pptx",), "Slides"),
    Trait("entries", "count", ("zip",), "Archive members"),
    Trait("width", "count", ("png", "jpeg", "gif"), "Pixels across"),
    Trait("height", "count", ("png", "jpeg", "gif"), "Pixels down"),
)

_BY_MEDIA_TYPE = {file_type.media_type: file_type for file_type in FILE_TYPES}


def file_type(media_type: str) -> FileType | None:
    return _BY_MEDIA_TYPE.get(media_type)


def family_of(media_type: str) -> Family:
    found = _BY_MEDIA_TYPE.get(media_type)
    return found.family if found is not None else "other"


def short_name(media_type: str) -> str:
    """The route name of a media type: ``pdf`` for ``application/pdf``; the media type itself
    when it has none."""
    found = _BY_MEDIA_TYPE.get(media_type)
    return found.short if found is not None else media_type


def route_key(media_type: str, traits: dict[str, object]) -> str:
    """The most specific parse route key for a document: its short type with every true trait,
    as ``pdf[scanned]``."""
    flags = sorted(name for name, value in traits.items() if value is True)
    base = short_name(media_type)
    return f"{base}[{','.join(flags)}]" if flags else base
