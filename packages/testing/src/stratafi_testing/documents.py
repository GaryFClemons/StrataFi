"""Synthetic documents for tests and fixtures: Markdown, DOCX, PDF, ZIP, email and Outlook
messages, the same bytes every time. Everything they write is invented.

- ``docx_bytes`` writes the smallest WordprocessingML package Word and the parsers open: the
  content types, the package and document relationships, ``word/document.xml``, a styles part
  whose heading styles carry outline levels (so a reader sees heading levels), and core
  properties with the title. ``docx_table`` makes a table for its ``extra_body_xml``.
- ``pdf_bytes`` draws a text PDF with reportlab's canvas: the title and headings in
  Helvetica-Bold, larger by level, and paragraphs in Helvetica, wrapped at ``wrap``
  characters, on US Letter pages with one-inch margins. reportlab's invariant mode fixes the
  dates and the document ID.
- ``blank_pdf_bytes`` is pages without a text layer, what a scan looks like to a text
  extractor; ``encrypted_pdf_bytes`` needs a password to open.
"""

import io
import itertools
import textwrap
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from xml.sax.saxutils import escape

_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class Block:
    kind: Literal["heading", "paragraph"]
    text: str
    level: int = 0


def markdown_text(title: str, blocks: Sequence[Block]) -> str:
    parts = [f"# {title}"]
    for block in blocks:
        if block.kind == "heading":
            parts.append(f"{'#' * min(block.level + 1, 6)} {block.text}")
        else:
            parts.append(block.text)
    return "\n\n".join(parts) + "\n"


# --- DOCX -----------------------------------------------------------------------------------

_MAIN = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_PACKAGE = "http://schemas.openxmlformats.org/package/2006"
_DOCUMENT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_WORD_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml"
_DECLARATION = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'


def _xml(body: str) -> bytes:
    return (_DECLARATION + body).encode("utf-8")


def _relationships(*links: tuple[str, str, str]) -> bytes:
    """A relationships part: an (ID, type, target) per link."""
    items = "".join(
        f'<Relationship Id="{rid}" Type="{kind}" Target="{target}"/>' for rid, kind, target in links
    )
    return _xml(f'<Relationships xmlns="{_PACKAGE}/relationships">{items}</Relationships>')


def _content_types() -> bytes:
    overrides = {
        "/word/document.xml": f"{_WORD_TYPE}.document.main+xml",
        "/word/styles.xml": f"{_WORD_TYPE}.styles+xml",
        "/docProps/core.xml": "application/vnd.openxmlformats-package.core-properties+xml",
    }
    defaults = (
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
    )
    parts = "".join(
        f'<Override PartName="{part}" ContentType="{kind}"/>' for part, kind in overrides.items()
    )
    return _xml(f'<Types xmlns="{_PACKAGE}/content-types">{defaults}{parts}</Types>')


def _style(style_id: str, name: str, size: int, outline: int | None = None) -> str:
    """A bold paragraph style of ``size`` half-points; a heading's has its outline level."""
    outline_xml = "" if outline is None else f'<w:pPr><w:outlineLvl w:val="{outline}"/></w:pPr>'
    return (
        f'<w:style w:type="paragraph" w:styleId="{style_id}"><w:name w:val="{name}"/>'
        f'<w:basedOn w:val="Normal"/>{outline_xml}'
        f'<w:rPr><w:b/><w:sz w:val="{size}"/></w:rPr></w:style>'
    )


def _styles() -> bytes:
    """Normal, Title, and Heading1 to Heading6 at outline levels 0 to 5."""
    normal = (
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
        '<w:name w:val="Normal"/><w:rPr><w:sz w:val="22"/></w:rPr></w:style>'
    )
    headings = "".join(
        _style(f"Heading{level}", f"heading {level}", max(36 - 4 * level, 22), level - 1)
        for level in range(1, 7)
    )
    title = _style("Title", "Title", 44)
    return _xml(f'<w:styles xmlns:w="{_MAIN}">{normal}{title}{headings}</w:styles>')


def _paragraph(text: str, style: str | None = None) -> str:
    """A paragraph of one run; each line after the first follows a line break."""
    properties = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    lines = [f'<w:t xml:space="preserve">{escape(line)}</w:t>' for line in text.split("\n")]
    return f"<w:p>{properties}<w:r>{'<w:br/>'.join(lines)}</w:r></w:p>"


def docx_table(rows: Sequence[Sequence[str]]) -> str:
    """A table's XML, a row per row and a cell per cell, for ``docx_bytes``'s
    ``extra_body_xml``."""
    body = "".join(
        "<w:tr>" + "".join(f"<w:tc>{_paragraph(cell)}</w:tc>" for cell in row) + "</w:tr>"
        for row in rows
    )
    return f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/></w:tblPr>{body}</w:tbl>'


def docx_bytes(
    title: str,
    blocks: Sequence[Block],
    *,
    extra_body_xml: str = "",
    core_title: bool = True,
) -> bytes:
    """A DOCX: the title in the Title style, each heading in the Heading style of its level
    (1 to 6), each paragraph in Normal, then ``extra_body_xml`` (tables, tracked changes).
    ``core_title`` puts the title in the core properties too."""
    paragraphs = [_paragraph(title, "Title")]
    for block in blocks:
        style = f"Heading{min(max(block.level, 1), 6)}" if block.kind == "heading" else None
        paragraphs.append(_paragraph(block.text, style))
    document = _xml(
        f'<w:document xmlns:w="{_MAIN}"><w:body>{"".join(paragraphs)}{extra_body_xml}'
        "<w:sectPr/></w:body></w:document>"
    )
    title_xml = f"<dc:title>{escape(title)}</dc:title>" if core_title else ""
    core = _xml(
        f'<cp:coreProperties xmlns:cp="{_PACKAGE}/metadata/core-properties" '
        f'xmlns:dc="http://purl.org/dc/elements/1.1/">{title_xml}'
        "<dc:creator>Stratafi test documents</dc:creator></cp:coreProperties>"
    )
    return zip_bytes(
        {
            "[Content_Types].xml": _content_types(),
            "_rels/.rels": _relationships(
                ("rId1", f"{_DOCUMENT}/officeDocument", "word/document.xml"),
                ("rId2", f"{_PACKAGE}/relationships/metadata/core-properties", "docProps/core.xml"),
            ),
            "word/_rels/document.xml.rels": _relationships(
                ("rId1", f"{_DOCUMENT}/styles", "styles.xml")
            ),
            "word/document.xml": document,
            "word/styles.xml": _styles(),
            "docProps/core.xml": core,
        }
    )


# --- PDF ------------------------------------------------------------------------------------

_MARGIN = 72.0  # points: an inch
_BODY_FONT, _BOLD_FONT = "Helvetica", "Helvetica-Bold"
_BODY_SIZE, _LEADING, _PARAGRAPH_GAP = 10.5, 13.0, 8.0
_TITLE_SIZE = 18.0
_HEADING_SIZES = {1: 14.0, 2: 12.0}
_AUTHOR = "Stratafi test documents"


def _canvas(buffer: io.BytesIO, title: str | None) -> Any:
    """A US Letter canvas in invariant mode: fixed dates and document ID."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen.canvas import Canvas

    canvas = Canvas(buffer, pagesize=letter, invariant=1)
    if title:
        canvas.setTitle(title)
    canvas.setAuthor(_AUTHOR)
    canvas.setCreator(_AUTHOR)
    return canvas


def pdf_bytes(title: str, blocks: Sequence[Block], *, wrap: int = 88) -> bytes:
    """A text PDF: the title, then each heading and each paragraph wrapped at ``wrap``
    characters, on a new page whenever the last one is full."""
    from reportlab.lib.pagesizes import letter

    buffer = io.BytesIO()
    canvas = _canvas(buffer, title)
    top = letter[1] - _MARGIN
    y = top

    def write(text: str, font: str, size: float, space: float) -> None:
        """A line ``space`` points below the last one's baseline, on a new page if needed."""
        nonlocal y
        if y - space < _MARGIN:
            canvas.showPage()
            y = top - size
        else:
            y -= space
        canvas.setFont(font, size)
        canvas.drawString(_MARGIN, y, text)

    write(title, _BOLD_FONT, _TITLE_SIZE, _TITLE_SIZE)
    for block in blocks:
        if block.kind == "heading":
            size = _HEADING_SIZES.get(block.level, 11.0)
            write(block.text, _BOLD_FONT, size, size * 1.25 + _PARAGRAPH_GAP)
            continue
        lines = textwrap.wrap(" ".join(block.text.split()), width=wrap) or [""]
        for number, line in enumerate(lines):
            write(line, _BODY_FONT, _BODY_SIZE, _LEADING + (_PARAGRAPH_GAP if number == 0 else 0))
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def blank_pdf_bytes(pages: int = 1) -> bytes:
    """Pages with nothing on them: no text layer, as a scan looks without OCR."""
    buffer = io.BytesIO()
    canvas = _canvas(buffer, None)
    for _ in range(pages):
        canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def encrypted_pdf_bytes(title: str = "Locked", password: str = "synthetic-owner") -> bytes:  # noqa: S107
    """A PDF that needs a password to open."""
    import pypdf  # from stratafi-components, which every test environment installs

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes(title, [Block("paragraph", "Sealed text.")])))
    writer = pypdf.PdfWriter(clone_from=reader)
    writer.encrypt(user_password=password, owner_password=password)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


# --- ZIP ------------------------------------------------------------------------------------


def zip_bytes(members: Mapping[str, bytes], *, compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    """A ZIP of ``members`` (name → bytes) with fixed timestamps."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
            info.compress_type = compression
            archive.writestr(info, content)
    return buffer.getvalue()


# --- email ----------------------------------------------------------------------------------


def eml_bytes(
    subject: str,
    body: str,
    attachments: Mapping[str, tuple[str, bytes]] | None = None,
    *,
    html: str | None = None,
) -> bytes:
    """A MIME message with a plain body (or only ``html``) and attachments (name → (media
    type, bytes)), with a fixed date and boundary."""
    from email.message import EmailMessage
    from email.policy import SMTP

    message = EmailMessage(policy=SMTP)
    message["Subject"] = subject
    message["From"] = "Policy Desk <policy@example.org>"
    message["To"] = "staff@example.org"
    message["Date"] = "Fri, 30 Oct 2026 09:00:00 +0000"
    message["Message-ID"] = "<synthetic-1@example.org>"
    if html is not None and not body:
        message.set_content(html, subtype="html")
    else:
        message.set_content(body)
    for name, (media_type, data) in (attachments or {}).items():
        maintype, _, subtype = media_type.partition("/")
        message.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
    boundary = "===============synthetic=="
    if message.is_multipart():
        message.set_boundary(boundary)
    return message.as_bytes()


_SECTOR = 512
_MINI = 64
_FREE, _END, _FATSECT = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD


def compound_file(streams: Mapping[str, bytes]) -> bytes:
    """A minimal Compound File Binary (version 3) holding ``streams``, named by paths such as
    ``__attach_version1.0_#00000000/__substg1.0_37010102``: enough of the format for readers
    such as olefile. Streams under 4,096 bytes live in the mini stream, as the format requires.
    For tests only."""
    import struct

    # The directory: the root, then storages and streams, each storage's children chained as
    # right siblings in the format's name order (length, then upper case).
    entries: list[dict] = [{"name": "Root Entry", "type": 5, "children": [], "data": b""}]
    index: dict[str, int] = {"": 0}
    for path in sorted(streams):
        parts = path.split("/")
        for depth in range(1, len(parts) + 1):
            key = "/".join(parts[:depth])
            if key in index:
                continue
            is_stream = depth == len(parts)
            entries.append(
                {
                    "name": parts[depth - 1],
                    "type": 2 if is_stream else 1,
                    "children": [],
                    "data": streams[path] if is_stream else b"",
                }
            )
            index[key] = len(entries) - 1
            entries[index["/".join(parts[: depth - 1])]]["children"].append(index[key])

    mini, mini_fat = bytearray(), []
    big: list[tuple[int, bytes]] = []
    for number, entry in enumerate(entries):
        data = entry["data"]
        if entry["type"] != 2:
            continue
        if len(data) < 4096:
            entry["start"] = len(mini) // _MINI
            count = max(1, -(-len(data) // _MINI)) if data else 0
            if not data:
                entry["start"] = _END
            for offset in range(count):
                mini_fat.append(entry["start"] + offset + 1 if offset < count - 1 else _END)
            mini += data.ljust(count * _MINI, b"\0")
        else:
            big.append((number, data))

    sectors: list[bytes] = []
    fat: list[int] = []

    def place(data: bytes) -> int:
        count = max(1, -(-len(data) // _SECTOR))
        start = len(sectors)
        for offset in range(count):
            sectors.append(data[offset * _SECTOR : (offset + 1) * _SECTOR].ljust(_SECTOR, b"\0"))
            fat.append(start + offset + 1 if offset < count - 1 else _END)
        return start

    for number, data in big:
        entries[number]["start"] = place(data)
    entries[0]["start"] = place(bytes(mini)) if mini else _END
    entries[0]["size"] = len(mini)
    mini_fat_start = (
        place(b"".join(struct.pack("<I", value) for value in mini_fat)) if mini_fat else _END
    )
    mini_fat_sectors = -(-len(mini_fat) * 4 // _SECTOR) if mini_fat else 0

    def sort_key(number: int) -> tuple[int, str]:
        name = entries[number]["name"]
        return (len(name), name.upper())

    raw = bytearray()
    for entry in entries:
        children = sorted(entry["children"], key=sort_key)
        entry["child"] = children[0] if children else _FREE
        for left, right in itertools.pairwise(children):
            entries[left]["right"] = right
    for entry in entries:
        name = entry["name"].encode("utf-16-le") + b"\0\0"
        size = entry.get("size", len(entry["data"]))
        raw += name.ljust(64, b"\0")
        raw += struct.pack("<HBB", len(name), entry["type"], 1)
        raw += struct.pack("<III", _FREE, entry.get("right", _FREE), entry["child"])
        raw += b"\0" * 16 + struct.pack("<I", 0) + b"\0" * 16
        raw += struct.pack("<IQ", entry.get("start", _END) if entry["type"] != 1 else 0, size)
    directory_start = place(bytes(raw))

    # The FAT covers every sector, including its own.
    fat_sectors = 1
    while (len(sectors) + fat_sectors) * 4 > fat_sectors * _SECTOR:
        fat_sectors += 1
    fat_start = len(sectors)
    fat += [_FATSECT] * fat_sectors
    table = b"".join(struct.pack("<I", value) for value in fat)
    table = table.ljust(fat_sectors * _SECTOR, b"\xff")
    for offset in range(fat_sectors):
        sectors.append(table[offset * _SECTOR : (offset + 1) * _SECTOR])

    difat = [fat_start + offset for offset in range(fat_sectors)] + [_FREE] * (109 - fat_sectors)
    header = bytearray(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 16)
    header += struct.pack("<HHHHH", 0x3E, 3, 0xFFFE, 9, 6) + b"\0" * 6
    header += struct.pack("<I", 0)  # directory sectors (0 in version 3)
    header += struct.pack("<II", fat_sectors, directory_start) + b"\0" * 4
    header += struct.pack("<IIIII", 4096, mini_fat_start, mini_fat_sectors, _END, 0)
    header += b"".join(struct.pack("<I", value) for value in difat)
    return bytes(header) + b"".join(sectors)


def msg_bytes(subject: str, body: str, attachments: Mapping[str, bytes] | None = None) -> bytes:
    """An Outlook message's essentials: its subject and plain body as Unicode MAPI properties,
    and attachments with their long names and data."""
    streams: dict[str, bytes] = {
        "__substg1.0_0037001F": subject.encode("utf-16-le"),
        "__substg1.0_1000001F": body.encode("utf-16-le"),
    }
    for number, (name, data) in enumerate((attachments or {}).items()):
        storage = f"__attach_version1.0_#{number:08X}"
        streams[f"{storage}/__substg1.0_3707001F"] = name.encode("utf-16-le")
        streams[f"{storage}/__substg1.0_37010102"] = data
    return compound_file(streams)
