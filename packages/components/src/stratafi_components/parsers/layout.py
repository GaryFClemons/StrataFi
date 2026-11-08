"""Layout and conversion parsers: markitdown, docling and unstructured.

Each adapter maps its library's output onto ``stratafi.parsed-document.v1``. The libraries are
optional extras: without one, its component stays in the registry, says which extra it needs,
and refuses to run (``extra_missing``).

- ``parse/markitdown`` (MIT, the ``markitdown`` extra) converts many formats to Markdown, which
  is read back into headings, list items, tables, code and paragraphs.
- ``parse/docling`` (MIT; its models carry their own licenses; the ``docling`` extra, Linux
  only) reads layout and reading order, with table structure and optional OCR. Its items keep
  their labels, levels, pages, boxes and table cells.
- ``parse/unstructured`` (Apache-2.0, the ``unstructured`` extra, Linux only) partitions a file
  with the ``fast``, ``hi_res`` or ``ocr_only`` strategy; its element categories map onto
  Stratafi's types, with pages and coordinates when it has them.

docling and unstructured download models at first use; a lab without that network access runs
markitdown, pypdf and pdfplumber instead.
"""

import tempfile
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.detect import DOCX, EML, MSG, PDF, PPTX, XLSX
from stratafi_components.parse import ParseComponent
from stratafi_components.parsers.common import Builder, markdown_elements
from stratafi_core.artifacts import ParsedDocument, RawDocument
from stratafi_core.components import ComponentError, ComponentInfo

EXTENSION_FOR = {
    PDF: ".pdf",
    DOCX: ".docx",
    PPTX: ".pptx",
    XLSX: ".xlsx",
    "text/html": ".html",
    "text/csv": ".csv",
    "application/json": ".json",
    "application/xml": ".xml",
    "text/plain": ".txt",
    "text/markdown": ".md",
    EML: ".eml",
    MSG: ".msg",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _named_copy(path: Path, media_type: str) -> Path:
    """The document under a name with its type's extension, which these libraries go by. The
    copy goes to the temporary directory, which in the sandbox is its scratch directory: the
    only place a confined parser may write."""
    target = Path(tempfile.mkdtemp(prefix="stratafi-parse-")) / (
        f"document{EXTENSION_FOR.get(media_type, '')}"
    )
    target.write_bytes(path.read_bytes())
    return target


def _version(module: str) -> dict[str, Any]:
    try:
        from importlib.metadata import version

        return {module: version(module)}
    except Exception:
        return {}


# --- parse/markitdown ------------------------------------------------------------------------


class MarkitdownParser(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/markitdown",
        stage="parse",
        version="1.0.0",
        title="markitdown",
        description="Microsoft's markitdown: many formats to Markdown, read back into headings, "
        "lists, tables, code and paragraphs. No pages or boxes.",
        accepts=(
            PDF,
            DOCX,
            PPTX,
            XLSX,
            "text/html",
            "text/csv",
            "application/json",
            "application/xml",
            "text/plain",
            "text/markdown",
        ),
        license="MIT",
    )
    requires: ClassVar[tuple[str, ...]] = ("markitdown",)
    extra: ClassVar[str | None] = "markitdown"

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        return _version("markitdown")

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        from markitdown import MarkItDown

        try:
            result = MarkItDown(enable_plugins=False).convert_local(
                str(_named_copy(path, raw.media_type))
            )
        except Exception:
            raise ComponentError("conversion_failed") from None
        builder = Builder(raw, self, params)
        markdown_elements(builder, result.text_content or "")
        title = getattr(result, "title", None)
        return builder.document(title=title.strip() if isinstance(title, str) and title else None)


# --- parse/docling ---------------------------------------------------------------------------

DOCLING_TYPES = {
    "title": "title",
    "section_header": "heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list_item": "list_item",
    "table": "table",
    "picture": "figure",
    "chart": "figure",
    "caption": "caption",
    "code": "code",
    "formula": "code",
    "page_header": "page_header",
    "page_footer": "page_footer",
    "footnote": "paragraph",
    "reference": "paragraph",
}


class DoclingParams(_Params):
    ocr: bool = Field(default=False, description="Run OCR on pages and images")
    table_structure: bool = Field(default=True, description="Recover table cells")


def _docling_box(item: Any, heights: dict[int, float]) -> tuple[int | None, list[float] | None]:
    prov = (getattr(item, "prov", None) or [None])[0]
    if prov is None:
        return None, None
    page = int(prov.page_no)
    box = prov.bbox
    left, right = float(box.l), float(box.r)
    top, bottom = float(box.t), float(box.b)
    origin = str(getattr(box, "coord_origin", "BOTTOMLEFT")).upper()
    if "TOPLEFT" in origin and page in heights:
        top, bottom = heights[page] - top, heights[page] - bottom
    return page, [left, min(top, bottom), right, max(top, bottom)]


class DoclingParser(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/docling",
        stage="parse",
        version="1.0.0",
        title="Docling",
        description="IBM's docling: layout, reading order, headings and table structure, with "
        "optional OCR; PDF, DOCX, PPTX, XLSX, HTML and images. Models download at first use.",
        accepts=(PDF, DOCX, PPTX, XLSX, "text/html", "text/markdown", "image/png", "image/jpeg"),
        resource_class="cpu",
        license="MIT (models vary)",
        platforms=("linux",),
        services=("model-files",),
    )
    Params: ClassVar[type[BaseModel]] = DoclingParams
    requires: ClassVar[tuple[str, ...]] = ("docling",)
    extra: ClassVar[str | None] = "docling"

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        return _version("docling")

    def _converter(self, params: DoclingParams) -> Any:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions(do_ocr=params.ocr, do_table_structure=params.table_structure)
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        params = cast(DoclingParams, params)
        try:
            result = self._converter(params).convert(str(_named_copy(path, raw.media_type)))
        except ComponentError:
            raise
        except Exception:
            raise ComponentError("conversion_failed") from None
        document = result.document
        heights: dict[int, float] = {}
        pages = []
        for number, page in sorted((getattr(document, "pages", None) or {}).items()):
            size = getattr(page, "size", None)
            width = float(size.width) if size is not None else None
            height = float(size.height) if size is not None else None
            if height is not None:
                heights[int(number)] = height
            pages.append({"number": int(number), "width": width, "height": height})
        builder = Builder(raw, self, params)
        for item, level in document.iterate_items():
            label = str(getattr(getattr(item, "label", ""), "value", getattr(item, "label", "")))
            kind = DOCLING_TYPES.get(label.lower(), "other")
            page, bbox = _docling_box(item, heights)
            if kind == "table":
                rows = _docling_rows(item)
                if rows:
                    builder.add_table(rows, page=page, bbox=bbox)
                continue
            text = str(getattr(item, "text", "") or "")
            heading_level = getattr(item, "level", None) or (level if kind == "heading" else None)
            builder.add(
                kind,
                text,
                level=int(heading_level) if kind == "heading" and heading_level else None,
                page=page,
                bbox=bbox,
            )
        return builder.document(pages=pages)


def _docling_rows(item: Any) -> list[list[str]]:
    data = getattr(item, "data", None)
    cells = getattr(data, "table_cells", None) or []
    rows = int(getattr(data, "num_rows", 0) or 0)
    cols = int(getattr(data, "num_cols", 0) or 0)
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        r = int(getattr(cell, "start_row_offset_idx", 0))
        c = int(getattr(cell, "start_col_offset_idx", 0))
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = str(getattr(cell, "text", "") or "")
    return grid


# --- parse/unstructured ----------------------------------------------------------------------

UNSTRUCTURED_TYPES = {
    "Title": "heading",
    "NarrativeText": "paragraph",
    "UncategorizedText": "paragraph",
    "Text": "paragraph",
    "ListItem": "list_item",
    "Table": "table",
    "Header": "page_header",
    "Footer": "page_footer",
    "PageNumber": "page_number",
    "FigureCaption": "caption",
    "Image": "figure",
    "Formula": "code",
    "CodeSnippet": "code",
    "Address": "paragraph",
    "EmailAddress": "paragraph",
}


class UnstructuredParams(_Params):
    strategy: Literal["fast", "hi_res", "ocr_only", "auto"] = Field(
        default="fast",
        description="fast: text layers; hi_res: layout models; ocr_only: OCR every page",
    )


class UnstructuredParser(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/unstructured",
        stage="parse",
        version="1.0.0",
        title="Unstructured",
        description="unstructured's partitioning with the fast, hi_res or ocr_only strategy, "
        "over many formats; element categories map onto Stratafi's types.",
        accepts=(
            PDF,
            DOCX,
            PPTX,
            XLSX,
            "text/html",
            "text/plain",
            "text/markdown",
            "text/csv",
            EML,
            MSG,
            "image/png",
            "image/jpeg",
        ),
        license="Apache-2.0",
        platforms=("linux",),
        services=("model-files",),
    )
    Params: ClassVar[type[BaseModel]] = UnstructuredParams
    requires: ClassVar[tuple[str, ...]] = ("unstructured",)
    extra: ClassVar[str | None] = "unstructured"

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        return _version("unstructured")

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        from unstructured.partition.auto import partition

        params = cast(UnstructuredParams, params)
        try:
            elements = partition(
                filename=str(_named_copy(path, raw.media_type)), strategy=params.strategy
            )
        except Exception:
            raise ComponentError("conversion_failed") from None
        builder = Builder(raw, self, params)
        sizes: dict[int, tuple[float, float]] = {}
        for element in elements:
            category = str(getattr(element, "category", type(element).__name__))
            kind = UNSTRUCTURED_TYPES.get(category, "other")
            metadata = getattr(element, "metadata", None)
            page = getattr(metadata, "page_number", None)
            bbox = _unstructured_box(metadata, sizes, page)
            text = str(getattr(element, "text", "") or "")
            if kind == "table":
                rows = [row.split("\t") for row in text.split("\n") if row.strip()]
                builder.add_table(rows or [[text]], page=page, bbox=bbox, header_rows=0)
                continue
            depth = getattr(metadata, "category_depth", None)
            builder.add(
                kind,
                text,
                level=(int(depth) + 1)
                if kind == "heading" and depth is not None
                else (1 if kind == "heading" else None),
                page=page,
                bbox=bbox,
            )
        pages = [
            {"number": number, "width": width, "height": height}
            for number, (width, height) in sorted(sizes.items())
        ]
        return builder.document(pages=pages)


def _unstructured_box(
    metadata: Any, sizes: dict[int, tuple[float, float]], page: int | None
) -> list[float] | None:
    coordinates = getattr(metadata, "coordinates", None)
    points = getattr(coordinates, "points", None)
    system = getattr(coordinates, "system", None)
    if not points or system is None or page is None:
        return None
    width, height = float(system.width), float(system.height)
    sizes.setdefault(int(page), (width, height))
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    # Pixel space has its origin at the top left; flip to the schema's bottom left.
    return [min(xs), height - max(ys), max(xs), height - min(ys)]


COMPONENTS = [MarkitdownParser, DoclingParser, UnstructuredParser]
