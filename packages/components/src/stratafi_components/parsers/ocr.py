"""OCR: scanned PDFs and images through RapidOCR or Tesseract.

``parse/ocr`` renders each PDF page with pypdfium2 at ``dpi`` and recognizes its lines with the
chosen engine; an image is recognized as it is. With ``pages: empty`` (the default), a page
that has a text layer keeps it (pypdf) and only pages without one are recognized, so a partly
scanned PDF comes out whole. Lines become paragraphs where the gap between them is small, lines
much taller than the page's median become headings, and every element keeps its box, in points
from the page's bottom left.

- **Tesseract** (Apache-2.0, the default) is the ``tesseract`` binary with its language data, a
  system install, called through pytesseract.
- **RapidOCR** (Apache-2.0; ``rapidocr-onnxruntime``, PP-OCRv4 models inside the wheel, so it
  needs no download) runs on onnxruntime, in process. Its bundled recognizer is the Chinese
  model, which drops the spaces between English words: on the scanned-PDF variant it found none
  of the five evidence phrases that Tesseract found. It suits scripts without
  spaces; English needs its English model, a download.

Both are in the ``ocr`` extra; Tesseract also needs its binary. Recognition on the same pages
with the same engine version gives the same text, so the parser is deterministic; the engine's
version is in the cache key.
"""

import io
import shutil
import statistics
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import pypdf
from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.detect import PDF
from stratafi_components.parse import ParseComponent
from stratafi_components.parsers.common import Builder
from stratafi_core.artifacts import ParsedDocument, RawDocument
from stratafi_core.components import ComponentError, ComponentInfo

IMAGE_TYPES = ("image/png", "image/jpeg")


class OcrParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: Literal["tesseract", "rapidocr"] = Field(
        default="tesseract",
        description="Tesseract (the system binary) or RapidOCR (models in the wheel; its "
        "bundled model drops spaces in English)",
    )
    pages: Literal["empty", "all"] = Field(
        default="empty",
        description="empty: recognize only pages without a text layer; all: every page",
    )
    dpi: int = Field(default=200, ge=72, le=400, description="Render resolution for PDF pages")
    language: str = Field(
        default="eng", pattern=r"^[a-z_]{3,16}(\+[a-z_]{3,16}){0,3}$", description="Tesseract's"
    )
    min_confidence: float = Field(default=0.5, ge=0, le=1, description="Lines below are dropped")


# (x0, top, x1, bottom) in pixels from the top left, the text, and a confidence in [0, 1].
Line = tuple[tuple[float, float, float, float], str, float]


class Recognizer:
    def lines(self, image: Any) -> list[Line]:
        raise NotImplementedError


class RapidRecognizer(Recognizer):
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def lines(self, image: Any) -> list[Line]:
        import numpy as np

        result, _ = self.engine(np.asarray(image.convert("RGB")))
        found: list[Line] = []
        for points, text, score in result or []:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            found.append(((min(xs), min(ys), max(xs), max(ys)), str(text), float(score)))
        return found


class TesseractRecognizer(Recognizer):
    def __init__(self, language: str) -> None:
        if shutil.which("tesseract") is None:
            raise ComponentError("tesseract_not_installed")
        self.language = language

    def lines(self, image: Any) -> list[Line]:
        import pytesseract

        data = pytesseract.image_to_data(
            image, lang=self.language, output_type=pytesseract.Output.DICT
        )
        grouped: dict[tuple[int, int, int], list[int]] = {}
        for index, word in enumerate(data["text"]):
            if str(word).strip() and float(data["conf"][index]) >= 0:
                key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
                grouped.setdefault(key, []).append(index)
        found: list[Line] = []
        for indexes in grouped.values():
            left = min(data["left"][i] for i in indexes)
            top = min(data["top"][i] for i in indexes)
            right = max(data["left"][i] + data["width"][i] for i in indexes)
            bottom = max(data["top"][i] + data["height"][i] for i in indexes)
            text = " ".join(str(data["text"][i]).strip() for i in indexes)
            confidence = statistics.mean(float(data["conf"][i]) for i in indexes) / 100
            found.append(((left, top, right, bottom), text, confidence))
        return found


def engine_version(engine: str) -> str:
    try:
        if engine == "rapidocr":
            from importlib.metadata import version

            return f"rapidocr-onnxruntime {version('rapidocr-onnxruntime')}"
        import pytesseract

        return f"tesseract {pytesseract.get_tesseract_version()}"
    except Exception:
        return "unavailable"


def add_lines(
    builder: Builder,
    lines: list[Line],
    *,
    page: int,
    scale: float,
    page_height: float,
    min_confidence: float,
) -> int:
    """Group recognized lines into paragraphs and headings; returns how many lines were kept.
    ``scale`` turns pixels into points."""
    kept = sorted(
        (line for line in lines if line[2] >= min_confidence and line[1].strip()),
        key=lambda line: (round(line[0][1] / 4), line[0][0]),
    )
    if not kept:
        return 0
    heights = [box[3] - box[1] for box, _, _ in kept]
    median = statistics.median(heights) or 1.0
    blocks: list[list[Line]] = []
    for line in kept:
        box = line[0]
        heading = (box[3] - box[1]) > median * 1.3
        if blocks:
            last = blocks[-1][-1][0]
            last_heading = (last[3] - last[1]) > median * 1.3
            gap = box[1] - last[3]
            if gap < median * 0.9 and heading == last_heading:
                blocks[-1].append(line)
                continue
        blocks.append([line])
    for block in blocks:
        x0 = min(box[0] for box, _, _ in block) * scale
        x1 = max(box[2] for box, _, _ in block) * scale
        top = min(box[1] for box, _, _ in block) * scale
        bottom = max(box[3] for box, _, _ in block) * scale
        bbox = (x0, max(page_height - bottom, 0.0), x1, max(page_height - top, 0.0))
        text = "\n".join(text.strip() for _, text, _ in block)
        is_heading = (block[0][0][3] - block[0][0][1]) > median * 1.3 and len(block) <= 2
        if is_heading:
            builder.add("heading", " ".join(text.split()), level=1, page=page, bbox=bbox)
        else:
            builder.add("paragraph", text, page=page, bbox=bbox)
    return len(kept)


class OcrParser(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/ocr",
        stage="parse",
        version="1.0.0",
        title="OCR (RapidOCR or Tesseract)",
        description="Scanned PDFs and images: pages rendered with pypdfium2 and recognized by "
        "RapidOCR or Tesseract, as paragraphs and headings with boxes. Pages with a text layer "
        "keep it unless every page is asked for.",
        accepts=(PDF, "pdf[scanned]", *IMAGE_TYPES),
        license="Apache-2.0",
    )
    Params: ClassVar[type[BaseModel]] = OcrParams
    requires: ClassVar[tuple[str, ...]] = ("pytesseract", "pypdfium2", "PIL")
    extra: ClassVar[str | None] = "ocr"
    # onnxruntime and OpenCV reserve several GiB of address space for a few hundred MiB used.
    memory_bytes: ClassVar[int | None] = 8 * 1024 * 1024 * 1024

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        params = cast(OcrParams, params)
        env = {"ocr": engine_version(params.engine)}
        if params.engine == "tesseract":
            env["language"] = params.language
        return env

    def recognizer(self, params: OcrParams) -> Recognizer:
        if params.engine == "tesseract":
            return TesseractRecognizer(params.language)
        try:
            return RapidRecognizer()
        except ImportError:
            raise ComponentError("extra_missing") from None

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        params = cast(OcrParams, params)
        builder = Builder(raw, self, params)
        recognizer: Recognizer | None = None
        self.local_metrics = {"ocr_pages": 0, "ocr_lines": 0}
        if raw.media_type in IMAGE_TYPES:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
                recognizer = self.recognizer(params)
                lines = recognizer.lines(image)
            kept = add_lines(
                builder,
                lines,
                page=1,
                scale=1.0,
                page_height=float(height),
                min_confidence=params.min_confidence,
            )
            self.local_metrics.update(ocr_pages=1, ocr_lines=kept)
            pages = [{"number": 1, "width": float(width), "height": float(height)}]
            return builder.document(pages=pages)
        import pypdfium2 as pdfium

        data = path.read_bytes()
        try:
            layers = pypdf.PdfReader(io.BytesIO(data))
            document = pdfium.PdfDocument(data)
        except Exception:
            raise ComponentError("malformed") from None
        pages = []
        scale = 72 / params.dpi
        for index in range(len(document)):
            page = document[index]
            width, height = page.get_size()
            number = index + 1
            pages.append({"number": number, "width": float(width), "height": float(height)})
            text = ""
            if params.pages == "empty" and index < len(layers.pages):
                try:
                    text = (layers.pages[index].extract_text() or "").strip()
                except Exception:
                    text = ""
            if text:
                for block in text.split("\n\n"):
                    builder.add("paragraph", block, page=number)
                continue
            recognizer = recognizer or self.recognizer(params)
            image = page.render(scale=params.dpi / 72).to_pil()
            kept = add_lines(
                builder,
                recognizer.lines(image),
                page=number,
                scale=scale,
                page_height=float(height),
                min_confidence=params.min_confidence,
            )
            self.local_metrics["ocr_pages"] += 1
            self.local_metrics["ocr_lines"] += kept
        if self.local_metrics["ocr_pages"]:
            builder.warn("ocr")
        return builder.document(pages=pages)


COMPONENTS = [OcrParser]
