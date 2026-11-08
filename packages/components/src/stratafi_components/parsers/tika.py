"""Apache Tika: ``parse/tika`` sends a document to a Tika server and reads its
XHTML as structure: headings, paragraphs, lists and tables, page by page for PDFs, with OCR
through Tika's Tesseract when asked. The server is Tika 3.2.3 (``tika`` in ``compose.yaml``).
Tika is a service, so the call runs in the worker, not in the parse sandbox: the server is the
isolation. ``TikaClient`` streams the answer and stops at its character limit.
"""

from pathlib import Path
from typing import Any, ClassVar, cast

import httpx2
from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.detect import DOCX, PDF, XLSX
from stratafi_components.parse import ParseComponent
from stratafi_components.parsers.common import Builder
from stratafi_core.artifacts import ParsedDocument, RawDocument
from stratafi_core.components import ComponentError, ComponentInfo, StageContext

TIKA_SERVICE = "tika"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_OUTPUT_CHARACTERS = 10_000_000


class TikaClient:
    """``PUT /tika`` on a Tika server. Ignores proxy settings: a proxy would see every file."""

    def __init__(self, base_url: str, *, client: httpx2.Client | None = None) -> None:
        self._client = client or httpx2.Client(base_url=base_url.rstrip("/"), trust_env=False)

    def close(self) -> None:
        self._client.close()

    def version(self) -> str:
        try:
            response = self._client.get("/version", timeout=5.0)
        except httpx2.HTTPError:
            raise ComponentError("tika_unreachable", retryable=True) from None
        return response.text.strip()

    def extract(
        self,
        path: Path,
        media_type: str,
        *,
        accept: str = "text/plain",
        ocr: bool = False,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_chars: int = DEFAULT_MAX_OUTPUT_CHARACTERS,
        extra_headers: dict[str, str] | None = None,
    ) -> str:
        headers = {
            "Accept": accept,
            "Content-Type": media_type,
            **({} if ocr else {"X-Tika-OCRskipOcr": "true"}),
            **(extra_headers or {}),
        }
        try:
            with (
                path.open("rb") as body,
                self._client.stream(
                    "PUT", "/tika", content=body, headers=headers, timeout=timeout
                ) as response,
            ):
                if response.status_code != 200:
                    raise ComponentError(f"tika_http_{response.status_code}")
                chunks: list[str] = []
                size = 0
                for chunk in response.iter_text():
                    size += len(chunk)
                    if size > max_chars:
                        raise ComponentError("limit_exceeded")
                    chunks.append(chunk)
                return "".join(chunks)
        except httpx2.TimeoutException:
            raise ComponentError("tika_timeout", retryable=True) from None
        except httpx2.HTTPError:
            raise ComponentError("tika_unreachable", retryable=True) from None


# --- parse/tika ------------------------------------------------------------------------------


class TikaParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ocr: bool = Field(default=False, description="Let Tika run Tesseract on images and scans")
    ocr_language: str = Field(default="eng", pattern=r"^[a-z_]{3,16}(\+[a-z_]{3,16}){0,3}$")
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0, le=600)
    max_output_chars: int = Field(default=DEFAULT_MAX_OUTPUT_CHARACTERS, ge=1, le=50_000_000)


class TikaXhtml(ParseComponent):
    isolated: ClassVar[bool] = False
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/tika",
        stage="parse",
        version="1.0.0",
        title="Tika (XHTML structure)",
        description="Tika's XHTML output read as structure: headings, paragraphs, lists and "
        "tables, page by page for PDFs; OCR through Tika's Tesseract when asked.",
        accepts=(
            PDF,
            DOCX,
            XLSX,
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/msword",
            "application/vnd.ms-excel",
            "application/vnd.ms-powerpoint",
            "text/html",
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/json",
            "message/rfc822",
            "image/png",
            "image/jpeg",
        ),
        resource_class="io",
        license="Apache-2.0",
        egress="provider:tika",
    )
    Params: ClassVar[type[BaseModel]] = TikaParams

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        return {"tika": "3.2.3"}

    def parse_service(
        self, path: Path, raw: RawDocument, params: BaseModel, context: StageContext
    ) -> ParsedDocument:
        from stratafi_components.parsers.office import html_elements

        params = cast(TikaParams, params)
        client: TikaClient = context.service(TIKA_SERVICE)
        headers = {"X-Tika-OCRLanguage": params.ocr_language} if params.ocr else {}
        with context.provider(TIKA_SERVICE):
            xhtml = client.extract(
                path,
                raw.media_type,
                accept="text/html",
                ocr=params.ocr,
                timeout=params.timeout_seconds,
                max_chars=params.max_output_chars,
                extra_headers=headers,
            )
        context.metric("tika.calls", 1, add=True)
        builder = Builder(raw, self, params)
        pages = tika_pages(xhtml)
        if pages:
            for number, markup in enumerate(pages, start=1):
                start = len(builder.elements)
                html_elements(builder, markup)
                for element in builder.elements[start:]:
                    element["page"] = number
            title = None
        else:
            title = html_elements(builder, xhtml)
        return builder.document(
            title=title, pages=[{"number": n} for n in range(1, len(pages) + 1)]
        )


def tika_pages(xhtml: str) -> list[str]:
    """Tika marks each PDF page as ``<div class="page">``; the markup of each, in order."""
    import re

    starts = [match.end() for match in re.finditer(r'<div class="page">', xhtml)]
    if not starts:
        return []
    ends = [*(start - len('<div class="page">') for start in starts[1:]), len(xhtml)]
    return [xhtml[start:end] for start, end in zip(starts, ends, strict=True)]


COMPONENTS = [TikaXhtml]
