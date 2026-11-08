"""Hosted parsers and the license-gated pymupdf4llm.

**Hosted adapters** send a document to a provider's API: Azure Document Intelligence (layout,
Markdown output), AWS Textract (lines), Google Document AI (paragraphs), LlamaParse (Markdown)
and Mistral OCR (Markdown per page). Each **refuses to run unless its provider is
allow-listed** (``providers.allowed``, empty by default: the provider receives the document's
text): the check comes before any byte leaves. Endpoints and the name of the
environment variable holding the key come from ``providers.endpoints``; keys never sit in
configuration. A run records which providers received which documents (``egress``).

**``parse/pymupdf4llm``** (fast PDF to Markdown) is AGPL-3.0: an optional extra that also
refuses to run until the owner allows the license (``licenses.allowed``).
"""

import base64
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import httpx2
from pydantic import BaseModel

from stratafi_components.detect import DOCX, PDF, PPTX, XLSX
from stratafi_components.parse import ParseComponent
from stratafi_components.parsers.common import Builder, markdown_elements
from stratafi_core.artifacts import ParsedDocument, RawDocument
from stratafi_core.components import ComponentError, ComponentInfo, StageContext

ALLOWED_SERVICE = "providers_allowed"
POLL_SECONDS = 2.0
MAX_POLLS = 150


@dataclass(frozen=True)
class ProviderAccess:
    """What the worker hands a hosted adapter for an allow-listed provider."""

    url: str | None
    key: str | None
    region: str | None = None
    client: httpx2.Client | None = None  # tests inject a mock transport's client
    sleep: Callable[[float], None] = time.sleep

    def http(self) -> httpx2.Client:
        return self.client or httpx2.Client(timeout=120.0, trust_env=False)


def hosted_services(settings: Any) -> dict[str, Any]:
    """What a process hands hosted and gated components: the allow-list, each allow-listed
    provider's access (its key read from the environment variable the configuration names),
    the licenses allowed, and the lab switches."""
    import os

    providers = settings.providers
    services: dict[str, Any] = {
        ALLOWED_SERVICE: frozenset(providers.allowed),
        "licenses": frozenset(settings.licenses.allowed),
        "enrichment_enabled": settings.models.enrichment_enabled,
        "llm_rerank_enabled": settings.models.llm_rerank_enabled,
        "in_process": settings.models.in_process,
    }
    for name in providers.allowed:
        endpoint = providers.endpoints.get(name)
        if endpoint is not None:
            key = os.environ.get(endpoint.api_key_env) if endpoint.api_key_env else None
            services[f"provider:{name}"] = ProviderAccess(endpoint.url, key, endpoint.region)
    return services


class HostedParser(ParseComponent):
    isolated: ClassVar[bool] = False
    provider: ClassVar[str]

    def access(self, context: StageContext) -> ProviderAccess:
        if self.provider not in context.services.get(ALLOWED_SERVICE, ()):
            raise ComponentError("provider_not_allowed")
        access = context.services.get(f"provider:{self.provider}")
        if not isinstance(access, ProviderAccess) or not access.key:
            raise ComponentError("provider_not_configured")
        return access

    def parse_service(
        self, path: Path, raw: RawDocument, params: BaseModel, context: StageContext
    ) -> ParsedDocument:
        access = self.access(context)
        builder = Builder(raw, self, params)
        with context.provider(self.provider):
            try:
                self.extract(access, path, raw, builder)
            except httpx2.TimeoutException:
                raise ComponentError("provider_timeout", retryable=True) from None
            except httpx2.HTTPError:
                raise ComponentError("provider_unreachable", retryable=True) from None
        context.metric(f"{self.provider.replace('-', '_')}.calls", 1, add=True)
        return builder.document()

    def extract(
        self, access: ProviderAccess, path: Path, raw: RawDocument, builder: Builder
    ) -> None:
        raise NotImplementedError


def _checked(response: httpx2.Response, *ok: int) -> httpx2.Response:
    if response.status_code not in (ok or (200,)):
        code = response.status_code
        raise ComponentError(f"provider_http_{code}", retryable=code == 429 or code >= 500)
    return response


def _info(component_id: str, title: str, provider: str, accepts: tuple[str, ...]) -> ComponentInfo:
    return ComponentInfo(
        id=component_id,
        stage="parse",
        version="1.0.0",
        title=title,
        description=f"Sends each document to {title}'s API. Refuses to run unless the provider "
        "is allow-listed (providers.allowed): it receives the document's text.",
        accepts=accepts,
        resource_class="io",
        deterministic=False,
        license="commercial",
        egress=f"provider:{provider}",
    )


class AzureDocumentIntelligence(HostedParser):
    provider: ClassVar[str] = "azure-document-intelligence"
    info: ClassVar[ComponentInfo] = _info(
        "parse/azure-document-intelligence",
        "Azure Document Intelligence",
        "azure-document-intelligence",
        (PDF, DOCX, PPTX, XLSX, "text/html", "image/png", "image/jpeg"),
    )

    def extract(self, access, path, raw, builder) -> None:
        http = access.http()
        url = (
            f"{(access.url or '').rstrip('/')}/documentintelligence/documentModels/"
            "prebuilt-layout:analyze?api-version=2024-11-30&outputContentFormat=markdown"
        )
        headers = {"Ocp-Apim-Subscription-Key": access.key or "", "Content-Type": raw.media_type}
        started = _checked(http.post(url, content=path.read_bytes(), headers=headers), 202)
        location = started.headers.get("operation-location")
        if not location:
            raise ComponentError("provider_bad_response")
        for _ in range(MAX_POLLS):
            body = _checked(http.get(location, headers=headers)).json()
            if body.get("status") == "succeeded":
                markdown_elements(builder, body["analyzeResult"]["content"])
                return
            if body.get("status") == "failed":
                raise ComponentError("provider_failed")
            access.sleep(POLL_SECONDS)
        raise ComponentError("provider_timeout", retryable=True)


class MistralOcr(HostedParser):
    provider: ClassVar[str] = "mistral"
    info: ClassVar[ComponentInfo] = _info(
        "parse/mistral-ocr", "Mistral OCR", "mistral", (PDF, "image/png", "image/jpeg")
    )

    def extract(self, access, path, raw, builder) -> None:
        data = base64.b64encode(path.read_bytes()).decode()
        kind = "image_url" if raw.media_type.startswith("image/") else "document_url"
        body = {
            "model": "mistral-ocr-latest",
            "document": {"type": kind, kind: f"data:{raw.media_type};base64,{data}"},
        }
        url = f"{(access.url or 'https://api.mistral.ai').rstrip('/')}/v1/ocr"
        response = _checked(
            access.http().post(url, json=body, headers={"Authorization": f"Bearer {access.key}"})
        )
        for page in response.json().get("pages", []):
            markdown_elements(builder, str(page.get("markdown", "")), page=int(page["index"]) + 1)


class LlamaParse(HostedParser):
    provider: ClassVar[str] = "llamaparse"
    info: ClassVar[ComponentInfo] = _info(
        "parse/llamaparse", "LlamaParse", "llamaparse", (PDF, DOCX, PPTX, XLSX, "text/html")
    )

    def extract(self, access, path, raw, builder) -> None:
        http = access.http()
        base = (access.url or "https://api.cloud.llamaindex.ai").rstrip("/")
        headers = {"Authorization": f"Bearer {access.key}"}
        upload = _checked(
            http.post(
                f"{base}/api/parsing/upload",
                files={"file": (f"document{Path(path).suffix}", path.read_bytes(), raw.media_type)},
                headers=headers,
            )
        ).json()
        job = upload["id"]
        for _ in range(MAX_POLLS):
            status = _checked(http.get(f"{base}/api/parsing/job/{job}", headers=headers)).json()
            if status.get("status") == "SUCCESS":
                result = _checked(
                    http.get(f"{base}/api/parsing/job/{job}/result/markdown", headers=headers)
                ).json()
                markdown_elements(builder, str(result.get("markdown", "")))
                return
            if status.get("status") in ("ERROR", "CANCELED"):
                raise ComponentError("provider_failed")
            access.sleep(POLL_SECONDS)
        raise ComponentError("provider_timeout", retryable=True)


class GoogleDocumentAi(HostedParser):
    provider: ClassVar[str] = "google-document-ai"
    info: ClassVar[ComponentInfo] = _info(
        "parse/google-document-ai",
        "Google Document AI",
        "google-document-ai",
        (PDF, "image/png", "image/jpeg"),
    )

    def extract(self, access, path, raw, builder) -> None:
        # access.url is the processor's :process URL; the key is an OAuth access token.
        body = {
            "rawDocument": {
                "content": base64.b64encode(path.read_bytes()).decode(),
                "mimeType": raw.media_type,
            }
        }
        response = _checked(
            access.http().post(
                access.url or "", json=body, headers={"Authorization": f"Bearer {access.key}"}
            )
        )
        document = response.json().get("document", {})
        text = str(document.get("text", ""))
        for number, page in enumerate(document.get("pages", []), start=1):
            for paragraph in page.get("paragraphs", []):
                segments = paragraph.get("layout", {}).get("textAnchor", {}).get("textSegments", [])
                piece = "".join(
                    text[int(s.get("startIndex", 0)) : int(s.get("endIndex", 0))] for s in segments
                )
                builder.add("paragraph", piece.strip(), page=number)


class AwsTextract(HostedParser):
    provider: ClassVar[str] = "aws-textract"
    info: ClassVar[ComponentInfo] = _info(
        "parse/aws-textract", "AWS Textract", "aws-textract", (PDF, "image/png", "image/jpeg")
    )
    requires: ClassVar[tuple[str, ...]] = ("boto3",)
    extra: ClassVar[str | None] = "textract"

    def textract(self, access: ProviderAccess) -> Any:
        import boto3

        return boto3.client("textract", region_name=access.region or "us-east-1")

    def extract(self, access, path, raw, builder) -> None:
        # Synchronous analysis: single-page documents and images. AWS credentials come from
        # the environment, as boto3 finds them.
        result = self.textract(access).analyze_document(
            Document={"Bytes": path.read_bytes()}, FeatureTypes=["LAYOUT"]
        )
        lines: dict[int, list[str]] = {}
        for block in result.get("Blocks", []):
            if block.get("BlockType") == "LINE":
                lines.setdefault(int(block.get("Page", 1)), []).append(str(block.get("Text", "")))
        for number, texts in sorted(lines.items()):
            builder.add("paragraph", "\n".join(texts), page=number)

    def access(self, context: StageContext) -> ProviderAccess:
        if self.provider not in context.services.get(ALLOWED_SERVICE, ()):
            raise ComponentError("provider_not_allowed")
        access = context.services.get(f"provider:{self.provider}")
        if not isinstance(access, ProviderAccess):
            raise ComponentError("provider_not_configured")
        return access


class Pymupdf4llm(ParseComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="parse/pymupdf4llm",
        stage="parse",
        version="1.0.0",
        title="PyMuPDF4LLM",
        description="Fast PDF to Markdown with PyMuPDF. AGPL-3.0: an optional extra that "
        "refuses to run until the owner allows the license (licenses.allowed).",
        accepts=(PDF,),
        license="AGPL-3.0",
    )
    requires: ClassVar[tuple[str, ...]] = ("pymupdf4llm",)
    extra: ClassVar[str | None] = "pymupdf4llm"
    license_gate: ClassVar[str | None] = "AGPL-3.0"

    def parse_local(self, path: Path, raw: RawDocument, params: BaseModel) -> ParsedDocument:
        import pymupdf4llm

        try:
            chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        except Exception:
            raise ComponentError("conversion_failed") from None
        builder = Builder(raw, self, params)
        for number, chunk in enumerate(chunks, start=1):
            markdown_elements(builder, str(chunk.get("text", "")), page=number)
        return builder.document()


COMPONENTS = [
    AzureDocumentIntelligence,
    AwsTextract,
    GoogleDocumentAi,
    LlamaParse,
    MistralOcr,
    Pymupdf4llm,
]
