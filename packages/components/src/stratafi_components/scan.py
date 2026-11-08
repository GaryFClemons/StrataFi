"""The scan stage.

Scan runs on each raw document before any parser opens it. Its checks run in the spec's order:

- ``scan/content-safety``: the bytes agree with the extension, archive limits, size and page
  limits, encryption and corruption probes
- ``scan/clamav``: clamd ``INSTREAM``; signatures more than 72 hours old refuse a pass
- ``scan/pii``: personal data in the document's text, found in process
  (``stratafi_core.pii``); block-only (redaction lives in Clean)
- ``scan/dedupe``: an earlier entry of the corpus version has the same bytes

Outcomes: ``clean``, ``infected``, ``encrypted``, ``malformed``, ``limit_exceeded``, ``pii``,
``error``, plus ``duplicate``. The spec's ``on_outcome`` policy maps each to ``continue``,
``flag`` or ``exclude``. Once a check's outcome excludes the document, the remaining checks are
skipped, so an infected or encrypted file is never opened to read its text. The ``scan_report``
artifact (``stratafi.scan-report.v1``) holds every check's outcome, reason and finding counts,
never a finding's value, and the decision. An excluded document gets no downstream tasks
(``stratafi_core.planner``).
"""

import html
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stratafi_components.archives import ArchiveLimits, ArchiveRefused, measure_zip
from stratafi_components.clamd import ClamdClient, ClamdError, parse_version
from stratafi_core.artifacts import (
    DECISION_RANK,
    OUTCOME_PRIORITY,
    RawDocument,
    ScanCheck,
    ScanReport,
)
from stratafi_core.components import (
    Cancelled,
    Component,
    ComponentError,
    ComponentInfo,
    ComponentRegistry,
    StageContext,
)
from stratafi_core.pii import (
    DEFAULT_CATEGORIES,
    PiiCategory,
    canonical_categories,
    count_by_category,
    decode_text,
    find_pii,
)
from stratafi_core.specs import ScanStage

OOXML_KINDS = frozenset({"docx", "xlsx", "pptx"})
CLAMD_SERVICE = "clamd"


@dataclass(frozen=True)
class ScanInput:
    path: Path
    raw: RawDocument
    duplicate: bool = False  # an earlier entry of the corpus version has the same bytes


@dataclass(frozen=True)
class CheckResult:
    outcome: str
    reason: str | None = None
    findings: dict[str, int] | None = None
    details: dict[str, str | int | bool | None] | None = None


CLEAN = CheckResult("clean")


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScanComponent(Component):
    """A scan check: one raw document → one outcome."""

    def check(self, scan: ScanInput, params: BaseModel, context: StageContext) -> CheckResult:
        raise NotImplementedError


# --- scan/content-safety ---------------------------------------------------------------------


class ContentSafetyParams(_Params):
    max_bytes: int = Field(
        default=256 * 1024 * 1024, ge=1, description="The largest file, in bytes (256 MiB)"
    )
    max_pages: int = Field(default=2000, ge=1, description="The most pages a document may have")
    unknown_types: Literal["allow", "refuse"] = Field(
        default="allow",
        description="Extensions Stratafi doesn't list: allowed, typed by their bytes, or "
        "refused as malformed",
    )


class ContentSafety(ScanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="scan/content-safety",
        stage="scan",
        version="1.0.0",
        title="Content safety",
        description="Magic bytes against the extension, archive limits, size and page limits, "
        "and encryption and corruption probes.",
        license="MIT",
        baseline=True,
    )
    Params: ClassVar[type[BaseModel]] = ContentSafetyParams

    def check(self, scan: ScanInput, params: BaseModel, context: StageContext) -> CheckResult:
        params = cast(ContentSafetyParams, params)
        raw = scan.raw
        if raw.mismatch:
            return CheckResult("malformed", "media_type_mismatch")
        if not raw.known and params.unknown_types == "refuse":
            return CheckResult("malformed", "unknown_type")
        if raw.size_bytes > params.max_bytes:
            return CheckResult("limit_exceeded", "file_size")
        if raw.traits.get("encrypted"):
            return CheckResult("encrypted", f"{raw.kind}_encrypted")
        if raw.traits.get("unreadable"):
            reason = "archive_malformed" if raw.kind == "zip" else f"{raw.kind}_unreadable"
            return CheckResult("malformed", reason)
        pages = raw.traits.get("pages")
        if isinstance(pages, int) and pages > params.max_pages:
            return CheckResult("limit_exceeded", "pages")
        if raw.kind == "zip" or raw.kind in OOXML_KINDS:
            # Office files are ZIPs of XML that compress well: they're allowed a ratio of 1,000.
            ratio = 1000 if raw.kind in OOXML_KINDS else 100
            try:
                measure_zip(
                    scan.path, ArchiveLimits(max_ratio=ratio), checkpoint=context.checkpoint
                )
            except ArchiveRefused as refused:
                return CheckResult(refused.outcome, refused.reason)
        return CLEAN


# --- scan/clamav -----------------------------------------------------------------------------


class ClamavParams(_Params):
    max_signature_age_hours: int = Field(
        default=72, ge=1, description="Signatures older than this refuse a pass"
    )
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)


class Clamav(ScanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="scan/clamav",
        stage="scan",
        version="1.0.0",
        title="ClamAV",
        description="clamd INSTREAM. Refuses to pass a file when its signatures are more than "
        "72 hours old, since a stale database misses new threats.",
        resource_class="io",
        license="MIT",
        baseline=True,
        services=("clamd",),
    )
    Params: ClassVar[type[BaseModel]] = ClamavParams

    def check(self, scan: ScanInput, params: BaseModel, context: StageContext) -> CheckResult:
        params = cast(ClamavParams, params)
        client = ClamdClient(context.service(CLAMD_SERVICE), timeout=params.timeout_seconds)
        now = context.clock()
        try:
            identity = parse_version(client.version(), now=now)
            details: dict[str, str | int | bool | None] = {
                "engine": identity.engine,
                "signature_revision": identity.revision,
            }
            age = (now - identity.built_at).total_seconds() / 3600
            if age > params.max_signature_age_hours:
                return CheckResult("error", "signatures_stale", details=details)
            context.checkpoint()
            reply = client.instream(scan.path)
            after = parse_version(client.version(), now=context.clock())
        except ClamdError as error:
            if error.reason in ("connection", "timeout"):
                raise ComponentError(f"clamd_{error.reason}", retryable=True) from None
            return CheckResult("error", error.reason)
        if after != identity:
            return CheckResult("error", "signature_identity_changed", details=details)
        if reply == "stream: OK":
            return CheckResult("clean", details=details)
        if reply.startswith("stream: ") and reply.endswith(" FOUND"):
            signature = reply.removeprefix("stream: ").removesuffix(" FOUND")
            # clamd reports a file it couldn't finish within its limits as a find.
            if signature.startswith("Heuristics.Limits.Exceeded"):
                return CheckResult("limit_exceeded", "scanner_limit", details=details)
            return CheckResult("infected", details={**details, "signature": signature[:128]})
        if "size limit exceeded" in reply.lower():
            return CheckResult("limit_exceeded", "scanner_limit", details=details)
        return CheckResult("error", "invalid_response", details=details)


# --- scan/pii ----------------------------------------------------------------------------------

# Kinds whose bytes are text (``stratafi_components.detect``): read as they are, markup included.
TEXT_KINDS = frozenset({"text", "html", "xml", "svg"})
# Where an Office file keeps its text: the XML parts under its main folder, and its properties.
OOXML_TEXT_FOLDERS = {"docx": "word/", "xlsx": "xl/", "pptx": "ppt/"}
# The most of one Office part the scan reads; a larger part is past its limit.
MAX_PART_BYTES = 64 * 1024 * 1024
# How an Office part becomes text: paragraph, row and line ends become line breaks, cell ends and
# tabs become tabs, and every other tag goes, so a value split across runs reads whole.
_XML_LINE_END = re.compile(rb"</(?:w:p|a:p|w:tr|a:tr|row|si)>|<(?:w:br|w:cr|a:br)\b[^>]*>")
_XML_CELL_END = re.compile(rb"</(?:w:tc|a:tc|c)>|<w:tab\b[^>]*>")
_XML_TAG = re.compile(rb"<[^>]*>")


class Unreadable(Exception):
    """A document whose text the scan can't read in process."""


class TooLong(Exception):
    """A document with more text than the scan reads."""


def _part_text(data: bytes) -> str:
    lines = _XML_CELL_END.sub(b"\t", _XML_LINE_END.sub(b"\n", data))
    return html.unescape(_XML_TAG.sub(b"", lines).decode("utf-8", errors="replace"))


def _office_text(path: Path, kind: str, max_chars: int, checkpoint: Callable[[], None]) -> str:
    folder = OOXML_TEXT_FOLDERS[kind]
    texts: list[str] = []
    total = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                name = info.filename
                if not name.endswith(".xml") or not name.startswith((folder, "docProps/")):
                    continue
                checkpoint()
                with archive.open(info) as part:
                    data = part.read(MAX_PART_BYTES + 1)
                if len(data) > MAX_PART_BYTES:
                    raise TooLong()
                text = _part_text(data)
                total += len(text) + 1
                if total > max_chars:
                    raise TooLong()
                texts.append(text)
    except (TooLong, Cancelled):
        raise
    except Exception:  # untrusted input: any failure means the text can't be read
        raise Unreadable() from None
    return "\n".join(texts)


def _pdf_text(path: Path, max_chars: int, checkpoint: Callable[[], None]) -> str:
    import pypdf

    texts: list[str] = []
    total = 0
    try:
        reader = pypdf.PdfReader(str(path))
        # An "encrypted" PDF with an empty user password still opens.
        if reader.is_encrypted and not reader.decrypt(""):
            raise Unreadable()
        for page in reader.pages:
            checkpoint()
            text = page.extract_text() or ""
            total += len(text) + 1
            if total > max_chars:
                raise TooLong()
            texts.append(text)
    except (Unreadable, TooLong, Cancelled):
        raise
    except Exception:  # untrusted input: any failure means the text can't be read
        raise Unreadable() from None
    return "\n".join(texts)


def document_text(
    path: Path, raw: RawDocument, *, max_chars: int, checkpoint: Callable[[], None] = lambda: None
) -> str:
    """The text ``scan/pii`` reads from a raw document: text and markup as they are, a PDF's
    text layer, an Office file's XML text. Raises ``Unreadable`` for anything else (images,
    legacy Office files, archives, a file that won't open) and ``TooLong`` past ``max_chars``."""
    if raw.traits.get("unreadable"):
        raise Unreadable()
    if raw.kind in TEXT_KINDS:
        # A character takes at most four bytes, so a larger file holds too much text.
        if raw.size_bytes > 4 * max_chars:
            raise TooLong()
        text = decode_text(path.read_bytes())
    elif raw.kind == "pdf":
        text = _pdf_text(path, max_chars, checkpoint)
    elif raw.kind in OOXML_TEXT_FOLDERS:
        text = _office_text(path, raw.kind, max_chars, checkpoint)
    else:
        raise Unreadable()
    if len(text) > max_chars:
        raise TooLong()
    return text


class PiiParams(_Params):
    categories: list[PiiCategory] = Field(
        default_factory=lambda: list(DEFAULT_CATEGORIES),
        min_length=1,
        description="What it looks for: ssn, card_number and iban by default; email, phone and "
        "ip_address on request",
    )
    max_chars: int = Field(
        default=10_000_000,
        ge=1000,
        le=500_000_000,
        description="The most text it reads from one document; a longer one is limit_exceeded",
    )
    unreadable: Literal["refuse", "allow"] = Field(
        default="refuse",
        description="A file whose text it can't read (an image, a legacy Office file, an "
        "archive): refused as malformed, or allowed through unscanned",
    )

    @field_validator("categories")
    @classmethod
    def _canonical(cls, value: list[str]) -> list[str]:
        return canonical_categories(value)


class Pii(ScanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="scan/pii",
        stage="scan",
        version="1.0.0",
        title="Personal data (PII)",
        description="Finds personal data in each document's text, in process: SSNs, payment "
        "card numbers (Luhn) and IBANs (mod 97) by default; email addresses, US phone numbers "
        "and IPv4 addresses on request. Reads text, PDF and Office files. Block-only: counts "
        "by category, never values; redaction lives in Clean.",
        license="MIT",
        baseline=True,
    )
    Params: ClassVar[type[BaseModel]] = PiiParams

    def check(self, scan: ScanInput, params: BaseModel, context: StageContext) -> CheckResult:
        params = cast(PiiParams, params)
        try:
            text = document_text(
                scan.path, scan.raw, max_chars=params.max_chars, checkpoint=context.checkpoint
            )
        except Unreadable:
            if params.unreadable == "allow":
                return CheckResult("clean", details={"scanned_chars": 0, "unscannable": True})
            return CheckResult("malformed", "unscannable")
        except TooLong:
            return CheckResult("limit_exceeded", "scan_limit")
        details: dict[str, str | int | bool | None] = {"scanned_chars": len(text)}
        found = count_by_category(find_pii(text, params.categories))
        if found:
            return CheckResult("pii", findings=found, details=details)
        return CheckResult("clean", details=details)


# --- scan/dedupe -----------------------------------------------------------------------------


class Dedupe(ScanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="scan/dedupe",
        stage="scan",
        version="1.0.0",
        title="Exact duplicates",
        description="Exact duplicates by SHA-256 within the corpus version: the first entry by "
        "name is kept, the others are duplicates.",
        license="MIT",
        baseline=True,
    )

    def check(self, scan: ScanInput, params: BaseModel, context: StageContext) -> CheckResult:
        return CheckResult("duplicate", "same_bytes_as_earlier_entry") if scan.duplicate else CLEAN


# --- the stage ---------------------------------------------------------------------------------


def run_scan(
    scan: ScanInput,
    stage: ScanStage,
    registry: ComponentRegistry,
    context: StageContext,
) -> ScanReport:
    """Run the stage's checks in order and apply its policy."""
    checks: list[ScanCheck] = []
    stopped = False
    for ref in stage.checks:
        component_type = registry.get(ref.component)
        if stopped:
            checks.append(
                ScanCheck(
                    component=ref.component,
                    version=component_type.info.version,
                    outcome="skipped",
                )
            )
            continue
        component = cast(ScanComponent, component_type())
        context.checkpoint()
        result = component.check(scan, component.resolve_params(ref.params), context)
        checks.append(
            ScanCheck(
                component=ref.component,
                version=component_type.info.version,
                outcome=result.outcome,  # type: ignore[arg-type]
                reason=result.reason,
                findings=dict(result.findings or {}),
                details=dict(result.details or {}),
            )
        )
        if result.outcome != "clean" and stage.on_outcome[result.outcome] == "exclude":
            stopped = True
    outcomes = [check.outcome for check in checks if check.outcome not in ("clean", "skipped")]
    outcome = next((o for o in OUTCOME_PRIORITY if o in outcomes), "clean")
    decision = max(
        (stage.on_outcome[o] for o in outcomes), key=DECISION_RANK.__getitem__, default="continue"
    )
    findings: dict[str, int] = {}
    for check in checks:
        for category, count in check.findings.items():
            findings[category] = findings.get(category, 0) + count
    return ScanReport(
        document_id=scan.raw.document_id,
        sha256=scan.raw.sha256,
        outcome=outcome,  # type: ignore[arg-type]
        decision=decision,  # type: ignore[arg-type]
        checks=checks,
        findings=findings,
    )


COMPONENTS: list[type[Component]] = [ContentSafety, Clamav, Pii, Dedupe]
