"""The scan stage: every outcome path, the policy, and no finding values in logs
or reports. ClamAV is the in-process clamd fake; the PII scan runs in process. Every value that
looks like personal data here is synthetic: test SSNs, published test card numbers and the
standard example IBAN."""

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stratafi_components import default_registry
from stratafi_components.extract import ExtractUpload
from stratafi_components.scan import ScanInput, run_scan
from stratafi_core.artifacts import ScanReport
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.specs import resolve_pipeline
from stratafi_testing.canary import CanaryCheck
from stratafi_testing.clamd import eicar, fake_clamd
from stratafi_testing.components import stub_registry
from stratafi_testing.documents import (
    Block,
    docx_bytes,
    encrypted_pdf_bytes,
    pdf_bytes,
    zip_bytes,
)

REGISTRY = default_registry()
for _component in stub_registry():
    if _component.info.id not in REGISTRY:
        REGISTRY.register(_component)

BASELINE_CHECKS = ["scan/content-safety", "scan/clamav", "scan/pii", "scan/dedupe"]
TEST_SSN = "123-45-6789"
TEST_CARD = "4111 1111 1111 1111"
TEST_IBAN = "GB82 WEST 1234 5698 7654 32"


def scan_stage(checks: list | None = None, **policy: str):
    spec = resolve_pipeline(
        {
            "schema": "stratafi.pipeline.v1",
            "name": "scan",
            "stages": {"scan": {"checks": checks or BASELINE_CHECKS, "on_outcome": policy}},
        },
        REGISTRY,
    )
    return spec.stages.scan


@pytest.fixture
def clamd() -> Iterator[tuple[str, int]]:
    with fake_clamd() as address:
        yield address


@pytest.fixture
def context(clamd) -> StageContext:
    return StageContext(services={"clamd": clamd})


def scan_bytes(
    tmp_path: Path,
    data: bytes,
    name: str,
    context: StageContext,
    *,
    duplicate: bool = False,
    stage=None,
) -> ScanReport:
    path = tmp_path / "raw"
    path.write_bytes(data)
    raw = ExtractUpload().extract(
        path,
        name,
        document_id="d-1",
        expected_sha256=hashlib.sha256(data).hexdigest(),
        params=ExtractUpload.resolve_params({}),
        context=StageContext(),
    )
    return run_scan(ScanInput(path, raw, duplicate), stage or scan_stage(), REGISTRY, context)


def outcomes(report: ScanReport) -> list[tuple[str, str]]:
    return [(check.component, check.outcome) for check in report.checks]


def pii_check(report: ScanReport):
    return next(check for check in report.checks if check.component == "scan/pii")


PDF = pdf_bytes("Policy", [Block("paragraph", "Keep records for seven years.")])


def test_a_clean_document_passes_every_check(tmp_path: Path, context) -> None:
    report = scan_bytes(tmp_path, PDF, "policy.pdf", context)
    assert (report.outcome, report.decision) == ("clean", "continue")
    assert all(outcome == "clean" for _, outcome in outcomes(report))
    clamav = report.checks[1]
    assert clamav.details["engine"] == "clamav-1.4.3"
    assert clamav.details["signature_revision"] == "27800"
    assert pii_check(report).details["scanned_chars"] > 0


def test_infected_eicar_through_the_fake(tmp_path: Path, context) -> None:
    report = scan_bytes(tmp_path, eicar(), "notes.txt", context)
    assert (report.outcome, report.decision) == ("infected", "exclude")
    assert report.checks[1].details["signature"] == "Eicar-Test-Signature"
    # Excluded at ClamAV: the file is never opened to read its text.
    assert outcomes(report)[2:] == [("scan/pii", "skipped"), ("scan/dedupe", "skipped")]


def test_encrypted(tmp_path: Path, context) -> None:
    report = scan_bytes(tmp_path, encrypted_pdf_bytes(), "locked.pdf", context)
    assert (report.outcome, report.checks[0].reason) == ("encrypted", "pdf_encrypted")
    zipped = zip_bytes({"a.txt": b"x"})
    flagged = bytearray(zipped)
    central = flagged.rfind(b"PK\x01\x02")
    flagged[central + 8] |= 0x1
    report = scan_bytes(tmp_path, bytes(flagged), "bundle.zip", context)
    assert (report.outcome, report.checks[0].reason) == ("encrypted", "archive_encrypted")


@pytest.mark.parametrize(
    ("data", "name", "reason"),
    [
        (b"Not a PDF at all.", "policy.pdf", "media_type_mismatch"),
        (b"%PDF-1.4\nbroken", "policy.pdf", "pdf_unreadable"),
        (b"PK\x03\x04 then garbage", "bundle.zip", "archive_malformed"),
    ],
)
def test_malformed(tmp_path: Path, context, data: bytes, name: str, reason: str) -> None:
    report = scan_bytes(tmp_path, data, name, context)
    assert (report.outcome, report.checks[0].reason) == ("malformed", reason)


def test_limit_exceeded(tmp_path: Path, context) -> None:
    stage = scan_stage(
        [{"component": "scan/content-safety", "params": {"max_bytes": 100}}, "scan/clamav"]
    )
    report = scan_bytes(tmp_path, PDF, "policy.pdf", context, stage=stage)
    assert (report.outcome, report.checks[0].reason) == ("limit_exceeded", "file_size")
    long = pdf_bytes("Long", [Block("paragraph", "Words. " * 400) for _ in range(12)])
    stage = scan_stage(
        [{"component": "scan/content-safety", "params": {"max_pages": 2}}, "scan/clamav"]
    )
    report = scan_bytes(tmp_path, long, "long.pdf", context, stage=stage)
    assert (report.outcome, report.checks[0].reason) == ("limit_exceeded", "pages")
    bomb = zip_bytes({"zeros.bin": b"\0" * (16 * 1024 * 1024)})
    report = scan_bytes(tmp_path, bomb, "bomb.zip", context)
    assert (report.outcome, report.checks[0].reason) == (
        "limit_exceeded",
        "archive_expansion_limit",
    )


def test_office_files_get_the_office_ratio(tmp_path: Path, context) -> None:
    # A DOCX of repetitive text compresses past 100:1, which office files are allowed.
    docx = docx_bytes("Repeats", [Block("paragraph", "same words " * 20000)])
    report = scan_bytes(tmp_path, docx, "repeats.docx", context)
    assert report.outcome == "clean", report.checks[0]


def test_pii_in_text_is_counted_by_category(tmp_path: Path, context) -> None:
    text = f"Test record: SSN {TEST_SSN}, card {TEST_CARD}, another SSN 234-56-7890."
    report = scan_bytes(tmp_path, text.encode(), "record.txt", context)
    assert (report.outcome, report.decision) == ("pii", "exclude")
    assert pii_check(report).findings == {"ssn": 2, "card_number": 1}
    assert report.findings == {"ssn": 2, "card_number": 1}


@pytest.mark.parametrize("kind", ["pdf", "docx"])
def test_pii_is_read_from_pdf_and_office_text(tmp_path: Path, context, kind: str) -> None:
    blocks = [Block("paragraph", "Payroll test record."), Block("paragraph", f"IBAN {TEST_IBAN}")]
    data = pdf_bytes("Payroll", blocks) if kind == "pdf" else docx_bytes("Payroll", blocks)
    report = scan_bytes(tmp_path, data, f"payroll.{kind}", context)
    assert report.outcome == "pii"
    assert pii_check(report).findings == {"iban": 1}


def test_a_value_split_across_word_runs_is_found(tmp_path: Path, context) -> None:
    runs = (
        '<w:p><w:r><w:t xml:space="preserve">SSN 123-45-</w:t></w:r>'
        "<w:r><w:t>6789</w:t></w:r></w:p>"
    )
    data = docx_bytes("Runs", [Block("paragraph", "A test record.")], extra_body_xml=runs)
    assert pii_check(scan_bytes(tmp_path, data, "runs.docx", context)).findings == {"ssn": 1}


def test_categories_choose_what_counts(tmp_path: Path, context) -> None:
    text = b"Write to test.user@example.com or call (312) 555-0142."
    assert scan_bytes(tmp_path, text, "contact.txt", context).outcome == "clean"
    stage = scan_stage(
        [
            "scan/content-safety",
            {"component": "scan/pii", "params": {"categories": ["phone", "email", "phone"]}},
        ]
    )
    assert stage.checks[1].params["categories"] == ["email", "phone"]
    report = scan_bytes(tmp_path, text, "contact.txt", context, stage=stage)
    assert report.findings == {"email": 1, "phone": 1}


def test_unreadable_files_are_refused_unless_allowed(tmp_path: Path, context) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
    report = scan_bytes(tmp_path, png, "scan.png", context)
    assert (report.outcome, pii_check(report).reason) == ("malformed", "unscannable")
    stage = scan_stage(
        ["scan/content-safety", {"component": "scan/pii", "params": {"unreadable": "allow"}}]
    )
    allowed = pii_check(scan_bytes(tmp_path, png, "scan.png", context, stage=stage))
    assert (allowed.outcome, allowed.details) == (
        "clean",
        {"scanned_chars": 0, "unscannable": True},
    )


def test_too_much_text_is_limit_exceeded(tmp_path: Path, context) -> None:
    stage = scan_stage(
        ["scan/content-safety", {"component": "scan/pii", "params": {"max_chars": 1000}}]
    )
    report = scan_bytes(tmp_path, b"word " * 300, "long.txt", context, stage=stage)
    assert (report.outcome, report.checks[1].reason) == ("limit_exceeded", "scan_limit")


def test_duplicate(tmp_path: Path, context) -> None:
    report = scan_bytes(tmp_path, PDF, "copy.pdf", context, duplicate=True)
    assert (report.outcome, report.decision) == ("duplicate", "exclude")
    assert report.checks[-1].reason == "same_bytes_as_earlier_entry"


def test_stale_signatures_refuse_a_pass(tmp_path: Path) -> None:
    week_ago = datetime.now(UTC) - timedelta(days=7)
    with fake_clamd(signature_time=week_ago) as address:
        context = StageContext(services={"clamd": address})
        report = scan_bytes(
            tmp_path, PDF, "policy.pdf", context, stage=scan_stage(BASELINE_CHECKS[:2])
        )
    assert (report.outcome, report.checks[1].reason) == ("error", "signatures_stale")
    assert report.decision == "exclude"


def test_a_long_signature_age_limit_accepts_older_signatures(tmp_path: Path) -> None:
    with fake_clamd(signature_time=datetime.now(UTC) - timedelta(days=7)) as address:
        stage = scan_stage(
            [
                "scan/content-safety",
                {"component": "scan/clamav", "params": {"max_signature_age_hours": 240}},
            ]
        )
        report = scan_bytes(
            tmp_path, PDF, "p.pdf", StageContext(services={"clamd": address}), stage=stage
        )
    assert report.outcome == "clean"


def test_an_unreachable_clamd_is_retryable(tmp_path: Path) -> None:
    context = StageContext(services={"clamd": ("127.0.0.1", 9)})
    with pytest.raises(ComponentError) as raised:
        scan_bytes(tmp_path, PDF, "p.pdf", context, stage=scan_stage(BASELINE_CHECKS[:2]))
    assert (raised.value.code, raised.value.retryable) == ("clamd_connection", True)


def test_a_missing_service_names_itself(tmp_path: Path) -> None:
    with pytest.raises(ComponentError) as raised:
        scan_bytes(tmp_path, PDF, "p.pdf", StageContext(), stage=scan_stage(BASELINE_CHECKS[:2]))
    assert raised.value.code == "clamd_not_configured"


def test_the_pii_scan_needs_no_service(tmp_path: Path) -> None:
    stage = scan_stage(["scan/content-safety", "scan/pii"])
    report = scan_bytes(tmp_path, f"SSN {TEST_SSN}".encode(), "a.txt", StageContext(), stage=stage)
    assert report.outcome == "pii"


def test_the_policy_can_flag_instead_of_exclude(tmp_path: Path, context) -> None:
    stage = scan_stage(duplicate="flag", pii="flag")
    report = scan_bytes(
        tmp_path, f"SSN {TEST_SSN}".encode(), "a.txt", context, duplicate=True, stage=stage
    )
    assert report.decision == "flag"
    assert report.outcome == "pii"  # it outranks duplicate
    assert [outcome for _, outcome in outcomes(report)] == ["clean", "clean", "pii", "duplicate"]


def test_unknown_types_pass_unless_the_policy_refuses_them(tmp_path: Path, context) -> None:
    assert scan_bytes(tmp_path, b"plain words", "README", context).outcome == "clean"
    stage = scan_stage(
        [{"component": "scan/content-safety", "params": {"unknown_types": "refuse"}}, "scan/clamav"]
    )
    report = scan_bytes(tmp_path, b"plain words", "README", context, stage=stage)
    assert (report.outcome, report.checks[0].reason) == ("malformed", "unknown_type")


def test_finding_values_never_reach_logs_or_reports(
    tmp_path: Path, context, canary: CanaryCheck
) -> None:
    text = f"{canary.value} test record SSN {TEST_SSN} and card {TEST_CARD} on file".encode()
    report = scan_bytes(tmp_path, text, "letter.txt", context)
    assert report.outcome == "pii"
    serialized = json.dumps(report.model_dump(mode="json", by_alias=True))
    for value in (TEST_SSN, TEST_CARD, TEST_CARD.replace(" ", "")):
        assert value not in serialized
        assert value not in canary.captured_text()
    canary.assert_absent(serialized, canary.captured_text())
