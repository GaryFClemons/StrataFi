"""``scan/clamav`` against a real clamd (integration tier): it detects EICAR, and
stale signatures refuse a pass.

The image's signatures may be older than 72 hours when freshclam can't reach its mirror (in a
sandbox, say), so the detection test allows any age; the staleness test moves the clock forward
instead, which works whatever the signatures' age."""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stratafi_components import default_registry
from stratafi_components.extract import ExtractUpload
from stratafi_components.scan import ScanInput, run_scan
from stratafi_core.components import StageContext
from stratafi_core.specs import resolve_pipeline
from stratafi_testing.clamd import eicar
from stratafi_testing.components import stub_registry
from stratafi_testing.documents import Block, pdf_bytes

pytestmark = pytest.mark.integration

REGISTRY = default_registry()
for _component in stub_registry():
    if _component.info.id not in REGISTRY:
        REGISTRY.register(_component)


def scan(path: Path, data: bytes, name: str, context: StageContext, max_age_hours: int):
    path.write_bytes(data)
    raw = ExtractUpload().extract(
        path,
        name,
        document_id="d-1",
        expected_sha256=hashlib.sha256(data).hexdigest(),
        params=ExtractUpload.resolve_params({}),
        context=StageContext(),
    )
    spec = resolve_pipeline(
        {
            "schema": "stratafi.pipeline.v1",
            "name": "clamav",
            "stages": {
                "scan": {
                    "checks": [
                        "scan/content-safety",
                        {
                            "component": "scan/clamav",
                            "params": {"max_signature_age_hours": max_age_hours},
                        },
                    ]
                }
            },
        },
        REGISTRY,
    )
    return run_scan(ScanInput(path, raw), spec.stages.scan, REGISTRY, context)


@pytest.fixture
def context(compose_services) -> StageContext:
    return StageContext(services={"clamd": compose_services("clamav").clamd_address()})


def test_real_clamav_detects_eicar_and_passes_clean_files(tmp_path: Path, context) -> None:
    any_age = 1_000_000
    infected = scan(tmp_path / "a", eicar(), "eicar.txt", context, any_age)
    assert (infected.outcome, infected.decision) == ("infected", "exclude")
    clamav = infected.checks[1]
    assert "Eicar" in str(clamav.details["signature"])
    assert str(clamav.details["engine"]).startswith("clamav-")
    clean = scan(
        tmp_path / "b",
        pdf_bytes("Policy", [Block("paragraph", "Fine.")]),
        "p.pdf",
        context,
        any_age,
    )
    assert clean.outcome == "clean"


def test_stale_signatures_refuse_a_pass(tmp_path: Path, context) -> None:
    later = datetime.now(UTC) + timedelta(days=400)
    context.clock = lambda: later
    report = scan(tmp_path / "a", eicar(), "eicar.txt", context, 72)
    assert (report.outcome, report.checks[1].reason) == ("error", "signatures_stale")
    assert report.decision == "exclude"
