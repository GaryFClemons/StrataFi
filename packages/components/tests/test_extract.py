"""The extract stage at run time: stored bytes are checked against their hash and
typed by their entry's name."""

import hashlib
from pathlib import Path

import pytest

from stratafi_components import default_registry
from stratafi_components.extract import ExtractUpload
from stratafi_core.components import ComponentError, StageContext
from stratafi_testing.documents import blank_pdf_bytes


def test_the_extract_stage_records_type_and_traits(tmp_path: Path) -> None:
    data = blank_pdf_bytes(2)
    path = tmp_path / "stored"
    path.write_bytes(data)
    component = ExtractUpload()
    context = StageContext()
    raw = component.extract(
        path,
        "scans/letter.pdf",
        document_id="d-1",
        expected_sha256=hashlib.sha256(data).hexdigest(),
        params=component.resolve_params({}),
        context=context,
    )
    assert (raw.media_type, raw.extension, raw.mismatch) == ("application/pdf", ".pdf", False)
    assert raw.traits == {"pages": 2, "scanned": True, "partly_scanned": False}
    assert raw.trait_names == {"scanned"}
    assert context.metrics == {"bytes": len(data)}
    same_bytes_other_name = component.extract(
        path,
        "letter.txt",
        document_id="d-1",
        expected_sha256=hashlib.sha256(data).hexdigest(),
        params=component.resolve_params({}),
        context=StageContext(),
    )
    assert same_bytes_other_name.mismatch is True


def test_bytes_that_dont_match_their_hash_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "stored"
    path.write_bytes(b"tampered")
    with pytest.raises(ComponentError) as raised:
        ExtractUpload().extract(
            path,
            "a.md",
            document_id="d-1",
            expected_sha256="0" * 64,
            params=ExtractUpload.resolve_params({}),
            context=StageContext(),
        )
    assert raised.value.code == "hash_mismatch"


def test_the_extract_components_are_registered() -> None:
    registry = default_registry()
    assert {"extract/upload", "extract/folder", "unpack/zip"} <= {c.info.id for c in registry}
    unpack = registry.describe("unpack/zip").params_schema["properties"]
    assert unpack["max_entries"]["default"] == 10_000
    assert unpack["max_ratio"]["maximum"] == 100
