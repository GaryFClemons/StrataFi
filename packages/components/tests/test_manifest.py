"""Corpus manifests (``extract/manifest``)."""

from datetime import date
from pathlib import Path

import pytest
import yaml

from stratafi_components import default_registry
from stratafi_components.manifest import load_corpus_manifest
from stratafi_core.specs import SpecInvalid

MANIFEST = Path(__file__).resolve().parents[3] / "fixtures" / "demo-corpus" / "corpus"


def test_the_demo_corpus_manifest_loads_with_every_field() -> None:
    manifest = load_corpus_manifest((MANIFEST / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest.tenant == "halcyon"
    assert len(manifest.documents) == 24
    superseded = next(d for d in manifest.documents if d.path.endswith("pto-leave-v2.md"))
    assert superseded.entry_fields() == {
        "doc_key": "pto-leave",
        "doc_version": "2",
        "collection": "handbook",
        "acl": {"groups": ["all-staff"], "users": []},
        "effective_from": date(2024, 1, 1),
        "effective_to": date(2026, 1, 1),
        "metadata": {
            "status": "published",
            "title": "Paid Time Off and Leave Policy",
            "source_uri": "corpus/handbook/pto-leave-v2.md",
            "expect": "indexed",
        },
    }
    restricted = {d.collection: d.acl.groups for d in manifest.documents}
    assert restricted["management"] == ("managers",)
    assert restricted["executive"] == ("executives",)
    for document in manifest.documents:
        assert (MANIFEST / document.path).is_file()


@pytest.mark.parametrize(
    ("change", "path", "code"),
    [
        ({"acl": {"groups": [], "users": []}}, "documents.0.acl", "value_error"),
        ({"effective_to": "2024-01-01"}, "documents.0", "value_error"),
        ({"doc_key": "Not A Key"}, "documents.0.doc_key", "string_pattern_mismatch"),
        ({"status": "archived"}, "documents.0.status", "literal_error"),
    ],
)
def test_invalid_manifests_fail_closed_with_paths(change: dict, path: str, code: str) -> None:
    document = {
        "path": "a.md",
        "collection": "policies",
        "doc_key": "a",
        "version": 1,
        "effective_from": "2025-01-01",
        "acl": {"groups": ["all-staff"]},
        **change,
    }
    with pytest.raises(SpecInvalid) as error:
        load_corpus_manifest(yaml.safe_dump({"documents": [document]}))
    assert (path, code) in [(issue.path, issue.code) for issue in error.value.issues]


def test_a_numeric_version_is_text_and_extract_manifest_is_registered() -> None:
    text = yaml.safe_dump(
        {
            "documents": [
                {
                    "path": "a.md",
                    "collection": "policies",
                    "doc_key": "a",
                    "version": 3,
                    "effective_from": "2025-01-01",
                    "acl": {"users": ["u-halcyon-staff"]},
                }
            ]
        }
    )
    assert load_corpus_manifest(text).documents[0].version == "3"
    info = default_registry().get("extract/manifest").info
    assert (info.stage, info.resource_class) == ("extract", "io")
