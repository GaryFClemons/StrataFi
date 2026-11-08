"""The Halcyon demo corpus's variants import as separate corpus versions that share document keys,
and a variant answers the same queries.

``scripts/corpus_variants.py --import-as`` renders the five variants and imports each through
a lab's API, served over HTTP, as its own corpus. The DOCX variant then runs through the
default components (DOCX through ``parse/markitdown``) and is evaluated with the demo set's
queries.
"""

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

from stratafi_testing.lab import (
    HALCYON_RETRIEVAL,
    baseline_pipeline,
    evaluate,
    hybrid_retrieval,
    running_lab,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "fixtures" / "demo-corpus" / "corpus"
MEDIA_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "scanned-pdf": "application/pdf",
    "html": "text/html",
    "pdf-table-images": "application/pdf",
}


def load() -> Any:
    path = ROOT / "scripts" / "corpus_variants.py"
    spec = importlib.util.spec_from_file_location("script_corpus_variants", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_halcyon_variants_import_as_corpora_that_share_document_keys(
    migrated_engine: Any,
    dev_identity: Any,
    tmp_path: Path,
    serve_app: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    variants = load()
    source = yaml.safe_load((CORPUS / "manifest.yaml").read_text(encoding="utf-8"))
    keys = {(entry["doc_key"], str(entry["version"])) for entry in source["documents"]}
    with running_lab(migrated_engine, dev_identity, tmp_path / "lab", serve_app) as lab:
        monkeypatch.setenv("STRATAFI_URL", serve_app(lab.app))
        monkeypatch.setenv("STRATAFI_DEV_SIGN_IN", "tenant0:editor")
        monkeypatch.delenv("STRATAFI_TOKEN", raising=False)
        argv = [str(CORPUS), str(tmp_path / "variants"), "--import-as", "halcyon"]
        assert variants.main(argv) == 0
        out = capsys.readouterr().out

        corpora = {corpus["name"]: corpus for corpus in lab.get("/corpora")["items"]}
        for variant, media_type in MEDIA_TYPES.items():
            name = f"halcyon-{variant}"
            assert f"{name} version 1: {len(keys)} documents" in out
            path = f"/corpora/{corpora[name]['id']}/versions/1/documents"
            entries = lab.get(path, limit=200)["items"]
            assert {(entry["doc_key"], entry["doc_version"]) for entry in entries} == keys
            assert {entry["media_type"] for entry in entries} == {media_type}
            assert all(entry["acl"] for entry in entries)  # the manifest's metadata came along

        # The DOCX variant runs through the default components and answers the same queries.
        detail = lab.get(f"/corpora/{corpora['halcyon-docx']['id']}")
        run = lab.wait_run(
            lab.start_run(lab.save_spec(baseline_pipeline()), detail["latest_version"]["id"])["id"]
        )
        assert run["status"] == "succeeded"
        imported = lab.import_eval_set(
            "halcyon-retrieval", HALCYON_RETRIEVAL.read_text(encoding="utf-8"), seed=0
        )
        inputs = {
            "run_id": run["id"],
            "retrieval_config_id": lab.save_retrieval(hybrid_retrieval()),
            "eval_set_version_id": imported["version"]["id"],
        }
        result = evaluate(lab, inputs)
        assert result["status"] == "succeeded"
        assert result["counts"]["queries"] == imported["version"]["split_counts"]["dev"]
        findability = result["metrics"]["findability"]
        assert findability["rate"] == 1.0  # every evidence phrase survived DOCX parsing
        assert set(findability["by_file_type"]) == {MEDIA_TYPES["docx"]}
