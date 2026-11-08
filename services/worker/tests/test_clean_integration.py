"""The cleaning rules in runs, against Postgres: near duplicates wait for the corpus and
stop after Clean; a redacted value appears in no artifact or response; the diff API. And the
Halcyon demo corpus passes the PII scan with its default categories: every document is indexed.
The SSN here is a synthetic test value."""

import json
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from stratafi_testing.canary import CanaryCheck
from stratafi_testing.documents import Block, markdown_text
from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import baseline_pipeline, running_lab

pytestmark = pytest.mark.integration

SSN = "123-45-6789"
RULES = [
    f"Rule {i}: managers confirm every refund request by calling the number on file."
    for i in range(30)
]


def document(title: str, paragraphs: list[str]) -> bytes:
    return markdown_text(title, [Block("paragraph", text) for text in paragraphs]).encode()


def spec(clean: list[Any]) -> dict[str, Any]:
    return {
        "schema": "stratafi.pipeline.v1",
        "name": "clean-test",
        "stages": {
            "clean": clean,
            "embed": {"component": "embed/fake-hash", "params": {"dimensions": 64}},
        },
    }


def test_near_duplicates_wait_for_the_corpus_and_keep_the_newest(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    files = {
        "policy-v1.md": document("Refund Policy", RULES),
        "policy-v2.md": document("Refund Policy", [*RULES[:-1], "Rule 29: revised wording."]),
        "hours.md": document("Hours", [f"Café {i} opens at seven." for i in range(30)]),
    }
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        version = lab.upload_corpus("near", files)
        clean = [
            "clean/whitespace",
            {"component": "clean/near-duplicates", "params": {"threshold": 0.8}},
        ]
        run = lab.wait_run(lab.start_run(lab.save_spec(spec(clean)), version)["id"])
        assert run["status"] == "succeeded", run
        entries = {e["name"]: e for e in lab.get(f"/runs/{run['id']}/entries")["items"]}
        assert entries["policy-v1.md"]["state"] == "excluded"
        assert entries["policy-v2.md"]["state"] == "done"
        assert entries["hours.md"]["state"] == "done"
        cleaned = {task["entry"]: task for task in lab.tasks(run["id"], "clean")}
        assert cleaned["policy-v1.md"]["summary"]["excluded"] == "near_duplicate"
        # Each clean task read every parsed document: its own first.
        assert all(len(task["inputs"]) == 3 for task in cleaned.values())
        assert not lab.tasks(run["id"], "chunk") or "policy-v1.md" not in {
            task["entry"] for task in lab.tasks(run["id"], "chunk")
        }
        # An identical second run is all cache hits, the barrier included.
        again = lab.wait_run(lab.start_run(lab.save_spec(spec(clean)), version)["id"])
        assert again["status"] == "succeeded"
        assert all(task["cache_hit"] for task in lab.tasks(again["id"], "clean"))


def test_a_redacted_value_reaches_no_artifact_or_response(
    migrated_engine: sa.Engine,
    dev_identity: DevIdentity,
    tmp_path: Path,
    serve_app: Any,
    canary: CanaryCheck,
) -> None:
    files = {
        "records.md": document("Records", [f"The test record lists SSN {SSN} for now.", *RULES])
    }
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        version = lab.upload_corpus("redact", files)
        clean = [
            "clean/whitespace",
            {"component": "clean/redact", "params": {"mode": "placeholder"}},
        ]
        run = lab.wait_run(lab.start_run(lab.save_spec(spec(clean)), version)["id"])
        assert run["status"] == "succeeded", run
        [cleaned] = lab.tasks(run["id"], "clean")
        assert cleaned["summary"]["redactions"] == 1
        [chunks] = lab.tasks(run["id"], "chunk")
        indexes = lab.tasks(run["id"], "index")
        payloads = [lab.artifact(cleaned["cache_key"]), lab.artifact(chunks["cache_key"])]
        payloads += [
            lab.store.get(f"ws/{lab.workspace_id}/artifacts/{task['cache_key']}")
            for task in indexes
            if task["summary"]["index_kind"] == "keyword"
        ]
        diff = lab.client.get(f"/api/v1/artifacts/{cleaned['cache_key']}/diff", headers=lab.editor)
        listed = lab.client.get(
            f"/api/v1/artifacts/{chunks['cache_key']}/chunks", headers=lab.editor
        )
        search = lab.client.post(
            "/api/v1/search",
            json={"run_id": run["id"], "query": "test record lists SSN", "mode": "keyword"},
            headers=lab.editor,
        )
        responses = [diff.text, listed.text, search.text]
        assert diff.status_code == listed.status_code == search.status_code == 200
        assert any(hunk["redaction"] for hunk in diff.json()["hunks"])
        assert "[SSN]" in search.text
        for text in [*payloads, *responses]:
            raw = text if isinstance(text, bytes) else text.encode()
            assert SSN.encode() not in raw
            assert SSN[:7].encode() not in raw
        # The parsed document still holds the value: owners only.
        [parsed] = lab.tasks(run["id"], "parse")
        content = f"/api/v1/artifacts/{parsed['cache_key']}/content"
        refused = lab.client.get(content, headers=lab.editor)
        assert (refused.status_code, refused.json()["code"]) == (403, "pre_redaction_owner_only")
        owner = lab.client.get(content, headers=dev_identity.headers(role="owner"))
        assert owner.status_code == 200
        assert SSN.encode() in owner.content
        # Nothing logged the value.
        canary.value = SSN
        canary.assert_not_logged()


def test_the_halcyon_corpus_passes_the_pii_scan(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """The demo corpus through the baseline pipeline, whose scan includes ``scan/pii`` with its
    default categories: none of the 24 documents is flagged, and all of them are indexed."""
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        version = lab.import_manifest("halcyon", "demo-corpus/corpus")
        pipeline = baseline_pipeline()
        assert {"component": "scan/pii"} in pipeline["stages"]["scan"]["checks"]
        run = lab.wait_run(lab.start_run(lab.save_spec(pipeline), version)["id"])
        assert run["status"] == "succeeded", run
        entries = lab.get(f"/runs/{run['id']}/entries", limit=200)["items"]
        assert len(entries) == 24
        assert {entry["state"] for entry in entries} == {"done"}
        scans = lab.tasks(run["id"], "scan")
        assert len(scans) == 24
        for task in scans:
            report = json.loads(lab.artifact(task["cache_key"]))
            pii = next(check for check in report["checks"] if check["component"] == "scan/pii")
            assert (pii["outcome"], pii["findings"]) == ("clean", {})
            assert pii["details"]["scanned_chars"] > 0
