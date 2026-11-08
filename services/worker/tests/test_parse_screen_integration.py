"""The Parse screen's API against Postgres: the original is served only for
PDFs and images the run's scan passed; authored items become draft versions."""

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import FIXTURES, running_lab

pytestmark = pytest.mark.integration


def test_the_original_is_served_only_after_scan_for_pdfs_and_images(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    pdf = (FIXTURES / "formats" / "tips.pdf").read_bytes()
    files = {"a.pdf": pdf, "b.txt": b"Plain text.\n", "copy.pdf": pdf}
    spec = {"schema": "stratafi.pipeline.v1", "name": "originals"}
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        version = lab.upload_corpus("originals", files)
        run = lab.wait_run(lab.start_run(lab.save_spec(spec), version, until="parse")["id"])
        entries = {e["name"]: e["ordinal"] for e in lab.get(f"/runs/{run['id']}/entries")["items"]}

        def original(name: str):
            return lab.client.get(
                f"/api/v1/runs/{run['id']}/entries/{entries[name]}/original", headers=lab.editor
            )

        served = original("a.pdf")
        assert served.status_code == 200
        assert served.content == pdf
        assert served.headers["content-type"] == "application/pdf"
        assert served.headers["content-security-policy"].startswith("sandbox")
        assert served.headers["x-content-type-options"] == "nosniff"
        assert (original("b.txt").status_code, original("b.txt").json()["code"]) == (
            409,
            "not_viewable",
        )
        # The duplicate was excluded by scan/dedupe: never served.
        assert original("copy.pdf").json()["code"] == "not_scanned"


def test_authored_items_become_draft_versions(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        body = {
            "eval_set": "authored",
            "query": "Who counts the tip jar at close?",
            "evidence": {
                "doc_key": "tips",
                "section": "3.1 Counting the tip jar",
                "evidence": "two people count the tip jar together",
            },
        }
        first = lab.post("/eval-sets/author", body)
        second = lab.post("/eval-sets/author", {**body, "query": "Who signs the tip log?"})
        assert (first["item_id"], second["item_id"]) == ("authored-1", "authored-2")
        assert (first["version"]["version"], second["version"]["version"]) == (1, 2)
        assert second["version"]["frozen"] is False
        items = lab.get(f"/eval-sets/{second['eval_set']['id']}/versions/2/items")["items"]
        assert [item["id"] for item in items] == ["authored-1", "authored-2"]
        assert items[1]["tags"][0] == "authored"
