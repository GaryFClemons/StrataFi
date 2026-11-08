"""The full step screens in a browser (browser tier).

Each test starts a lab (the API, a worker and the fakes on a fresh database), serves the built
web app beside it, writes the files its flow uploads, and runs the flow in ``web/e2e``. The
flows use the fake gateway and the in-process components, so they need no models.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from stratafi_testing.documents import Block, eml_bytes, markdown_text, zip_bytes
from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import FIXTURES, WEB, running_lab, serve_web

pytestmark = pytest.mark.integration

FEES = markdown_text(
    "Wholesale Prices",
    [Block("heading", "Tiers", 1), Block("paragraph", "Halcyon charges 11.80 a pound for Tier 1.")],
).encode()
# A question about the format fixtures' Tip Policy, for the search playground.
TIP_QUESTIONS = {
    "name": "tip-questions",
    "items": [
        {
            "id": "tip-jar-count",
            "query": "Who counts the tip jar at close?",
            "expected": [{"doc_key": "tips", "evidence": "count the tip jar together"}],
        }
    ],
}


def run_flow(flow: str, *args: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.fail("the browser flows need Node 24 and Playwright")
    result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
        [node, str(WEB / "e2e" / flow), *args],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=WEB,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("ok"), result.stdout


def intake_files(folder: Path) -> Path:
    folder.mkdir()
    (folder / "bundle.zip").write_bytes(
        zip_bytes({"policies/fees.md": FEES, "wires.txt": b"Count the tip jar twice.\n"})
    )
    (folder / "notice.eml").write_bytes(
        eml_bytes(
            "Hours",
            "The cafe hours change in May.",
            {"hours.md": ("text/markdown", b"# Hours\n")},
        )
    )
    (folder / "copy.md").write_bytes(FEES)
    (folder / "fake.pdf").write_bytes(b"This is plain text, not a PDF.\n")
    return folder


@pytest.mark.browser
def test_the_extract_screen(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Extract: types against extensions, the mismatch, duplicates, the archive and email
    tree, and a refused metadata URL (web/e2e/extract-flow.mjs)."""
    folder = intake_files(tmp_path / "files")
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        run_flow("extract-flow.mjs", serve_web(lab, serve_app), str(folder))


@pytest.mark.browser
def test_the_scan_and_clean_screen(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Scan + Clean: the policy editor, adding and reordering the redaction rule, and the diff
    viewer, which shows the token and never the value (web/e2e/clean-flow.mjs)."""
    folder = tmp_path / "files"
    folder.mkdir()
    (folder / "records.md").write_bytes(
        markdown_text(
            "Records",
            [
                Block("paragraph", "The test record lists SSN 123-45-6789 for now."),
                Block("paragraph", "Refunds  are   confirmed by a manager."),
            ],
        ).encode()
    )
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        run_flow("clean-flow.mjs", serve_web(lab, serve_app), str(folder))


@pytest.mark.browser
def test_the_parse_screen(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Parse: the routing table, the pdf.js page with element boxes, the table viewer,
    comparing parsers on one page, and authoring an eval item (web/e2e/parse-flow.mjs)."""
    folder = tmp_path / "files"
    folder.mkdir()
    shutil.copy(FIXTURES / "formats" / "wholesale.pdf", folder / "wholesale.pdf")
    (folder / "tiers.csv").write_bytes(b"Tier,Price per lb\nNeighbor,11.80\nFlagship,9.10\n")
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        run_flow("parse-flow.mjs", serve_web(lab, serve_app), str(folder))
        sets = lab.get("/eval-sets")["items"]
        assert [item["name"] for item in sets] == ["authored"]


@pytest.mark.browser
def test_the_chunk_screen(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Chunk: boundaries with hatched overlaps, the histogram with the limit line, warnings,
    the chunk list, and a second chunker side by side (web/e2e/chunk-flow.mjs)."""
    folder = tmp_path / "files"
    folder.mkdir()
    shutil.copy(FIXTURES / "formats" / "food-safety.md", folder / "food-safety.md")
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        run_flow("chunk-flow.mjs", serve_web(lab, serve_app), str(folder))


@pytest.mark.browser
def test_the_embed_screen(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Embed: the model catalog, embed options (truncation, int8, an input limit), the
    projection with nearest neighbours, and pgvector HNSW's ANN recall (web/e2e/embed-flow.mjs)."""
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        run_flow("embed-flow.mjs", serve_web(lab, serve_app), str(FIXTURES / "formats"))


@pytest.mark.browser
def test_the_search_playground(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Search: an eval item fills the playground, two configs search side by side with
    their checkpoints, and the evidence is highlighted (web/e2e/playground-flow.mjs)."""
    import yaml

    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        lab.import_eval_set("tip-questions", yaml.safe_dump(TIP_QUESTIONS))
        run_flow("playground-flow.mjs", serve_web(lab, serve_app), str(FIXTURES / "formats"))
