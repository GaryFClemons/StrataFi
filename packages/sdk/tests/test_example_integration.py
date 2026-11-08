"""The SDK end to end: the example script runs against the compose stack with fakes.

A lab serves the API over HTTP with its worker, against Postgres from the compose stack and fakes
for the model gateway and clamd. The ``stratafi`` command loads the
Halcyon demo corpus under the names the example defaults to, the demo seed's: the corpus with its
manifest, the default components as ``local-baseline``, ``local-hybrid`` and the eval set.
``examples/chunk_size_sweep.py`` then runs as its own process with its defaults, signing in as a
developer, and ``stratafi compare`` compares two of its eval runs.
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from stratafi.cli import main
from stratafi_testing.lab import (
    FIXTURES,
    HALCYON_RETRIEVAL,
    baseline_pipeline,
    hybrid_retrieval,
    running_lab,
)

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "examples" / "chunk_size_sweep.py"
PIPELINE = "local-baseline"
VARIANT = PIPELINE + ".chunk.max_tokens-{size}.min_tokens-{half}"


def table(text: str) -> list[dict[str, str]]:
    """A Markdown table's rows by its header."""
    lines = [line for line in text.splitlines() if line.startswith("| ")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    return [
        dict(zip(header, (cell.strip() for cell in line.strip("|").split("|")), strict=True))
        for line in lines[2:]
    ]


def test_the_chunk_size_sweep_runs_end_to_end(
    migrated_engine: Any,
    dev_identity: Any,
    tmp_path: Path,
    serve_app: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        monkeypatch.setenv("STRATAFI_URL", serve_app(lab.app))
        monkeypatch.setenv("STRATAFI_DEV_SIGN_IN", "tenant0:editor")
        monkeypatch.delenv("STRATAFI_TOKEN", raising=False)
        pipeline = tmp_path / "pipeline.yaml"
        pipeline.write_text(yaml.safe_dump(baseline_pipeline(PIPELINE)))
        retrieval = tmp_path / "retrieval.yaml"
        retrieval.write_text(yaml.safe_dump(hybrid_retrieval("local-hybrid")))
        manifest = FIXTURES / "demo-corpus" / "corpus" / "manifest.yaml"

        for argv in (
            ["corpus", "create", "halcyon"],
            ["corpus", "upload", "halcyon", "--manifest", str(manifest)],
            ["corpus", "freeze", "halcyon"],
            ["pipeline", "validate", str(pipeline)],
            ["pipeline", "push", str(pipeline)],
            ["retrieval", "push", str(retrieval)],
            [
                "evalset",
                "import",
                str(HALCYON_RETRIEVAL),
                "--name",
                "halcyon-retrieval",
                "--split-test",
                "0.3",
            ],
        ):
            assert main(argv) == 0, argv
        out = capsys.readouterr().out
        assert "halcyon version 1 (" in out
        assert f"{PIPELINE} version 1 (" in out
        assert "halcyon-retrieval version 1 (" in out
        corpus = lab.get("/corpora")["items"][0]
        entries = lab.get(f"/corpora/{corpus['id']}/versions/1/documents")["items"]
        assert {entry["name"] for entry in entries} >= {"operations/loyalty-v4.md"}
        assert all(entry["doc_key"] for entry in entries)  # the manifest's metadata arrived

        # A quick run through the command, waiting over server-sent events.
        quick = ["--until", "chunk", "--sample", "3", "--wait"]
        argv = [
            "run",
            "start",
            "--pipeline",
            PIPELINE,
            "--corpus",
            "halcyon@latest",
        ]
        assert main([*argv, *quick]) == 0
        assert " succeeded" in capsys.readouterr().out

        swept = subprocess.run(  # noqa: S603
            [sys.executable, str(EXAMPLE), "--sizes", "256,512"],
            capture_output=True,
            text=True,
            env=dict(os.environ),
            cwd=REPO,
            timeout=240,
            check=False,
        )
        assert swept.returncode == 0, swept.stderr
        rows = table(swept.stdout)
        assert [row["max_tokens"] for row in rows] == ["256", "512"]
        for row in rows:
            assert 0.0 <= float(row["recall@5"]) <= 1.0
            assert 0.0 <= float(row["mrr@10"]) <= 1.0
            uuid.UUID(row["eval_run"])
        for size in (256, 512):
            saved = lab.get(f"/pipeline-specs/{VARIANT.format(size=size, half=size // 2)}")
            params = saved["document"]["stages"]["chunk"]["params"]
            assert (params["max_tokens"], params["min_tokens"]) == (size, size // 2)
        assert lab.get(f"/pipeline-specs/{PIPELINE}")["version"] == 1  # untouched

        assert main(["compare", rows[0]["eval_run"], rows[1]["eval_run"]]) == 0
        compared = capsys.readouterr().out
        assert "| recall@5 |" in compared
        assert "- `stages.chunk.params.max_tokens` changed" in compared
