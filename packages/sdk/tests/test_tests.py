"""The SDK's test suites and API tokens: tokens go as bearer tokens, a suite
file goes as text, and ``stratafi test run`` writes the reports and exits by the outcome."""

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from stratafi import Client
from stratafi.cli import main, suite_exit_code
from stratafi.resources import TestRun
from stratafi_core.suites import parse_suite

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "00000000-0000-0000-0000-0000000000a1"
SUITE_ID = "00000000-0000-0000-0000-0000000000b1"


def run_record(status: str, error_code: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "id": RUN_ID,
        "suite_id": SUITE_ID,
        "suite_name": "policies-regression",
        "suite_version": 1,
        "status": status,
        "outcome": status if status in ("passed", "failed") else None,
        "trigger": "ci",
        "triggers": ["ci"],
        "resolved": {},
        "pipeline_run_id": None,
        "eval_run_id": None,
        "baseline_pipeline_run_id": None,
        "baseline_eval_run_id": None,
        "baseline_test_run_id": None,
        "summary": {"checks": 1, "passed": status == "passed", "failed": status == "failed"},
        "checks": [],
        "record_hash": None,
        "error_code": error_code,
        "created_by": "token:x",
        "created_at": "2026-10-31T14:00:00Z",
        "started_at": None,
        "finished_at": None,
        **extra,
    }


def test_api_tokens_go_as_bearer_tokens_and_identity_tokens_as_before(api: Any) -> None:
    api.on("GET", "/corpora", api.page())
    token = "sft_" + "a" * 43
    with Client("http://stratafi.test", token=token, transport=api.transport) as client:
        client.corpora.list()
    with Client("http://stratafi.test", token="eyJ.c1.token", transport=api.transport) as client:
        client.corpora.list()
    first, second = api.sent("GET", "/corpora")
    assert first.headers["Authorization"] == f"Bearer {token}"
    assert "X-Stratafi-Identity" not in first.headers
    assert second.headers["X-Stratafi-Identity"] == "eyJ.c1.token"


@pytest.mark.parametrize(
    ("status", "code", "exit_code"),
    [
        ("passed", None, 0),
        ("failed", None, 1),
        ("error", "corpus_not_found", 2),
        ("error", "eval_set_not_frozen", 2),
        ("error", "evaluation_failed", 3),
        ("cancelled", "cancelled", 3),
    ],
)
def test_suite_exit_codes_follow_the_outcome(status: str, code: str | None, exit_code: int) -> None:
    run = TestRun.__new__(TestRun)
    run.record = TestRun.model.model_validate(run_record(status, code))
    assert suite_exit_code(run) == exit_code


def test_test_run_sends_the_file_waits_and_writes_the_reports(
    api: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text("schema: stratafi.test-suite.v1\nname: policies-regression\n")
    api.on(
        "POST",
        "/test-runs",
        lambda request: httpx2.Response(202, json={**run_record("queued"), "coalesced": False}),
    )
    api.on(
        "GET",
        f"/test-runs/{RUN_ID}",
        lambda request: httpx2.Response(200, json=run_record("failed")),
    )
    reports = {
        "junit": "<testsuites/>",
        "json": '{"schema": "stratafi.test-run.v1"}',
        "markdown": "## failed\n",
    }
    api.on(
        "GET",
        f"/test-runs/{RUN_ID}/report",
        lambda request: httpx2.Response(200, text=reports[request.url.params["format"]]),
    )
    api.on("GET", "/events", lambda request: httpx2.Response(404))
    monkeypatch.setenv("STRATAFI_URL", "http://stratafi.test")
    monkeypatch.setenv("STRATAFI_TOKEN", "sft_" + "b" * 43)
    code = main(
        [
            "test",
            "run",
            str(suite),
            "--junit",
            str(tmp_path / "junit.xml"),
            "--json",
            str(tmp_path / "run.json"),
            "--markdown",
            str(tmp_path / "summary.md"),
        ],
        transport=api.transport,
    )
    assert code == 1
    body = api.body("POST", "/test-runs")
    assert body == {"content": suite.read_text(), "trigger": "ci"}
    assert (tmp_path / "junit.xml").read_text() == "<testsuites/>"
    assert json.loads((tmp_path / "run.json").read_text())["schema"] == "stratafi.test-run.v1"
    assert (tmp_path / "summary.md").read_text() == "## failed\n"


def test_the_example_suite_is_valid() -> None:
    document = yaml.safe_load(
        (ROOT / "examples" / "suites" / "policies-regression.yaml").read_text()
    )
    suite = parse_suite(document)
    assert suite.name == "policies-regression"
    assert [trigger.on for trigger in suite.triggers] == ["corpus_version_frozen", "target_changed"]


CUSTOM_SUITE = """schema: stratafi.test-suite.v1
name: tables
target: {pipeline: p}
corpus: c
checks:
  - id: tables-kept-whole
    python: checks/tables.py:tables_not_split
  - id: again
    python: checks/tables.py:tables_not_split
"""


def test_a_suite_file_brings_the_files_its_custom_checks_name(
    api: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRATAFI_URL", "http://stratafi.test")
    monkeypatch.setenv("STRATAFI_TOKEN", "sft_" + "b" * 43)
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "tables.py").write_text("def tables_not_split(ctx): ...\n")
    suite = tmp_path / "suite.yaml"
    suite.write_text(CUSTOM_SUITE)
    api.on(
        "POST",
        "/test-suites/validate",
        lambda request: httpx2.Response(
            200, json={"valid": True, "issues": [], "document": {}, "needs": {}}
        ),
    )
    assert main(["test", "validate", str(suite)], transport=api.transport) == 0
    body = api.body("POST", "/test-suites/validate")
    assert body == {
        "content": CUSTOM_SUITE,
        "code": {"checks/tables.py": "def tables_not_split(ctx): ...\n"},
    }


def test_a_custom_check_cant_name_a_file_outside_the_suites_folder(
    api: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("STRATAFI_URL", "http://stratafi.test")
    monkeypatch.setenv("STRATAFI_TOKEN", "sft_" + "b" * 43)
    (tmp_path / "secret.py").write_text("TOKEN = 'x'\n")
    folder = tmp_path / "suites"
    folder.mkdir()
    suite = folder / "suite.yaml"
    suite.write_text(CUSTOM_SUITE.replace("checks/tables.py", "../secret.py"))
    assert main(["test", "validate", str(suite)], transport=api.transport) == 2
    assert "a check's file is a path beside the suite" in capsys.readouterr().err
    assert api.sent("POST", "/test-suites/validate") == []


def test_the_custom_check_example_validates_and_its_checks_run(tmp_path: Path) -> None:
    from stratafi.checks import CheckContext
    from stratafi.resources import _suite_body

    example = ROOT / "examples" / "suites" / "tables.yaml"
    body = _suite_body(example)
    assert list(body["code"]) == ["checks/tables.py"]
    suite = parse_suite({**yaml.safe_load(body["content"]), "code": body["code"]})
    assert [check.kind for check in suite.checks] == ["stage", "custom", "custom"]

    namespace: dict[str, Any] = {}
    exec(compile(body["code"]["checks/tables.py"], "tables.py", "exec"), namespace)  # noqa: S102
    chunk = {"document_id": "d1", "ordinal": 0, "text": "t", "heading_path": ["Price tiers"]}
    ctx = CheckContext.from_records(
        tmp_path,
        chunks=[{**chunk, "chunk_id": "c1", "metadata": {"table_part": 1}}],
        documents=[
            {"document_id": "d1", "name": "wholesale-v4.md", "parse_status": "succeeded"},
            {"document_id": "d2", "name": "hiring-pay-v3.md", "parse_status": "succeeded"},
        ],
    )
    split = namespace["tables_not_split"](ctx).to_dict()
    assert (split["passed"], split["detail"]) == (False, {"chunks": ["c1"]})
    empty = namespace["every_document_has_chunks"](ctx).to_dict()
    assert (empty["passed"], empty["code"], empty["detail"]) == (
        False,
        "documents_unchunked",
        {"documents": ["d2"]},
    )
