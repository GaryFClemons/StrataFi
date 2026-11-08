"""The ``stratafi`` command: argument parsing and exit codes.

Exit codes: 0 passed, 1 a check failed, 2 a usage or configuration error, 3 a platform error.
"""

import json
from typing import Any

import httpx2
import pytest

from stratafi.cli import build_parser, main


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATAFI_URL", "http://stratafi.test")
    monkeypatch.setenv("STRATAFI_TOKEN", "token")
    monkeypatch.delenv("STRATAFI_DEV_SIGN_IN", raising=False)


# --- parsing ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        "corpus create halcyon",
        "corpus upload halcyon ./docs/a.pdf ./docs/b.pdf",
        "corpus upload halcyon --manifest corpus/manifest.yaml",
        "corpus freeze halcyon",
        "corpus list",
        "pipeline validate pipeline.yaml",
        "pipeline push pipeline.yaml",
        "pipeline show baseline-v1@2",
        "retrieval push retrieval.yaml",
        "run start --pipeline baseline-v1 --corpus halcyon@latest --until chunk --sample 20 --wait",
        "run wait 00000000-0000-0000-0000-000000000001 --timeout 60",
        "evalset import demo-corpus/retrieval.yaml --name halcyon-retrieval --split-test 0.3",
        "evalset show halcyon-retrieval@1",
        "eval run --run RUN_ID --retrieval hybrid-v1 --eval-set halcyon-retrieval@1 "
        "--split dev --wait",
        "compare EVAL_RUN_A EVAL_RUN_B --format markdown",
    ],
)
def test_the_documented_commands_parse(argv: str) -> None:
    """The commands in "SDK and CLI" (v1) parse, each to its handler."""
    args = build_parser().parse_args(argv.split())
    assert callable(args.handler)


def test_options_parse_to_their_types() -> None:
    args = build_parser().parse_args(
        [
            "evalset",
            "import",
            "r.yaml",
            "--name",
            "halcyon",
            "--split-test",
            "0.25",
            "--seed",
            "7",
            "--as-of",
            "2026-11-02",
            "--k",
            "5",
            "--no-freeze",
        ]
    )
    assert (args.split_test, args.seed, str(args.as_of), args.k, args.no_freeze) == (
        0.25,
        7,
        "2026-11-02",
        5,
        True,
    )
    args = build_parser().parse_args(
        [
            "compare",
            "A",
            "B",
            "--metric",
            "recall@5",
            "--metric",
            "mrr@10",
            "--rule",
            "doc-v1",
            "--format",
            "json",
        ]
    )
    assert (args.metric, args.rule, args.checkpoint, args.format) == (
        ["recall@5", "mrr@10"],
        "doc-v1",
        "final",
        "json",
    )


@pytest.mark.parametrize(
    "argv",
    [
        "",
        "corpus",
        "frobnicate",
        "run start --corpus halcyon@latest",
        "run start --pipeline p --corpus c --until nowhere",
        "run start --pipeline p --corpus c --sample 0",
        "run wait RUN --timeout -1",
        "evalset import r.yaml --name halcyon --split-test 0.95",
        "evalset import r.yaml --name halcyon --as-of yesterday",
        "evalset import r.yaml",
        "eval run --run R --retrieval x --eval-set y --split holdout",
        "eval run --run R --retrieval x --eval-set y --resamples 10",
        "compare A",
        "compare A B --format html",
        "compare A B --checkpoint everywhere",
    ],
)
def test_usage_errors_exit_2(argv: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv.split())
    assert exit_info.value.code == 2
    assert "usage: stratafi" in capsys.readouterr().err


def test_help_lists_the_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "exit codes:" in capsys.readouterr().out


# --- configuration ------------------------------------------------------------------------


def test_no_url_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("STRATAFI_URL")
    assert main(["corpus", "list"]) == 2
    assert "STRATAFI_URL" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("name", "value"),
    [("STRATAFI_URL", "ftp://stratafi.test"), ("STRATAFI_DEV_SIGN_IN", "tenant0")],
)
def test_bad_settings_are_configuration_errors(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.delenv("STRATAFI_TOKEN")
    monkeypatch.setenv(name, value)
    assert main(["corpus", "list"]) == 2


def test_the_url_option_overrides_the_environment(api: Any) -> None:
    api.on("GET", "/corpora", api.page())
    assert main(["--url", "http://other.test:8100", "corpus", "list"], transport=api.transport) == 0
    assert api.requests[0].url.host == "other.test"
    assert api.requests[0].headers["X-Stratafi-Identity"] == "token"


def test_a_missing_file_is_a_configuration_error(api: Any, tmp_path: Any) -> None:
    missing = str(tmp_path / "missing.yaml")
    assert main(["pipeline", "validate", missing], transport=api.transport) == 2
    assert api.requests == []


def test_a_bad_reference_is_a_configuration_error(api: Any) -> None:
    assert main(["run", "show", "not-an-id"], transport=api.transport) == 2
    argv = ["run", "start", "--pipeline", "Bad Name", "--corpus", "halcyon@latest"]
    assert main(argv, transport=api.transport) == 2
    assert api.requests == []


def test_a_name_that_doesnt_exist_is_a_configuration_error(
    api: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    api.on("GET", "/corpora", api.page(api.corpus("other")))
    assert main(["corpus", "freeze", "halcyon"], transport=api.transport) == 2
    assert "no corpus is called halcyon" in capsys.readouterr().err


# --- checks: validation and runs ----------------------------------------------------------


@pytest.fixture
def spec_file(tmp_path: Any, pipeline: Any) -> str:
    path = tmp_path / "pipeline.yaml"
    path.write_text(json.dumps(pipeline()), encoding="utf-8")
    return str(path)


def test_a_valid_spec_passes(api: Any, spec_file: str, capsys: pytest.CaptureFixture[str]) -> None:
    api.on("POST", "/pipeline-specs/validate", {"valid": True, "issues": [], "spec_hash": "a" * 64})
    assert main(["pipeline", "validate", spec_file], transport=api.transport) == 0
    assert capsys.readouterr().out.startswith("valid (spec hash aaaaaaaaaaaa)")
    assert api.body("POST", "/pipeline-specs/validate")["name"] == "baseline-v1"


def test_an_invalid_spec_fails_the_check(
    api: Any, spec_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    issues = [{"path": "stages.chunk.params.max_tokens", "code": "greater_than_equal"}]
    api.on("POST", "/pipeline-specs/validate", {"valid": False, "issues": issues})
    assert main(["pipeline", "validate", spec_file], transport=api.transport) == 1
    assert "stages.chunk.params.max_tokens: greater_than_equal" in capsys.readouterr().out


def test_push_reports_a_new_or_unchanged_version(
    api: Any, spec_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = api.spec("baseline-v1", 3)
    api.on("POST", "/pipeline-specs", httpx2.Response(201, json=saved), saved)
    assert main(["pipeline", "push", spec_file], transport=api.transport) == 0
    assert main(["pipeline", "push", spec_file], transport=api.transport) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("baseline-v1 version 3")
    assert out[1].endswith(", unchanged")


def start_run(api: Any) -> None:
    api.on("GET", "/pipeline-specs/baseline-v1", api.spec("baseline-v1"))
    api.on("GET", "/corpora", api.page(api.corpus()))
    latest = api.corpus_version(2)
    api.on("GET", f"/corpora/{api.corpus()['id']}", api.corpus(latest=latest))
    api.on("POST", "/runs", httpx2.Response(202, json=api.run("queued")))


START = ["run", "start", "--pipeline", "baseline-v1", "--corpus", "halcyon@latest"]


def test_run_start_without_waiting_passes(api: Any, capsys: pytest.CaptureFixture[str]) -> None:
    start_run(api)
    assert main(START, transport=api.transport) == 0
    body = api.body("POST", "/runs")
    assert body["pipeline_spec_id"] == api.spec("x")["id"]
    assert body["corpus_version_id"] == api.corpus_version(2)["id"]
    assert capsys.readouterr().out.startswith(f"run {api.run()['id']} queued")


@pytest.mark.parametrize(("status", "code"), [("succeeded", 0), ("failed", 1), ("cancelled", 1)])
def test_waiting_for_a_run_passes_only_when_it_succeeds(
    api: Any, sse_response: Any, status: str, code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    start_run(api)
    run_id = api.run()["id"]
    api.on("GET", f"/runs/{run_id}", api.run("running"), api.run("running"), api.run(status))
    api.on("GET", "/events", sse_response(("run", {"run_id": run_id, "status": status})))
    assert main([*START, "--wait"], transport=api.transport) == code
    captured = capsys.readouterr()
    assert captured.out.startswith(f"run {run_id} {status}")
    assert f"run {run_id}: {status}" in captured.err  # progress


def test_a_run_that_outlasts_the_timeout_is_a_platform_error(
    api: Any, sse_response: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = api.run()["id"]
    api.on("GET", f"/runs/{run_id}", api.run("running"))
    api.on("GET", "/events", sse_response())
    assert main(["run", "wait", run_id, "--timeout", "0.05"], transport=api.transport) == 3
    assert "hasn't finished" in capsys.readouterr().err


def eval_run_answers(api: Any, status: str) -> None:
    retrieval = {"schema": "stratafi.retrieval.v1", "name": "hybrid-v1"}
    api.on(
        "GET",
        "/retrieval-configs/hybrid-v1",
        api.spec("hybrid-v1", document=retrieval),
    )
    api.on("GET", "/eval-sets", api.page(api.eval_set()))
    api.on("GET", f"/eval-sets/{api.eval_set()['id']}/versions/1", api.eval_set_version(1))
    api.on("POST", "/eval-runs", httpx2.Response(202, json=api.eval_run("queued")))
    api.on("GET", f"/eval-runs/{api.eval_run()['id']}", api.eval_run(status))


EVAL = (
    f"eval run --run {'0' * 8}-0000-0000-0000-{'0' * 11}1 --retrieval hybrid-v1 "
    "--eval-set halcyon-retrieval@1 --wait"
).split()


def test_waiting_for_an_eval_run_prints_its_metrics(
    api: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_run_answers(api, "succeeded")
    assert main(EVAL, transport=api.transport) == 0
    out = capsys.readouterr().out
    assert "| recall@5 | 0.750 | [0.600, 0.900] | 63 |" in out
    assert api.body("POST", "/eval-runs")["eval_set_version_id"] == api.eval_set_version(1)["id"]


def test_a_failed_eval_run_fails_the_check(api: Any) -> None:
    eval_run_answers(api, "failed")
    assert main(EVAL, transport=api.transport) == 1


def test_compare_prints_markdown(api: Any, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = "00000000-0000-0000-0000-00000000000a", "00000000-0000-0000-0000-00000000000b"
    ref = {
        "split": "dev",
        "retrieval_ref": None,
        "eval_set": None,
        "pipeline_run_id": a,
        "created_at": "2026-10-30T12:00:00Z",
    }
    row = {
        "metric": "recall@5",
        "n": 63,
        "mean_a": 0.7,
        "mean_b": 0.75,
        "delta": 0.05,
        "low": 0.01,
        "high": 0.09,
        "p_value": 0.02,
        "p_holm": 0.04,
        "wins": 5,
        "losses": 1,
        "ties": 57,
    }
    api.on(
        "GET",
        "/eval-comparisons",
        {
            "a": {"id": a, **ref},
            "b": {"id": b, **ref},
            "rule": "knowledge-hit-v1",
            "checkpoint": "final",
            "paired": 63,
            "resamples": 2000,
            "seed": 0,
            "metrics": [row],
            "queries": [],
            "retrieval_diff": [],
            "pipeline_diff": [
                {"path": "stages.chunk.params.max_tokens", "op": "changed", "before": 800}
            ],
            "warnings": [],
        },
    )
    argv = ["compare", a, b, "--metric", "recall@5", "--format", "markdown"]
    assert main(argv, transport=api.transport) == 0
    out = capsys.readouterr().out
    assert "| recall@5 | 0.700 | 0.750 | +0.050 | [0.010, 0.090] | 0.020 | 0.040 | 5/1/57 |" in out
    assert "- `stages.chunk.params.max_tokens` changed" in out
    assert api.sent("GET", "/eval-comparisons")[0].url.params.get_list("metric") == ["recall@5"]


def test_json_output_is_the_api_record(api: Any, capsys: pytest.CaptureFixture[str]) -> None:
    run_id = api.run()["id"]
    api.on("GET", f"/runs/{run_id}", api.run("succeeded"))
    assert main(["run", "show", run_id, "--format", "json"], transport=api.transport) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"


# --- platform errors ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code"),
    [(400, 2), (401, 2), (403, 2), (404, 2), (409, 2), (429, 3), (500, 3), (503, 3)],
)
def test_api_errors_map_to_exit_codes(
    api: Any, status: int, code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    api.problem("GET", "/corpora", status, code="some_reason")
    assert main(["corpus", "list"], transport=api.transport) == code
    err = capsys.readouterr().err
    assert f"stratafi: {status} Problem: It went wrong. [some_reason]" in err


def test_an_unreachable_api_is_a_platform_error(capsys: pytest.CaptureFixture[str]) -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    assert main(["corpus", "list"], transport=httpx2.MockTransport(refuse)) == 3
    assert "didn't answer (ConnectError)" in capsys.readouterr().err
