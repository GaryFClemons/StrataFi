"""The custom-check sandbox's runner, without Docker: the isolation flags it
asks for, what it makes of a container's output, the bundle it writes, its limits against a
fake ``docker``, and how ``test.evaluate`` gates and judges custom checks.
``test_sandbox_system.py`` runs escape attempts in the real sandbox."""

import json
import os
import re
import stat
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import stratafi.checks as sdk_checks
from stratafi_core import logs
from stratafi_core.artifacts import Chunk
from stratafi_core.settings import SandboxSettings, SuiteSettings
from stratafi_core.suites import EntryState, QueryRow, RunData, TestSuite, parse_suite
from stratafi_store.testruns import TestRun
from stratafi_worker import testruns as worker_testruns
from stratafi_worker.sandbox import (
    CPU_LIMIT_EXIT,
    RESULT_MARKER,
    Bundle,
    DockerSandbox,
    SandboxError,
    parse_output,
)
from stratafi_worker.testruns import TestEvaluator, _CustomChecks


def settings(**changes: Any) -> SandboxSettings:
    return SandboxSettings(backend="docker", **changes)


# --- the command ----------------------------------------------------------------------------------


def test_the_command_isolates_the_check(tmp_path: Path) -> None:
    sandbox = DockerSandbox(settings(memory_mb=256, cpus=0.5, pids=32, scratch_mb=16))
    command = sandbox.command(
        "stratafi-check-1", tmp_path / "b", tmp_path / "c", "checks/t.py", "whole", 30, "runc"
    )
    text = " ".join(command)
    for flag in (
        "run --rm --init --pull never",
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--user 65534:65534",
        "--ipc none",
        "--log-driver none",
        "--pids-limit 32",
        "--memory 256m --memory-swap 256m",
        "--cpus 0.5",
        "--ulimit nofile=256:256",
        "--ulimit core=0:0",
        "--ulimit cpu=30:31",
        "--tmpfs /scratch:rw,noexec,nosuid,nodev,size=16m,mode=1777",
        f"--mount type=bind,source={tmp_path / 'b'},target=/bundle,readonly",
        f"--mount type=bind,source={tmp_path / 'c'},target=/check,readonly",
    ):
        assert flag in text
    assert command[-3:] == ["stratafi/check-sandbox:1", "/check/checks/t.py", "whole"]
    assert "--runtime" not in command
    # Nothing of the worker's: no environment but these, no other mounts, no privileges.
    assert [command[i + 1] for i, part in enumerate(command) if part == "--env"] == [
        "TMPDIR=/scratch",
        "HOME=/scratch",
    ]
    assert command.count("--mount") == 2
    assert not {"--privileged", "--volume", "-v", "--cap-add", "--network=host"} & set(command)
    gvisor = sandbox.command("n", tmp_path, tmp_path, "t.py", "f", 5, "runsc")
    assert gvisor[gvisor.index("--runtime") + 1] == "runsc"


# --- the output -----------------------------------------------------------------------------------


def marked(payload: Any) -> bytes:
    return f"{RESULT_MARKER}{json.dumps(payload)}\n".encode()


def test_the_last_marked_line_is_the_verdict() -> None:
    output = (
        b"the check's own printing\n"
        + marked({"result": {"passed": False}})
        + b"more printing\n"
        + marked({"result": {"passed": True}})
    )
    assert parse_output(output, 0) == {"result": {"passed": True}}
    assert parse_output(marked({"error": "check_raised"}), 1) == {"error": "check_raised"}


@pytest.mark.parametrize(
    ("output", "exit_code", "code"),
    [
        (f"{RESULT_MARKER}{{not json\n".encode(), 0, "output_invalid"),
        (f"{RESULT_MARKER}[1, 2]\n".encode(), 0, "output_invalid"),
        (f"{RESULT_MARKER}{'[' * 200_000}\n".encode(), 0, "output_invalid"),
        (b"", 125, "sandbox_failed"),
        (b"", 127, "sandbox_failed"),
        (b"half a line", 137, "killed"),
        (b"", CPU_LIMIT_EXIT, "timeout"),
        (b"", -9, "killed"),
        (b"exited early\n", 0, "no_result"),
        (b"\xff\xfe", 3, "no_result"),
    ],
)
def test_output_without_a_verdict_says_why(output: bytes, exit_code: int, code: str) -> None:
    with pytest.raises(SandboxError) as raised:
        parse_output(output, exit_code)
    assert raised.value.code == code


# --- the bundle -----------------------------------------------------------------------------------


def a_chunk(ordinal: int, **metadata: Any) -> Chunk:
    return Chunk(
        chunk_id=f"{ordinal:032x}",
        document_id="d1",
        ordinal=ordinal,
        text=f"chunk {ordinal}",
        index_text=f"Rates > chunk {ordinal}",
        char_start=0,
        char_end=7,
        heading_path=["Rates"],
        metadata=metadata,
    )


ENTRY = EntryState(
    name="rates.pdf",
    sha256="a" * 64,
    document_id="d1",
    media_type="application/pdf",
    state="indexed",
    scan_outcome="clean",
    parse_status="succeeded",
)
QUERY = QueryRow(
    "q1", True, ["rates"], ["application/pdf"], {"metrics": {"knowledge-hit-v1": {"final": {}}}}
)


def test_the_bundle_holds_the_run_and_the_code_readable_by_the_sandbox(tmp_path: Path) -> None:
    bundle = Bundle(tmp_path / "work")
    bundle.write_code({"checks/tables.py": "X = 1\n", "top.py": "Y = 2\n"})
    counts = bundle.write_data(
        chunks=[a_chunk(0), a_chunk(1, table_part=1)], documents=[ENTRY], queries=[QUERY]
    )
    assert counts == {"chunks": 2, "documents": 1, "queries": 1}
    first = bundle.for_check({"check_id": "a", "runs": {"pipeline_run": "r1"}})
    second = bundle.for_check({"check_id": "b"})
    assert json.loads((first / "context.json").read_text())["check_id"] == "a"
    assert json.loads((second / "context.json").read_text())["check_id"] == "b"
    assert os.path.samefile(first / "chunks.jsonl", second / "chunks.jsonl")

    context = sdk_checks.CheckContext(first)
    assert [chunk.metadata for chunk in context.chunks()] == [{}, {"table_part": 1}]
    assert [document.parse_status for document in context.documents()] == ["succeeded"]
    assert [query.tags for query in context.queries()] == [["rates"]]
    assert context.runs == {"pipeline_run": "r1"}

    # The container's user (nobody) reads it all and writes none of it.
    for path in [first, bundle.code, bundle.code / "checks", *bundle.root.rglob("*")]:
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir() and path != bundle.root:
            assert mode == 0o755, path
        elif path.is_file():
            assert mode == 0o644, path
    assert (bundle.code / "checks" / "tables.py").read_text() == "X = 1\n"
    bundle.close()
    assert not bundle.root.exists()


def test_the_bundle_refuses_a_path_out_of_its_code(tmp_path: Path) -> None:
    bundle = Bundle(tmp_path)
    with pytest.raises(SandboxError) as raised:
        bundle.write_code({"../escape.py": ""})
    assert raised.value.code == "code_path_invalid"
    bundle.close()


# --- the runner, against a fake docker ------------------------------------------------------------

FAKE_DOCKER = """\
#!{python}
import json, os, signal, sys, time
command = sys.argv[1:]
state = os.environ["FAKE_DOCKER_STATE"]
with open(os.path.join(state, "calls"), "a") as log:
    log.write(" ".join(command[:2]) + "\\n")
if command[0] == "info":
    print(json.dumps({{name: {{}} for name in os.environ["FAKE_RUNTIMES"].split(",")}}))
    sys.exit(0)
if command[0] == "image":
    sys.exit(0 if os.environ.get("FAKE_IMAGE") == "present" else 1)
if command[0] == "kill":
    with open(os.path.join(state, "pid")) as handle:
        os.kill(int(handle.read()), signal.SIGKILL)
    sys.exit(0)
with open(os.path.join(state, "pid"), "w") as handle:
    handle.write(str(os.getpid()))
behaviour = os.environ["FAKE_RUN"]
if behaviour == "verdict":
    print("{marker}" + json.dumps({{"result": {{"passed": True, "value": 1}}}}), flush=True)
elif behaviour == "flood":
    while True:
        sys.stdout.write("x" * 65536)
elif behaviour == "hang":
    time.sleep(120)
elif behaviour == "oom":
    sys.exit(137)
"""


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path / "state"
    state.mkdir()
    binary = tmp_path / "docker"
    binary.write_text(FAKE_DOCKER.format(python=sys.executable, marker=RESULT_MARKER))
    binary.chmod(0o755)
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    monkeypatch.setenv("FAKE_RUNTIMES", "runc")
    monkeypatch.setenv("FAKE_IMAGE", "present")
    monkeypatch.setenv("FAKE_RUN", "verdict")
    return binary


def calls(fake: Path) -> list[str]:
    return (fake.parent / "state" / "calls").read_text().splitlines()


def run(sandbox: DockerSandbox, tmp_path: Path, timeout_s: int = 5) -> Any:
    return sandbox.run(tmp_path, tmp_path, "checks/t.py", "whole", timeout_s)


def test_a_verdict_comes_back_from_the_container(fake_docker: Path, tmp_path: Path) -> None:
    sandbox = DockerSandbox(settings(docker=str(fake_docker)))
    assert run(sandbox, tmp_path) == {"result": {"passed": True, "value": 1}}
    assert run(sandbox, tmp_path)["result"]["passed"] is True
    # Docker's runtimes and the image are looked up once.
    assert calls(fake_docker) == ["info --format", "image inspect", "run --rm", "run --rm"]
    assert sandbox.runtime() == "runc"


def test_gvisor_runs_the_check_where_docker_has_it(
    fake_docker: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_RUNTIMES", "runc,runsc")
    assert DockerSandbox(settings(docker=str(fake_docker))).runtime() == "runsc"
    assert DockerSandbox(settings(docker=str(fake_docker), runtime="runc")).runtime() == "runc"


@pytest.mark.parametrize(
    ("variable", "value", "configured"),
    [
        ("FAKE_IMAGE", "missing", {}),
        ("FAKE_RUNTIMES", "runc", {"runtime": "runsc"}),
    ],
)
def test_a_sandbox_without_its_image_or_runtime_is_unavailable(
    fake_docker: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
    configured: dict[str, Any],
) -> None:
    monkeypatch.setenv(variable, value)
    sandbox = DockerSandbox(settings(docker=str(fake_docker), **configured))
    with pytest.raises(SandboxError) as raised:
        run(sandbox, tmp_path)
    assert raised.value.code == "sandbox_unavailable"
    assert not any(call.startswith("run") for call in calls(fake_docker))


def test_no_docker_means_no_sandbox(tmp_path: Path) -> None:
    sandbox = DockerSandbox(settings(docker=str(tmp_path / "no-docker-here")))
    with pytest.raises(SandboxError) as raised:
        run(sandbox, tmp_path)
    assert raised.value.code == "sandbox_unavailable"


def test_too_much_output_kills_the_container(
    fake_docker: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_RUN", "flood")
    sandbox = DockerSandbox(settings(docker=str(fake_docker), output_bytes=65536))
    started = time.monotonic()
    with pytest.raises(SandboxError) as raised:
        run(sandbox, tmp_path)
    assert raised.value.code == "output_too_large"
    assert time.monotonic() - started < 10
    assert "kill stratafi-check-" in "\n".join(calls(fake_docker))


def test_the_deadline_kills_the_container(
    fake_docker: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_RUN", "hang")
    sandbox = DockerSandbox(settings(docker=str(fake_docker), start_seconds=0.5))
    started = time.monotonic()
    with pytest.raises(SandboxError) as raised:
        run(sandbox, tmp_path, timeout_s=1)
    assert raised.value.code == "timeout"
    assert 1.4 < time.monotonic() - started < 10
    assert any(call.startswith("kill stratafi-check-") for call in calls(fake_docker))


def test_a_killed_container_says_so(
    fake_docker: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_RUN", "oom")
    with pytest.raises(SandboxError) as raised:
        run(DockerSandbox(settings(docker=str(fake_docker))), tmp_path)
    assert raised.value.code == "killed"


# --- the lab's rules, in the SDK ------------------------------------------------------------------


def test_the_sdk_keeps_the_labs_rule_for_text() -> None:
    assert sdk_checks.TEXT_FIELD_NAMES == logs.TEXT_FIELD_NAMES
    assert sdk_checks.METADATA_SUFFIXES == logs.METADATA_SUFFIXES
    assert sdk_checks.SAFE_STRING_RE.pattern == logs._SAFE_STRING_RE.pattern
    assert sdk_checks.RESULT_MARKER == RESULT_MARKER
    source = Path(worker_testruns.__file__).parents[4] / "docker" / "sandbox" / "Dockerfile"
    assert re.search(r"COPY packages/sdk/src/stratafi/checks\.py ", source.read_text())


# --- test.evaluate --------------------------------------------------------------------------------

CODE = {"checks/tables.py": "def whole(ctx): ...\n"}


class FakeSandbox:
    def __init__(self, payload: Any = None, error: str | None = None) -> None:
        self.payload = payload or {"result": {"passed": True, "value": 2}}
        self.error = error
        self.seen: list[dict[str, Any]] = []

    def run(self, bundle: Path, code: Path, path: str, function: str, timeout_s: int) -> Any:
        context = sdk_checks.CheckContext(bundle)
        self.seen.append(
            {
                "context": json.loads((bundle / "context.json").read_text()),
                "chunks": [chunk.chunk_id for chunk in context.chunks()],
                "documents": [document.name for document in context.documents()],
                "code": (code / path).read_text(),
                "call": (path, function, timeout_s),
            }
        )
        if self.error:
            raise SandboxError(self.error)
        return self.payload


def a_run() -> TestRun:
    now = datetime(2026, 10, 31, tzinfo=UTC)
    return TestRun(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        suite_id=uuid.uuid4(),
        suite_name="tables",
        suite_version=2,
        status="evaluating",
        trigger="api",
        triggers=["api"],
        resolved={},
        pipeline_run_id=uuid.uuid4(),
        eval_run_id=None,
        baseline_pipeline_run_id=None,
        baseline_eval_run_id=None,
        baseline_test_run_id=None,
        summary={},
        record=None,
        record_hash=None,
        error_code=None,
        created_by="u",
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=None,
    )


def custom_suite(timeout_s: int = 60) -> TestSuite:
    return parse_suite(
        {
            "schema": "stratafi.test-suite.v1",
            "name": "tables",
            "target": {"pipeline": "p"},
            "corpus": "c",
            "checks": [{"id": "whole", "python": "checks/tables.py:whole", "timeout_s": timeout_s}],
            "code": CODE,
        }
    )


def checks_for(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    sandbox: Any,
    feature_on: bool = True,
    review: str | None = "SEC-12",
    backend: str = "docker",
) -> _CustomChecks:
    suites = SuiteSettings(sandbox={"backend": backend, "review": review, "workdir": tmp_path})
    evaluator = TestEvaluator(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        SimpleNamespace(suites=suites),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        sandbox,
    )
    monkeypatch.setattr(worker_testruns.features, "enabled", lambda conn, w, f: feature_on)
    monkeypatch.setattr(evaluator, "chunks_of", lambda conn, run: iter([a_chunk(0), a_chunk(1)]))
    suite = custom_suite()
    stored = SimpleNamespace(name="tables", version=2)
    target = RunData(entries=[ENTRY], queries=[QUERY], metrics={"metrics": {}})
    return _CustomChecks(evaluator, None, a_run(), suite, stored, target)  # type: ignore[arg-type]


def test_a_custom_check_runs_in_the_sandbox_over_the_runs_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sandbox = FakeSandbox()
    custom = checks_for(monkeypatch, tmp_path, sandbox=sandbox)
    [check] = custom.suite.custom_checks
    result = custom.judge(check)
    assert (result.result, result.value) == ("pass", 2.0)
    assert [link.kind for link in result.evidence] == ["run"]
    [seen] = sandbox.seen
    assert seen["context"]["check_id"] == "whole"
    assert seen["context"]["suite"] == {"name": "tables", "version": 2}
    assert seen["context"]["runs"]["pipeline_run"] == str(custom.run.pipeline_run_id)
    assert seen["chunks"] == [f"{0:032x}", f"{1:032x}"]
    assert seen["documents"] == ["rates.pdf"]
    assert seen["code"] == CODE["checks/tables.py"]
    assert seen["call"] == ("checks/tables.py", "whole", 60)
    # The bundle is written once for the run's checks, and deleted afterwards.
    custom.judge(check)
    assert len(list(tmp_path.glob("stratafi-checks-*/check-*"))) == 2
    custom.close()
    assert list(tmp_path.glob("stratafi-checks-*")) == []


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"backend": "none"}, "sandbox_not_configured"),
        ({"review": None}, "sandbox_not_reviewed"),
        ({"sandbox": None}, "sandbox_not_configured"),
        ({"feature_on": False}, "feature_off"),
    ],
)
def test_a_custom_check_is_disabled_until_every_gate_is_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changes: dict[str, Any], reason: str
) -> None:
    sandbox = FakeSandbox()
    options: dict[str, Any] = {"sandbox": sandbox, **changes}
    custom = checks_for(monkeypatch, tmp_path, **options)
    [check] = custom.suite.custom_checks
    result = custom.judge(check)
    assert (result.result, result.code, result.detail) == (
        "error",
        "custom_checks_disabled",
        {"reason": reason},
    )
    assert sandbox.seen == []
    assert list(tmp_path.glob("stratafi-checks-*")) == []


def test_a_sandbox_error_is_the_checks_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom = checks_for(monkeypatch, tmp_path, sandbox=FakeSandbox(error="timeout"))
    [check] = custom.suite.custom_checks
    assert custom.judge(check).code == "timeout"
    custom.close()


def test_the_runner_logs_no_chunk_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, canary: Any
) -> None:
    # Text as a check would see it: a sentence. The rule keeps tokens such as IDs, so a check
    # written to leak a single word could; it stops text, as in logs.
    sentence = f"the rate table says {canary.value}"
    sandbox = FakeSandbox(payload={"result": {"passed": False, "detail": {"x": sentence}}})
    custom = checks_for(monkeypatch, tmp_path, sandbox=sandbox)
    monkeypatch.setattr(
        custom.evaluator,
        "chunks_of",
        lambda conn, run: iter([a_chunk(0).model_copy(update={"text": sentence})]),
    )
    [check] = custom.suite.custom_checks
    result = custom.judge(check)
    custom.close()
    canary.assert_absent(result.model_dump_json())
    canary.assert_not_logged()
