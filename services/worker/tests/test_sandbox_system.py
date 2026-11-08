"""Escape attempts in the real custom-check sandbox fail safely: network access, writes outside
scratch, fork bombs, huge output.

System tier: ``STRATAFI_TEST_SYSTEM=1`` with a Docker daemon. The module builds the sandbox's
image from docker/sandbox/Dockerfile as ``stratafi/check-sandbox:test``, then runs checks
written to break out through ``DockerSandbox``, as ``test.evaluate`` does. Each must end in a
verdict or an error code, within its limits, and leave no container behind.
"""

import json
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from stratafi_core.settings import SandboxSettings
from stratafi_testing.check_sandbox import build_sandbox_image
from stratafi_worker.sandbox import CONTAINER_LABEL, Bundle, DockerSandbox, SandboxError

pytestmark = pytest.mark.system

ESCAPES = r"""
import os
import socket
import sys
import time

from stratafi.checks import CheckContext, CheckResult


def reads_its_bundle(ctx: CheckContext) -> CheckResult:
    chunks = [chunk.chunk_id for chunk in ctx.chunks()]
    return CheckResult.assert_true(
        len(chunks) == 2 and ctx.check_id == "escape",
        value=len(chunks),
        detail={"chunks": chunks, "suite": ctx.suite.get("name")},
    )


def network(ctx: CheckContext) -> CheckResult:
    found = {}
    targets = {
        "public": ("1.1.1.1", 443),
        "docker_gateway": ("172.17.0.1", 80),
        "metadata_service": ("169.254.169.254", 80),
    }
    for name, address in targets.items():
        try:
            socket.create_connection(address, timeout=2).close()
            found[name] = "connected"
        except OSError as error:
            found[name] = type(error).__name__
    try:
        socket.getaddrinfo("example.com", 443)
        found["dns"] = "resolved"
    except OSError as error:
        found["dns"] = type(error).__name__
    try:
        socket.socket(socket.AF_PACKET, socket.SOCK_RAW).close()
        found["raw_socket"] = "opened"
    except OSError as error:
        found["raw_socket"] = type(error).__name__
    interfaces = sorted(name for _, name in socket.if_nameindex())
    return CheckResult.passing(detail={**found, "interfaces": interfaces})


def writes(ctx: CheckContext) -> CheckResult:
    found = {}
    places = ["/check/x", "/bundle/x", "/etc/x", "/tmp/x", "/var/tmp/x", "/x",
              "/opt/sandbox/x", "/opt/sandbox/stratafi/checks.py", "/dev/shm/x", "/scratch/x"]
    for path in places:
        try:
            with open(path, "a") as handle:
                handle.write("x")
            found[path.strip("/").replace("/", "_")] = "written"
        except OSError as error:
            found[path.strip("/").replace("/", "_")] = type(error).__name__
    return CheckResult.passing(detail=found)


def fills_scratch(ctx: CheckContext) -> CheckResult:
    written = 0
    try:
        with open(ctx.scratch / "big", "wb") as handle:
            for _ in range(1024):
                handle.write(b"\0" * (1024 * 1024))
                written += 1
    except OSError as error:
        return CheckResult.passing(value=written, detail={"stopped": type(error).__name__})
    return CheckResult.failing("scratch_unbounded", value=written)


def fork_bomb(ctx: CheckContext) -> CheckResult:
    while True:
        os.fork()


def stubborn_fork_bomb(ctx: CheckContext) -> CheckResult:
    while True:
        try:
            os.fork()
        except OSError:
            pass


def huge_output(ctx: CheckContext) -> CheckResult:
    line = "x" * 1023 + "\n"
    while True:
        sys.stdout.write(line)


def huge_detail(ctx: CheckContext) -> CheckResult:
    return CheckResult.passing(detail={f"k{i}": ["y" * 200] * 50 for i in range(200)})


def memory(ctx: CheckContext) -> CheckResult:
    blocks = []
    while True:
        blocks.append(bytearray(64 * 1024 * 1024))


def spins(ctx: CheckContext) -> CheckResult:
    while True:
        pass


def sleeps(ctx: CheckContext) -> CheckResult:
    time.sleep(3600)
    return CheckResult.passing()


def credentials(ctx: CheckContext) -> CheckResult:
    found = {"uid": os.getuid(), "gid": os.getgid(), "environment": sorted(os.environ)}
    found["docker_socket"] = any(
        os.path.exists(path) for path in ("/var/run/docker.sock", "/run/docker.sock")
    )
    try:
        os.setuid(0)
        found["setuid"] = "worked"
    except OSError as error:
        found["setuid"] = type(error).__name__
    with open("/proc/self/status") as handle:
        status = dict(line.rstrip("\n").split(":\t", 1) for line in handle if ":\t" in line)
    found["capabilities"] = [status[key].strip() for key in ("CapEff", "CapPrm", "CapBnd")]
    found["no_new_privileges"] = status.get("NoNewPrivs", "").strip()
    found["compilers"] = [
        name for name in ("gcc", "cc", "clang", "pip", "pip3", "curl", "wget")
        if any(os.path.exists(os.path.join(d, name)) for d in os.environ["PATH"].split(":"))
    ]
    return CheckResult.passing(detail=found)


def fakes_a_verdict(ctx: CheckContext) -> CheckResult:
    print("STRATAFI-CHECK-RESULT not json")
    return CheckResult.failing("the_real_verdict")
"""


DOCKER = shutil.which("docker")


def docker(*args: str) -> subprocess.CompletedProcess[str]:
    assert DOCKER is not None
    return subprocess.run(  # noqa: S603 - fixed arguments
        [DOCKER, *args], capture_output=True, text=True, timeout=600, check=False
    )


@pytest.fixture(scope="module")
def sandbox(repo_root: Path) -> DockerSandbox:
    image = build_sandbox_image(repo_root)
    if image is None:
        pytest.skip("the sandbox's system tests need a Docker daemon")
    return DockerSandbox(
        SandboxSettings(
            backend="docker",
            image=image,
            memory_mb=256,
            pids=64,
            scratch_mb=16,
            output_bytes=1 << 20,
            start_seconds=10,
        )
    )


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Bundle]:
    from stratafi_core.artifacts import Chunk

    made = Bundle(tmp_path_factory.mktemp("work"))
    made.write_code({"checks/escapes.py": ESCAPES})
    made.write_data(
        chunks=[
            Chunk(
                chunk_id=f"{i:032x}",
                document_id="d1",
                ordinal=i,
                text=f"chunk {i}",
                index_text=f"chunk {i}",
                char_start=0,
                char_end=7,
            )
            for i in range(2)
        ],
        documents=[],
        queries=[],
    )
    yield made
    made.close()


def run(
    sandbox: DockerSandbox, bundle: Bundle, function: str, timeout_s: int = 20
) -> tuple[dict[str, Any] | str, float]:
    """The check's payload, or the sandbox's error code; and how long it took."""
    directory = bundle.for_check({"check_id": "escape", "suite": {"name": "escapes"}})
    started = time.monotonic()
    try:
        found: dict[str, Any] | str = dict(
            sandbox.run(directory, bundle.code, "checks/escapes.py", function, timeout_s)
        )
    except SandboxError as error:
        found = error.code
    return found, time.monotonic() - started


def detail(found: dict[str, Any] | str) -> dict[str, Any]:
    assert isinstance(found, dict), found
    return found["result"]["detail"]


@pytest.fixture(autouse=True)
def no_container_left(sandbox: DockerSandbox) -> Iterator[None]:
    yield
    left = docker("ps", "-aq", "--filter", f"label={CONTAINER_LABEL}").stdout.split()
    assert left == [], "a custom-check container was left behind"


def test_a_check_reads_its_bundle(sandbox: DockerSandbox, bundle: Bundle) -> None:
    found, _ = run(sandbox, bundle, "reads_its_bundle")
    assert found == {
        "result": {
            "passed": True,
            "value": 2.0,
            "code": None,
            "detail": {"chunks": [f"{0:032x}", f"{1:032x}"], "suite": "escapes"},
        }
    }


def test_the_network_is_out_of_reach(sandbox: DockerSandbox, bundle: Bundle) -> None:
    found = detail(run(sandbox, bundle, "network")[0])
    assert found["interfaces"] == ["lo"]
    for name in ("public", "docker_gateway", "metadata_service", "dns", "raw_socket"):
        assert found[name] not in ("connected", "resolved", "opened"), name


def test_only_scratch_is_writable(sandbox: DockerSandbox, bundle: Bundle) -> None:
    found = detail(run(sandbox, bundle, "writes")[0])
    assert found.pop("scratch_x") == "written"
    assert found
    assert all(outcome != "written" for outcome in found.values()), found
    stopped = run(sandbox, bundle, "fills_scratch")[0]
    assert isinstance(stopped, dict)
    assert stopped["result"]["passed"] is True
    assert stopped["result"]["value"] <= 16


@pytest.mark.parametrize("function", ["fork_bomb", "stubborn_fork_bomb"])
def test_a_fork_bomb_fails_safely(sandbox: DockerSandbox, bundle: Bundle, function: str) -> None:
    found, seconds = run(sandbox, bundle, function, timeout_s=5)
    assert found in (
        {"error": "check_raised", "exception": "BlockingIOError"},
        "timeout",
        "killed",
    )
    assert seconds < 5 + 10 + 30
    after, _ = run(sandbox, bundle, "reads_its_bundle")
    assert isinstance(after, dict)
    assert after["result"]["passed"] is True


def test_huge_output_is_cut_off(sandbox: DockerSandbox, bundle: Bundle) -> None:
    found, seconds = run(sandbox, bundle, "huge_output")
    assert found == "output_too_large"
    assert seconds < 15
    assert detail(run(sandbox, bundle, "huge_detail")[0]) == {"truncated": True}


def test_memory_cpu_and_time_are_limited(sandbox: DockerSandbox, bundle: Bundle) -> None:
    assert run(sandbox, bundle, "memory")[0] == "killed"
    found, seconds = run(sandbox, bundle, "spins", timeout_s=2)
    assert found == "timeout"
    assert seconds < 2 + 10
    found, seconds = run(sandbox, bundle, "sleeps", timeout_s=1)
    assert found == "timeout"
    assert 1 + 10 <= seconds < 1 + 10 + 30


def test_the_check_has_no_credentials_or_privileges(sandbox: DockerSandbox, bundle: Bundle) -> None:
    found = detail(run(sandbox, bundle, "credentials")[0])
    assert (found["uid"], found["gid"]) == (65534, 65534)
    assert set(found["environment"]) <= {
        "GPG_KEY",
        "HOME",
        "HOSTNAME",
        "LANG",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "PYTHON_SHA256",
        "PYTHON_VERSION",
        "TMPDIR",
    }, json.dumps(found["environment"])
    assert found["docker_socket"] is False
    assert found["setuid"] == "PermissionError"
    assert found["capabilities"] == ["0000000000000000"] * 3
    assert found["no_new_privileges"] == "1"
    assert found["compilers"] == []


def test_a_check_cant_fake_another_verdict(sandbox: DockerSandbox, bundle: Bundle) -> None:
    found, _ = run(sandbox, bundle, "fakes_a_verdict")
    assert isinstance(found, dict)
    assert found["result"]["code"] == "the_real_verdict"
