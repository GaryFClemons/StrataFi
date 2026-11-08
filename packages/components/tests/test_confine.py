"""The parse sandbox's confinement: a parser can't reach the network, the worker
or the files outside what it needs."""

import hashlib
import os
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from stratafi_components.confine import (
    _ARCHES,
    DENIED,
    Confinement,
    install_seccomp,
    landlock_abi,
    probe_targets,
    seccomp_program,
    self_test,
)
from stratafi_components.detect import detect
from stratafi_components.extract import raw_document
from stratafi_components.parsers.markdown import MarkdownParser
from stratafi_components.sandbox import run_isolated
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.settings import LimitSettings, OfflineSettings

LINUX = sys.platform.startswith("linux")
PROBE = "stratafi_testing.sandbox_targets:reach"
FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"

# --- unit tier: the program ------------------------------------------------------------------


@pytest.mark.parametrize("arch", sorted(_ARCHES))
def test_the_seccomp_program_denies_each_call_and_checks_the_architecture(arch: str) -> None:
    table = _ARCHES[arch]
    program = seccomp_program(table, x32=arch == "x86_64")
    instructions = [struct.unpack("=HBBI", program[i : i + 8]) for i in range(0, len(program), 8)]
    # Load the architecture, compare, kill otherwise; then load the call's number.
    assert instructions[0] == (0x20, 0, 0, 4)
    assert instructions[1][3] == table["arch"]
    assert instructions[2] == (0x06, 0, 0, 0x80000000)
    compared = {k for op, _, _, k in instructions if op == 0x15}
    assert {table[name] for name in DENIED} <= compared
    # Every jump lands inside the program, and it ends allow, EPERM, ENOSYS.
    for index, (op, true, false, _) in enumerate(instructions):
        if op in (0x15, 0x35, 0x45):
            assert index + 1 + max(true, false) < len(instructions)
    assert [k for _, _, _, k in instructions[-3:]] == [0x7FFF0000, 0x50001, 0x50026]


def test_confinement_blocks_the_network_with_seccomp_or_a_namespace() -> None:
    assert Confinement(network="seccomp", seccomp=True).blocks_network
    assert Confinement(network="namespace").blocks_network
    assert not Confinement().blocks_network


def test_a_prod_deployment_must_require_confinement() -> None:
    loose = OfflineSettings(env="prod")
    assert any("parse_confinement" in problem for problem in loose.deployment_problems())
    strict = OfflineSettings(env="prod", limits={"parse_confinement": "required"})
    assert not any("parse_confinement" in problem for problem in strict.deployment_problems())


# --- unit tier: on this machine --------------------------------------------------------------


def _seccomp_works() -> bool:
    if not LINUX:
        return False
    # Try it in a throwaway child, never in the test process.
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child
        os._exit(0 if install_seccomp() else 1)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status) == 0


SECCOMP = _seccomp_works()
needs_seccomp = pytest.mark.skipif(not SECCOMP, reason="seccomp isn't available here")


@pytest.fixture
def targets(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """A TCP and a UDP listener, a Unix socket, a secret file and a path outside the sandbox."""
    with probe_targets(tmp_path) as made:
        yield made


@needs_seccomp
def test_without_confinement_the_probe_reaches_out(targets: dict[str, Any]) -> None:
    # The control: the probe works, so the confined run's refusals mean something.
    reached = run_isolated(PROBE, targets, timeout=30, confinement="off")
    assert all(reached.values()), reached


@needs_seccomp
def test_a_confined_parse_cannot_reach_the_network_or_the_worker(
    targets: dict[str, Any], tmp_path: Path
) -> None:
    stats: dict[str, Any] = {}
    reached = run_isolated(PROBE, targets, timeout=30, confinement="required", stats=stats)
    assert reached == {
        "tcp": False,
        "udp": False,
        "unix": False,
        "secret": False,
        "parent_environ": False,
        "write_outside": False,
    }
    assert not (tmp_path / "planted").exists()
    assert stats["confinement"]["seccomp"] is True
    if landlock_abi():
        assert stats["confinement"]["landlock"] >= 1


@needs_seccomp
def test_a_confined_parse_still_parses(tmp_path: Path) -> None:
    path = tmp_path / "tips.md"
    path.write_bytes((FIXTURES / "formats" / "tips.md").read_bytes())
    context = StageContext(limits=LimitSettings(parse_confinement="required"))
    component = MarkdownParser()
    data = path.read_bytes()
    raw = raw_document("d-1", hashlib.sha256(data).hexdigest(), len(data), detect(path, path.name))
    document = component.parse(path, raw, component.resolve_params({}), context)
    assert document.elements
    assert context.metrics["sandbox_seccomp"] == 1


@pytest.mark.skipif(LINUX and SECCOMP, reason="seccomp is available here")
def test_required_confinement_refuses_where_seccomp_is_missing() -> None:
    with pytest.raises(ComponentError) as raised:
        run_isolated(
            "stratafi_testing.sandbox_targets:echo", {}, timeout=30, confinement="required"
        )
    assert raised.value.code == "sandbox_unconfined"


@needs_seccomp
def test_the_self_test_reports_a_confined_sandbox() -> None:
    # What a deployment runs: python -m stratafi_components.confine (exit 0 when confined).
    report = self_test()
    assert report["confinement"]["seccomp"] is True
    assert not any(report["reached"].values())
