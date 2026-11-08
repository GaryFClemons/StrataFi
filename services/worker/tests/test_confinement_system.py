"""A parser cannot reach the network, in the image the worker runs.

System tier: ``STRATAFI_TEST_SYSTEM=1`` with a Docker daemon and the app image built:

    docker compose --profile app build api

The test runs the confinement's self-test (``python -m stratafi_components.confine``) in that
image as compose runs the worker: its non-root user, no capabilities, no privilege gain, and
Docker's default seccomp profile. The self-test starts TCP, UDP and Unix listeners and a secret
file, then runs the probe in the parse sandbox; nothing may be reached.
"""

import json
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.system

IMAGE = "stratafi/app:dev"


def _image_exists() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    found = subprocess.run(  # noqa: S603 - fixed arguments
        [docker, "image", "inspect", IMAGE], capture_output=True, check=False
    )
    return found.returncode == 0


def test_a_parser_in_the_worker_image_cannot_reach_out() -> None:
    docker = shutil.which("docker")
    if docker is None or not _image_exists():
        pytest.skip(f"build {IMAGE} first: docker compose --profile app build api")
    completed = subprocess.run(  # noqa: S603 - fixed arguments
        [
            docker,
            "run",
            "--rm",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            IMAGE,
            "python",
            "-m",
            "stratafi_components.confine",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert completed.returncode == 0, report
    assert report["confinement"]["seccomp"] is True
    assert report["confinement"]["landlock"] >= 1
    assert report["reached"] == dict.fromkeys(
        ("tcp", "udp", "unix", "secret", "parent_environ", "write_outside"), False
    )
