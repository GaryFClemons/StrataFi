"""The custom-check sandbox's image for tests: built from
docker/sandbox/Dockerfile under a test tag, so system tests run the sandbox this checkout
defines."""

import shutil
import subprocess
from pathlib import Path

TEST_IMAGE = "stratafi/check-sandbox:test"


def build_sandbox_image(repo_root: Path, tag: str = TEST_IMAGE) -> str | None:
    """Build the image and return its tag; None when there's no Docker daemon to build with.
    Raises when the build fails."""
    docker = shutil.which("docker")
    if docker is None:
        return None
    info = subprocess.run(  # noqa: S603 - fixed arguments
        [docker, "info"], capture_output=True, timeout=60, check=False
    )
    if info.returncode != 0:
        return None
    built = subprocess.run(  # noqa: S603 - fixed arguments
        [docker, "build", "-q", "-f", "docker/sandbox/Dockerfile", "-t", tag, "."],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if built.returncode != 0:
        raise RuntimeError("the sandbox image didn't build: " + built.stderr[-2000:])
    return tag
