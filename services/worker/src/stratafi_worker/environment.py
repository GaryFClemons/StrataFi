"""What a run records about where it ran: its environment and its hardware.

- Environment: Stratafi's version and commit (``STRATAFI_COMMIT``, or ``git`` when the code is
  a checkout), Python, the libraries whose versions can change output, the OS, and the
  container image (``STRATAFI_IMAGE``).
- Hardware: the host, CPU model and cores, RAM, GPUs through ``pynvml`` when it's installed (its
  absence just means no GPUs are listed), and the worker's resource classes.
"""

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import stratafi_core

LIBRARIES = (
    "defusedxml",
    "fastapi",
    "httpx2",
    "numpy",
    "psycopg",
    "pydantic",
    "pypdf",
    "snowballstemmer",
    "sqlalchemy",
    "tiktoken",
)
REPO_ROOT = Path(__file__).resolve().parents[4]


def _commit() -> str | None:
    if commit := os.environ.get("STRATAFI_COMMIT"):
        return commit
    if not (REPO_ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=12", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def environment() -> dict[str, Any]:
    libraries = {}
    for name in LIBRARIES:
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "stratafi_version": stratafi_core.__version__,
        "commit": _commit(),
        "python": platform.python_version(),
        "libraries": libraries,
        "os": platform.platform(),
        "container_image": os.environ.get("STRATAFI_IMAGE"),
    }


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip() or None
    return platform.processor() or None


def _ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def _gpus() -> list[dict[str, Any]]:
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:  # no driver, no GPUs
        return []
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        gpus = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            gpus.append(
                {
                    "model": str(pynvml.nvmlDeviceGetName(handle)),
                    "memory_bytes": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                    "driver": str(driver),
                }
            )
        return gpus
    finally:
        pynvml.nvmlShutdown()


def hardware(resource_classes: list[str]) -> dict[str, Any]:
    return {
        "host": platform.node() or None,
        "cpu_model": _cpu_model(),
        "cpu_cores": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "gpus": _gpus(),
        "resource_classes": sorted(resource_classes),
    }
