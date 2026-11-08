"""Run untrusted-input work in a constrained subprocess.

Parsers are an attack surface, so each parse runs in its own Python process:

- **a time limit**: the process is killed when it passes it (``timeout``), and its CPU time is
  capped a little above it (``RLIMIT_CPU``)
- **memory limits**: its address space is capped with ``RLIMIT_AS`` and its writable memory with
  ``RLIMIT_DATA`` on Linux and macOS; on Windows only the time limit applies. A parser whose
  runtime reserves more address space than it uses (OCR) raises the first, never the
  second
- **file limits**: no core files, at most 512 open files, and no file over 1 GiB
- **a scratch directory**: the process's working directory and its temporary directory,
  deleted afterwards
- **a small environment**: no inherited variables but the ones Python needs, so no proxy or
  credential settings reach it, and Python runs isolated (``-I``)
- **confinement** (``stratafi_components.confine``): no network, no system calls that
  reach the worker, and a filesystem it can read only where Python and the system live, and
  write only in its scratch directory. ``confinement`` is ``required`` (a parse fails with
  ``sandbox_unconfined`` when seccomp isn't available), ``best_effort`` or ``off``
- **failures stay with one document**: a crash, a kill or an exception becomes a
  ``ComponentError`` with a short code for that document; the worker carries on

The child runs ``target`` (``"module:function"``), which takes a JSON payload and returns a
JSON-able result; both travel through files in the scratch directory, never through pipes that
could fill. The child also reports its peak memory and its confinement, which a run records for
the task. ``python -m stratafi_components.sandbox`` is the child's entry point.
"""

import contextlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from stratafi_core.components import ComponentError

DEFAULT_MEMORY_BYTES = 2 * 1024 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024 * 1024
MAX_OPEN_FILES = 512
_PASSED_ENV = ("PATH", "SYSTEMROOT", "HOME", "LANG", "LC_ALL", "TZ")
EXIT_ERROR = 3

ConfinementMode = Literal["required", "best_effort", "off"]


def run_isolated(
    target: str,
    payload: Mapping[str, Any],
    *,
    timeout: float,
    memory_bytes: int | None = DEFAULT_MEMORY_BYTES,
    address_space_bytes: int | None = None,
    scratch_parent: Path | None = None,
    stats: dict[str, Any] | None = None,
    confinement: ConfinementMode = "best_effort",
    read_files: tuple[str, ...] = (),
) -> Any:
    """Run ``target(payload)`` in a subprocess and return its result. Raises
    ``ComponentError``: ``sandbox_timeout``, ``sandbox_memory``, ``sandbox_crashed``,
    ``sandbox_unconfined``, or the code the target raised. ``memory_bytes`` caps writable
    memory and, unless ``address_space_bytes`` is larger, the address space. ``read_files`` are
    the files outside the scratch directory the target may read (the document). ``stats``, if
    given, gets ``peak_memory_bytes`` when the platform reports it, and ``confinement``."""
    scratch = Path(tempfile.mkdtemp(prefix="stratafi-sandbox-", dir=scratch_parent))
    try:
        input_path, output_path = scratch / "input.json", scratch / "output.json"
        options = {
            "data_bytes": memory_bytes or 0,
            "address_space_bytes": max(memory_bytes or 0, address_space_bytes or 0),
            "cpu_seconds": math.ceil(timeout) + 1,
            "confinement": confinement,
            "read_files": list(read_files),
        }
        input_path.write_text(
            json.dumps({"payload": dict(payload), "options": options}), encoding="utf-8"
        )
        environment = {name: os.environ[name] for name in _PASSED_ENV if name in os.environ}
        environment.update(
            PYTHONHASHSEED="0",
            PYTHONDONTWRITEBYTECODE="1",
            TMPDIR=str(scratch),
            TEMP=str(scratch),
            TMP=str(scratch),
        )
        command = [
            sys.executable,
            "-I",
            "-m",
            "stratafi_components.sandbox",
            target,
            str(input_path),
            str(output_path),
        ]
        try:
            process = subprocess.run(  # noqa: S603 - fixed arguments, no shell
                command,
                cwd=scratch,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # subprocess.run kills the child when the timeout passes.
            raise ComponentError("sandbox_timeout") from None
        if not output_path.is_file():
            raise ComponentError("sandbox_crashed")
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if stats is not None and isinstance(result.get("confinement"), dict):
            stats["confinement"] = result["confinement"]
        if process.returncode == EXIT_ERROR or "error" in result:
            raise ComponentError(str(result.get("error", "sandbox_failed")))
        if process.returncode != 0:
            raise ComponentError("sandbox_crashed")
        if stats is not None and isinstance(result.get("peak_memory_bytes"), int):
            stats["peak_memory_bytes"] = result["peak_memory_bytes"]
        return result["result"]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _limit_resources(options: Mapping[str, Any]) -> None:
    try:
        import resource
    except ImportError:  # Windows: the time limit still applies
        return

    def cap(limit: int, value: int) -> None:
        # Above the hard limit already in force: keep that one.
        if value > 0:
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(limit, (value, value))

    cap(resource.RLIMIT_AS, int(options.get("address_space_bytes", 0)))
    if sys.platform.startswith("linux"):
        cap(resource.RLIMIT_DATA, int(options.get("data_bytes", 0)))
    cpu = int(options.get("cpu_seconds", 0))
    if cpu > 0:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cap(resource.RLIMIT_FSIZE, MAX_FILE_BYTES)
    cap(resource.RLIMIT_NOFILE, MAX_OPEN_FILES)


def _peak_memory_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS, bytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _error_code(error: BaseException) -> str:
    if isinstance(error, MemoryError):
        return "sandbox_memory"
    code = getattr(error, "code", None) or getattr(error, "reason", None)
    if isinstance(code, str) and code.replace("_", "").isalnum() and len(code) <= 64:
        return code
    return "sandbox_failed"


def _confine(options: Mapping[str, Any]) -> dict[str, Any]:
    from stratafi_components.confine import Confinement, confine

    mode = options.get("confinement", "best_effort")
    if mode == "off":
        return Confinement().as_dict()
    files = [str(path) for path in options.get("read_files", ())]
    applied = confine(document=files[0] if files else None, scratch=os.getcwd())
    if mode == "required" and not applied.seccomp:
        raise ComponentError("sandbox_unconfined")
    return applied.as_dict()


def main(argv: list[str]) -> int:
    target, input_path, output_path = argv
    output = Path(output_path)
    confinement: dict[str, Any] | None = None
    try:
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        options = data.get("options", {})
        _limit_resources(options)
        confinement = _confine(options)
        module_name, _, function_name = target.partition(":")
        function = getattr(importlib.import_module(module_name), function_name)
        result = function(data["payload"])
    except BaseException as error:
        # Never the message: it can quote the document.
        output.write_text(
            json.dumps({"error": _error_code(error), "confinement": confinement}),
            encoding="utf-8",
        )
        return EXIT_ERROR
    output.write_text(
        json.dumps(
            {
                "result": result,
                "peak_memory_bytes": _peak_memory_bytes(),
                "confinement": confinement,
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
