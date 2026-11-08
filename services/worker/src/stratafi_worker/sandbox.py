"""The custom-check sandbox. Each custom check runs in a container of its own:

    docker run --rm --init --pull never --network none --read-only --cap-drop ALL
      --security-opt no-new-privileges --user 65534:65534 --ipc none --log-driver none
      --pids-limit 64 --memory 512m --memory-swap 512m --cpus 1
      --ulimit nofile=256:256 --ulimit core=0:0 --ulimit cpu=<timeout>
      --tmpfs /scratch:rw,noexec,nosuid,nodev,size=64m,mode=1777
      --mount type=bind,source=<bundle>,target=/bundle,readonly
      --mount type=bind,source=<code>,target=/check,readonly
      [--runtime runsc] stratafi/check-sandbox:1 /check/<path.py> <function>

The image (docker/sandbox/Dockerfile) holds Python's standard library, ``stratafi.checks`` and
the entry point: no compiler, no pip. The container has no network, no credentials (only
``TMPDIR`` and ``HOME`` are set), no Docker socket and none of the worker's files: the run's
bundle and the suite's code, read-only, and a small ``/scratch``. gVisor (``runsc``) runs it
where Docker lists that runtime. ``--init`` keeps the check from being the container's PID 1,
which the kernel shields from signals such as the CPU limit's SIGXCPU. The worker keeps at most
``output_bytes`` of what it prints and kills it at its deadline: ``timeout_s`` of CPU time, or
``timeout_s`` plus ``start_seconds`` of wall-clock time.

``Bundle`` writes what a check reads, through ``CheckContext``: ``context.json`` (the check,
the suite, the run IDs and the eval run's aggregates), and ``chunks.jsonl``, ``documents.jsonl``
and ``queries.jsonl``. They hold the run's chunk text, so they live only in the worker's
working directory while the checks run, and are deleted afterwards.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from stratafi_core.artifacts import Chunk
from stratafi_core.settings import SandboxSettings
from stratafi_core.suites import EntryState, QueryRow

logger = logging.getLogger("stratafi.worker.sandbox")

RESULT_MARKER = "STRATAFI-CHECK-RESULT "  # stratafi.checks.RESULT_MARKER
CONTEXT_FILE = "context.json"
DATA_FILES = ("chunks.jsonl", "documents.jsonl", "queries.jsonl")
SANDBOX_USER = "65534:65534"
CPU_LIMIT_EXIT = 128 + 24  # SIGXCPU: the cpu ulimit, the check's timeout_s of CPU time
CONTAINER_LABEL = "stratafi.sandbox=custom-check"


class SandboxError(Exception):
    """The sandbox couldn't give a verdict. The message is a code, never output."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Sandbox(Protocol):
    def run(
        self, bundle: Path, code: Path, path: str, function: str, timeout_s: int
    ) -> Mapping[str, Any]:
        """The check's payload: ``{"result": ...}`` or ``{"error": code}``. Raises
        ``SandboxError`` when there is none."""
        ...


# --- the bundle ---------------------------------------------------------------------------------


def _write_lines(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    path.chmod(0o644)
    return count


def chunk_record(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "index_text": chunk.index_text,
        "heading_path": list(chunk.heading_path),
        "metadata": dict(chunk.metadata),
        "tokens": dict(chunk.tokens),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "context": chunk.context,
    }


def document_record(entry: EntryState) -> dict[str, Any]:
    return {
        "document_id": entry.document_id,
        "name": entry.name,
        "media_type": entry.media_type,
        "state": entry.state,
        "scan_outcome": entry.scan_outcome,
        "parse_status": entry.parse_status,
    }


def query_record(row: QueryRow) -> dict[str, Any]:
    return {
        "item_id": row.item_id,
        "has_evidence": row.has_evidence,
        "tags": list(row.tags),
        "file_types": list(row.file_types),
        "metrics": dict((row.score or {}).get("metrics") or {}),
    }


class Bundle:
    """A test run's custom-check inputs on disk: the run's data and the suite's code, written
    once, and a directory per check with its ``context.json``. ``close`` deletes it all."""

    def __init__(self, workdir: Path | None = None) -> None:
        if workdir is not None:
            workdir.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="stratafi-checks-", dir=workdir))
        self.data = self.root / "data"
        self.code = self.root / "code"
        for directory in (self.data, self.code):
            directory.mkdir()
            directory.chmod(0o755)
        self._checks = 0

    def write_data(
        self,
        *,
        chunks: Iterable[Chunk],
        documents: Sequence[EntryState],
        queries: Sequence[QueryRow],
    ) -> dict[str, int]:
        return {
            "chunks": _write_lines(
                self.data / "chunks.jsonl", (chunk_record(chunk) for chunk in chunks)
            ),
            "documents": _write_lines(
                self.data / "documents.jsonl", (document_record(entry) for entry in documents)
            ),
            "queries": _write_lines(
                self.data / "queries.jsonl", (query_record(row) for row in queries)
            ),
        }

    def write_code(self, code: Mapping[str, str]) -> None:
        """The suite's files under ``code/``; their paths are relative, validated by the
        suite."""
        root = self.code.resolve()
        for relative, source in code.items():
            target = (self.code / relative).resolve()
            if not target.is_relative_to(root) or target == root:
                raise SandboxError("code_path_invalid")
            for parent in reversed(target.relative_to(root).parents):
                directory = root / parent
                if directory != root and not directory.exists():
                    directory.mkdir()
                    directory.chmod(0o755)
            target.write_text(source, encoding="utf-8")
            target.chmod(0o644)

    def for_check(self, context: Mapping[str, Any]) -> Path:
        """A directory for one check: its ``context.json`` and the run's data, linked."""
        self._checks += 1
        directory = self.root / f"check-{self._checks}"
        directory.mkdir()
        directory.chmod(0o755)
        context_file = directory / CONTEXT_FILE
        context_file.write_text(json.dumps(dict(context), ensure_ascii=False), encoding="utf-8")
        context_file.chmod(0o644)
        for name in DATA_FILES:
            source = self.data / name
            if source.exists():
                os.link(source, directory / name)
        return directory

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


# --- the container ------------------------------------------------------------------------------


@dataclass
class _Outcome:
    output: bytes
    exit_code: int | None
    timed_out: bool = False
    overflow: bool = False


def parse_output(output: bytes, exit_code: int | None) -> Mapping[str, Any]:
    """The payload in the container's output: the last line the entry point marked. Without
    one, why: ``timeout`` (its CPU time ran out), ``killed`` (another signal, often the memory
    limit), ``sandbox_failed`` (Docker couldn't start it) or ``no_result``."""
    for line in reversed(output.decode("utf-8", "replace").splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                payload = json.loads(line[len(RESULT_MARKER) :])
            except (ValueError, RecursionError):
                raise SandboxError("output_invalid") from None
            if not isinstance(payload, dict):
                raise SandboxError("output_invalid")
            return payload
    if exit_code is not None and exit_code in (125, 126, 127):
        raise SandboxError("sandbox_failed")
    if exit_code == CPU_LIMIT_EXIT:
        raise SandboxError("timeout")
    if exit_code is not None and (exit_code >= 128 or exit_code < 0):
        raise SandboxError("killed")
    raise SandboxError("no_result")


class DockerSandbox:
    """Runs each check with ``docker run`` (``SandboxSettings``)."""

    def __init__(self, settings: SandboxSettings) -> None:
        self.settings = settings
        self._runtime: str | None = None
        self._lock = threading.Lock()

    # --- the host ---------------------------------------------------------------------------------

    def _docker(self, *args: str, timeout: float = 30) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(  # noqa: S603 - a fixed command; arguments aren't a shell's
                [self.settings.docker, *args],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise SandboxError("sandbox_unavailable") from None

    def runtime(self) -> str:
        """The container runtime: gVisor's ``runsc`` when Docker has it and ``runtime`` allows
        it, else ``runc``. Checked once; the image must be there too (no pulls)."""
        with self._lock:
            if self._runtime is not None:
                return self._runtime
            found = self._docker("info", "--format", "{{json .Runtimes}}")
            if found.returncode != 0:
                raise SandboxError("sandbox_unavailable")
            try:
                runtimes = set(json.loads(found.stdout or b"{}"))
            except ValueError:
                runtimes = set()
            wanted = self.settings.runtime
            if wanted == "runsc" and "runsc" not in runtimes:
                raise SandboxError("sandbox_unavailable")
            runtime = "runsc" if wanted != "runc" and "runsc" in runtimes else "runc"
            if self._docker("image", "inspect", self.settings.image).returncode != 0:
                raise SandboxError("sandbox_unavailable")
            self._runtime = runtime
            logger.info("custom-check sandbox ready", extra={"runtime": runtime})
            return runtime

    def command(
        self,
        name: str,
        bundle: Path,
        code: Path,
        path: str,
        function: str,
        timeout_s: int,
        runtime: str,
    ) -> list[str]:
        """The ``docker run`` command for one check."""
        limits = self.settings
        memory = f"{limits.memory_mb}m"
        command = [
            limits.docker,
            "run",
            "--rm",
            "--init",
            "--pull",
            "never",
            "--name",
            name,
            "--label",
            CONTAINER_LABEL,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            SANDBOX_USER,
            "--ipc",
            "none",
            "--log-driver",
            "none",
            "--pids-limit",
            str(limits.pids),
            "--memory",
            memory,
            "--memory-swap",
            memory,
            "--cpus",
            f"{limits.cpus:g}",
            "--ulimit",
            "nofile=256:256",
            "--ulimit",
            "core=0:0",
            "--ulimit",
            f"cpu={timeout_s}:{timeout_s + 1}",
            "--tmpfs",
            f"/scratch:rw,noexec,nosuid,nodev,size={limits.scratch_mb}m,mode=1777",
            "--mount",
            f"type=bind,source={bundle},target=/bundle,readonly",
            "--mount",
            f"type=bind,source={code},target=/check,readonly",
            "--env",
            "TMPDIR=/scratch",
            "--env",
            "HOME=/scratch",
        ]
        if runtime != "runc":
            command += ["--runtime", runtime]
        return [*command, limits.image, f"/check/{path}", function]

    def _kill(self, name: str) -> None:
        try:
            self._docker("kill", name, timeout=30)
        except SandboxError:
            logger.warning("couldn't kill a custom-check container", extra={"container": name})

    def _execute(self, command: list[str], name: str, seconds: float) -> _Outcome:
        limit = self.settings.output_bytes
        try:
            process = subprocess.Popen(  # noqa: S603 - a fixed command; arguments aren't a shell's
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            raise SandboxError("sandbox_unavailable") from None
        assert process.stdout is not None  # noqa: S101 - stdout is a pipe
        stream = process.stdout
        buffer = bytearray()
        overflow = threading.Event()
        wake = threading.Event()

        def read() -> None:
            # Past the limit, it keeps reading and drops what it reads, so neither the docker
            # CLI nor the daemon blocks on a full pipe while the container is killed.
            try:
                while chunk := stream.read1(65536):
                    if overflow.is_set():
                        continue
                    if len(buffer) + len(chunk) > limit:
                        overflow.set()
                        wake.set()
                    else:
                        buffer.extend(chunk)
            except (OSError, ValueError):
                pass
            finally:
                wake.set()

        reader = threading.Thread(target=read, name=f"sandbox-{name}", daemon=True)
        reader.start()
        wake.wait(seconds)
        timed_out = not overflow.is_set() and reader.is_alive()
        if overflow.is_set() or timed_out:
            self._kill(name)
        try:
            exit_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            exit_code = process.wait()
        reader.join(30)
        return _Outcome(bytes(buffer), exit_code, timed_out, overflow.is_set())

    def run(
        self, bundle: Path, code: Path, path: str, function: str, timeout_s: int
    ) -> Mapping[str, Any]:
        runtime = self.runtime()
        name = f"stratafi-check-{uuid.uuid4().hex}"
        command = self.command(name, bundle, code, path, function, timeout_s, runtime)
        started = time.monotonic()
        outcome = self._execute(command, name, timeout_s + self.settings.start_seconds)
        logger.info(
            "custom check ran",
            extra={
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
                "exit_code": outcome.exit_code,
                "output_bytes": len(outcome.output),
                "timed_out": outcome.timed_out,
                "overflow": outcome.overflow,
            },
        )
        if outcome.overflow:
            raise SandboxError("output_too_large")
        if outcome.timed_out:
            raise SandboxError("timeout")
        return parse_output(outcome.output, outcome.exit_code)


def sandbox_from_settings(settings: SandboxSettings) -> Sandbox | None:
    if settings.backend == "docker":
        return DockerSandbox(settings)
    return None
