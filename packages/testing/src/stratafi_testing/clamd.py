"""A minimal clamd client and an in-process clamd fake, for the integration tier.

The client speaks clamd's TCP protocol (``zPING``, ``zINSTREAM``). The real scan component is
``scan/clamav``; this is just enough to prove ClamAV answers and detects the EICAR test file.

``FakeClamd`` is the no-Docker stand-in (docs/guides/local-development.md): it answers ``PING``
and ``VERSION`` (with a signature database built at ``signature_time``, now by default), flags a
stream that is the EICAR test file, and passes everything else. Like ClamAV, whose
EICAR signature is a hash of the exact file, it doesn't flag EICAR inside other content; trailing
whitespace is allowed.

The EICAR string is assembled at runtime, so no file in this repo trips a developer's antivirus.
"""

import socket
import socketserver
import struct
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

_EICAR_HEAD = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
_EICAR_TAIL = "EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
CHUNK_BYTES = 8192
MAX_STREAM_BYTES = 64 * 1024 * 1024
FAKE_SIGNATURE = "Eicar-Test-Signature"

Address = tuple[str, int]


def eicar() -> bytes:
    """The EICAR anti-malware test file: harmless, and detected by every scanner."""
    return (_EICAR_HEAD + _EICAR_TAIL).encode("ascii")


class ClamdError(RuntimeError):
    pass


def _command(address: Address, payload: bytes, *, timeout: float) -> str:
    with socket.create_connection(address, timeout=timeout) as sock:
        sock.sendall(payload)
        reply = bytearray()
        while not reply.endswith(b"\0"):
            data = sock.recv(4096)
            if not data:
                break
            reply += data
    if not reply:
        raise ClamdError("clamd closed the connection without a reply")
    return reply.rstrip(b"\0").decode("utf-8", errors="replace")


def ping(address: Address, *, timeout: float = 5.0) -> bool:
    return _command(address, b"zPING\0", timeout=timeout) == "PONG"


def instream(address: Address, data: bytes, *, timeout: float = 60.0) -> str:
    """Scan ``data``. Returns clamd's reply: ``stream: OK`` or ``stream: <signature> FOUND``."""
    frames = [b"zINSTREAM\0"]
    for start in range(0, len(data), CHUNK_BYTES):
        chunk = data[start : start + CHUNK_BYTES]
        frames.append(struct.pack("!I", len(chunk)) + chunk)
    frames.append(struct.pack("!I", 0))
    return _command(address, b"".join(frames), timeout=timeout)


class _FakeHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        command = self._read_command()
        if command in (b"zPING", b"nPING"):
            self._reply("PONG")
        elif command in (b"zVERSION", b"nVERSION"):
            built = self.server.signature_time  # type: ignore[attr-defined]
            self._reply(f"ClamAV 1.4.3/27800/{built.astimezone(UTC):%a %b %d %H:%M:%S %Y}")
        elif command in (b"zINSTREAM", b"nINSTREAM"):
            data = self._read_stream()
            if data is None:
                self._reply("INSTREAM size limit exceeded. ERROR")
            elif data.rstrip() == eicar():
                self._reply(f"stream: {FAKE_SIGNATURE} FOUND")
            else:
                self._reply("stream: OK")
        else:
            self._reply("UNKNOWN COMMAND")

    def _read_exact(self, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = self.request.recv(size - len(data))
            if not chunk:
                raise ConnectionError("client closed the stream")
            data += chunk
        return bytes(data)

    def _read_command(self) -> bytes:
        command = bytearray()
        while not command.endswith(b"\0") and not command.endswith(b"\n") and len(command) < 64:
            byte = self.request.recv(1)
            if not byte:
                break
            command += byte
        return bytes(command).rstrip(b"\0\n")

    def _read_stream(self) -> bytes | None:
        data = bytearray()
        while True:
            (size,) = struct.unpack("!I", self._read_exact(4))
            if size == 0:
                return bytes(data)
            if len(data) + size > MAX_STREAM_BYTES:
                return None
            data += self._read_exact(size)

    def _reply(self, text: str) -> None:
        self.request.sendall(text.encode("utf-8") + b"\0")


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


@contextmanager
def fake_clamd(signature_time: datetime | None = None) -> Iterator[Address]:
    """Run ``FakeClamd`` on 127.0.0.1 with a free port; yields its address. Its signature
    database was built at ``signature_time`` (when it started, if None), and like a real
    clamd's, it stays that age: a ``VERSION`` reply doesn't change during a scan."""
    server = _Server(("127.0.0.1", 0), _FakeHandler)
    server.signature_time = signature_time or datetime.now(UTC)  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, name="fake-clamd", daemon=True)
    thread.start()
    try:
        yield server.server_address[:2]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
