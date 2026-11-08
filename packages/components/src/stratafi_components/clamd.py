"""A clamd client for ``scan/clamav``: ``VERSION`` and ``INSTREAM`` over TCP.

- ``VERSION`` names the engine, the signature database's revision and when it was built. A
  database older than the limit (72 hours by default) refuses a pass: the result is an error,
  never ``clean``.
- ``INSTREAM`` sends the file from disk in chunks, so a large file never sits in memory.
- ``VERSION`` is read again afterwards; if the database reloaded during the scan, the result is
  an error (``signature_identity_changed``).
"""

import re
import socket
import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CHUNK_BYTES = 64 * 1024
_VERSION_RE = re.compile(
    r"ClamAV (\d+\.\d+\.\d+)/([1-9][0-9]{0,9})/"
    r"((?:Sun|Mon|Tue|Wed|Thu|Fri|Sat) (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"[ 0-3][0-9] [0-2][0-9]:[0-5][0-9]:[0-5][0-9] [0-9]{4})"
)

Address = tuple[str, int]


class ClamdError(Exception):
    """clamd couldn't be asked, or answered something unusable. ``reason`` is a short code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class SignatureIdentity:
    engine: str
    revision: str
    built_at: datetime


def parse_version(reply: str, *, now: datetime) -> SignatureIdentity:
    """The engine and signature database from a ``VERSION`` reply. clamd writes the database's
    build time in UTC (the image sets TZ=UTC)."""
    match = _VERSION_RE.fullmatch(reply.strip())
    if match is None:
        raise ClamdError("signature_identity_invalid")
    try:
        built = datetime.strptime(match.group(3), "%a %b %d %H:%M:%S %Y").replace(tzinfo=UTC)
    except ValueError:
        raise ClamdError("signature_identity_invalid") from None
    if built.year < 2020 or (built - now).total_seconds() > 300:
        raise ClamdError("signature_identity_invalid")
    return SignatureIdentity(f"clamav-{match.group(1)}", match.group(2), built)


def format_version(built_at: datetime, *, engine: str = "1.4.3", revision: str = "27800") -> str:
    """A ``VERSION`` reply, as clamd writes it. For fakes."""
    return f"ClamAV {engine}/{revision}/{built_at.astimezone(UTC):%a %b %d %H:%M:%S %Y}"


class ClamdClient:
    def __init__(self, address: Address, *, timeout: float = 30.0) -> None:
        self.address = address
        self.timeout = timeout

    def _connect(self, timeout: float) -> socket.socket:
        try:
            return socket.create_connection(self.address, timeout=timeout)
        except TimeoutError:
            raise ClamdError("timeout") from None
        except OSError:
            raise ClamdError("connection") from None

    @staticmethod
    def _reply(sock: socket.socket) -> str:
        reply = bytearray()
        try:
            while not reply.endswith(b"\0"):
                data = sock.recv(4096)
                if not data:
                    break
                reply += data
        except TimeoutError:
            raise ClamdError("timeout") from None
        except OSError:
            raise ClamdError("connection") from None
        if not reply:
            raise ClamdError("invalid_response")
        return reply.rstrip(b"\0").decode("utf-8", errors="replace")

    def version(self) -> str:
        with self._connect(self.timeout) as sock:
            try:
                sock.sendall(b"zVERSION\0")
            except OSError:
                raise ClamdError("connection") from None
            return self._reply(sock)

    def instream(self, path: Path) -> str:
        """Scan the file at ``path``. Returns clamd's reply, such as ``stream: OK``. Sending and
        scanning take time in proportion to size: the timeout grows a second per MiB."""
        size = path.stat().st_size
        timeout = self.timeout + size // (1024 * 1024)
        with self._connect(timeout) as sock:
            try:
                sock.sendall(b"zINSTREAM\0")
                with path.open("rb") as handle:
                    while chunk := handle.read(CHUNK_BYTES):
                        sock.sendall(struct.pack("!I", len(chunk)) + chunk)
                sock.sendall(struct.pack("!I", 0))
            except TimeoutError:
                raise ClamdError("timeout") from None
            except OSError:
                # clamd closes the connection when a stream passes its StreamMaxLength.
                raise ClamdError("connection") from None
            return self._reply(sock)
