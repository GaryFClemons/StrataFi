"""Email messages: ``.eml`` and ``.msg`` bodies and attachments become documents
(``unpack/email``).

Like ZIP archives, messages unpack when they arrive. A message yields:

- ``message.txt``: its subject and date, then its plain-text body. A message with only an HTML
  body yields ``message.html`` instead, which parsers read as HTML. Addresses are left out: they
  are about people, not about what the message says.
- one member per attachment, under its file name (normalized as archive members are; the name
  is a label, never a path on disk), or ``attachment-<n>`` when it has none.

The archive limits apply (``ArchiveLimits``): at most ``max_entries`` members and
``max_expanded_bytes`` in total. An attached archive refuses the message (``archive_nested``),
as a nested archive refuses a ZIP. An attached message is a member like any other file; it is
not unpacked again.

- ``.eml`` (RFC 5322 / MIME) is read with the standard library's ``email`` package.
- ``.msg`` (Outlook's compound file) is read with ``olefile`` (BSD-2-Clause): the subject,
  body and attachments are fixed streams of the MAPI property storage. Its HTML or RTF-only
  bodies are not decoded; such a message yields its attachments and an empty body.

License: the standard library, and olefile under BSD-2-Clause, so the component is cleared for
any use.
"""

import email
import email.policy
import hashlib
from collections.abc import Iterator
from email.message import EmailMessage
from pathlib import Path

from stratafi_components.archives import (
    FILES_LIMITS,
    NESTED_ARCHIVE_NAME,
    ArchiveLimits,
    ArchiveRefused,
    Member,
    member_name,
)

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
BODY_TEXT = "message.txt"
BODY_HTML = "message.html"
# MAPI property streams of an Outlook message (MS-OXMSG): Unicode (001F) or 8-bit (001E).
_SUBJECT = ("__substg1.0_0037001F", "__substg1.0_0037001E")
_BODY = ("__substg1.0_1000001F", "__substg1.0_1000001E")
_ATTACH_DATA = "__substg1.0_37010102"
_ATTACH_LONG_NAME = ("__substg1.0_3707001F", "__substg1.0_3707001E")
_ATTACH_NAME = ("__substg1.0_3704001F", "__substg1.0_3704001E")


class _Writer:
    """Writes members under generated names, enforcing the limits."""

    def __init__(self, destination: Path, limits: ArchiveLimits) -> None:
        self.destination = destination
        self.limits = limits
        self.total = 0
        self.count = 0
        self.names: set[str] = set()
        destination.mkdir(parents=True, exist_ok=True)

    def _name(self, raw: str, fallback: str) -> str:
        name = member_name(raw) or fallback
        stem, dot, suffix = name.rpartition(".")
        candidate, index = name, 1
        while candidate in self.names:
            index += 1
            candidate = f"{stem}-{index}.{suffix}" if dot and stem else f"{name}-{index}"
        self.names.add(candidate)
        return candidate

    def write(self, raw_name: str, data: bytes, fallback: str) -> Member:
        if not self.limits.allow_nested and NESTED_ARCHIVE_NAME.search(raw_name or ""):
            raise ArchiveRefused("limit_exceeded", "archive_nested")
        self.count += 1
        if self.count > self.limits.max_entries:
            raise ArchiveRefused("limit_exceeded", "archive_entry_limit")
        self.total += len(data)
        if self.total > self.limits.max_expanded_bytes:
            raise ArchiveRefused("limit_exceeded", "archive_expansion_limit")
        target = self.destination / f"member-{self.count - 1:05d}"
        target.write_bytes(data)
        return Member(
            self._name(raw_name, fallback), target, len(data), hashlib.sha256(data).hexdigest()
        )


def _body_text(subject: str | None, date: str | None, body: str) -> bytes:
    header = [f"Subject: {subject}" if subject else "", f"Date: {date}" if date else ""]
    lines = [line for line in header if line]
    return ("\n".join(lines) + ("\n\n" if lines else "") + body.strip() + "\n").encode("utf-8")


def _read(path: Path) -> bytes:
    if path.stat().st_size > MAX_MESSAGE_BYTES:
        raise ArchiveRefused("limit_exceeded", "message_too_large")
    return path.read_bytes()


def extract_eml(
    path: Path, destination: Path, limits: ArchiveLimits = FILES_LIMITS
) -> Iterator[Member]:
    """An ``.eml`` message's body and attachments. Raises ``ArchiveRefused``."""
    try:
        message = email.message_from_bytes(_read(path), policy=email.policy.default)
    except (ValueError, TypeError, IndexError):
        raise ArchiveRefused("malformed", "message_malformed") from None
    if not isinstance(message, EmailMessage) or not message.keys():
        raise ArchiveRefused("malformed", "message_malformed")
    writer = _Writer(destination, limits)
    try:
        body = message.get_body(preferencelist=("plain", "html"))
        subject, date = message.get("subject"), message.get("date")
        if body is not None and body.get_content_type() == "text/html":
            yield writer.write(BODY_HTML, str(body.get_content()).encode("utf-8"), BODY_HTML)
        else:
            text = str(body.get_content()) if body is not None else ""
            yield writer.write(
                BODY_TEXT, _body_text(str(subject or ""), str(date or ""), text), BODY_TEXT
            )
        for index, part in enumerate(message.iter_attachments(), start=1):
            if part.get_content_type() == "message/rfc822":
                inner = part.get_payload(0) if part.is_multipart() else part.get_payload()
                data = inner.as_bytes() if hasattr(inner, "as_bytes") else bytes(inner)
                yield writer.write(part.get_filename() or "", data, f"attachment-{index}.eml")
                continue
            payload = part.get_payload(decode=True)
            if not isinstance(payload, bytes):
                continue
            yield writer.write(part.get_filename() or "", payload, f"attachment-{index}")
    except ArchiveRefused:
        raise
    except (ValueError, TypeError, LookupError, AttributeError):
        raise ArchiveRefused("malformed", "message_malformed") from None


def _stream(ole: object, names: tuple[str, ...], prefix: str = "") -> str | None:
    for name in names:
        path = f"{prefix}{name}"
        if ole.exists(path):  # type: ignore[attr-defined]
            data = ole.openstream(path).read()  # type: ignore[attr-defined]
            encoding = "utf-16-le" if name.endswith("001F") else "latin-1"
            return data.decode(encoding, errors="replace").rstrip("\0")
    return None


def extract_msg(
    path: Path, destination: Path, limits: ArchiveLimits = FILES_LIMITS
) -> Iterator[Member]:
    """An Outlook ``.msg`` message's body and attachments. Raises ``ArchiveRefused``."""
    import olefile

    data = _read(path)
    if len(data) < olefile.MINIMAL_OLEFILE_SIZE or not data.startswith(olefile.MAGIC):
        raise ArchiveRefused("malformed", "message_malformed")
    try:
        ole = olefile.OleFileIO(data)
    except (OSError, ValueError):
        raise ArchiveRefused("malformed", "message_malformed") from None
    writer = _Writer(destination, limits)
    try:
        subject = _stream(ole, _SUBJECT)
        body = _stream(ole, _BODY) or ""
        yield writer.write(BODY_TEXT, _body_text(subject, None, body), BODY_TEXT)
        storages = sorted(
            {entry[0] for entry in ole.listdir() if entry[0].startswith("__attach_version1.0_")}
        )
        for index, storage in enumerate(storages, start=1):
            prefix = f"{storage}/"
            if not ole.exists(prefix + _ATTACH_DATA):
                continue  # an embedded message or OLE object, not a file
            name = _stream(ole, _ATTACH_LONG_NAME, prefix) or _stream(ole, _ATTACH_NAME, prefix)
            payload = ole.openstream(prefix + _ATTACH_DATA).read()
            yield writer.write(name or "", payload, f"attachment-{index}")
    except ArchiveRefused:
        raise
    except (OSError, ValueError, TypeError):
        raise ArchiveRefused("malformed", "message_malformed") from None
    finally:
        ole.close()
