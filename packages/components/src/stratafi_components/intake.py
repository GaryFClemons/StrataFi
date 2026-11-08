"""Staging files as they arrive: hash, detect, and unpack archives into a tree.

Intake (the API's uploads, folder, bucket and URL imports) streams each file to disk and hashes
it on the way, then ``stage`` detects its type and, for a ZIP archive (``unpack/zip``) or an
email message (``unpack/email``), unpacks its members within the archive limits. The result is
a tree of ``StagedFile``: the caller stores the bytes and records the documents. Nothing here
touches the database or the artifact store.

An archive or message that breaks a limit is staged with a ``refusal`` and no members; it is
recorded so the user can see why, and its scan excludes it.
"""

import hashlib
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO

from stratafi_components.archives import (
    FILES_LIMITS,
    ArchiveLimits,
    ArchiveRefused,
    extract_members,
)
from stratafi_components.detect import MSG, Detection, detect
from stratafi_components.messages import extract_eml, extract_msg

CHUNK_BYTES = 1024 * 1024
MAX_NAME_LENGTH = 512
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class InvalidName(ValueError):
    """A file name that can't be a corpus entry. The message never repeats the name."""


def entry_name(raw: str) -> str:
    """A file's name as a corpus entry: a relative POSIX path with no empty, ``.`` or ``..``
    parts and no control characters, at most 512 characters."""
    if not isinstance(raw, str) or _CONTROL_RE.search(raw):
        raise InvalidName("a file name has control characters")
    parts = [
        part
        for part in PurePosixPath(raw.replace("\\", "/")).parts
        if part not in ("", ".", "..", "/")
    ]
    name = "/".join(part.strip() for part in parts if part.strip())
    if not name or len(name) > MAX_NAME_LENGTH:
        raise InvalidName("a file name is empty or too long")
    return name


@dataclass
class StagedFile:
    name: str
    path: Path
    sha256: str
    size: int
    detection: Detection
    member_path: str | None = None
    refusal: tuple[str, str] | None = None  # (outcome, reason) for a refused archive
    members: list["StagedFile"] = field(default_factory=list)

    def walk(self) -> Iterator["StagedFile"]:
        yield self
        for member in self.members:
            yield from member.walk()


class HashingFile:
    """A temporary file that hashes and counts what's written to it, refusing more than
    ``max_bytes``."""

    def __init__(self, directory: Path, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        handle = tempfile.NamedTemporaryFile(dir=directory, prefix="upload-", delete=False)  # noqa: SIM115
        self.path = Path(handle.name)
        self._handle: IO[bytes] | None = handle
        self._digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> None:
        if self._handle is None:
            raise ValueError("the file is closed")
        self.size += len(data)
        if self.size > self.max_bytes:
            raise FileTooLarge()
        self._digest.update(data)
        self._handle.write(data)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    def discard(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)


class FileTooLarge(Exception):
    pass


def hash_file(path: Path, max_bytes: int | None = None) -> tuple[str, int]:
    """The SHA-256 and size of a file, read in chunks."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise FileTooLarge()
            digest.update(chunk)
    return digest.hexdigest(), size


def stage(
    path: Path,
    name: str,
    *,
    sha256: str,
    size: int,
    scratch: Path,
    unpack: bool = True,
    unpack_email: bool = True,
    limits: ArchiveLimits = FILES_LIMITS,
) -> StagedFile:
    """Detect a file and unpack it into ``scratch`` when it's a ZIP archive and ``unpack`` is
    on, or an email message and ``unpack_email`` is. Members that are archives themselves
    refuse the whole file."""
    detection = detect(path, name)
    staged = StagedFile(name=name, path=path, sha256=sha256, size=size, detection=detection)
    if detection.is_archive and unpack:
        extract = extract_members
    elif detection.is_message and unpack_email:
        extract = extract_msg if detection.media_type == MSG else extract_eml
    else:
        return staged
    destination = Path(tempfile.mkdtemp(dir=scratch, prefix="unpack-"))
    members: list[StagedFile] = []
    try:
        for member in extract(path, destination, limits):
            member_detection = detect(member.path, member.name)
            if member_detection.kind in ("zip", "gzip") and not limits.allow_nested:
                raise ArchiveRefused("limit_exceeded", "archive_nested")
            members.append(
                StagedFile(
                    name=entry_name(f"{name}/{member.name}"),
                    path=member.path,
                    sha256=member.sha256,
                    size=member.size,
                    detection=member_detection,
                    member_path=member.name,
                )
            )
    except ArchiveRefused as refused:
        shutil.rmtree(destination, ignore_errors=True)
        staged.refusal = (refused.outcome, refused.reason)
        return staged
    staged.members = members
    return staged


@contextmanager
def scratch_directory(parent: Path | None = None) -> Iterator[Path]:
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(dir=parent, prefix="stratafi-intake-"))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
