"""ZIP archives: safety limits, then unpacking (``unpack/zip``).

``inspect_zip`` first checks the central directory as it declares itself:

- at most 10,000 entries; no ZIP64 (its sizes are what a bomb would lie about)
- no encrypted entries, and only stored or deflated ones
- at most 512 MiB expanded in total, and at most 100 times the compressed size
- no nested archives, judged by name (``.zip``, ``.7z``, ``.tar``…)

``extract_members`` then inflates each entry for real, counting what comes out, so the limits
hold for what the archive contains, not what it claims: an entry that expands past its
declared size, or fails its CRC, is refused. Unpacking stops at the first refusal and keeps
nothing. ``measure_zip`` makes the same checks without keeping anything.

Member names are labels, never paths on disk: members are written under generated names, so a
name like ``../../etc/passwd`` can't escape (it's normalized to ``etc/passwd`` for display).
"""

import hashlib
import re
import struct
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

CHUNK_BYTES = 64 * 1024
NESTED_ARCHIVE_NAME = re.compile(r"\.(?:zip|jar|7z|rar|tar|tgz|gz|bz2|xz)$", re.IGNORECASE)
ZIP64_EXTRA = 0x0001
STORED, DEFLATED = 0, 8
# Resource forks macOS adds to archives: not documents.
_SKIPPED_PREFIXES = ("__MACOSX/",)

RefusalOutcome = Literal["encrypted", "malformed", "limit_exceeded"]


@dataclass(frozen=True)
class ArchiveLimits:
    """What an archive may hold: entries, expanded bytes, compression ratio and nesting."""

    max_entries: int = 10_000
    max_expanded_bytes: int = 512 * 1024 * 1024
    max_ratio: int = 100
    allow_nested: bool = False


FILES_LIMITS = ArchiveLimits()


class ArchiveRefused(Exception):
    """The archive breaks a limit or can't be read. ``reason`` is its code (``archive_*``)."""

    def __init__(self, outcome: RefusalOutcome, reason: str) -> None:
        super().__init__(reason)
        self.outcome = outcome
        self.reason = reason


@dataclass(frozen=True)
class Member:
    name: str  # the normalized path inside the archive
    path: Path  # where its bytes were written
    size: int
    sha256: str


def member_name(raw: str) -> str:
    """A member's path as a label: forward slashes, no empty, ``.`` or ``..`` parts."""
    parts = [
        part
        for part in PurePosixPath(raw.replace("\\", "/")).parts
        if part not in ("", ".", "..", "/")
    ]
    return "/".join(parts)


def _has_zip64(info: zipfile.ZipInfo) -> bool:
    extra, offset = info.extra, 0
    while offset + 4 <= len(extra):
        field_id, size = struct.unpack_from("<HH", extra, offset)
        if field_id == ZIP64_EXTRA:
            return True
        offset += 4 + size
    return False


def _documents(infos: list[zipfile.ZipInfo]) -> list[zipfile.ZipInfo]:
    return [
        info
        for info in infos
        if not info.is_dir() and not info.filename.startswith(_SKIPPED_PREFIXES)
    ]


def inspect_zip(path: Path, limits: ArchiveLimits = FILES_LIMITS) -> list[zipfile.ZipInfo]:
    """The archive's document entries, after the central directory's checks. Raises
    ``ArchiveRefused``."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError, EOFError):
        raise ArchiveRefused("malformed", "archive_malformed") from None
    if len(infos) >= 0xFFFF:
        raise ArchiveRefused("limit_exceeded", "archive_zip64_unsupported")
    if len(infos) > limits.max_entries:
        raise ArchiveRefused("limit_exceeded", "archive_entry_limit")
    compressed = expanded = 0
    for info in infos:
        if (
            _has_zip64(info)
            or max(info.file_size, info.compress_size, info.header_offset) >= 0xFFFFFFFF
        ):
            raise ArchiveRefused("limit_exceeded", "archive_zip64_unsupported")
        if info.flag_bits & 0x1:
            raise ArchiveRefused("encrypted", "archive_encrypted")
        if not limits.allow_nested and NESTED_ARCHIVE_NAME.search(info.filename):
            raise ArchiveRefused("limit_exceeded", "archive_nested")
        compressed += info.compress_size
        expanded += info.file_size
        if expanded > limits.max_expanded_bytes or (
            expanded > 0 and expanded / max(1, compressed) > limits.max_ratio
        ):
            raise ArchiveRefused("limit_exceeded", "archive_expansion_limit")
        if info.compress_type not in (STORED, DEFLATED):
            raise ArchiveRefused("malformed", "archive_method_unsupported")
        if info.compress_type == STORED and info.compress_size != info.file_size:
            raise ArchiveRefused("malformed", "archive_size_mismatch")
    return _documents(infos)


def _inflate(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    limits: ArchiveLimits,
    total: int,
    sink: Callable[[bytes], object] | None,
) -> tuple[int, int, str]:
    """Inflate one entry, counting what comes out. Returns its size, the running total and its
    SHA-256."""
    digest = hashlib.sha256()
    size = 0
    try:
        with archive.open(info) as source:
            while chunk := source.read(CHUNK_BYTES):
                size += len(chunk)
                total += len(chunk)
                if size > info.file_size:
                    raise ArchiveRefused("malformed", "archive_size_mismatch")
                if total > limits.max_expanded_bytes:
                    raise ArchiveRefused("limit_exceeded", "archive_expansion_limit")
                digest.update(chunk)
                if sink is not None:
                    sink(chunk)
    except ArchiveRefused:
        raise
    except (zipfile.BadZipFile, OSError, EOFError, ValueError):
        # zipfile checks each entry's CRC, and stops at its declared size.
        raise ArchiveRefused("malformed", "archive_size_mismatch") from None
    if size != info.file_size:
        raise ArchiveRefused("malformed", "archive_size_mismatch")
    return size, total, digest.hexdigest()


def _open(path: Path) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise ArchiveRefused("malformed", "archive_malformed") from None


def measure_zip(
    path: Path,
    limits: ArchiveLimits = FILES_LIMITS,
    *,
    checkpoint: Callable[[], None] = lambda: None,
) -> None:
    """Check an archive's limits, then inflate every entry without keeping it. Raises
    ``ArchiveRefused``."""
    inspect_zip(path, limits)
    total = 0
    with _open(path) as archive:
        for info in archive.infolist():
            if not info.is_dir():
                checkpoint()
                _, total, _ = _inflate(archive, info, limits, total, None)


def extract_members(
    path: Path, destination: Path, limits: ArchiveLimits = FILES_LIMITS
) -> Iterator[Member]:
    """Inflate each document entry into ``destination``, checking real sizes as they come out.
    Raises ``ArchiveRefused``; the caller discards what was written."""
    infos = inspect_zip(path, limits)
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    with _open(path) as archive:
        for index, info in enumerate(infos):
            target = destination / f"member-{index:05d}"
            with target.open("wb") as handle:
                size, total, sha256 = _inflate(archive, info, limits, total, handle.write)
            yield Member(member_name(info.filename) or f"member-{index}", target, size, sha256)
