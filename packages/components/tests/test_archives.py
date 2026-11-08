"""ZIP archives: the limits refuse bombs and nested archives, and members unpack with
normalized names."""

import io
import struct
import zipfile
from pathlib import Path

import pytest

from stratafi_components.archives import (
    ArchiveLimits,
    ArchiveRefused,
    extract_members,
    inspect_zip,
    member_name,
)
from stratafi_testing.documents import zip_bytes


def write(tmp_path: Path, data: bytes, name: str = "archive.zip") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


FILES = ArchiveLimits()


def refusal(path: Path, limits: ArchiveLimits = FILES) -> tuple[str, str]:
    with pytest.raises(ArchiveRefused) as raised:
        list(extract_members(path, path.parent / "out", limits))
    return raised.value.outcome, raised.value.reason


def test_members_unpack_with_their_hashes(tmp_path: Path) -> None:
    path = write(tmp_path, zip_bytes({"docs/a.md": b"# A\n", "b.txt": b"B", "docs/": b""}))
    members = list(extract_members(path, tmp_path / "out"))
    assert [(m.name, m.size, m.path.read_bytes()) for m in members] == [
        ("docs/a.md", 4, b"# A\n"),
        ("b.txt", 1, b"B"),
    ]
    import hashlib

    assert members[1].sha256 == hashlib.sha256(b"B").hexdigest()


def test_a_ratio_bomb_is_refused(tmp_path: Path) -> None:
    # 20 MiB of zeros deflates to about 20 KiB: a ratio near 1,000, over the limit of 100.
    path = write(tmp_path, zip_bytes({"zeros.txt": b"\0" * (20 * 1024 * 1024)}))
    assert refusal(path) == ("limit_exceeded", "archive_expansion_limit")


def test_too_much_expanded_data_is_refused(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        zip_bytes({"a.txt": b"a" * 3000, "b.txt": b"b" * 3000}, compression=zipfile.ZIP_STORED),
    )
    assert refusal(path, ArchiveLimits(max_expanded_bytes=5000)) == (
        "limit_exceeded",
        "archive_expansion_limit",
    )


def test_too_many_entries_are_refused(tmp_path: Path) -> None:
    path = write(tmp_path, zip_bytes({f"{n}.txt": b"x" for n in range(6)}))
    assert refusal(path, ArchiveLimits(max_entries=5)) == ("limit_exceeded", "archive_entry_limit")


@pytest.mark.parametrize("inner", ["inner.zip", "deep/inner.7z", "logs.tar", "x.GZ", "a.jar"])
def test_nested_archives_are_refused_by_name(tmp_path: Path, inner: str) -> None:
    path = write(tmp_path, zip_bytes({"ok.md": b"# ok", inner: b"anything"}))
    assert refusal(path) == ("limit_exceeded", "archive_nested")


def test_encrypted_entries_are_refused(tmp_path: Path) -> None:
    data = bytearray(zip_bytes({"secret.txt": b"hidden"}))
    # Set the encryption flag in the central directory record (offset 8 of its header).
    central = data.rfind(b"PK\x01\x02")
    flags = struct.unpack_from("<H", data, central + 8)[0]
    struct.pack_into("<H", data, central + 8, flags | 0x1)
    path = write(tmp_path, bytes(data))
    assert refusal(path) == ("encrypted", "archive_encrypted")


def test_an_entry_that_lies_about_its_size_is_refused(tmp_path: Path) -> None:
    data = bytearray(zip_bytes({"big.txt": b"x" * 5000}))
    # Claim 50 bytes in both headers; the real stream inflates to 5,000 and fails its CRC.
    local = data.find(b"PK\x03\x04")
    central = data.rfind(b"PK\x01\x02")
    struct.pack_into("<I", data, local + 22, 50)
    struct.pack_into("<I", data, central + 24, 50)
    path = write(tmp_path, bytes(data))
    outcome, reason = refusal(path, ArchiveLimits(max_ratio=1000))
    assert (outcome, reason) == ("malformed", "archive_size_mismatch")


def test_a_damaged_archive_is_malformed(tmp_path: Path) -> None:
    path = write(tmp_path, b"PK\x03\x04 then nothing useful")
    assert refusal(path) == ("malformed", "archive_malformed")


def test_unsupported_compression_methods_are_refused(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_BZIP2) as archive:
        archive.writestr("a.txt", "text " * 10)
    path = write(tmp_path, buffer.getvalue())
    assert refusal(path, ArchiveLimits(max_ratio=1000)) == (
        "malformed",
        "archive_method_unsupported",
    )


def test_inspection_skips_folders_and_macos_resource_forks(tmp_path: Path) -> None:
    path = write(tmp_path, zip_bytes({"a/": b"", "a/b.md": b"b", "__MACOSX/a/._b.md": b"fork"}))
    assert [info.filename for info in inspect_zip(path)] == ["a/b.md"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("docs/a.md", "docs/a.md"),
        ("../../etc/passwd", "etc/passwd"),
        ("/abs/path.txt", "abs/path.txt"),
        ("win\\style\\name.txt", "win/style/name.txt"),
        ("./a/./b.txt", "a/b.txt"),
    ],
)
def test_member_names_are_labels_that_cant_escape(raw: str, expected: str) -> None:
    assert member_name(raw) == expected
