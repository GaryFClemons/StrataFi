"""Staging arrivals: hashing with a size limit, archives unpacked into members
with provenance, and refused archives kept with their reason."""

import hashlib
from pathlib import Path

import pytest

from stratafi_components.intake import (
    FileTooLarge,
    HashingFile,
    InvalidName,
    entry_name,
    hash_file,
    stage,
)
from stratafi_testing.documents import Block, docx_bytes, zip_bytes


def put(tmp_path: Path, name: str, data: bytes) -> tuple[Path, str, int]:
    path = tmp_path / name
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest(), len(data)


def test_a_plain_file_stages_as_itself(tmp_path: Path) -> None:
    path, sha256, size = put(tmp_path, "a.md", b"# A\n")
    staged = stage(path, "notes/a.md", sha256=sha256, size=size, scratch=tmp_path)
    assert (staged.name, staged.detection.media_type, staged.members) == (
        "notes/a.md",
        "text/markdown",
        [],
    )


def test_an_archive_stages_its_members_with_provenance(tmp_path: Path) -> None:
    docx = docx_bytes("Fees", [Block("paragraph", "Fees apply.")])
    path, sha256, size = put(
        tmp_path, "bundle.zip", zip_bytes({"policies/fees.docx": docx, "notes.txt": b"Notes"})
    )
    staged = stage(path, "bundle.zip", sha256=sha256, size=size, scratch=tmp_path)
    assert staged.refusal is None
    names = [(m.name, m.member_path, m.detection.kind) for m in staged.members]
    assert names == [
        ("bundle.zip/policies/fees.docx", "policies/fees.docx", "docx"),
        ("bundle.zip/notes.txt", "notes.txt", "text"),
    ]
    assert staged.members[0].sha256 == hashlib.sha256(docx).hexdigest()
    assert next(node.name for node in staged.walk()) == "bundle.zip"


def test_a_docx_is_never_unpacked(tmp_path: Path) -> None:
    path, sha256, size = put(tmp_path, "fees.docx", docx_bytes("Fees", []))
    assert stage(path, "fees.docx", sha256=sha256, size=size, scratch=tmp_path).members == []


def test_a_zip_bomb_is_staged_as_refused(tmp_path: Path) -> None:
    path, sha256, size = put(tmp_path, "bomb.zip", zip_bytes({"z": b"\0" * (8 * 1024 * 1024)}))
    staged = stage(path, "bomb.zip", sha256=sha256, size=size, scratch=tmp_path)
    assert staged.refusal == ("limit_exceeded", "archive_expansion_limit")
    assert staged.members == []


def test_an_archive_inside_an_archive_is_refused_even_when_renamed(tmp_path: Path) -> None:
    inner = zip_bytes({"a.md": b"# a"})
    path, sha256, size = put(tmp_path, "outer.zip", zip_bytes({"innocent.txt": inner}))
    staged = stage(path, "outer.zip", sha256=sha256, size=size, scratch=tmp_path)
    assert staged.refusal == ("limit_exceeded", "archive_nested")


def test_unpacking_can_be_off(tmp_path: Path) -> None:
    path, sha256, size = put(tmp_path, "bundle.zip", zip_bytes({"a.md": b"# a"}))
    staged = stage(path, "bundle.zip", sha256=sha256, size=size, scratch=tmp_path, unpack=False)
    assert staged.members == []
    assert staged.detection.is_archive


def test_hashing_files_refuse_more_than_their_limit(tmp_path: Path) -> None:
    writer = HashingFile(tmp_path, max_bytes=10)
    writer.write(b"12345")
    with pytest.raises(FileTooLarge):
        writer.write(b"678901")
    writer.discard()
    assert not writer.path.exists()
    path, sha256, size = put(tmp_path, "x.bin", b"0123456789")
    assert hash_file(path) == (sha256, size)
    with pytest.raises(FileTooLarge):
        hash_file(path, max_bytes=5)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a.md", "a.md"),
        ("folder/sub/b.pdf", "folder/sub/b.pdf"),
        ("C:\\Users\\x\\c.txt", "C:/Users/x/c.txt"),
        ("../../d.txt", "d.txt"),
        ("/e.txt", "e.txt"),
        (" spaced / name .md ", "spaced/name .md"),
    ],
)
def test_entry_names_are_relative_paths(raw: str, expected: str) -> None:
    assert entry_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "/", "..", "a\x00b", "line\nbreak.md", "x" * 600])
def test_bad_entry_names_are_refused(raw: str) -> None:
    with pytest.raises(InvalidName):
        entry_name(raw)
