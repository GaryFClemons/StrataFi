"""The streaming multipart reader (unit tier): each file part goes to its own
spooled file, hashed, within the limits."""

import hashlib
from pathlib import Path

import pytest

from stratafi_api.intake import MultipartFiles, UploadError, boundary_of, folder_files

BOUNDARY = "stratafi-boundary"


def part(name: str, filename: str | None, data: bytes) -> bytes:
    disposition = f'form-data; name="{name}"'
    if filename is not None:
        disposition += f'; filename="{filename}"'
    return (
        (
            f"--{BOUNDARY}\r\nContent-Disposition: {disposition}\r\n"
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + data
        + b"\r\n"
    )


def body(*parts: bytes) -> bytes:
    return b"".join(parts) + f"--{BOUNDARY}--\r\n".encode()


def read(data: bytes, tmp_path: Path, *, max_bytes: int = 1000, max_files: int = 10, step: int = 7):
    reader = MultipartFiles(BOUNDARY.encode(), tmp_path, max_bytes=max_bytes, max_files=max_files)
    try:
        # Small, odd-sized pieces: parts and headers straddle chunk boundaries.
        for start in range(0, len(data), step):
            reader.feed(data[start : start + step])
        return reader.finish()
    except UploadError:
        reader.discard()
        raise


def test_each_file_part_is_spooled_and_hashed(tmp_path: Path) -> None:
    files = read(
        body(part("file", "a.md", b"# A\n"), part("file", "dir/b.pdf", b"%PDF-1.4 b")), tmp_path
    )
    assert [(f.name, f.size) for f in files] == [("a.md", 4), ("dir/b.pdf", 10)]
    assert files[0].sha256 == hashlib.sha256(b"# A\n").hexdigest()
    assert files[1].path.read_bytes() == b"%PDF-1.4 b"


def test_a_file_over_the_limit_is_refused_as_it_streams(tmp_path: Path) -> None:
    with pytest.raises(UploadError) as raised:
        read(body(part("file", "big.txt", b"x" * 2000)), tmp_path, max_bytes=1000)
    assert (raised.value.kind, raised.value.code) == ("payload_too_large", "file_too_large")
    assert list(tmp_path.iterdir()) == []


def test_too_many_files_are_refused(tmp_path: Path) -> None:
    parts = [part("file", f"{n}.txt", b"x") for n in range(3)]
    with pytest.raises(UploadError) as raised:
        read(body(*parts), tmp_path, max_files=2)
    assert raised.value.code == "too_many_files"


@pytest.mark.parametrize(
    ("parts", "code"),
    [
        ([part("comment", None, b"hello")], "unexpected_part"),
        ([part("file", None, b"no name")], "unexpected_part"),
        ([part("file", "../../", b"x")], "invalid_file_name"),
        ([], "no_files"),
    ],
)
def test_bad_parts_are_refused(tmp_path: Path, parts: list[bytes], code: str) -> None:
    with pytest.raises(UploadError) as raised:
        read(body(*parts), tmp_path)
    assert raised.value.code == code


def test_a_truncated_body_is_refused(tmp_path: Path) -> None:
    data = part("file", "a.txt", b"x" * 50)[:-10]
    with pytest.raises(UploadError) as raised:
        read(data, tmp_path)
    assert raised.value.code in ("multipart_incomplete", "multipart_invalid")


def test_only_multipart_form_data_is_accepted() -> None:
    assert boundary_of(f"multipart/form-data; boundary={BOUNDARY}") == BOUNDARY.encode()
    for content_type in ("application/json", "multipart/form-data", "multipart/mixed; boundary=x"):
        with pytest.raises(UploadError) as raised:
            boundary_of(content_type)
        assert raised.value.kind == "unsupported_media_type"


def test_folder_imports_stay_inside_their_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "docs" / ".hidden").mkdir(parents=True)
    (root / "docs" / "a.md").write_bytes(b"# a")
    (root / "docs" / "sub").mkdir()
    (root / "docs" / "sub" / "b.txt").write_bytes(b"b")
    (root / "docs" / ".hidden" / "c.txt").write_bytes(b"c")
    (root / "docs" / ".dotfile").write_bytes(b"d")
    found = folder_files(root, "docs", max_files=10, max_bytes=100)
    assert [item.name for item in found] == ["a.md", "sub/b.txt"]
    assert found[0].path == root / "docs" / "a.md"
    for escape in ("..", "../root/../..", "/etc"):
        with pytest.raises(UploadError) as raised:
            folder_files(root, escape, max_files=10, max_bytes=100)
        assert raised.value.code in ("path_outside_root", "folder_not_found")
    with pytest.raises(UploadError) as raised:
        folder_files(root, "docs", max_files=1, max_bytes=100)
    assert raised.value.code == "too_many_files"
    with pytest.raises(UploadError) as raised:
        folder_files(root, "docs", max_files=10, max_bytes=1)
    assert raised.value.code == "file_too_large"
