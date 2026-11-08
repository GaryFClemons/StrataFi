"""Bucket and URL intake (unit tier): ``extract/s3`` lists a configured prefix and
streams each object into the spool; ``extract/url`` goes through the SSRF guard."""

import hashlib
import io
from pathlib import Path

import httpx2
import pytest

from stratafi_api.intake import UploadError, s3_files, url_file
from stratafi_components.fetch import UrlFetcher, UrlGuard
from stratafi_core.settings import OfflineSettings, S3Source

SOURCE = S3Source(bucket="policies", prefix="corpus/v1")


class Paginator:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def paginate(self, Bucket: str, Prefix: str):
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        # Two pages, as a long listing would be.
        middle = len(keys) // 2
        for page in (keys[:middle], keys[middle:]):
            yield {"Contents": [{"Key": key, "Size": len(self.objects[key])} for key in page]}


class FakeS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.listed: list[tuple[str, str]] = []

    def get_paginator(self, name: str) -> Paginator:
        assert name == "list_objects_v2"
        return Paginator(self.objects)

    def get_object(self, Bucket: str, Key: str):
        assert Bucket == "policies"
        return {"Body": io.BytesIO(self.objects[Key])}


OBJECTS = {
    "corpus/v1/fees.md": b"# Fees\n",
    "corpus/v1/wires/callback.txt": b"Call back.\n",
    "corpus/v1/.hidden/secret.txt": b"no",
    "corpus/v1/folder/": b"",
    "corpus/v2/other.md": b"# Other\n",
}


def test_every_object_under_the_prefix_is_spooled_and_named_below_it(tmp_path: Path) -> None:
    found = s3_files(SOURCE, "", tmp_path, max_files=10, max_bytes=1000, client=FakeS3(OBJECTS))
    assert [(item.name, item.size) for item in found] == [
        ("fees.md", 7),
        ("wires/callback.txt", 11),
    ]
    assert found[0].sha256 == hashlib.sha256(b"# Fees\n").hexdigest()
    assert found[1].path.read_bytes() == b"Call back.\n"


def test_a_caller_prefix_narrows_but_never_climbs(tmp_path: Path) -> None:
    client = FakeS3(OBJECTS)
    found = s3_files(SOURCE, "wires", tmp_path, max_files=10, max_bytes=1000, client=client)
    assert [item.name for item in found] == ["callback.txt"]
    with pytest.raises(UploadError) as refused:
        s3_files(SOURCE, "../v2", tmp_path, max_files=10, max_bytes=1000, client=client)
    assert refused.value.code == "path_outside_root"


def test_bucket_limits_hold(tmp_path: Path) -> None:
    client = FakeS3(OBJECTS)
    with pytest.raises(UploadError) as refused:
        s3_files(SOURCE, "", tmp_path, max_files=1, max_bytes=1000, client=client)
    assert refused.value.code == "too_many_files"
    with pytest.raises(UploadError) as refused:
        s3_files(SOURCE, "", tmp_path, max_files=10, max_bytes=8, client=client)
    assert refused.value.code == "file_too_large"
    with pytest.raises(UploadError) as refused:
        s3_files(SOURCE, "empty", tmp_path, max_files=10, max_bytes=1000, client=client)
    assert refused.value.code == "no_files"


def test_a_url_lands_in_the_spool_through_the_guard(tmp_path: Path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "text/markdown"}, content=b"# Hi\n")

    fetcher = UrlFetcher(
        guard=UrlGuard(resolver=lambda host, port: ["93.184.215.14"]),
        transport=httpx2.MockTransport(handler),
    )
    settings = OfflineSettings()
    received = url_file("https://docs.example/p/hours", tmp_path, settings, fetcher=fetcher)
    assert (received.name, received.size) == ("hours.md", 5)

    with pytest.raises(UploadError) as refused:
        url_file("http://169.254.169.254/latest/", tmp_path, settings)
    assert (refused.value.kind, refused.value.code) == ("invalid_request", "metadata_address")
    with pytest.raises(UploadError) as refused:
        url_file("http://10.0.0.1/a.pdf", tmp_path, settings)
    assert refused.value.code == "private_address"
