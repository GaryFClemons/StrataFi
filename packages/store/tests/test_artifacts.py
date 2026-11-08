"""The artifact store's contract, run against both drivers.

The local driver runs in the unit tier; the S3 driver runs against SeaweedFS in the integration
tier.
"""

import io
import secrets
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from stratafi_core.settings import StorageSettings
from stratafi_store.artifacts import (
    ArtifactNotFound,
    ArtifactStore,
    ArtifactStoreError,
    LocalArtifactStore,
    S3ArtifactStore,
    artifact_key,
    artifact_store_from_settings,
    check_key,
    quarantine_key,
    sha256_hex,
    workspace_prefix,
)


@pytest.fixture(params=["local", pytest.param("s3", marks=pytest.mark.integration)])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[ArtifactStore]:
    if request.param == "local":
        yield LocalArtifactStore(tmp_path / "artifacts")
        return
    services = request.getfixturevalue("compose_services")("seaweedfs")
    settings = StorageSettings(
        driver="s3",
        endpoint=services.s3_endpoint(),
        access_key=services.s3_access_key(),
        secret_key=services.s3_secret_key(),
        bucket=services.s3_bucket(),
    )
    yield artifact_store_from_settings(settings)


@pytest.fixture
def workspace() -> uuid.UUID:
    return uuid.uuid4()


def a_key(workspace: uuid.UUID) -> str:
    return artifact_key(workspace, secrets.token_hex(32))


def test_put_get_exists_delete(store: ArtifactStore, workspace: uuid.UUID) -> None:
    key = a_key(workspace)
    data = secrets.token_bytes(3000)
    assert not store.exists(key)
    info = store.put(key, data, media_type="application/json")
    assert (info.key, info.size, info.sha256) == (key, 3000, sha256_hex(data))
    assert info.media_type == "application/json"
    assert store.exists(key)
    assert store.get(key) == data
    with store.open(key) as handle:
        assert handle.read() == data
    store.delete(key)
    assert not store.exists(key)
    store.delete(key)  # deleting again is fine
    with pytest.raises(ArtifactNotFound):
        store.get(key)


def test_put_streams_large_objects(store: ArtifactStore, workspace: uuid.UUID) -> None:
    key = a_key(workspace)
    data = secrets.token_bytes(9 * 1024 * 1024)  # several chunks, and a multipart upload on S3
    info = store.put(key, io.BytesIO(data))
    assert (info.size, info.sha256) == (len(data), sha256_hex(data))
    assert store.get(key) == data
    store.delete(key)


def test_writing_a_key_again_is_idempotent(store: ArtifactStore, workspace: uuid.UUID) -> None:
    key = a_key(workspace)
    store.put(key, b"same bytes")
    store.put(key, b"same bytes")
    assert store.get(key) == b"same bytes"
    store.delete(key)


def test_raw_uploads_wait_in_quarantine_under_their_hash(
    store: ArtifactStore, workspace: uuid.UUID
) -> None:
    data = b"a synthetic upload " + secrets.token_bytes(16)
    info = store.put_quarantined(workspace, data, media_type="text/plain")
    assert info.key == quarantine_key(workspace, sha256_hex(data))
    assert info.key.startswith(f"ws/{workspace}/quarantine/")
    assert store.get(info.key) == data
    store.delete(info.key)


def test_listing_stays_inside_one_workspace(store: ArtifactStore) -> None:
    first, second = uuid.uuid4(), uuid.uuid4()
    keys_first = sorted(a_key(first) for _ in range(3))
    key_second = a_key(second)
    for key in [*keys_first, key_second]:
        store.put(key, b"x")
    assert sorted(store.keys(workspace_prefix(first))) == keys_first
    assert list(store.keys(f"ws/{second}/artifacts/")) == [key_second]
    assert list(store.keys(f"ws/{second}/quarantine/")) == []
    for key in [*keys_first, key_second]:
        store.delete(key)


def test_presigned_urls_read_the_object(store: ArtifactStore, workspace: uuid.UUID) -> None:
    key = a_key(workspace)
    store.put(key, b"presigned body")
    url = store.presign(key, expires_seconds=60)
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - our own URL
        assert response.read() == b"presigned body"
    store.delete(key)


def test_uris_name_the_driver(store: ArtifactStore, workspace: uuid.UUID) -> None:
    key = a_key(workspace)
    uri = store.uri(key)
    assert uri.endswith(key)
    assert uri.startswith(("local:", "s3://"))


def test_the_store_is_ready(store: ArtifactStore) -> None:
    store.check()


@pytest.mark.parametrize(
    "key",
    [
        "ws/not-a-uuid/artifacts/" + "0" * 64,
        "ws/00000000-0000-0000-0000-000000000000/artifacts/../../etc/passwd",
        "ws/00000000-0000-0000-0000-000000000000/other/" + "0" * 64,
        "ws/00000000-0000-0000-0000-000000000000/artifacts/" + "A" * 64,
        "/ws/00000000-0000-0000-0000-000000000000/artifacts/" + "0" * 64,
        "",
    ],
)
def test_keys_cannot_escape_their_prefix(key: str) -> None:
    with pytest.raises(ArtifactStoreError):
        check_key(key)


def test_key_builders_refuse_bad_parts() -> None:
    with pytest.raises(ArtifactStoreError):
        artifact_key("not-a-uuid", "0" * 64)
    with pytest.raises(ArtifactStoreError):
        artifact_key(uuid.uuid4(), "short")
    with pytest.raises(ArtifactStoreError):
        quarantine_key(uuid.uuid4(), "g" * 64)


def test_listing_needs_a_workspace_prefix(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ArtifactStoreError):
        list(store.keys("ws/"))


def test_local_writes_leave_no_temporary_files(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    workspace = uuid.uuid4()
    store.put(a_key(workspace), b"x" * 10)
    leftovers = [path for path in tmp_path.rglob(".tmp-*")]
    assert leftovers == []


def test_a_failed_local_write_leaves_nothing(tmp_path: Path) -> None:
    class Broken(io.RawIOBase):
        def read(self, size: int = -1) -> bytes:
            raise OSError("disk went away")

    store = LocalArtifactStore(tmp_path)
    key = a_key(uuid.uuid4())
    with pytest.raises(OSError, match="disk went away"):
        store.put(key, Broken())
    assert not store.exists(key)
    assert list(tmp_path.rglob(".tmp-*")) == []


def test_settings_pick_the_driver(tmp_path: Path) -> None:
    local = artifact_store_from_settings(StorageSettings(driver="local", local_root=tmp_path))
    assert isinstance(local, LocalArtifactStore)
    s3 = artifact_store_from_settings(
        StorageSettings(driver="s3", endpoint="http://127.0.0.1:1", access_key="k", secret_key="s")
    )
    assert isinstance(s3, S3ArtifactStore)
    assert s3.uri(artifact_key(uuid.uuid4(), "0" * 64)).startswith("s3://stratafi/ws/")


def test_an_unreachable_bucket_fails_the_check() -> None:
    store = artifact_store_from_settings(
        StorageSettings(driver="s3", endpoint="http://127.0.0.1:9", access_key="k", secret_key="s")
    )
    with pytest.raises(ArtifactStoreError, match="bucket"):
        store.check()
