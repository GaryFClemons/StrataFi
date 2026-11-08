"""The artifact store: immutable payloads under content-addressed keys.

Keys:

- ``ws/<workspace>/artifacts/<cache-key>``: stage outputs (parsed documents, chunks, vectors),
  addressed by their cache key (``stratafi_core.hashing.cache_key``), and
  ``ws/<workspace>/artifacts/<cache-key>.<part>`` for a named part beside one, such as the JSON
  sidecar of a vectors array
- ``ws/<workspace>/quarantine/<sha256>``: raw uploads, addressed by their content hash, until
  Scan passes

Both hashes are 64 lowercase hex characters, and the workspace is its UUID, so a key can never
escape its prefix. Writes are idempotent: the same key always holds the same bytes, so a retried
or duplicated job just writes them again.

Two drivers share ``ArtifactStore``: ``S3ArtifactStore`` (SeaweedFS in development, any S3 API in
production) and ``LocalArtifactStore`` (a directory, for tests and single-machine labs).
"""

import hashlib
import os
import re
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any

from stratafi_core.settings import StorageSettings

_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_KEY_RE = re.compile(
    r"ws/[0-9a-f-]{36}/(artifacts/[0-9a-f]{64}(\.[a-z0-9_-]{1,32})?|quarantine/[0-9a-f]{64})"
)
_PART_RE = re.compile(r"[a-z0-9_-]{1,32}")
CHUNK_BYTES = 1024 * 1024
DEFAULT_MEDIA_TYPE = "application/octet-stream"


class ArtifactStoreError(Exception):
    """The store failed or refused. Messages name keys, never content."""


class ArtifactNotFound(ArtifactStoreError):
    pass


def _workspace(workspace_id: uuid.UUID | str) -> str:
    try:
        return str(uuid.UUID(str(workspace_id)))
    except ValueError:
        raise ArtifactStoreError("invalid workspace ID") from None


def _hex64(value: str, what: str) -> str:
    if not isinstance(value, str) or not _HEX64_RE.fullmatch(value):
        raise ArtifactStoreError(f"invalid {what}")
    return value


def artifact_key(workspace_id: uuid.UUID | str, cache_key: str) -> str:
    return f"ws/{_workspace(workspace_id)}/artifacts/{_hex64(cache_key, 'cache key')}"


def artifact_part_key(workspace_id: uuid.UUID | str, cache_key: str, part: str) -> str:
    """A named part stored beside an artifact: ``<cache-key>.<part>``."""
    if not isinstance(part, str) or not _PART_RE.fullmatch(part):
        raise ArtifactStoreError("invalid artifact part")
    return f"{artifact_key(workspace_id, cache_key)}.{part}"


def quarantine_key(workspace_id: uuid.UUID | str, sha256: str) -> str:
    return f"ws/{_workspace(workspace_id)}/quarantine/{_hex64(sha256, 'content hash')}"


def workspace_prefix(workspace_id: uuid.UUID | str) -> str:
    return f"ws/{_workspace(workspace_id)}/"


def check_key(key: str) -> str:
    if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
        raise ArtifactStoreError("invalid artifact store key")
    return key


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    sha256: str
    media_type: str


class _HashingReader:
    """Wraps a binary stream; counts and hashes what passes through."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        self._hash = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        data = self._stream.read(size)
        self._hash.update(data)
        self.size += len(data)
        return data

    @property
    def sha256(self) -> str:
        return self._hash.hexdigest()


class ArtifactStore(ABC):
    """put, get, exists, delete and presign over content-addressed keys."""

    @abstractmethod
    def put(
        self, key: str, data: bytes | IO[bytes], *, media_type: str = DEFAULT_MEDIA_TYPE
    ) -> ObjectInfo:
        """Write ``data`` (bytes or a binary stream) at ``key``. Returns its size and SHA-256."""

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def open(self, key: str) -> IO[bytes]:
        """A stream of the object's bytes; the caller closes it."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete ``key``. Deleting a missing key is not an error."""

    @abstractmethod
    def presign(self, key: str, *, expires_seconds: int | None = None) -> str:
        """A time-limited URL that reads the object without other credentials."""

    @abstractmethod
    def keys(self, prefix: str) -> Iterator[str]:
        """Every key under ``prefix``, which must be a workspace prefix or below it."""

    @abstractmethod
    def uri(self, key: str) -> str:
        """Where ``key`` lives, for the ``artifacts.storage_uri`` column."""

    @abstractmethod
    def check(self) -> None:
        """Raise ``ArtifactStoreError`` unless the store is reachable and writable in principle."""

    def put_quarantined(
        self, workspace_id: uuid.UUID | str, data: bytes, *, media_type: str = DEFAULT_MEDIA_TYPE
    ) -> ObjectInfo:
        """Store a raw upload in quarantine under its own SHA-256."""
        return self.put(quarantine_key(workspace_id, sha256_hex(data)), data, media_type=media_type)


def _check_prefix(prefix: str) -> str:
    if not re.fullmatch(r"ws/[0-9a-f-]{36}/((artifacts|quarantine)/[0-9a-f]{0,64})?", prefix):
        raise ArtifactStoreError("listing needs a workspace prefix")
    return prefix


class LocalArtifactStore(ArtifactStore):
    """Objects as files under ``root``. Writes are atomic: a temporary file, then a rename."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        return self.root.joinpath(*PurePosixPath(check_key(key)).parts)

    def put(
        self, key: str, data: bytes | IO[bytes], *, media_type: str = DEFAULT_MEDIA_TYPE
    ) -> ObjectInfo:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as handle:
                chunks = (
                    [data] if isinstance(data, bytes) else iter(lambda: data.read(CHUNK_BYTES), b"")
                )
                for chunk in chunks:
                    digest.update(chunk)
                    size += len(chunk)
                    handle.write(chunk)
            os.replace(temp_name, path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
        return ObjectInfo(key, size, digest.hexdigest(), media_type)

    def get(self, key: str) -> bytes:
        with self.open(key) as handle:
            return handle.read()

    def open(self, key: str) -> IO[bytes]:
        try:
            return self._path(key).open("rb")
        except FileNotFoundError:
            raise ArtifactNotFound(key) from None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def presign(self, key: str, *, expires_seconds: int | None = None) -> str:
        # A local directory has no signer: the URL is the file's, readable on this machine only.
        return self._path(key).as_uri()

    def keys(self, prefix: str) -> Iterator[str]:
        _check_prefix(prefix)
        base = self.root.joinpath(*PurePosixPath(prefix.rstrip("/")).parts[:3])
        if not base.exists():
            return
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.startswith(".tmp-"):
                key = path.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    yield key

    def uri(self, key: str) -> str:
        return f"local:{check_key(key)}"

    def check(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise ArtifactStoreError("the local artifact directory can't be created") from None
        if not os.access(self.root, os.W_OK):
            raise ArtifactStoreError("the local artifact directory isn't writable")


class S3ArtifactStore(ArtifactStore):
    """Objects in one S3 bucket: SeaweedFS in development. Addressing is path-style: SeaweedFS
    serves virtual-host style only under a domain name it's started with (-s3.domainName)."""

    def __init__(self, client: Any, bucket: str, *, presign_seconds: int = 900) -> None:
        self.client = client
        self.bucket = bucket
        self.presign_seconds = presign_seconds

    @classmethod
    def from_settings(cls, settings: StorageSettings) -> "S3ArtifactStore":
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint,
            region_name=settings.region,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=(
                settings.secret_key.get_secret_value() if settings.secret_key else None
            ),
            config=Config(
                s3={"addressing_style": "path"},
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=5,
                read_timeout=60,
            ),
        )
        return cls(client, settings.bucket, presign_seconds=settings.presign_seconds)

    def _error(self, exc: Exception) -> str:
        response = getattr(exc, "response", None) or {}
        return str(response.get("Error", {}).get("Code") or type(exc).__name__)

    def put(
        self, key: str, data: bytes | IO[bytes], *, media_type: str = DEFAULT_MEDIA_TYPE
    ) -> ObjectInfo:
        check_key(key)
        try:
            if isinstance(data, bytes):
                self.client.put_object(
                    Bucket=self.bucket, Key=key, Body=data, ContentType=media_type
                )
                return ObjectInfo(key, len(data), sha256_hex(data), media_type)
            reader = _HashingReader(data)
            self.client.upload_fileobj(
                reader, self.bucket, key, ExtraArgs={"ContentType": media_type}
            )
            return ObjectInfo(key, reader.size, reader.sha256, media_type)
        except Exception as exc:
            if isinstance(exc, ArtifactStoreError):
                raise
            raise ArtifactStoreError(f"put failed ({self._error(exc)})") from None

    def get(self, key: str) -> bytes:
        with self.open(key) as handle:
            return handle.read()

    def open(self, key: str) -> IO[bytes]:
        check_key(key)
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        except Exception as exc:
            if self._error(exc) in ("NoSuchKey", "404", "NotFound"):
                raise ArtifactNotFound(key) from None
            raise ArtifactStoreError(f"get failed ({self._error(exc)})") from None

    def exists(self, key: str) -> bool:
        check_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._error(exc) in ("NoSuchKey", "404", "NotFound"):
                return False
            raise ArtifactStoreError(f"head failed ({self._error(exc)})") from None
        return True

    def delete(self, key: str) -> None:
        check_key(key)
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            raise ArtifactStoreError(f"delete failed ({self._error(exc)})") from None

    def presign(self, key: str, *, expires_seconds: int | None = None) -> str:
        check_key(key)
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_seconds or self.presign_seconds,
        )

    def keys(self, prefix: str) -> Iterator[str]:
        _check_prefix(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    yield item["Key"]
        except Exception as exc:
            raise ArtifactStoreError(f"list failed ({self._error(exc)})") from None

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{check_key(key)}"

    def check(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception as exc:
            raise ArtifactStoreError(f"the bucket isn't reachable ({self._error(exc)})") from None


def artifact_store_from_settings(settings: StorageSettings) -> ArtifactStore:
    if settings.driver == "local":
        return LocalArtifactStore(settings.local_root)
    return S3ArtifactStore.from_settings(settings)
