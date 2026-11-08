"""Intake: uploads and folder imports into the quarantine prefix.

An upload is a ``multipart/form-data`` body with one or more ``file`` parts. ``MultipartFiles``
parses it as it streams in: each part goes straight to a temporary file in the spool directory,
hashed and counted on the way, so a 200 MB upload never sits in memory, and a part over
``limits.upload_bytes`` is refused as soon as it passes the limit.

``ingest`` then stages each file (``stratafi_components.intake``: detect, unpack archives),
stores its bytes under ``ws/<workspace>/quarantine/<sha256>``, and records the documents, their
origins and their artifacts in one transaction. Bytes already in the workspace are the same
document: the result says it was deduplicated. With a corpus, the new entries join its working
set: an archive's members, never the archive, unless the archive was refused.

A manifest (``extract/manifest``, ``stratafi_components.manifest``) describes the files of an
upload or a server folder; each file it lists joins the corpus with that metadata: key,
version, collection, ACL, effective dates, title and status.

A bucket the configuration names (``extract/s3``) imports every object under a prefix; a URL
(``extract/url``) is fetched through the SSRF guard (``stratafi_components.fetch``). Both land in
the spool like uploads, with the same limits, and go through ``ingest``.

Logs and audit events carry counts and IDs, never file names.
"""

import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from python_multipart.multipart import MultipartParser, parse_options_header
from sqlalchemy.engine import Engine

from stratafi_components.archives import ArchiveLimits
from stratafi_components.fetch import FetchRefused, UrlFetcher, UrlGuard
from stratafi_components.intake import (
    FileTooLarge,
    HashingFile,
    InvalidName,
    StagedFile,
    entry_name,
    hash_file,
    scratch_directory,
    stage,
)
from stratafi_components.manifest import MANIFEST_NAME, CorpusManifest
from stratafi_core.settings import S3Source, Settings
from stratafi_store.artifacts import ArtifactStore, quarantine_key
from stratafi_store.audit import record_audit_event
from stratafi_store.corpora import NewEntry, add_entries
from stratafi_store.documents import DocumentRow, create_source, record_document, record_origin
from stratafi_store.lineage import ArtifactRecord, record_artifact

logger = logging.getLogger("stratafi.api.intake")

FILE_FIELD = b"file"
MANIFEST_FIELD = b"manifest"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class UploadError(Exception):
    """An upload can't be accepted. ``kind`` is a problem kind, ``code`` a short reason."""

    def __init__(self, kind: str, code: str) -> None:
        super().__init__(code)
        self.kind = kind
        self.code = code


@dataclass
class Received:
    name: str
    path: Path
    sha256: str
    size: int


@dataclass
class MultipartFiles:
    """Writes each ``file`` part of a streaming multipart body to its own spooled file."""

    boundary: bytes
    spool: Path
    max_bytes: int
    max_files: int
    files: list[Received] = field(default_factory=list)
    manifest: bytes | None = None
    _manifest: bytearray | None = None
    _current: HashingFile | None = None
    _current_name: str = ""
    _header_field: bytearray = field(default_factory=bytearray)
    _header_value: bytearray = field(default_factory=bytearray)
    _headers: dict[bytes, bytes] = field(default_factory=dict)
    _parser: MultipartParser = field(init=False)

    def __post_init__(self) -> None:
        self._parser = MultipartParser(
            self.boundary,
            {
                "on_part_begin": self._part_begin,
                "on_header_field": self._header_field_data,
                "on_header_value": self._header_value_data,
                "on_header_end": self._header_end,
                "on_headers_finished": self._headers_finished,
                "on_part_data": self._part_data,
                "on_part_end": self._part_end,
            },
        )

    def feed(self, chunk: bytes) -> None:
        try:
            self._parser.write(chunk)
        except FileTooLarge:
            raise UploadError("payload_too_large", "file_too_large") from None
        except UploadError:
            raise
        except Exception:
            raise UploadError("invalid_request", "multipart_invalid") from None

    def finish(self) -> list[Received]:
        try:
            self._parser.finalize()
        except Exception:
            raise UploadError("invalid_request", "multipart_invalid") from None
        if self._current is not None:
            raise UploadError("invalid_request", "multipart_incomplete")
        if not self.files:
            raise UploadError("invalid_request", "no_files")
        return self.files

    def discard(self) -> None:
        if self._current is not None:
            self._current.discard()
            self._current = None
        for received in self.files:
            received.path.unlink(missing_ok=True)

    # --- parser callbacks ---------------------------------------------------------------------

    def _part_begin(self) -> None:
        self._headers = {}

    def _header_field_data(self, data: bytes, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def _header_value_data(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _header_end(self) -> None:
        self._headers[bytes(self._header_field).lower()] = bytes(self._header_value)
        self._header_field.clear()
        self._header_value.clear()

    def _headers_finished(self) -> None:
        _, options = parse_options_header(self._headers.get(b"content-disposition", b""))
        filename = options.get(b"filename")
        if options.get(b"name") == MANIFEST_FIELD:
            if self.manifest is not None or self._manifest is not None:
                raise UploadError("invalid_request", "manifest_repeated")
            self._manifest = bytearray()
            return
        if options.get(b"name") != FILE_FIELD or filename is None:
            raise UploadError("invalid_request", "unexpected_part")
        if len(self.files) >= self.max_files:
            raise UploadError("payload_too_large", "too_many_files")
        try:
            self._current_name = entry_name(filename.decode("utf-8", errors="replace"))
        except InvalidName:
            raise UploadError("invalid_request", "invalid_file_name") from None
        self._current = HashingFile(self.spool, self.max_bytes)

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        if self._manifest is not None:
            self._manifest += data[start:end]
            if len(self._manifest) > MAX_MANIFEST_BYTES:
                raise UploadError("payload_too_large", "manifest_too_large")
        elif self._current is not None:
            self._current.write(data[start:end])

    def _part_end(self) -> None:
        if self._manifest is not None:
            self.manifest, self._manifest = bytes(self._manifest), None
            return
        if self._current is None:
            return
        current, self._current = self._current, None
        current.close()
        self.files.append(Received(self._current_name, current.path, current.sha256, current.size))


def boundary_of(content_type: str) -> bytes:
    kind, options = parse_options_header(content_type.encode("latin-1", errors="replace"))
    boundary = options.get(b"boundary")
    if kind != b"multipart/form-data" or not boundary:
        raise UploadError("unsupported_media_type", "multipart_required")
    return boundary


# --- recording ---------------------------------------------------------------------------------


@dataclass
class IngestedFile:
    name: str
    document: DocumentRow
    deduplicated: bool
    refusal: tuple[str, str] | None = None
    members: list["IngestedFile"] = field(default_factory=list)


@dataclass
class IngestReport:
    files: list[IngestedFile]
    source_id: uuid.UUID
    added_to_corpus: int = 0

    def counts(self) -> dict[str, int]:
        nodes = [node for item in self.files for node in _walk(item)]
        return {
            "files": len(self.files),
            "documents": len(nodes),
            "new": sum(1 for node in nodes if not node.deduplicated),
            "deduplicated": sum(1 for node in nodes if node.deduplicated),
            "refused": sum(1 for node in nodes if node.refusal is not None),
        }


def _walk(item: IngestedFile) -> list[IngestedFile]:
    return [item, *(node for member in item.members for node in _walk(member))]


def archive_limits(settings: Settings) -> ArchiveLimits:
    limits = settings.limits
    return ArchiveLimits(
        max_entries=limits.archive_entries,
        max_expanded_bytes=limits.archive_expanded_bytes,
        max_ratio=limits.archive_ratio,
    )


def ingest(
    *,
    engine: Engine,
    store: ArtifactStore,
    settings: Settings,
    workspace_id: uuid.UUID,
    actor_kind: str,
    subject: str,
    request_id: str | None,
    received: list[Received],
    source_kind: str,
    source_details: dict[str, Any] | None = None,
    corpus_id: uuid.UUID | None = None,
    entry_fields: Mapping[str, Mapping[str, Any]] | None = None,
    checkpoint: Callable[[], None] = lambda: None,
) -> IngestReport:
    """Stage, store and record ``received`` files. Runs in a worker thread. ``entry_fields``
    gives corpus entries their metadata, by entry name (a manifest's)."""
    with scratch_directory(settings.intake.spool_dir) as scratch:
        staged = [
            stage(
                item.path,
                item.name,
                sha256=item.sha256,
                size=item.size,
                scratch=scratch,
                unpack=settings.intake.unpack_archives,
                unpack_email=settings.intake.unpack_email,
                limits=archive_limits(settings),
            )
            for item in received
        ]
        # Bytes first: writes are idempotent, so a failed transaction leaves only reusable bytes.
        for tree in staged:
            for node in tree.walk():
                checkpoint()
                key = quarantine_key(workspace_id, node.sha256)
                if not store.exists(key):
                    with node.path.open("rb") as handle:
                        store.put(key, handle, media_type=node.detection.media_type)
        with engine.begin() as conn:
            source_id = create_source(conn, workspace_id, source_kind, subject, source_details)

            def record(node: StagedFile, parent: DocumentRow | None) -> IngestedFile:
                key = quarantine_key(workspace_id, node.sha256)
                document, created = record_document(
                    conn,
                    workspace_id,
                    sha256=node.sha256,
                    size_bytes=node.size,
                    name=node.name,
                    detection=node.detection.as_dict(),
                    storage_key=key,
                    created_by=subject,
                    status="refused" if node.refusal else "stored",
                    reason_code=node.refusal[1] if node.refusal else None,
                )
                if created:
                    record_artifact(
                        conn,
                        ArtifactRecord(
                            workspace_id=workspace_id,
                            cache_key=node.sha256,
                            kind="document",
                            storage_uri=store.uri(key),
                            size_bytes=node.size,
                            content_sha256=node.sha256,
                            media_type=node.detection.media_type,
                        ),
                    )
                record_origin(
                    conn,
                    workspace_id,
                    document_id=document.id,
                    source_id=source_id,
                    name=node.name,
                    parent_document_id=parent.id if parent else None,
                    member_path=node.member_path,
                )
                result = IngestedFile(node.name, document, not created, node.refusal)
                result.members = [record(member, document) for member in node.members]
                return result

            files = [record(tree, None) for tree in staged]
            report = IngestReport(files=files, source_id=source_id)
            if corpus_id is not None:
                fields = entry_fields or {}
                entries = [
                    NewEntry(member.name, member.document.id, fields.get(member.name, {}))
                    for item in files
                    for member in (item.members or [item])
                ]
                report.added_to_corpus = add_entries(
                    conn, workspace_id, corpus_id, entries, subject
                )
            record_audit_event(
                conn,
                workspace_id=workspace_id,
                actor_kind=actor_kind,
                actor_subject=subject,
                action=f"document.{source_kind}",
                resource_type="source",
                resource_id=str(source_id),
                request_id=request_id,
                details={**report.counts(), "added_to_corpus": report.added_to_corpus},
            )
    logger.info("documents ingested", extra=report.counts())
    return report


def folder_files(root: Path, relative: str, *, max_files: int, max_bytes: int) -> list[Received]:
    """Every file under ``root/relative`` (hidden ones skipped), named by its path below it,
    hashed in place. Refuses a path that leaves ``root``."""
    base = root.resolve()
    target = folder_under(root, relative)
    found: list[Received] = []
    for path in sorted(target.rglob("*")):
        relative_parts = path.relative_to(target).parts
        if not path.is_file() or any(part.startswith(".") for part in relative_parts):
            continue
        if path.is_symlink() and base not in path.resolve().parents:
            continue
        if len(found) >= max_files:
            raise UploadError("payload_too_large", "too_many_files")
        try:
            sha256, size = hash_file(path, max_bytes)
        except FileTooLarge:
            raise UploadError("payload_too_large", "file_too_large") from None
        try:
            name = entry_name("/".join(relative_parts))
        except InvalidName:
            continue
        found.append(Received(name, path, sha256, size))
    if not found:
        raise UploadError("invalid_request", "no_files")
    return found


def manifest_entry_fields(manifest: CorpusManifest) -> dict[str, dict[str, Any]]:
    """Each listed file's corpus entry metadata, by its entry name. Raises ``UploadError`` for a
    path that can't be an entry name or that two documents share."""
    fields: dict[str, dict[str, Any]] = {}
    for document in manifest.documents:
        try:
            name = entry_name(document.path)
        except InvalidName:
            raise UploadError("invalid_request", "invalid_manifest_path") from None
        if name in fields:
            raise UploadError("invalid_request", "manifest_path_repeated")
        fields[name] = document.entry_fields()
    return fields


def folder_under(root: Path, relative: str) -> Path:
    """The folder ``root/relative``, refusing a path that leaves ``root``."""
    base = root.resolve()
    target = (base / relative).resolve() if relative else base
    if target != base and base not in target.parents:
        raise UploadError("forbidden", "path_outside_root")
    if not target.is_dir():
        raise UploadError("not_found", "folder_not_found")
    return target


def read_manifest_text(folder: Path) -> str:
    path = folder / MANIFEST_NAME
    if not path.is_file():
        raise UploadError("not_found", "manifest_not_found")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise UploadError("payload_too_large", "manifest_too_large")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise UploadError("invalid_request", "manifest_unreadable") from None


def manifest_files(
    folder: Path, manifest: CorpusManifest, *, max_files: int, max_bytes: int
) -> list[Received]:
    """The files a manifest lists, named by their paths, hashed in place. Refuses a path that
    leaves the folder, and a listed file that isn't there."""
    if len(manifest.documents) > max_files:
        raise UploadError("payload_too_large", "too_many_files")
    base = folder.resolve()
    found: list[Received] = []
    for document in manifest.documents:
        try:
            name = entry_name(document.path)
        except InvalidName:
            raise UploadError("invalid_request", "invalid_manifest_path") from None
        path = (base / name).resolve()
        if base not in path.parents:
            raise UploadError("forbidden", "path_outside_root")
        if not path.is_file():
            raise UploadError("not_found", "manifest_file_missing")
        try:
            sha256, size = hash_file(path, max_bytes)
        except FileTooLarge:
            raise UploadError("payload_too_large", "file_too_large") from None
        found.append(Received(name, path, sha256, size))
    return found


# --- buckets and URLs --------------------------------------------------------------------------


def s3_client(source: S3Source) -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=source.endpoint,
        region_name=source.region,
        aws_access_key_id=source.access_key,
        aws_secret_access_key=source.secret_key.get_secret_value() if source.secret_key else None,
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def _s3_prefix(source: S3Source, relative: str) -> str:
    """The key prefix to list: the source's own, then the caller's, which can't climb out."""
    parts = [part for part in relative.replace("\\", "/").split("/") if part not in ("", ".")]
    if ".." in parts:
        raise UploadError("forbidden", "path_outside_root")
    base = source.prefix.strip("/")
    joined = "/".join([base, *parts]) if base else "/".join(parts)
    return f"{joined}/" if joined else ""


def s3_files(
    source: S3Source,
    relative: str,
    spool: Path,
    *,
    max_files: int,
    max_bytes: int,
    client: Any = None,
    checkpoint: Callable[[], None] = lambda: None,
) -> list[Received]:
    """Every object under the source's prefix and ``relative``, streamed into ``spool`` and
    named by its key below the listed prefix. Hidden keys (a part starting with a dot) and
    folder markers are skipped."""
    from botocore.exceptions import BotoCoreError, ClientError

    client = client or s3_client(source)
    prefix = _s3_prefix(source, relative)
    keys: list[tuple[str, int]] = []
    try:
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=source.bucket, Prefix=prefix
        ):
            for item in page.get("Contents", []):
                key, size = str(item["Key"]), int(item.get("Size", 0))
                below = key[len(prefix) :]
                if not below or below.endswith("/"):
                    continue
                if any(part.startswith(".") for part in below.split("/")):
                    continue
                if size > max_bytes:
                    raise UploadError("payload_too_large", "file_too_large")
                keys.append((key, size))
                if len(keys) > max_files:
                    raise UploadError("payload_too_large", "too_many_files")
    except (BotoCoreError, ClientError):
        raise UploadError("unavailable", "s3_unavailable") from None
    if not keys:
        raise UploadError("invalid_request", "no_files")
    found: list[Received] = []
    try:
        for key, _ in sorted(keys):
            checkpoint()
            try:
                name = entry_name(key[len(prefix) :])
            except InvalidName:
                continue
            sink = HashingFile(spool, max_bytes)
            try:
                body = client.get_object(Bucket=source.bucket, Key=key)["Body"]
                while chunk := body.read(1024 * 1024):
                    sink.write(chunk)
            except FileTooLarge:
                sink.discard()
                raise UploadError("payload_too_large", "file_too_large") from None
            except (BotoCoreError, ClientError):
                sink.discard()
                raise UploadError("unavailable", "s3_unavailable") from None
            sink.close()
            found.append(Received(name, sink.path, sink.sha256, sink.size))
    except UploadError:
        for item in found:
            item.path.unlink(missing_ok=True)
        raise
    return found


def url_file(
    url: str, spool: Path, settings: Settings, *, fetcher: UrlFetcher | None = None
) -> Received:
    """Fetch ``url`` through the SSRF guard into ``spool``."""
    fetcher = fetcher or UrlFetcher(
        guard=UrlGuard(allowed_ports=tuple(settings.intake.url_ports)),
        max_bytes=settings.limits.upload_bytes,
    )
    try:
        fetched = fetcher.fetch(url, spool)
    except FetchRefused as refused:
        kind = "payload_too_large" if refused.code == "file_too_large" else "invalid_request"
        if refused.code in ("fetch_timeout", "fetch_failed", "host_not_found"):
            kind = "unavailable"
        raise UploadError(kind, refused.code) from None
    try:
        name = entry_name(fetched.name)
    except InvalidName:
        name = "download"
    return Received(name, fetched.path, fetched.sha256, fetched.size)
