"""The extract stage's components.

Documents arrive through intake: an upload, a server folder, a bucket (``extract/s3``)
or a URL (``extract/url``) stores each file's bytes in the quarantine prefix under its SHA-256,
detects its type and unpacks archives (``unpack/zip``) and email (``unpack/email``). In a run,
the extract stage then reads each entry's stored bytes, checks their hash, and records what they
are: the ``raw`` artifact, ``stratafi.raw-document.v1``. Detection depends on the bytes and the
entry's extension, so both are in its cache key.

Version 1.1.0 of every extractor adds traits (``stratafi_components.taxonomy``): Office files'
pages, sheets, slides, tables and images, image sizes, ``partly_scanned`` PDFs.
"""

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.detect import Detection, detect
from stratafi_components.intake import hash_file
from stratafi_core.artifacts import RawDocument
from stratafi_core.components import Component, ComponentError, ComponentInfo, StageContext


def raw_document(document_id: str, sha256: str, size: int, detection: Detection) -> RawDocument:
    return RawDocument(
        document_id=document_id,
        sha256=sha256,
        size_bytes=size,
        extension=detection.extension,
        media_type=detection.media_type,
        kind=detection.kind,
        declared_type=detection.declared_type,
        mismatch=detection.mismatch,
        known=detection.known,
        traits=dict(detection.traits),
    )


class ExtractComponent(Component):
    """The extract stage: stored bytes → ``RawDocument``."""

    def extract(
        self,
        path: Path,
        name: str,
        *,
        document_id: str,
        expected_sha256: str,
        params: BaseModel,
        context: StageContext,
    ) -> RawDocument:
        sha256, size = hash_file(path)
        if sha256 != expected_sha256:
            raise ComponentError("hash_mismatch")
        context.checkpoint()
        detection = detect(path, name)
        context.metric("bytes", size)
        return raw_document(document_id, sha256, size, detection)


class ExtractUpload(ExtractComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="extract/upload",
        stage="extract",
        version="1.1.0",
        title="Upload",
        description="Files uploaded through the API: streamed to quarantine with a size limit, "
        "addressed by SHA-256, and typed by their bytes, which must agree with their extension.",
        license="MIT",
        baseline=True,
    )


class ExtractFolder(ExtractComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="extract/folder",
        stage="extract",
        version="1.1.0",
        title="Server folder",
        description="Files imported from a folder on the server that the configuration names "
        "(intake.folder_roots): for development and fixtures.",
        license="MIT",
        resource_class="io",
        services=("server-folder",),
    )


class ExtractManifest(ExtractComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="extract/manifest",
        stage="extract",
        version="1.1.0",
        title="Corpus manifest",
        description="A folder described by a manifest.yaml: each listed file becomes an entry "
        "with its doc_key, version, effective dates, ACL, collection, title and status, as the "
        "Halcyon demo corpus is described.",
        license="MIT",
        resource_class="io",
        services=("server-folder",),
    )


class ExtractS3(ExtractComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="extract/s3",
        stage="extract",
        version="1.1.0",
        title="S3 bucket",
        description="Every object under a prefix of a bucket the configuration names "
        "(intake.s3_sources), streamed to quarantine with the upload limits and named by its "
        "key below the prefix.",
        license="MIT (boto3: Apache-2.0)",
        resource_class="io",
        services=("s3-source",),
    )


class ExtractUrl(ExtractComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="extract/url",
        stage="extract",
        version="1.1.0",
        title="URL",
        description="A document fetched from a URL through an SSRF guard: public addresses "
        "only, checked after DNS resolution and on every redirect; a size cap; content types "
        "Stratafi can process.",
        license="MIT",
        resource_class="io",
        services=("web",),
    )


class UnpackParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_entries: int = Field(default=10_000, ge=1, le=10_000)
    max_expanded_bytes: int = Field(default=512 * 1024 * 1024, ge=1, le=512 * 1024 * 1024)
    max_ratio: int = Field(default=100, ge=1, le=100)


class UnpackZip(Component):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="unpack/zip",
        stage="extract",
        version="1.0.0",
        title="ZIP archives",
        description="Archive members become documents with provenance, within the archive "
        "limits: at most 10,000 entries and 512 MiB expanded, a ratio of 100, no nested "
        "archives.",
        accepts=("application/zip",),
        license="MIT",
        baseline=True,
    )
    Params: ClassVar[type[BaseModel]] = UnpackParams


class UnpackEmail(Component):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="unpack/email",
        stage="extract",
        version="1.0.0",
        title="Email messages",
        description="An .eml or .msg message becomes its body (subject, date and text) and one "
        "document per attachment, with the archive limits; an attached archive refuses it.",
        accepts=("message/rfc822", "application/vnd.ms-outlook"),
        license="MIT (olefile: BSD-2-Clause)",
    )
    Params: ClassVar[type[BaseModel]] = UnpackParams


COMPONENTS: list[type[Component]] = [
    ExtractUpload,
    ExtractFolder,
    ExtractManifest,
    ExtractS3,
    ExtractUrl,
    UnpackZip,
    UnpackEmail,
]
