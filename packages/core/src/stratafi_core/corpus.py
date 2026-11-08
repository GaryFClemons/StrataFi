"""Corpus versions and their manifests.

A corpus is an editable set of entries: a name (a relative path such as ``policies/fees.md``)
for a document. Freezing it makes a corpus version, which never changes. A version's
``manifest_hash`` is the hash of its entries, sorted by name: each name with its content hash,
size, detected type and traits, and the metadata a manifest import adds. It holds no
workspace-specific IDs, so the same files frozen in two workspaces hash the same.
"""

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from stratafi_core.hashing import content_hash


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    media_type: str
    traits: dict[str, Any] = Field(default_factory=dict)
    doc_key: str | None = None
    doc_version: str | None = None
    collection: str | None = None
    acl: dict[str, Any] | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def manifest_hash(entries: list[ManifestEntry]) -> str:
    return content_hash(sorted(entries, key=lambda entry: entry.name))
