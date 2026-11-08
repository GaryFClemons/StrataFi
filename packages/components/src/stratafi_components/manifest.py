"""Corpus manifests (``extract/manifest``).

A ``manifest.yaml`` describes a folder of documents: each document's path, collection, key,
version, effective dates, ACL, title, source URI and status. ``extract/manifest`` imports such a
folder into a corpus with that metadata on every entry, so evaluation can filter as a knowledge
base does: by collection, ACL, effective dates and status. The Halcyon demo corpus
(``fixtures/demo-corpus/corpus``) is one such folder.

An ACL needs a group or a user (an empty one would mean nobody, so it fails closed), and a
version ends after it starts. Unknown keys are ignored. Errors are paths and codes, never
values.
"""

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from stratafi_core.evalsets import load_yaml
from stratafi_core.specs import SpecInvalid, SpecIssue, validation_issues

MANIFEST_NAME = "manifest.yaml"
MAX_DOCUMENTS = 10_000

Collection = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,62}$")]
DocKey = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")]
Version = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")]
Principal = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[^\x00-\x1f\x7f]+$")
]
PrintableText = Annotated[
    str, StringConstraints(min_length=1, max_length=512, pattern=r"^[^\x00-\x1f\x7f]+$")
]


class ManifestAcl(BaseModel):
    """Who may read a document: any listed group, or any listed user (a token's ``sub``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    groups: tuple[Principal, ...] = Field(default=(), max_length=256)
    users: tuple[Principal, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def _not_empty(self) -> "ManifestAcl":
        if not self.groups and not self.users:
            raise ValueError("an ACL needs at least one group or user")
        return self


class ManifestDocument(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    path: str = Field(min_length=1, max_length=512)
    collection: Collection
    doc_key: DocKey
    version: Version
    effective_from: date
    effective_to: date | None = None
    acl: ManifestAcl
    source_uri: PrintableText | None = None
    title: PrintableText | None = None
    status: Literal["published", "draft"] = "published"
    expect: Literal["indexed", "rejected"] | None = None

    @field_validator("version", mode="before")
    @classmethod
    def _version_as_text(cls, value: Any) -> Any:
        return str(value) if isinstance(value, int) and not isinstance(value, bool) else value

    @model_validator(mode="after")
    def _dates_in_order(self) -> "ManifestDocument":
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be after effective_from")
        return self

    def entry_fields(self) -> dict[str, Any]:
        """The corpus entry's metadata (``stratafi_store.corpora.NewEntry.fields``)."""
        metadata: dict[str, Any] = {"status": self.status}
        for key in ("title", "source_uri", "expect"):
            value = getattr(self, key)
            if value is not None:
                metadata[key] = value
        return {
            "doc_key": self.doc_key,
            "doc_version": self.version,
            "collection": self.collection,
            "acl": {"groups": list(self.acl.groups), "users": list(self.acl.users)},
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "metadata": metadata,
        }


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant: str | None = Field(default=None, max_length=64)
    documents: list[ManifestDocument] = Field(min_length=1, max_length=MAX_DOCUMENTS)


def load_corpus_manifest(text: str) -> CorpusManifest:
    """Parse a ``manifest.yaml``. Raises ``SpecInvalid``."""
    data = load_yaml(text)
    if not isinstance(data, dict):
        raise SpecInvalid([SpecIssue("(root)", "not_an_object")])
    try:
        return CorpusManifest.model_validate(data)
    except ValidationError as exc:
        raise SpecInvalid(validation_issues(exc)) from None
