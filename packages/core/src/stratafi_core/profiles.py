"""Knowledge profiles: a winning configuration, exported with its evidence.

``stratafi.knowledge-profile.v1`` holds a trial's resolved pipeline spec and retrieval config,
the models it used, and the evidence: the experiment, the trial, and the objective on dev and on
the confirming test-split read, each with its interval. At export it carries its review: who
approved or rejected it. Neither the review nor who exported it and when are part of the
profile's hash, so approving it doesn't make it another profile.
"""

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from stratafi_core.hashing import content_hash
from stratafi_core.specs import (
    NAME_PATTERN,
    PipelineSpec,
    RetrievalConfig,
    SpecInvalid,
    SpecIssue,
    validation_issues,
)

PROFILE_SCHEMA = "stratafi.knowledge-profile.v1"
Name = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ResultRef(_M):
    """The objective on one split: its mean and 95% interval, and the eval run behind it."""

    mean: float
    low: float
    high: float
    n: int
    eval_run_id: str | None = None


class Evidence(_M):
    experiment: dict[str, Any] | None = Field(default=None, description="ID, name, hash")
    trial: dict[str, Any] | None = Field(default=None, description="ID, ordinal, factor values")
    corpus: dict[str, Any] | None = None
    eval_set: dict[str, Any] | None = None
    objective: dict[str, Any] = Field(description="Metric, rule, checkpoint and goal")
    dev: ResultRef | None = None
    test: ResultRef | None = None
    test_read: int | None = Field(default=None, description="Which read of the test split")
    guardrails: dict[str, Any] | None = Field(
        default=None, description="Its group against the leader, equivalence, recommendation"
    )


class Approval(_M):
    by: str
    at: datetime
    decision: Literal["approved", "rejected"]
    self_review: bool = Field(
        default=False, description="The version's author decided it themselves"
    )


class Review(_M):
    status: Literal["draft", "approved", "rejected"]
    approvals: list[Approval] = Field(default_factory=list)


class KnowledgeProfile(_M):
    """``stratafi.knowledge-profile.v1``."""

    schema_: Literal["stratafi.knowledge-profile.v1"] = Field(
        default=PROFILE_SCHEMA, alias="schema"
    )
    name: Name
    description: str = Field(default="", max_length=2000)
    pipeline: PipelineSpec
    retrieval: RetrievalConfig
    models: dict[str, Any] = Field(default_factory=dict, description="Aliases and revisions")
    evidence: Evidence
    created_by: str
    created_at: datetime
    review: Review | None = Field(
        default=None, description="Who approved it, at export; not part of its hash"
    )


IDENTITY_EXCLUDES = frozenset({"review", "created_by", "created_at"})


def profile_hash(profile: KnowledgeProfile) -> str:
    """The profile's identity: its configuration, models and evidence. Who exported it, when,
    and its review don't change it, so exporting or importing the same profile again adds no
    version."""
    return content_hash(
        profile.model_dump(mode="json", by_alias=True, exclude=set(IDENTITY_EXCLUDES))
    )


def dump_profile(profile: KnowledgeProfile, format: Literal["yaml", "json"] = "yaml") -> str:
    data = profile.model_dump(mode="json", by_alias=True, exclude_none=True)
    if format == "json":
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    header = f"# {PROFILE_SCHEMA}: {profile.name}\n"
    return header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def load_profile(source: str | Mapping[str, Any]) -> KnowledgeProfile:
    """A profile from YAML or JSON text, or a mapping. Raises ``SpecInvalid``."""
    if isinstance(source, str):
        try:
            data = yaml.safe_load(source)
        except yaml.YAMLError:
            raise SpecInvalid([SpecIssue("(root)", "invalid_yaml")]) from None
    else:
        data = dict(source)
    if not isinstance(data, Mapping):
        raise SpecInvalid([SpecIssue("(root)", "not_an_object")])
    try:
        return KnowledgeProfile.model_validate(dict(data))
    except ValidationError as error:
        raise SpecInvalid(validation_issues(error)) from None
