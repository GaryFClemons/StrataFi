"""Knowledge profiles: the winning pipeline and retrieval config with their
evidence, and the round trip through export and import."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from stratafi_core.components import Component, ComponentInfo, ComponentRegistry
from stratafi_core.profiles import (
    Approval,
    Evidence,
    KnowledgeProfile,
    Review,
    dump_profile,
    load_profile,
    profile_hash,
)
from stratafi_core.specs import SpecInvalid, resolve_pipeline, resolve_retrieval
from stratafi_testing.components import ChunkParams, EmbedParams, FixedParams, RerankParams


def component(
    component_id: str, stage: str, params: type[BaseModel] | None = None
) -> type[Component]:
    info = ComponentInfo(
        id=component_id,
        stage=stage,  # type: ignore[arg-type]
        version="1.0.0",
        title=component_id,
        description="For tests.",
        license="MIT",
    )
    attributes: dict[str, object] = {"info": info}
    if params is not None:
        attributes["Params"] = params
    return type(component_id.replace("/", "_").replace("-", "_"), (Component,), attributes)


REGISTRY = ComponentRegistry(
    [
        component("extract/upload", "extract"),
        component("unpack/zip", "extract"),
        component("scan/content-safety", "scan"),
        component("scan/clamav", "scan"),
        component("parse/markdown", "parse"),
        component("clean/whitespace", "clean"),
        component("chunk/sections", "chunk", ChunkParams),
        component("chunk/fixed-tokens", "chunk", FixedParams),
        component("enrich/none", "enrich"),
        component("embed/gateway", "embed", EmbedParams),
        component("index/exact", "index"),
        component("index/bm25", "index"),
        component("rerank/none", "rerank"),
        component("rerank/gateway", "rerank", RerankParams),
    ]
)


def pipeline(**stages: Any):
    base: dict[str, Any] = {
        "scan": {"checks": ["scan/content-safety", "scan/clamav"]},
        "parse": {"route": {"default": "parse/markdown"}},
        "embed": {"component": "embed/gateway", "params": {"alias": "embed-default"}},
    }
    return resolve_pipeline(
        {"schema": "stratafi.pipeline.v1", "name": "p", "stages": {**base, **stages}}, REGISTRY
    )


def retrieval(**changes: Any):
    document: dict[str, Any] = {
        "schema": "stratafi.retrieval.v1",
        "name": "r",
        "rerank": {"component": "rerank/gateway", "params": {"alias": "rerank-default"}},
    }
    return resolve_retrieval({**document, **changes}, REGISTRY)


def profile(**changes: Any) -> KnowledgeProfile:
    fields: dict[str, Any] = {
        "name": "kp-policies",
        "pipeline": pipeline(
            chunk={"component": "chunk/fixed-tokens", "params": {"size": 256, "overlap": 32}}
        ),
        "retrieval": retrieval(),
        "models": {"embed": {"alias": "embed-default", "revision": "r1"}},
        "evidence": Evidence(
            experiment={"id": "e1", "name": "chunking"},
            trial={"id": "t1", "ordinal": 3, "values": {"chunk": "chunk/fixed-tokens size=256"}},
            objective={"metric": "recall@5", "rule": "knowledge-hit-v1", "checkpoint": "final"},
            dev={"mean": 0.86, "low": 0.77, "high": 0.94, "n": 52, "eval_run_id": "ev1"},
            test={"mean": 0.81, "low": 0.66, "high": 0.93, "n": 22, "eval_run_id": "ev2"},
            test_read=1,
        ),
        "created_by": "u-tenant0-editor",
        "created_at": datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
    }
    fields.update(changes)
    return KnowledgeProfile(**fields)


def test_a_profile_round_trips_through_export_and_import() -> None:
    original = profile()
    for format in ("yaml", "json"):
        text = dump_profile(original, format)  # type: ignore[arg-type]
        loaded = load_profile(text)
        assert loaded == original
        assert profile_hash(loaded) == profile_hash(original)
    assert dump_profile(original).startswith("# stratafi.knowledge-profile.v1: kp-policies\n")
    # A review travels with the export without changing what the profile is.
    reviewed = original.model_copy(
        update={
            "review": Review(
                status="approved",
                approvals=[
                    Approval(by="u-tenant0-owner", at=original.created_at, decision="approved")
                ],
            )
        }
    )
    again = load_profile(dump_profile(reviewed))
    assert again.review == reviewed.review
    assert profile_hash(again) == profile_hash(original)
    # Exported again by someone else, later: the same profile.
    later = original.model_copy(
        update={"created_by": "u-tenant0-owner", "created_at": datetime(2026, 11, 2, tzinfo=UTC)}
    )
    assert profile_hash(later) == profile_hash(original)
    assert profile_hash(original.model_copy(update={"name": "kp-other"})) != profile_hash(original)
    with pytest.raises(SpecInvalid):
        load_profile("schema: stratafi.knowledge-profile.v1\nname: Bad Name\n")
    with pytest.raises(SpecInvalid):
        load_profile("- not\n- a mapping\n")


def test_a_profile_is_its_configuration_and_evidence() -> None:
    document = profile().model_dump(mode="json", by_alias=True, exclude_none=True)
    assert set(document) == {
        "schema",
        "name",
        "description",
        "pipeline",
        "retrieval",
        "models",
        "evidence",
        "created_by",
        "created_at",
    }
    # A different winning configuration is a different profile.
    other = profile(retrieval=retrieval(rerank="rerank/none"))
    assert profile_hash(other) != profile_hash(profile())
