"""The component registry: identity, parameters and their JSON Schemas."""

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from stratafi_core.components import (
    Cancelled,
    Component,
    ComponentError,
    ComponentInfo,
    ComponentRegistry,
    StageContext,
    UnknownComponent,
)
from stratafi_testing.components import stub, stub_registry


def test_the_registry_exposes_each_components_parameter_json_schema() -> None:
    registry = stub_registry()
    for description in registry.descriptions():
        schema = description.params_schema
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False, description.info.id
    chunk = registry.describe("chunk/sections").params_schema
    assert chunk["properties"]["max_tokens"] == {
        "default": 400,
        "maximum": 100000,
        "minimum": 1,
        "title": "Max Tokens",
        "type": "integer",
    }
    assert chunk["properties"]["token_estimator"]["const"] == "chars/4"


def test_components_resolve_parameters_with_defaults() -> None:
    component = stub_registry().get("embed/gateway")
    params = component.resolve_params({"alias": "embed-local"})
    assert component.dump_params(params) == {"alias": "embed-local", "batch_size": 32}
    assert component.params_hash(params) == component.params_hash(
        component.resolve_params({"batch_size": 32, "alias": "embed-local"})
    )
    with pytest.raises(ValidationError):
        component.resolve_params({"batch_size": 0})


def test_the_registry_lists_by_id_and_by_stage() -> None:
    registry = stub_registry()
    ids = [component.info.id for component in registry]
    assert ids == sorted(ids)
    assert [c.info.id for c in registry.for_stage("index")] == ["index/bm25", "index/exact"]
    assert "parse/markdown" in registry
    with pytest.raises(UnknownComponent):
        registry.get("parse/nope")


def test_a_component_can_be_registered_once() -> None:
    registry = ComponentRegistry([stub("clean/whitespace", "clean")])
    with pytest.raises(ValueError, match="twice"):
        registry.register(stub("clean/whitespace", "clean"))


def test_components_need_info_and_a_pydantic_params_model() -> None:
    class NoInfo(Component):
        pass

    with pytest.raises(TypeError):
        ComponentRegistry([NoInfo])

    class BadParams(Component):
        info = stub("clean/bad", "clean").info
        Params = dict  # type: ignore[assignment]

    with pytest.raises(TypeError):
        ComponentRegistry([BadParams])


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "NoSlash"},
        {"id": "Upper/Case"},
        {"version": "1.0"},
        {"egress": "the internet"},
        {"stage": "nowhere"},
    ],
)
def test_component_info_is_checked(changes: dict) -> None:
    base = stub("clean/whitespace", "clean").info.model_dump()
    with pytest.raises(ValidationError):
        ComponentInfo(**{**base, **changes})


def test_egress_names_a_provider() -> None:
    base = stub("embed/hosted", "embed").info.model_dump()
    assert ComponentInfo(**{**base, "egress": "provider:cohere"}).egress == "provider:cohere"


def test_the_stage_context_checks_cancellation_and_records_metrics() -> None:
    context = StageContext()
    context.checkpoint()
    context.metric("pages", 3)
    context.metric("pages", 2, add=True)
    assert context.metrics == {"pages": 5}
    with pytest.raises(ValueError, match="metric"):
        context.metric("Bad Name", 1)
    with pytest.raises(ComponentError) as raised:
        context.service("clamd")
    assert raised.value.code == "clamd_not_configured"
    scratch = context.scratch_dir()
    assert scratch.is_dir()
    context.close()
    assert not scratch.exists()
    context.cancelled.set()
    with pytest.raises(Cancelled):
        context.checkpoint()


def test_params_models_that_allow_extra_fields_still_resolve() -> None:
    class Loose(BaseModel):
        model_config = ConfigDict(extra="ignore")
        size: int = 1

    component = stub("chunk/loose", "chunk", Loose)
    assert component.dump_params(component.resolve_params({"size": 2, "other": 3})) == {"size": 2}
