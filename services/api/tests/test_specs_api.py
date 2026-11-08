"""Components and specs through the API, unit tier: the registry with its
parameter schemas, and validation with precise paths. No database: saving answers 503."""

import pytest
from fastapi.testclient import TestClient

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, StaticWorkspaces
from stratafi_core.settings import OfflineSettings
from stratafi_testing.canary import CanaryCheck
from stratafi_testing.components import stub_registry
from stratafi_testing.identity import DevIdentity


@pytest.fixture
def client(dev_identity: DevIdentity) -> TestClient:
    app = create_app(
        OfflineSettings(),
        identity=DevIdentityProvider(dev_identity.verifier),
        workspaces=StaticWorkspaces(),
        registry=stub_registry(),
        connect=False,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_components_list_with_their_parameter_schemas(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    response = client.get("/api/v1/components", headers=dev_identity.headers(role="viewer"))
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert [item["id"] for item in items] == sorted(item["id"] for item in items)
    chunk = next(item for item in items if item["id"] == "chunk/sections")
    assert chunk["stage"] == "chunk"
    assert chunk["params_schema"]["properties"]["max_tokens"]["default"] == 400
    assert set(chunk) >= {
        "version",
        "accepts",
        "resource_class",
        "deterministic",
        "license",
        "platforms",
        "egress",
        "baseline",
    }
    only_index = client.get(
        "/api/v1/components", params={"stage": "index"}, headers=dev_identity.headers()
    )
    assert [item["id"] for item in only_index.json()["items"]] == ["index/bm25", "index/exact"]


def test_one_component_by_its_id(client: TestClient, dev_identity: DevIdentity) -> None:
    found = client.get("/api/v1/components/embed/gateway", headers=dev_identity.headers())
    assert found.status_code == 200
    assert found.json()["params_schema"]["properties"]["alias"]["default"] == "embed-default"
    missing = client.get("/api/v1/components/embed/nope", headers=dev_identity.headers())
    assert missing.status_code == 404
    assert missing.json()["code"] == "component_not_found"


def test_components_need_a_role(client: TestClient, dev_identity: DevIdentity) -> None:
    assert (
        client.get("/api/v1/components", headers=dev_identity.headers(role="guest")).status_code
        == 403
    )


def test_validation_resolves_a_valid_spec(client: TestClient, dev_identity: DevIdentity) -> None:
    response = client.post(
        "/api/v1/pipeline-specs/validate",
        json={"schema": "stratafi.pipeline.v1", "name": "mine", "stages": {"clean": []}},
        headers=dev_identity.headers(role="viewer"),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is True
    assert body["issues"] == []
    assert len(body["spec_hash"]) == 64
    assert body["document"]["schema"] == "stratafi.pipeline.v1"
    assert body["document"]["stages"]["embed"]["params"] == {
        "alias": "embed-default",
        "batch_size": 32,
    }


def test_validation_names_each_problem_by_path(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    response = client.post(
        "/api/v1/pipeline-specs/validate",
        json={
            "schema": "stratafi.pipeline.v1",
            "name": "mine",
            "stages": {"chunk": {"params": {"max_tokens": 0}}, "clean": ["clean/nope"]},
        },
        headers=dev_identity.headers(role="viewer"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert {(i["path"], i["code"]) for i in body["issues"]} == {
        ("stages.chunk.params.max_tokens", "greater_than_equal"),
        ("stages.clean.0.component", "unknown_component"),
    }
    assert body["document"] is None


def test_retrieval_configs_validate_too(client: TestClient, dev_identity: DevIdentity) -> None:
    response = client.post(
        "/api/v1/retrieval-configs/validate",
        json={"schema": "stratafi.retrieval.v1", "name": "r", "top_k": 0},
        headers=dev_identity.headers(),
    )
    assert response.json()["issues"] == [{"path": "top_k", "code": "greater_than_equal"}]


def test_saving_an_invalid_spec_is_a_problem_with_issues(
    client: TestClient, dev_identity: DevIdentity, canary: CanaryCheck
) -> None:
    response = client.post(
        "/api/v1/pipeline-specs",
        json={
            "schema": "stratafi.pipeline.v1",
            "name": "mine",
            "description": canary.value,
            "stages": {
                "embed": {"component": "embed/gateway", "params": {"alias": [canary.value]}}
            },
        },
        headers=dev_identity.headers(role="editor"),
    )
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["code"] == "invalid_spec"
    assert body["issues"] == [{"path": "stages.embed.params.alias", "code": "string_type"}]
    assert body["errors"] == ["stages.embed.params.alias"]
    canary.assert_absent(response.text, canary.captured_text())


def test_saving_needs_the_editor_role_and_a_database(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    spec = {"schema": "stratafi.pipeline.v1", "name": "mine"}
    viewer = client.post(
        "/api/v1/pipeline-specs", json=spec, headers=dev_identity.headers(role="viewer")
    )
    assert viewer.status_code == 403
    editor = client.post(
        "/api/v1/pipeline-specs", json=spec, headers=dev_identity.headers(role="editor")
    )
    assert editor.status_code == 503
    assert editor.json()["code"] == "database_not_configured"
