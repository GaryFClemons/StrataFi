"""Workspace features: custom checks are off until an owner turns them on, and
they can't be turned on before the sandbox is configured and its security review signed off."""

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, SqlWorkspaceDirectory
from stratafi_core.settings import OfflineSettings
from stratafi_store.artifacts import LocalArtifactStore
from stratafi_testing.identity import DevIdentity

pytestmark = pytest.mark.integration

FEATURE = "/api/v1/workspace/features/custom_checks"


def client_for(
    engine: sa.Engine, identity: DevIdentity, tmp_path: Path, **sandbox: Any
) -> TestClient:
    app = create_app(
        OfflineSettings(suites={"sandbox": sandbox}),
        identity=DevIdentityProvider(identity.verifier),
        workspaces=SqlWorkspaceDirectory(engine, auto_provision=True),
        engine=engine,
        store=LocalArtifactStore(tmp_path / "store"),
        connect=False,
    )
    return TestClient(app)


def feature(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    response = client.get("/api/v1/workspace/features", headers=headers)
    assert response.status_code == 200
    [found] = response.json()["items"]
    return found


@pytest.mark.parametrize(
    ("sandbox", "code"),
    [
        ({}, "sandbox_not_configured"),
        ({"backend": "docker"}, "sandbox_not_reviewed"),
    ],
)
def test_custom_checks_cant_be_turned_on_before_the_review(
    migrated_engine: sa.Engine,
    dev_identity: DevIdentity,
    tmp_path: Path,
    sandbox: dict[str, Any],
    code: str,
) -> None:
    client = client_for(migrated_engine, dev_identity, tmp_path, **sandbox)
    viewer = dev_identity.headers(role="viewer")
    owner = dev_identity.headers(role="owner")
    assert feature(client, viewer) == {
        "name": "custom_checks",
        "enabled": False,
        "blocked": code,
        "review": None,
        "changed_by": None,
        "changed_at": None,
    }
    refused = client.put(FEATURE, json={"enabled": True}, headers=owner)
    assert refused.status_code == 409
    assert refused.json()["code"] == code
    # Turning it off always works.
    assert client.put(FEATURE, json={"enabled": False}, headers=owner).status_code == 200


def test_an_owner_turns_custom_checks_on_under_the_signed_off_review(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path
) -> None:
    client = client_for(migrated_engine, dev_identity, tmp_path, backend="docker", review="SEC-77")
    owner = dev_identity.headers(role="owner")
    editor = dev_identity.headers(role="editor")
    assert feature(client, editor)["blocked"] is None
    refused = client.put(FEATURE, json={"enabled": True}, headers=editor)
    assert (refused.status_code, refused.json()["code"]) == (403, "missing_role")
    assert client.put(
        "/api/v1/workspace/features/telepathy", json={"enabled": True}, headers=owner
    ).status_code in (400, 422)

    turned = client.put(FEATURE, json={"enabled": True}, headers=owner)
    assert turned.status_code == 200
    on = turned.json()
    assert (on["enabled"], on["review"], on["changed_by"]) == (True, "SEC-77", on["changed_by"])
    assert on["changed_by"]
    assert feature(client, editor)["enabled"] is True

    off = client.put(FEATURE, json={"enabled": False}, headers=owner).json()
    assert (off["enabled"], off["review"]) == (False, None)
    audit = client.get("/api/v1/audit-events", headers=owner).json()["items"]
    actions = [event["action"] for event in audit if event["resource_type"] == "workspace_feature"]
    assert actions == ["workspace.feature_off", "workspace.feature_on"]


def test_an_api_token_cant_change_a_feature(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path
) -> None:
    client = client_for(migrated_engine, dev_identity, tmp_path, backend="docker", review="SEC-77")
    owner = dev_identity.headers(role="owner")
    made = client.post(
        "/api/v1/api-tokens",
        json={"name": "ci", "scopes": ["read", "runs:write", "tests:run"]},
        headers=owner,
    ).json()
    bearer = {"Authorization": f"Bearer {made['token']}"}
    assert client.get("/api/v1/workspace/features", headers=bearer).status_code == 200
    refused = client.put(FEATURE, json={"enabled": True}, headers=bearer)
    assert (refused.status_code, refused.json()["code"]) == (403, "token_not_allowed")
