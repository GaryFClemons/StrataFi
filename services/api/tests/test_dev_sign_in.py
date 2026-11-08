"""Development sign-in (unit tier): the standalone UI's short-lived tokens exist
only with a key and outside prod, never in the contract, and with no body sign in the local
owner."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, StaticWorkspaces
from stratafi_core.identity import HEADER, MAX_TTL_SECONDS
from stratafi_core.settings import OfflineSettings
from stratafi_testing.identity import DevIdentity


def client(dev_identity: DevIdentity, **settings: object) -> TestClient:
    app = create_app(
        OfflineSettings(**settings),
        identity=DevIdentityProvider(dev_identity.verifier),
        workspaces=StaticWorkspaces(),
        connect=False,
    )
    return TestClient(app)


def test_signing_in_mints_a_token_the_api_accepts(dev_identity: DevIdentity) -> None:
    lab = client(dev_identity, identity={"dev_sign_in_key": dev_identity.private_key_path})
    response = lab.post("/api/v1/dev/token", json={"tenant": "tenant0", "role": "viewer"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["subject"], body["expires_in"]) == ("u-tenant0-viewer", MAX_TTL_SECONDS)
    me = lab.get("/api/v1/me", headers={HEADER: body["token"]})
    assert me.status_code == 200, me.text
    assert me.json()["roles"] == ["viewer"]
    identity = dev_identity.verifier.verify(body["token"])
    assert (identity.tid, identity.roles, identity.groups) == (
        "tenant0",
        ("viewer",),
        ("all-staff",),
    )


def test_with_no_body_it_signs_in_the_local_owner(dev_identity: DevIdentity) -> None:
    lab = client(dev_identity, identity={"dev_sign_in_key": dev_identity.private_key_path})
    body = lab.post("/api/v1/dev/token", json={}).json()
    assert body["subject"] == "u-local-owner"
    me = lab.get("/api/v1/me", headers={HEADER: body["token"]}).json()
    assert (me["tenant"], me["roles"]) == ("local", ["owner"])


def test_a_guest_has_no_role(dev_identity: DevIdentity) -> None:
    lab = client(dev_identity, identity={"dev_sign_in_key": dev_identity.private_key_path})
    token = lab.post("/api/v1/dev/token", json={"role": "guest"}).json()["token"]
    assert dev_identity.verifier.verify(token).roles == ()
    headers = {HEADER: token}
    assert lab.get("/api/v1/me", headers=headers).json()["roles"] == []
    assert lab.get("/api/v1/components", headers=headers).status_code == 403


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"env": "prod", "identity": {"dev_sign_in_key": "KEY"}},
        {"identity": {"dev_sign_in_key": "missing.pem"}},
    ],
    ids=["no key", "prod", "unreadable key"],
)
def test_sign_in_is_off_unless_dev_with_a_key(
    dev_identity: DevIdentity, settings: dict, tmp_path: Path
) -> None:
    identity = dict(settings.get("identity", {}))
    if identity.get("dev_sign_in_key") == "KEY":
        identity["dev_sign_in_key"] = dev_identity.private_key_path
    elif "dev_sign_in_key" in identity:
        identity["dev_sign_in_key"] = tmp_path / identity["dev_sign_in_key"]
    lab = client(dev_identity, **{**settings, "identity": identity})
    assert lab.post("/api/v1/dev/token", json={}).status_code == 404
    assert lab.get("/api/v1/auth/config").json() == {"dev_sign_in": False}


def test_the_ui_learns_that_it_can_sign_in(dev_identity: DevIdentity) -> None:
    lab = client(dev_identity, identity={"dev_sign_in_key": dev_identity.private_key_path})
    response = lab.get("/api/v1/auth/config")
    assert response.status_code == 200
    assert response.json() == {"dev_sign_in": True}


def test_sign_in_is_not_in_the_contract(dev_identity: DevIdentity) -> None:
    lab = client(dev_identity, identity={"dev_sign_in_key": dev_identity.private_key_path})
    assert not any("/dev/" in path for path in lab.get("/api/v1/openapi.json").json()["paths"])


def test_requests_are_checked(dev_identity: DevIdentity) -> None:
    lab = client(dev_identity, identity={"dev_sign_in_key": dev_identity.private_key_path})
    assert lab.post("/api/v1/dev/token", json={"role": "admin"}).status_code == 400
    assert lab.post("/api/v1/dev/token", json={"tenant": "Not A Tenant"}).status_code == 400
