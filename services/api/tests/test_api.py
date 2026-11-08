"""The API skeleton in the unit tier: identity, roles, problems, request IDs,
pagination and the contract. No database: workspaces live in memory."""

import logging
import time
import uuid
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, Principal, StaticWorkspaces, require_role
from stratafi_api.pagination import decode_cursor, encode_cursor
from stratafi_api.problems import Problem
from stratafi_core.identity import AUDIENCE, ISSUER, TokenSigner
from stratafi_core.settings import OfflineSettings
from stratafi_store.artifacts import LocalArtifactStore
from stratafi_store.workspaces import Workspace
from stratafi_testing.canary import CanaryCheck
from stratafi_testing.identity import DevIdentity


def build_app(
    dev_identity: DevIdentity,
    *,
    workspaces: StaticWorkspaces | None = None,
    settings: OfflineSettings | None = None,
    store: LocalArtifactStore | None = None,
) -> FastAPI:
    app = create_app(
        settings or OfflineSettings(),
        identity=DevIdentityProvider(dev_identity.verifier),
        workspaces=workspaces or StaticWorkspaces(),
        store=store,
        connect=False,
    )

    @app.get("/api/v1/_test/editor")
    def editor_only(principal: Principal = Depends(require_role("editor"))) -> dict[str, str]:
        return {"subject": principal.subject}

    @app.get("/api/v1/_test/boom")
    def boom() -> None:
        raise RuntimeError("the document said: a private sentence")

    @app.get("/api/v1/_test/problem")
    def problem() -> None:
        raise Problem("conflict", code="already_running")

    return app


@pytest.fixture
def client(dev_identity: DevIdentity) -> TestClient:
    return TestClient(build_app(dev_identity), raise_server_exceptions=False)


def assert_problem(response, status: int, kind: str, code: str | None = None) -> dict:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["type"] == f"urn:stratafi:problem:{kind}"
    assert body["status"] == status
    assert body["request_id"] == response.headers["x-request-id"]
    assert body["instance"] == f"urn:stratafi:request:{body['request_id']}"
    if code is not None:
        assert body.get("code") == code
    return body


# --- identity -----------------------------------------------------------------------------


def test_a_valid_token_reaches_its_workspace(client: TestClient, dev_identity: DevIdentity) -> None:
    response = client.get("/api/v1/me", headers=dev_identity.headers("tenant0", "editor"))
    assert response.status_code == 200
    body = response.json()
    assert body["subject"] == "u-tenant0-editor"
    assert body["tenant"] == "tenant0"
    assert body["roles"] == ["editor"]
    assert body["workspace"]["slug"] == "tenant0"


def test_expired_tokens_are_refused(client: TestClient, dev_identity: DevIdentity) -> None:
    old = int(time.time()) - 900
    token = dev_identity.sign_claims(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "u-tenant0-owner",
            "tid": "tenant0",
            "roles": ["owner"],
            "rid": "r-0000000000000000",
            "iat": old,
            "exp": old + 60,
        }
    )
    response = client.get("/api/v1/me", headers={"X-Stratafi-Identity": token})
    assert_problem(response, 401, "unauthorized", "expired")


def test_a_token_signed_with_the_wrong_key_is_refused(client: TestClient) -> None:
    stranger = TokenSigner(Ed25519PrivateKey.generate())
    token = stranger.mint(sub="u-tenant0-owner", tenant="tenant0", roles=["owner"])
    response = client.get("/api/v1/me", headers={"X-Stratafi-Identity": token})
    assert_problem(response, 401, "unauthorized", "unknown_key")


def test_a_token_without_a_tenant_is_refused(client: TestClient, dev_identity: DevIdentity) -> None:
    now = int(time.time())
    token = dev_identity.sign_claims(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "u-tenant0-owner",
            "roles": ["owner"],
            "rid": "r-0000000000000000",
            "iat": now,
            "exp": now + 60,
        }
    )
    response = client.get("/api/v1/me", headers={"X-Stratafi-Identity": token})
    assert_problem(response, 401, "unauthorized", "missing_tenant")


def test_missing_and_doubled_tokens_are_refused(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    assert_problem(client.get("/api/v1/me"), 401, "unauthorized", "missing")
    token = dev_identity.token()
    doubled = client.get(
        "/api/v1/me", headers=[("X-Stratafi-Identity", token), ("X-Stratafi-Identity", token)]
    )
    assert_problem(doubled, 401, "unauthorized", "ambiguous")


@pytest.mark.parametrize(
    ("role", "allowed"),
    [("owner", True), ("editor", True), ("viewer", False), ("guest", False)],
)
def test_roles_nest_owner_over_editor_over_viewer(
    client: TestClient, dev_identity: DevIdentity, role: str, allowed: bool
) -> None:
    response = client.get("/api/v1/_test/editor", headers=dev_identity.headers("tenant0", role))
    if allowed:
        assert response.status_code == 200
        assert response.json() == {"subject": f"u-tenant0-{role}"}
    else:
        assert_problem(response, 403, "forbidden", "missing_role")


@pytest.mark.parametrize("role", ["editor", "viewer", "guest"])
def test_only_owners_read_audit_events(
    client: TestClient, dev_identity: DevIdentity, role: str
) -> None:
    response = client.get("/api/v1/audit-events", headers=dev_identity.headers("tenant0", role))
    assert_problem(response, 403, "forbidden", "missing_role")


def test_platform_roles_map_through_the_role_map(dev_identity: DevIdentity) -> None:
    settings = OfflineSettings(identity={"role_map": {"kb_admin": "editor", "staff": "viewer"}})
    client = TestClient(build_app(dev_identity, settings=settings))
    headers = dev_identity.headers("tenant0", "guest", roles=("kb_admin", "staff", "unknown"))
    assert client.get("/api/v1/me", headers=headers).json()["roles"] == ["editor", "viewer"]
    assert client.get("/api/v1/_test/editor", headers=headers).status_code == 200


def test_a_local_membership_adds_a_role(dev_identity: DevIdentity) -> None:
    workspaces = StaticWorkspaces(memberships={("tenant0", "u-tenant0-viewer"): "editor"})
    client = TestClient(build_app(dev_identity, workspaces=workspaces))
    headers = dev_identity.headers("tenant0", "viewer")
    assert client.get("/api/v1/me", headers=headers).json()["roles"] == ["editor", "viewer"]
    assert client.get("/api/v1/_test/editor", headers=headers).status_code == 200


def test_tenants_land_in_their_own_workspaces(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    first = client.get("/api/v1/me", headers=dev_identity.headers("tenant0")).json()
    second = client.get("/api/v1/me", headers=dev_identity.headers("halcyon")).json()
    again = client.get("/api/v1/me", headers=dev_identity.headers("tenant0", "viewer")).json()
    assert first["workspace"]["id"] != second["workspace"]["id"]
    assert first["workspace"]["id"] == again["workspace"]["id"]


def test_without_provisioning_an_unknown_tenant_is_refused(dev_identity: DevIdentity) -> None:
    known = Workspace(uuid.uuid4(), "tenant0", "Tenant 0", "tenant0")
    workspaces = StaticWorkspaces(workspaces={"tenant0": known}, auto_provision=False)
    client = TestClient(build_app(dev_identity, workspaces=workspaces))
    assert client.get("/api/v1/me", headers=dev_identity.headers("tenant0")).status_code == 200
    response = client.get("/api/v1/me", headers=dev_identity.headers("halcyon"))
    assert_problem(response, 403, "forbidden", "unknown_workspace")


# --- problems and request IDs -------------------------------------------------------------


def test_request_ids_are_kept_when_well_formed_and_made_otherwise(client: TestClient) -> None:
    kept = client.get("/api/v1/health/live", headers={"X-Request-Id": "req-123:abc"})
    assert kept.headers["x-request-id"] == "req-123:abc"
    replaced = client.get("/api/v1/health/live", headers={"X-Request-Id": "has spaces"})
    assert replaced.headers["x-request-id"].startswith("r-")
    made = client.get("/api/v1/health/live")
    assert made.headers["x-request-id"].startswith("r-")
    assert made.headers["x-request-id"] != replaced.headers["x-request-id"]


def test_each_request_logs_its_route_template_not_its_path(
    client: TestClient, dev_identity: DevIdentity, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="stratafi.api.requests")
    client.get("/api/v1/me", headers=dev_identity.headers())
    client.get("/api/v1/nothing-here?q=secret")
    lines = [r for r in caplog.records if r.name == "stratafi.api.requests"]
    assert [(r.method, r.route, r.status) for r in lines] == [
        ("GET", "/me", 200),
        ("GET", "unmatched", 404),
    ]
    assert all("secret" not in r.getMessage() for r in lines)


def test_unknown_routes_and_methods_are_problems(client: TestClient) -> None:
    assert_problem(client.get("/api/v1/nothing-here"), 404, "not_found")
    assert_problem(client.delete("/api/v1/health"), 405, "method_not_allowed")


def test_raised_problems_keep_their_code(client: TestClient) -> None:
    assert_problem(client.get("/api/v1/_test/problem"), 409, "conflict", "already_running")


def test_validation_errors_name_fields_and_never_echo_input(
    client: TestClient, dev_identity: DevIdentity, canary: CanaryCheck
) -> None:
    response = client.get(
        "/api/v1/audit-events",
        params={"limit": canary.value},
        headers=dev_identity.headers("tenant0", "owner"),
    )
    body = assert_problem(response, 400, "invalid_request", "validation")
    assert body["errors"] == ["query.limit"]
    canary.assert_absent(response.text)
    canary.assert_not_logged()


def test_bad_cursors_are_invalid_requests_without_echo(
    client: TestClient, dev_identity: DevIdentity, canary: CanaryCheck
) -> None:
    headers = dev_identity.headers("tenant0", "owner")
    for cursor in [encode_cursor([canary.value]), "AAAA", canary.value.replace("-", "_")]:
        response = client.get("/api/v1/audit-events", params={"cursor": cursor}, headers=headers)
        assert_problem(response, 400, "invalid_request", "invalid_cursor")
        canary.assert_absent(response.text)
    canary.assert_not_logged()


def test_an_unhandled_exception_is_a_plain_500(client: TestClient, canary: CanaryCheck) -> None:
    response = client.get("/api/v1/_test/boom")
    body = assert_problem(response, 500, "internal")
    assert body["detail"] == "Something went wrong."
    assert "private" not in response.text


def test_a_canary_token_never_reaches_logs_or_bodies(
    client: TestClient, canary: CanaryCheck
) -> None:
    response = client.get("/api/v1/me", headers={"X-Stratafi-Identity": canary.value})
    assert_problem(response, 401, "unauthorized", "malformed")
    canary.assert_absent(response.text)
    canary.assert_not_logged()


# --- health -------------------------------------------------------------------------------


def test_health_reports_each_dependency(dev_identity: DevIdentity, tmp_path: Path) -> None:
    client = TestClient(build_app(dev_identity, store=LocalArtifactStore(tmp_path)))
    response = client.get("/api/v1/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["checks"] == {"database": {"status": "not_configured"}, "storage": {"status": "ok"}}
    assert client.get("/api/v1/health/live").json() == {"status": "ok"}


# --- pagination ---------------------------------------------------------------------------


def test_cursors_round_trip() -> None:
    cursor = encode_cursor(["2026-10-30T12:00:00+00:00", 41])
    assert "=" not in cursor
    assert decode_cursor(cursor, size=2) == ["2026-10-30T12:00:00+00:00", 41]


@pytest.mark.parametrize(
    "cursor",
    ["", "!!", "bm90IGpzb24", encode_cursor([1]), "W10", "eyJ2IjoyLCJrIjpbMSwyXX0"],
)
def test_malformed_cursors_are_refused(cursor: str) -> None:
    with pytest.raises(Problem) as problem:
        decode_cursor(cursor, size=2)
    assert problem.value.code == "invalid_cursor"


# --- the contract -------------------------------------------------------------------------


def test_the_contract_matches_what_the_api_sends(dev_identity: DevIdentity) -> None:
    spec = build_app(dev_identity).openapi()
    assert spec["servers"] == [{"url": "/api/v1"}]
    for path, operations in spec["paths"].items():
        assert not path.startswith("/api/v1")
        for operation in operations.values():
            assert "422" not in operation["responses"]
            for status, response in operation["responses"].items():
                if int(status) >= 400 and status != "503":
                    assert list(response["content"]) == ["application/problem+json"]
    assert "HTTPValidationError" not in spec["components"]["schemas"]


def test_the_exported_contract_is_current(repo_root: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "script_export_openapi", repo_root / "scripts" / "export_openapi.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


def test_job_routes_need_a_database(client: TestClient, dev_identity: DevIdentity) -> None:
    response = client.get("/api/v1/jobs", headers=dev_identity.headers("tenant0", "viewer"))
    assert_problem(response, 503, "unavailable", "database_not_configured")
    events = client.get("/api/v1/events", headers=dev_identity.headers("tenant0", "viewer"))
    assert_problem(events, 503, "unavailable", "events_not_configured")
