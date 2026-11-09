"""Suites that run on their own, against Postgres with the fakes (integration
tier).

Freezing a corpus version runs a suite on that corpus once, and a new target version runs it
again; a due schedule fires in the worker; a regression compares with the last passing run; a
finished run is announced to a webhook, signed, and retried until the receiver takes it.
"""

import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from stratafi.webhooks import verify
from stratafi_store.scope import system_scope
from stratafi_store.tables import test_schedules
from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import (
    HALCYON_RETRIEVAL,
    Lab,
    baseline_pipeline,
    hybrid_retrieval,
    running_lab,
)

pytestmark = pytest.mark.integration


def prepare(lab: Lab) -> str:
    """The Halcyon corpus (frozen once), its pipeline, retrieval config and eval set; its ID."""
    corpus = lab.post("/corpora", {"name": "halcyon"})
    lab.post(
        "/documents/import-manifest",
        {"root": "fixtures", "path": "demo-corpus/corpus", "corpus_id": corpus["id"]},
    )
    lab.post(f"/corpora/{corpus['id']}/versions")
    lab.save_spec(baseline_pipeline())
    lab.save_retrieval(hybrid_retrieval())
    lab.import_eval_set("halcyon-retrieval", HALCYON_RETRIEVAL.read_text(encoding="utf-8"), seed=0)
    return corpus["id"]


def suite(*checks: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {
        "schema": "stratafi.test-suite.v1",
        "name": "halcyon-auto",
        "target": {"pipeline": "baseline-v1", "retrieval": "hybrid-v1"},
        "corpus": "halcyon",
        "eval_set": {"name": "halcyon-retrieval"},
        "checks": list(checks) or [{"id": "chunked", "stage": "chunk", "assert": "all_succeeded"}],
        **fields,
    }


def runs_of(lab: Lab, name: str = "halcyon-auto") -> list[dict[str, Any]]:
    return lab.get("/test-runs", suite=name)["items"]


def wait_for(found, *, timeout: float = 120.0, poll: float = 0.1) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = found()
        if value:
            return value
        time.sleep(poll)
    raise AssertionError("timed out")


def test_freezing_a_corpus_version_runs_the_suite_once(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        corpus_id = prepare(lab)
        triggers = [{"on": "corpus_version_frozen"}, {"on": "target_changed"}]
        lab.post("/test-suites", {"document": suite(triggers=triggers)})
        assert runs_of(lab) == []
        entries = lab.get(f"/corpora/{corpus_id}/documents", limit=200)["items"]
        lab.post(f"/corpora/{corpus_id}/documents/remove", {"names": [entries[0]["name"]]})
        frozen = lab.client.post(f"/api/v1/corpora/{corpus_id}/versions", headers=lab.editor)
        assert frozen.status_code == 201
        found = runs_of(lab)
        assert [(run["trigger"], run["triggers"]) for run in found] == [
            ("corpus_version_frozen", ["corpus_version_frozen"])
        ]
        done = lab.wait_test_run(found[0]["id"])
        assert done["status"] == "passed"
        assert done["resolved"]["corpus"]["version"] == frozen.json()["version"] == 2
        # The same working set again freezes nothing new, and runs nothing.
        again = lab.client.post(f"/api/v1/corpora/{corpus_id}/versions", headers=lab.editor)
        assert again.status_code == 200
        assert len(runs_of(lab)) == 1
        # A new version of the target's pipeline runs it too; a pinned suite doesn't run.
        lab.post(
            "/test-suites",
            {"document": suite(name="halcyon-pinned", corpus="halcyon@1", triggers=triggers)},
        )
        changed = baseline_pipeline()
        changed["stages"]["chunk"]["params"] = {"max_tokens": 700}
        lab.save_spec(changed)
        latest = runs_of(lab)
        assert [run["trigger"] for run in latest] == ["target_changed", "corpus_version_frozen"]
        assert lab.wait_test_run(latest[0]["id"])["status"] == "passed"
        assert [run["trigger"] for run in runs_of(lab, "halcyon-pinned")] == ["target_changed"]
        audit = lab.client.get(
            "/api/v1/audit-events", params={"limit": 200}, headers=dev_identity.headers()
        ).json()["items"]
        assert sum(event["action"] == "test_run.trigger" for event in audit) == 3


def test_a_due_schedule_fires_in_the_worker(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        lab.post("/test-suites", {"document": suite(triggers=[{"schedule": "17 6 * * 1-5"}])})
        automation = lab.get("/test-suites/halcyon-auto/automation")
        assert [item["cron"] for item in automation["schedules"]] == ["17 6 * * 1-5"]
        # In UTC, the suite's default, whatever time zone the database answers in.
        schedule = automation["schedules"][0]
        upcoming = datetime.fromisoformat(schedule["next_fire_at"]).astimezone(UTC)
        assert upcoming > datetime.now(UTC)
        assert (upcoming.hour, upcoming.minute) == (6, 17)
        with migrated_engine.begin() as conn:
            conn.execute(
                sa.update(test_schedules).values(
                    next_fire_at=datetime.now(UTC) - timedelta(seconds=1)
                )
            )
        fired = wait_for(lambda: runs_of(lab))
        assert [run["trigger"] for run in fired] == ["schedule"]
        assert fired[0]["created_by"] == "system:scheduler"
        assert lab.wait_test_run(fired[0]["id"])["status"] == "passed"
        after = lab.get("/test-suites/halcyon-auto/automation")["schedules"][0]
        assert after["last_test_run_id"] == fired[0]["id"]
        assert datetime.fromisoformat(after["next_fire_at"]) > datetime.now(UTC)
        time.sleep(0.5)
        assert len(runs_of(lab)) == 1


def test_a_regression_compares_with_the_last_passing_run(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        document = suite(
            {"id": "mrr", "metric": "mrr@10", "vs_baseline": {"max_drop": 0.02}},
            {"id": "changes", "type": "data_change", "max": {"removed": 0}},
            {"id": "same-embedding", "allow_change": False},
            baseline="last_passing",
        )
        first = lab.wait_test_run(
            lab.post("/test-runs", {"document": document}, expect=(202,))["id"]
        )
        assert first["status"] == "passed"
        assert {check["result"] for check in first["checks"]} == {"skipped"}
        second = lab.wait_test_run(
            lab.post("/test-runs", {"suite": "halcyon-auto"}, expect=(202,))["id"]
        )
        assert second["status"] == "passed"
        assert second["baseline_test_run_id"] == first["id"]
        checks = {check["id"]: check for check in second["checks"]}
        assert checks["mrr"]["result"] == "pass"
        assert checks["mrr"]["baseline_value"] == checks["mrr"]["value"]
        assert checks["changes"]["detail"]["counts"]["removed"] == 0
        assert checks["same-embedding"]["detail"]["changed"] == []


@pytest.mark.parametrize("secret_key", [None, "a-deployment-key"], ids=["stored", "derived"])
def test_a_webhook_delivery_is_signed_and_retried(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app, secret_key
) -> None:
    received: list[tuple[dict[str, str], bytes]] = []

    async def hook(request: Request) -> JSONResponse:
        received.append((dict(request.headers), await request.body()))
        return JSONResponse({}, status_code=500 if len(received) == 1 else 200)

    receiver = serve_app(Starlette(routes=[Route("/hook", hook, methods=["POST"])]))
    suites = {"webhook_secret_key": secret_key} if secret_key else None
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app, suites=suites) as lab:
        prepare(lab)
        owner = dev_identity.headers(role="owner")
        refused = lab.client.post(
            "/api/v1/webhooks", json={"url": "http://169.254.169.254/latest"}, headers=owner
        )
        assert (refused.status_code, refused.json()["code"]) == (400, "metadata_address")
        created = lab.client.post(
            "/api/v1/webhooks",
            json={"url": f"{receiver}/hook", "suite": "halcyon-auto"},
            headers=owner,
        )
        assert created.status_code == 201
        secret = created.json()["secret"]
        listed = lab.client.get("/api/v1/webhooks", headers=owner).json()["items"]
        assert "secret" not in listed[0]
        # With the deployment's key, Postgres keeps no secret.
        with system_scope(), migrated_engine.connect() as conn:
            stored = conn.execute(sa.text("SELECT secret FROM stratafi.webhooks")).scalar_one()
        assert (stored is None) == bool(secret_key)
        run = lab.wait_test_run(lab.post("/test-runs", {"document": suite()}, expect=(202,))["id"])
        wait_for(lambda: len(received) >= 2)
        deliveries = wait_for(
            lambda: [
                item
                for item in lab.client.get(
                    f"/api/v1/webhooks/{created.json()['id']}/deliveries", headers=owner
                ).json()["items"]
                if item["status"] == "delivered"
            ]
        )
        assert (deliveries[0]["attempts"], deliveries[0]["response_status"]) == (2, 200)
        assert deliveries[0]["test_run_id"] == run["id"]
        (first_headers, first_body), (headers, body) = received[:2]
        assert first_body == body
        assert first_headers["stratafi-delivery"] == headers["stratafi-delivery"]
        assert uuid.UUID(headers["stratafi-delivery"]) == uuid.UUID(deliveries[0]["id"])
        assert headers["stratafi-event"] == "test_run.finished"
        assert verify(secret, body, headers["stratafi-signature"])
        assert not verify(secret, body.replace(b"passed", b"failed"), headers["stratafi-signature"])
        assert not verify("whsec_other", body, headers["stratafi-signature"])
        assert b'"status":"passed"' in body
