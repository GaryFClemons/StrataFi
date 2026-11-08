"""API tokens and the CI recipe against a running lab (integration tier).

Tokens are stored hashed, carry scopes, expire and can be revoked. The CI recipe
(``examples/ci/github-actions.yml``) runs as GitHub would run its steps: its ``Run the suite``
step's script, with the lab's address and a token, then its last step, which fails the job. A
suite that passes exits 0; the same suite with a threshold above what the lab measures exits 1.
"""

import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
import yaml
from lxml import etree

from stratafi_store.tables import api_tokens
from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import (
    HALCYON_RETRIEVAL,
    Lab,
    baseline_pipeline,
    hybrid_retrieval,
    running_lab,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "examples" / "ci" / "github-actions.yml"
EXAMPLE_SUITE = ROOT / "examples" / "suites" / "policies-regression.yaml"
BASH = shutil.which("bash") or "/bin/bash"
JUNIT_XSD = etree.XMLSchema(etree.parse(str(ROOT / "contracts" / "junit" / "junit.xsd")))


def prepare(lab: Lab) -> None:
    corpus = lab.post("/corpora", {"name": "halcyon"})
    lab.post(
        "/documents/import-manifest",
        {"root": "fixtures", "path": "demo-corpus/corpus", "corpus_id": corpus["id"]},
    )
    lab.post(f"/corpora/{corpus['id']}/versions")
    # The example suite's targets, the demo seed's names.
    lab.save_spec(baseline_pipeline("local-baseline"))
    lab.save_retrieval(hybrid_retrieval("local-hybrid"))
    lab.import_eval_set("halcyon-retrieval", HALCYON_RETRIEVAL.read_text(encoding="utf-8"), seed=0)


def make_token(lab: Lab, owner: dict[str, str], *scopes: str, name: str = "ci-main") -> dict:
    created = lab.client.post(
        "/api/v1/api-tokens", json={"name": name, "scopes": list(scopes)}, headers=owner
    )
    assert created.status_code == 201, created.text
    return created.json()


def test_tokens_are_hashed_scoped_expiring_and_revocable(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        owner = dev_identity.headers(role="owner")
        reader = make_token(lab, owner, "read", name="reader")
        runner = make_token(lab, owner, "read", "tests:run", name="runner")
        assert reader["token"].startswith("sft_")
        with migrated_engine.connect() as conn:
            stored = conn.execute(sa.select(api_tokens.c.token_hash, api_tokens.c.prefix)).all()
        assert all(reader["token"] not in row.token_hash for row in stored)
        listed = lab.client.get("/api/v1/api-tokens", headers=owner).json()["items"]
        assert {item["name"] for item in listed} == {"reader", "runner"}
        assert all("token" not in item for item in listed)

        def call(token: str, method: str, path: str, **kwargs: Any) -> Any:
            headers = {"Authorization": f"Bearer {token}"}
            return lab.client.request(method, f"/api/v1{path}", headers=headers, **kwargs)

        me = call(reader["token"], "GET", "/me").json()
        assert me["subject"] == f"token:{reader['id']}"
        assert me["scopes"] == ["read"]
        assert call(reader["token"], "GET", "/test-runs").status_code == 200
        refused = call(reader["token"], "POST", "/test-runs", json={"suite": "x"})
        assert (refused.status_code, refused.json()["code"]) == (403, "missing_scope")
        spec = call(runner["token"], "POST", "/pipeline-specs", json=baseline_pipeline())
        assert (spec.status_code, spec.json()["code"]) == (403, "missing_scope")
        owner_only = call(
            runner["token"], "POST", "/api-tokens", json={"name": "x", "scopes": ["read"]}
        )
        assert (owner_only.status_code, owner_only.json()["code"]) == (403, "token_not_allowed")
        missing = call(runner["token"], "POST", "/test-runs", json={"suite": "absent"})
        assert (missing.status_code, missing.json()["code"]) == (404, "test_suite_not_found")
        bogus = call("sft_" + "x" * 43, "GET", "/me")
        assert (bogus.status_code, bogus.json()["code"]) == (401, "token_invalid")
        with migrated_engine.begin() as conn:
            conn.execute(
                sa.update(api_tokens)
                .where(api_tokens.c.name == "runner")
                .values(
                    created_at=datetime.now(UTC) - timedelta(days=2),
                    expires_at=datetime.now(UTC) - timedelta(days=1),
                )
            )
        expired = call(runner["token"], "GET", "/me")
        assert (expired.status_code, expired.json()["code"]) == (401, "token_expired")
        revoked = lab.client.delete(f"/api/v1/api-tokens/{reader['id']}", headers=owner)
        assert revoked.json()["active"] is False
        gone = call(reader["token"], "GET", "/me")
        assert (gone.status_code, gone.json()["code"]) == (401, "token_revoked")
        audit = lab.client.get("/api/v1/audit-events", headers=owner).text
        assert reader["token"] not in audit
        assert runner["token"] not in audit


def recipe_step(name: str) -> str:
    workflow = yaml.safe_load(RECIPE.read_text())
    for step in workflow["jobs"]["suite"]["steps"]:
        if step.get("name") == name:
            return step["run"]
    raise AssertionError(f"the recipe has no step {name!r}")


def run_recipe(workdir: Path, url: str, token: str) -> tuple[int, int]:
    """The recipe's suite step and its last step, as a runner would run them: the exit code the
    suite step recorded, and the job's."""
    output = workdir / "github-output"
    output.write_text("")
    env = {
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(workdir),
        "STRATAFI_URL": url,
        "STRATAFI_TOKEN": token,
        "SUITE": "stratafi/suites/policies-regression.yaml",
        "GITHUB_OUTPUT": str(output),
    }
    subprocess.run(  # noqa: S603 - the recipe's own script, in a scratch directory
        [BASH, "--noprofile", "--norc", "-eo", "pipefail", "-c", recipe_step("Run the suite")],
        cwd=workdir,
        env=env,
        check=True,
        timeout=600,
    )
    recorded = dict(line.split("=", 1) for line in output.read_text().splitlines())
    fail_step = recipe_step("Fail on a failed suite").replace(
        "${{ steps.suite.outputs.exit_code }}", recorded["exit_code"]
    )
    job = subprocess.run(  # noqa: S603
        [BASH, "--noprofile", "--norc", "-eo", "pipefail", "-c", fail_step],
        cwd=workdir,
        env=env,
        check=False,
    )
    return int(recorded["exit_code"]), job.returncode


def test_the_ci_recipe_passes_then_exits_1_on_a_broken_threshold(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        url = serve_app(lab.app)
        token = make_token(lab, dev_identity.headers(role="owner"), "read", "tests:run")["token"]
        workdir = tmp_path / "checkout"
        suites = workdir / "stratafi" / "suites"
        suites.mkdir(parents=True)
        suite = yaml.safe_load(EXAMPLE_SUITE.read_text())
        (suites / "policies-regression.yaml").write_text(EXAMPLE_SUITE.read_text())

        assert run_recipe(workdir, url, token) == (0, 0)
        junit = etree.parse(str(workdir / "stratafi-junit.xml"))
        JUNIT_XSD.assertValid(junit)
        assert junit.getroot().get("failures") == "0"
        assert (
            (workdir / "stratafi-summary.md")
            .read_text()
            .startswith("## policies-regression: passed")
        )

        suite["checks"][0]["min"] = 1.01  # recall@5 can't reach it
        (suites / "policies-regression.yaml").write_text(yaml.safe_dump(suite, sort_keys=False))
        assert run_recipe(workdir, url, token) == (1, 1)
        failed = [
            case.get("name")
            for case in etree.parse(str(workdir / "stratafi-junit.xml")).iter("testcase")
            if case.find("failure") is not None
        ]
        assert failed == ["recall-at-5"]
        record = yaml.safe_load((workdir / "stratafi-run.json").read_text())
        assert record["outcome"] == "failed"
        assert record["test_run"]["trigger"] == "ci"
        assert record["test_run"]["created_by"].startswith("token:")
