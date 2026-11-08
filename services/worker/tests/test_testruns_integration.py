"""Test suites against Postgres, with the fakes (integration tier).

A suite over the Halcyon demo corpus passes, then fails when one threshold changes; its second
run reuses every artifact. Its reports are the contract's record, JUnit XML that validates, and
Markdown. A reference that can't be resolved ends the run ``error``. A golden query's text
reaches no log, result or report.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from lxml import etree

from stratafi_core.contracts import validate_contract
from stratafi_testing.canary import CanaryCheck
from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import (
    HALCYON_RETRIEVAL,
    WEB,
    Lab,
    baseline_pipeline,
    hybrid_retrieval,
    running_lab,
    serve_web,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
JUNIT_XSD = etree.XMLSchema(etree.parse(str(ROOT / "contracts" / "junit" / "junit.xsd")))


def prepare(lab: Lab) -> None:
    corpus = lab.post("/corpora", {"name": "halcyon"})
    lab.post(
        "/documents/import-manifest",
        {"root": "fixtures", "path": "demo-corpus/corpus", "corpus_id": corpus["id"]},
    )
    lab.post(f"/corpora/{corpus['id']}/versions")
    lab.save_spec(baseline_pipeline())
    lab.save_retrieval(hybrid_retrieval())
    lab.import_eval_set("halcyon-retrieval", HALCYON_RETRIEVAL.read_text(encoding="utf-8"), seed=0)


def suite(*checks: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {
        "schema": "stratafi.test-suite.v1",
        "name": "halcyon-guard",
        "target": {"pipeline": "baseline-v1", "retrieval": "hybrid-v1"},
        "corpus": "halcyon@latest",
        "eval_set": {"name": "halcyon-retrieval", "version": "latest", "split": "dev"},
        "checks": list(checks),
        **fields,
    }


def guard(
    min_recall: float, query: str = "How fast do hourly baristas build up vacation time?"
) -> dict[str, Any]:
    return suite(
        {"id": "recall-at-5", "metric": "recall@5", "min": min_recall},
        {"id": "no-superseded", "metric": "must_not_return_violations", "max": 1000},
        {"id": "latency", "metric": "latency_ms.p95", "max": 60_000},
        {"id": "parse-success", "stage": "parse", "metric": "parse.success_rate", "min": 1.0},
        {"id": "all-chunked", "stage": "chunk", "assert": "all_succeeded"},
        {"id": "nothing-infected", "stage": "scan", "assert": {"no_outcome": ["infected"]}},
        {
            "id": "no-pii",
            "stage": "chunk",
            "assert": "no_findings",
            "detector": "scan/pii",
            "categories": ["ssn", "card_number", "iban"],
        },
        {
            "id": "pto-golden",
            "query": query,
            # A bare name: the eval set's own staff asker (all-staff), who may read the policy.
            "asker": "staff",
            "expect": {"in_top_k": 10, "doc_key": "pto-leave", "version": "3"},
        },
    )


def run(lab: Lab, document: dict[str, Any], **body: Any) -> dict[str, Any]:
    started = lab.post("/test-runs", {"document": document, **body}, expect=(202,))
    return lab.wait_test_run(started["id"])


def test_a_halcyon_suite_passes_then_fails_when_a_threshold_changes(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        lenient = run(lab, guard(0.0))
        assert lenient["status"] == "passed", lenient
        checks = {check["id"]: check for check in lenient["checks"]}
        assert {check["result"] for check in checks.values()} == {"pass"}
        recall = checks["recall-at-5"]["value"]
        assert 0 < recall <= 1
        assert checks["pto-golden"]["value"] is not None
        assert checks["no-pii"]["detail"]["scanned"] > 0
        assert lenient["summary"] == {
            "checks": 8,
            "passed": 8,
            "failed": 0,
            "errored": 0,
            "skipped": 0,
        }

        strict = run(lab, guard(round(recall + 0.05, 6)))
        assert strict["status"] == "failed"
        assert strict["suite_version"] == 2
        failed = [check for check in strict["checks"] if check["result"] != "pass"]
        assert [(check["id"], check["code"]) for check in failed] == [("recall-at-5", "below_min")]
        # Nothing changed but the threshold: every task of the second pipeline run was cached.
        tasks = lab.get(f"/runs/{strict['pipeline_run_id']}/tasks", limit=200)["items"]
        assert tasks
        assert {task["status"] for task in tasks} == {"cached"}

        report = lab.client.get(
            f"/api/v1/test-runs/{strict['id']}/report",
            params={"format": "junit"},
            headers=lab.editor,
        )
        assert report.status_code == 200
        assert report.headers["content-type"].startswith("application/xml")
        document = etree.fromstring(report.content)
        JUNIT_XSD.assertValid(document)
        assert document.get("failures") == "1"
        assert [case.get("name") for case in document.iter("testcase")][:2] == [
            "recall-at-5",
            "no-superseded",
        ]
        record = lab.get(f"/test-runs/{strict['id']}/report", format="json")
        validate_contract("stratafi.test-run.v1", record)
        assert record["outcome"] == "failed"
        assert record["resolved"]["eval_set"]["split"] == "dev"
        markdown = lab.client.get(
            f"/api/v1/test-runs/{strict['id']}/report",
            params={"format": "markdown"},
            headers=lab.editor,
        ).text
        assert markdown.startswith("## halcyon-guard: failed")

        history = lab.get("/test-suites/halcyon-guard/history")
        assert [entry["status"] for entry in history["runs"]] == ["passed", "failed"]
        listed = lab.get("/test-runs", suite="halcyon-guard")["items"]
        assert [item["status"] for item in listed] == ["failed", "passed"]


def test_unresolvable_references_end_the_run_as_an_error(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        document = suite({"id": "chunked", "stage": "chunk", "assert": "all_succeeded"})
        saved = lab.post("/test-suites", {"document": document})
        again = lab.client.post(
            "/api/v1/test-suites", json={"document": document}, headers=lab.editor
        )
        assert (saved["version"], again.status_code, again.json()["version"]) == (1, 200, 1)
        missing = run(
            lab,
            suite(
                {"id": "c", "stage": "chunk", "assert": "all_succeeded"}, corpus="no-such-corpus"
            ),
        )
        assert (missing["status"], missing["error_code"]) == ("error", "corpus_not_found")
        report = lab.client.get(
            f"/api/v1/test-runs/{missing['id']}/report",
            params={"format": "junit"},
            headers=lab.editor,
        )
        document = etree.fromstring(report.content)
        JUNIT_XSD.assertValid(document)
        assert [error.get("type") for error in document.iter("error")] == ["corpus_not_found"]
        invalid = lab.client.post(
            "/api/v1/test-runs",
            json={"document": suite({"id": "c", "metric": "recall@5"})},
            headers=lab.editor,
        )
        assert invalid.status_code == 400
        assert invalid.json()["issues"] == [{"path": "checks.0", "code": "threshold_missing"}]
        validated = lab.post("/test-suites/validate", {"content": "schema: x\n"})
        assert validated["valid"] is False


def test_a_golden_query_reaches_no_log_result_or_report(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app, canary: CanaryCheck
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        document = suite(
            {
                "id": "golden",
                "query": f"{canary.value} who counts the tip jar at close",
                "asker": "staff",
                "expect": {
                    "in_top_k": 10,
                    "doc_key": "tips",
                    "contains": canary.value,
                },
            }
        )
        finished = run(lab, document)
        assert finished["status"] == "failed"
        assert finished["checks"][0]["code"] == "not_found"
        outputs = [
            lab.client.get(
                f"/api/v1/test-runs/{finished['id']}/report",
                params={"format": kind},
                headers=lab.editor,
            ).text
            for kind in ("json", "junit", "markdown")
        ]
        audit = lab.client.get(
            "/api/v1/audit-events", headers=dev_identity.headers(role="owner")
        ).text
        canary.assert_absent(str(finished), *outputs, audit)
        canary.assert_not_logged()


@pytest.mark.browser
def test_the_tests_screen_in_a_browser(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app
) -> None:
    """The Tests screen's flows work: the editor's validation and completion, save and run,
    a run's checks with their evidence and reports, the suite's runs, trend and automation, and
    API tokens and webhooks (web/e2e/tests-flow.mjs)."""
    node = shutil.which("node")
    if node is None:
        pytest.fail("the browser flow needs Node 24 and Playwright")
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        url = serve_web(lab, serve_app)
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [node, str(WEB / "e2e" / "tests-flow.mjs"), url],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=WEB,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("ok"), result.stdout


TABLES = """
from stratafi.checks import CheckContext, CheckResult


def rate_tables_not_split(ctx: CheckContext) -> CheckResult:
    split = [
        c.chunk_id
        for c in ctx.chunks()
        if c.metadata.get("table_part") and "rate" in (c.heading_path[-1:] or [""])[0].lower()
    ]
    return CheckResult.assert_true(not split, value=len(split), detail={"chunks": split[:20]})


def every_chunk_has_its_document(ctx: CheckContext) -> CheckResult:
    documents = {document.document_id for document in ctx.documents()}
    chunks = list(ctx.chunks())
    orphans = [c.chunk_id for c in chunks if c.document_id not in documents]
    return CheckResult.assert_true(
        bool(chunks) and not orphans,
        value=len(chunks),
        detail={"documents": len(documents), "orphans": orphans[:20], "run": ctx.runs},
    )
"""


@pytest.mark.system
def test_custom_checks_run_in_the_sandbox_once_the_owner_turns_them_on(
    migrated_engine, dev_identity: DevIdentity, tmp_path: Path, serve_app, repo_root: Path
) -> None:
    """Custom checks end to end: needs Docker for the sandbox (system tier) besides the services."""
    from stratafi_testing.check_sandbox import build_sandbox_image

    image = build_sandbox_image(repo_root)
    if image is None:
        pytest.skip("custom checks need Docker for their sandbox")
    sandbox = {"backend": "docker", "image": image, "review": "SEC-TEST-1", "workdir": tmp_path}
    document = suite(
        {"id": "parse-success", "stage": "parse", "metric": "parse.success_rate", "min": 1.0},
        {"id": "rate-tables-whole", "python": "checks/tables.py:rate_tables_not_split"},
        {"id": "chunks-documented", "python": "checks/tables.py:every_chunk_has_its_document"},
        code={"checks/tables.py": TABLES},
    )
    with running_lab(
        migrated_engine, dev_identity, tmp_path, serve_app, suites={"sandbox": sandbox}
    ) as lab:
        prepare(lab)
        off = run(lab, document)
        disabled = [check for check in off["checks"] if check["type"] == "custom"]
        assert [(c["result"], c["code"], c["detail"]) for c in disabled] == [
            ("error", "custom_checks_disabled", {"reason": "feature_off"})
        ] * 2
        assert off["code"][0]["path"] == "checks/tables.py"

        owner = dev_identity.headers(role="owner")
        refused = lab.client.put(
            "/api/v1/workspace/features/custom_checks", json={"enabled": True}, headers=lab.editor
        )
        assert refused.status_code == 403
        turned = lab.client.put(
            "/api/v1/workspace/features/custom_checks", json={"enabled": True}, headers=owner
        )
        assert turned.status_code == 200, turned.text
        assert (turned.json()["enabled"], turned.json()["review"]) == (True, "SEC-TEST-1")

        on = run(lab, document)
        assert on["status"] == "passed", on
        checks = {check["id"]: check for check in on["checks"]}
        assert checks["rate-tables-whole"]["value"] == 0
        documented = checks["chunks-documented"]
        assert documented["value"] > 0
        assert documented["detail"]["documents"] > 0
        assert documented["detail"]["run"]["pipeline_run"] == on["pipeline_run_id"]
        # The suite scores no queries, so there's no eval run: the evidence is the pipeline run.
        assert on["eval_run_id"] is None
        assert documented["evidence"] == [{"kind": "run", "id": on["pipeline_run_id"]}]
        record = lab.get(f"/test-runs/{on['id']}/report", format="json")
        validate_contract("stratafi.test-run.v1", record)
        assert record["code"] == on["code"]
        assert len(record["code"][0]["sha256"]) == 64
        markdown = lab.client.get(
            f"/api/v1/test-runs/{on['id']}/report",
            params={"format": "markdown"},
            headers=lab.editor,
        ).text
        assert "`checks/tables.py`: SHA-256" in markdown
        assert "rate_tables_not_split" not in markdown
        # The bundles were deleted with the evaluation.
        assert list(tmp_path.glob("stratafi-checks-*")) == []
