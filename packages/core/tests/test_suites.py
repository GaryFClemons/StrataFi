"""Test suites: the spec, every check type passing, failing and erroring, the
record, and the reports, JUnit XML validated against its schema."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from lxml import etree

from stratafi_core.specs import SpecInvalid
from stratafi_core.suites import (
    CheckResult,
    EntryState,
    GoldenPassage,
    QueryRow,
    RunData,
    SuiteRef,
    Summary,
    TaskState,
    TestRunInfo,
    TestRunRecord,
    data_change,
    dump_suite,
    evaluate_data_change,
    evaluate_metric,
    evaluate_regression,
    evaluate_revision,
    evaluate_stage,
    judge_golden,
    junit_xml,
    markdown_summary,
    normalized,
    parse_suite,
    result_message,
    suite_hash,
)
from stratafi_testing.canary import new_canary

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "contracts" / "fixtures" / "test-suite.v1" / "policies-regression.yaml"
JUNIT_XSD = etree.XMLSchema(etree.parse(str(ROOT / "contracts" / "junit" / "junit.xsd")))


def suite_with(*checks: dict, **fields) -> dict:
    return {
        "schema": "stratafi.test-suite.v1",
        "name": "s",
        "target": {"pipeline": "p", "retrieval": "r"},
        "corpus": "c",
        "eval_set": {"name": "e"},
        "checks": list(checks),
        **fields,
    }


def check(**fields):
    return parse_suite(suite_with({"id": "c", **fields})).checks[0]


def score(rule: str = "knowledge-hit-v1", **values: float) -> dict:
    metrics = {name.replace("_at_", "@"): value for name, value in values.items()}
    return {"metrics": {rule: {"final": metrics}}}


def rows(*recalls: float, tag: str = "t", file_type: str = "text/markdown") -> list[QueryRow]:
    return [
        QueryRow(f"q{i}", True, [tag], [file_type], score(recall_at_5=value))
        for i, value in enumerate(recalls)
    ]


def eval_data(recall: float = 0.9, **extra) -> RunData:
    metrics = {
        "metrics": {
            "knowledge-hit-v1": {
                "final": {"recall@5": {"mean": recall, "low": 0.8, "high": 1.0, "n": 4}}
            }
        },
        "latency_ms": {"total_ms": {"p50": 40.0, "p95": 120.0, "mean": 50.0}},
        "violations": {"must_not_return": {"count": 0, "of": 4}, "avoid": {"count": 1, "of": 2}},
        "cost": {"per_1k_queries": 0.5},
        "findability": {"rate": 0.95, "findable": 19, "of": 20, "by_file_type": {}},
    }
    return RunData(
        eval_run_id="e1",
        eval_status="succeeded",
        eval_set_version_id="v1",
        metrics=metrics,
        queries=rows(1.0, 1.0, 1.0, 0.6),
        **extra,
    )


# --- the spec -------------------------------------------------------------------------------------


def test_the_plan_example_parses_with_each_type_inferred() -> None:
    suite = parse_suite(yaml.safe_load(EXAMPLE.read_text()))
    assert [c.kind for c in suite.checks] == [
        "metric",
        "regression",
        "safety",
        "safety",
        "stage",
        "stage",
        "stage",
        "safety",
        "golden",
        "budget",
        "data_change",
        "revision_guard",
    ]
    assert suite.baseline_ref is not None
    assert suite.baseline_ref.kind == "profile"
    assert suite.needs_eval_run
    assert suite.needs_baseline
    dumped = dump_suite(suite)
    assert [c["type"] for c in dumped["checks"]] == [c.kind for c in suite.checks]
    assert dump_suite(parse_suite(dumped)) == dumped


def test_a_bare_on_key_reads_as_a_trigger_although_yaml_says_true() -> None:
    text = "- on: corpus_version_frozen\n- schedule: '17 6 * * 1-5'\n"
    triggers = yaml.safe_load(text)
    assert True in triggers[0]
    suite = parse_suite(suite_with({"id": "a", "type": "data_change"}, triggers=triggers))
    assert suite.triggers[0].on == "corpus_version_frozen"


def test_the_configuration_hash_ignores_the_name_and_description() -> None:
    first = parse_suite(suite_with({"id": "a", "type": "data_change"}))
    second = parse_suite(suite_with({"id": "a", "type": "data_change"}, name="t", description="x"))
    assert suite_hash(first) == suite_hash(second)


@pytest.mark.parametrize(
    ("bad", "path", "code"),
    [
        ({"id": "a", "metric": "recall@5"}, "checks.0", "threshold_missing"),
        ({"id": "a", "metric": "speed", "max": 1}, "checks.0", "unknown_metric"),
        ({"id": "a", "metric": "latency_ms.p95", "min": 1}, "checks.0", "budget_min"),
        ({"id": "a", "bogus": 1}, "checks.0", "unknown_check_type"),
        (
            {"id": "a", "stage": "parse", "metric": "chunk.count", "min": 1},
            "checks.0",
            "wrong_stage",
        ),
        ({"id": "a", "stage": "scan", "assert": "no_outcome"}, "checks.0", "outcomes_missing"),
        ({"id": "a", "query": "q"}, "checks.0", "expect_missing"),
        (
            {"id": "a", "metric": "mrr@10", "vs_baseline": {}},
            "checks.0.vs_baseline",
            "drop_missing",
        ),
        ({"id": "a", "python": "../x.py:f"}, "checks.0.python", "invalid_reference"),
        (
            {"id": "a", "type": "data_change", "max": {"removed": -1}},
            "checks.0.max.removed",
            "greater_than_equal",
        ),
        ({"id": "A b", "type": "data_change"}, "checks.0.id", "string_pattern_mismatch"),
    ],
)
def test_invalid_checks_report_a_path_and_a_code(bad: dict, path: str, code: str) -> None:
    with pytest.raises(SpecInvalid) as raised:
        parse_suite(suite_with(bad))
    assert (path, code) in [(issue.path, issue.code) for issue in raised.value.issues]


def test_suite_level_rules() -> None:
    def codes(**fields) -> list[str]:
        with pytest.raises(SpecInvalid) as raised:
            parse_suite(suite_with(**fields))
        return [issue.code for issue in raised.value.issues]

    same = {"id": "a", "type": "data_change"}
    assert "duplicate_check_id" in codes(checks=[same, same])
    assert "eval_set_missing" in codes(
        checks=[{"id": "a", "metric": "recall@5", "min": 0.5}], eval_set=None
    )
    assert "retrieval_missing" in codes(
        checks=[{"id": "a", "metric": "recall@5", "min": 0.5}], target={"pipeline": "p"}
    )
    assert "target_ambiguous" in codes(checks=[same], target={"profile": "kp", "pipeline": "p"})
    assert "out_of_range" in codes(checks=[same], triggers=[{"schedule": "61 * * * *"}])
    assert "unknown_time_zone" in codes(checks=[same], time_zone="Mars/Olympus")
    with pytest.raises(SpecInvalid) as raised:
        parse_suite(suite_with(same, eval_set={"name": "e", "split": "test"}))
    assert raised.value.paths == ["eval_set.split"]


def test_a_golden_query_never_appears_in_a_validation_problem() -> None:
    canary = new_canary()
    with pytest.raises(SpecInvalid) as raised:
        parse_suite(
            suite_with({"id": "g", "query": canary, "asker": 5, "expect": {"doc_key": "a"}})
        )
    assert canary not in json.dumps([vars(issue) for issue in raised.value.issues])
    assert raised.value.paths == ["checks.0.asker"]


# --- metric, budget and safety --------------------------------------------------------------------


def test_a_metric_threshold_passes_fails_and_errors() -> None:
    passing = evaluate_metric(check(metric="recall@5", min=0.85), eval_data(0.9))
    assert (passing.result, passing.value, passing.detail["n"]) == ("pass", 0.9, 4)
    failing = evaluate_metric(check(metric="recall@5", min=0.95), eval_data(0.9))
    assert (failing.result, failing.code) == ("fail", "below_min")
    assert [link.id for link in failing.evidence if link.kind == "query"] == ["q3"]
    missing = evaluate_metric(check(metric="ndcg@3", min=0.5), eval_data())
    assert (missing.result, missing.code) == ("error", "metric_not_recorded")
    broken = evaluate_metric(check(metric="recall@5", min=0.5), RunData(eval_status="failed"))
    assert (broken.result, broken.code) == ("error", "eval_run_failed")


def test_a_metric_over_a_tag_or_file_type_reads_the_queries() -> None:
    data = eval_data()
    data.queries = [*rows(1.0, 0.0, tag="fees"), *rows(1.0, file_type="application/pdf")]
    by_tag = evaluate_metric(check(metric="recall@5", min=0.6, where={"tag": "fees"}), data)
    assert (by_tag.result, by_tag.value) == ("fail", 0.5)
    by_type = evaluate_metric(
        check(metric="recall@5", min=0.6, where={"file_type": "application/pdf"}), data
    )
    assert (by_type.result, by_type.value) == ("pass", 1.0)
    nothing = evaluate_metric(check(metric="recall@5", min=0.6, where={"tag": "absent"}), data)
    assert (nothing.result, nothing.code) == ("error", "no_queries")


def test_budgets_and_safety_counts() -> None:
    latency = check(metric="latency_ms.p95", max=800)
    assert latency.kind == "budget"
    assert evaluate_metric(latency, eval_data()).result == "pass"
    slow = evaluate_metric(check(metric="latency_ms.p95", max=100), eval_data())
    assert (slow.result, slow.code, slow.value) == ("fail", "above_max", 120.0)
    avoid = check(metric="avoid_violations", max=0)
    assert avoid.kind == "safety"
    assert evaluate_metric(avoid, eval_data()).result == "fail"
    ingest = evaluate_metric(
        check(metric="ingest_cost_usd", max=1), RunData(run_status="succeeded")
    )
    assert (ingest.result, ingest.code) == ("error", "no_value")
    run = RunData(run_id="r1", run_status="succeeded", run_manifest={"cost": {"estimated": 0.2}})
    paid = evaluate_metric(check(metric="ingest_cost_usd", max=1), run)
    assert (paid.result, paid.value, paid.evidence[0].kind) == ("pass", 0.2, "run")


# --- regression -----------------------------------------------------------------------------------


def test_a_regression_passes_fails_is_noise_and_errors() -> None:
    def data(*values: float, version: str = "v1") -> RunData:
        found = RunData(eval_run_id="e", eval_status="succeeded", eval_set_version_id=version)
        found.queries = [
            QueryRow(f"q{i}", True, [], [], score(mrr_at_10=v)) for i, v in enumerate(values)
        ]
        return found

    rule = check(metric="mrr@10", vs_baseline={"max_drop": 0.02, "significance": 0.05})
    baseline = data(*([1.0] * 40))
    same = evaluate_regression(rule, data(*([1.0] * 40)), baseline)
    assert (same.result, same.baseline_value, same.value) == ("pass", 1.0, 1.0)
    worse = evaluate_regression(rule, data(*([0.5] * 40)), baseline)
    assert (worse.result, worse.code) == ("fail", "regression")
    assert worse.detail["p_value"] <= 0.05
    noisy = evaluate_regression(rule, data(0.0, *([1.0] * 39)), baseline)
    assert (noisy.result, noisy.code) == ("pass", "not_significant")
    strict = check(metric="mrr@10", vs_baseline={"max_drop": 0.02})
    assert evaluate_regression(strict, data(0.0, *([1.0] * 39)), baseline).result == "fail"
    relative = check(metric="mrr@10", vs_baseline={"max_relative_drop": 0.5})
    assert evaluate_regression(relative, data(*([0.6] * 40)), baseline).result == "pass"
    assert evaluate_regression(rule, data(1.0), None).result == "skipped"
    unpaired = evaluate_regression(rule, data(1.0, version="v2"), baseline)
    assert (unpaired.result, unpaired.code) == ("error", "baseline_not_paired")


# --- stages ---------------------------------------------------------------------------------------


def stage_data() -> RunData:
    entries = [
        EntryState("a.pdf", "1" * 64, "d1", "application/pdf", "done", scan_outcome="clean"),
        EntryState("b.pdf", "2" * 64, "d2", "application/pdf", "failed", scan_outcome="clean"),
        EntryState("c.md", "3" * 64, "d3", "text/markdown", "excluded", scan_outcome="infected"),
    ]
    tasks = [
        TaskState("scan", "succeeded", "a.pdf", {"decision": "continue"}),
        TaskState("scan", "succeeded", "b.pdf", {"decision": "continue"}),
        TaskState("scan", "succeeded", "c.md", {"decision": "exclude"}),
        TaskState("parse", "succeeded", "a.pdf", {"ocr_pages": 2}),
        TaskState("parse", "failed", "b.pdf"),
        TaskState("chunk", "cached", "a.pdf", {"chunk_count": 5, "tokens_max": 700}),
        TaskState("embed", "succeeded", "a.pdf", {"chunks": 5, "truncated_inputs": 1}),
    ]
    return RunData(run_id="r1", run_status="failed", entries=entries, tasks=tasks)


def test_stage_metrics_pass_fail_and_error() -> None:
    data = stage_data()
    rate = check(stage="parse", metric="parse.success_rate", min=0.9)
    assert (evaluate_stage(rate, data).result, evaluate_stage(rate, data).value) == ("fail", 0.5)
    pdf_ok = check(
        stage="parse", metric="parse.ocr_pages", max=2, where={"file_type": "application/pdf"}
    )
    assert evaluate_stage(pdf_ok, data).result == "pass"
    over = check(stage="chunk", metric="chunk.share_over_embed_limit", max=0)
    result = evaluate_stage(over, data)
    assert (result.result, result.value, result.detail) == (
        "fail",
        0.2,
        {"over_limit": 1, "chunks": 5},
    )
    none = check(
        stage="parse", metric="parse.success_rate", min=0.9, where={"file_type": "image/png"}
    )
    assert (evaluate_stage(none, data).result, evaluate_stage(none, data).code) == (
        "error",
        "no_documents",
    )
    assert evaluate_stage(rate, RunData()).code == "run_missing"
    findable = check(stage="parse", metric="parse.evidence_findability", min=0.97)
    assert evaluate_stage(findable, eval_data()).result == "fail"


def test_stage_assertions() -> None:
    data = stage_data()
    succeeded = evaluate_stage(check(stage="parse", **{"assert": "all_succeeded"}), data)
    assert (succeeded.result, succeeded.value) == ("fail", 1.0)
    assert [link.id for link in succeeded.evidence if link.kind == "document"] == ["b.pdf"]
    chunks = evaluate_stage(check(stage="chunk", **{"assert": "all_succeeded"}), data)
    assert chunks.result == "pass"
    infected = evaluate_stage(check(stage="scan", **{"assert": {"no_outcome": ["infected"]}}), data)
    assert (infected.result, infected.detail["by_outcome"]) == ("fail", {"infected": 1})
    clean = evaluate_stage(check(stage="scan", **{"assert": {"no_outcome": ["encrypted"]}}), data)
    assert clean.result == "pass"
    nothing = evaluate_stage(check(stage="enrich", **{"assert": "all_succeeded"}), data)
    assert (nothing.result, nothing.code) == ("error", "no_tasks")


def test_no_findings_reads_what_the_detector_found() -> None:
    safety = check(
        stage="chunk", detector="scan/pii", categories=["ssn"], **{"assert": "no_findings"}
    )
    assert safety.kind == "safety"
    data = stage_data()
    assert evaluate_stage(safety, data).code == "not_scanned"
    data.findings = {"c": {"findings": 0, "by_category": {}, "scanned": 5, "chunks": []}}
    assert evaluate_stage(safety, data).result == "pass"
    data.findings = {
        "c": {"findings": 2, "by_category": {"ssn": 2}, "scanned": 5, "chunks": ["k1"]}
    }
    found = evaluate_stage(safety, data)
    assert (found.result, found.value, found.evidence[-1].id) == ("fail", 2.0, "k1")
    data.findings = {"c": {"error": "run_missing"}}
    assert evaluate_stage(safety, data).code == "run_missing"


def test_no_findings_takes_known_pii_categories_in_a_fixed_order() -> None:
    both = check(
        stage="chunk",
        detector="scan/pii",
        categories=["phone", "ssn", "phone"],
        **{"assert": "no_findings"},
    )
    assert both.categories == ["ssn", "phone"]
    with pytest.raises(SpecInvalid):
        check(
            stage="chunk", detector="scan/pii", categories=["passport"], **{"assert": "no_findings"}
        )


# --- golden queries -------------------------------------------------------------------------------


def passage(
    doc_key: str, text: str = "", version: str | None = "3", chunk: str = "k"
) -> GoldenPassage:
    return GoldenPassage(chunk, doc_key, version, "Holds", text)


def test_golden_queries_find_and_avoid() -> None:
    golden = check(
        query="What's the hold?",
        asker={"roles": ["staff"]},
        expect={
            "in_top_k": 2,
            "doc_key": "funds",
            "version": 3,
            "contains": "FIRST 30 calendar  days",
        },
    )
    found = judge_golden(
        golden, [passage("other"), passage("funds", "In the first 30 calendar days.", chunk="k2")]
    )
    assert (found.result, found.value, found.evidence[0].id) == ("pass", 2.0, "k2")
    late = judge_golden(
        golden, [passage("a"), passage("b"), passage("funds", "first 30 calendar days")]
    )
    assert (late.result, late.code) == ("fail", "not_found")
    wrong_text = judge_golden(golden, [passage("funds", "first 60 calendar days")])
    assert wrong_text.result == "fail"
    avoid = check(query="q", expect_not={"doc_key": "old-fees"})
    assert judge_golden(avoid, [passage("fees")]).result == "pass"
    returned = judge_golden(avoid, [passage("fees"), passage("old-fees", chunk="bad")])
    assert (returned.result, returned.code, returned.detail["unwanted_rank"]) == (
        "fail",
        "returned_unwanted",
        2,
    )
    assert normalized("Café  DÉJÀ\nvu") == "cafe deja vu"


def test_a_golden_result_never_carries_the_query_or_passage_text() -> None:
    canary = new_canary()
    golden = check(query=canary, expect={"doc_key": "funds", "contains": canary})
    result = judge_golden(golden, [passage("funds", f"text {canary} text")])
    assert result.result == "pass"
    assert canary not in result.model_dump_json()


# --- data change and the revision guard -----------------------------------------------------------


def entry(
    name: str, sha: str, *, parse: str = "succeeded", decision: str = "continue"
) -> EntryState:
    return EntryState(
        name, sha * 64, name, "text/markdown", "done", scan_decision=decision, parse_status=parse
    )


def test_data_change_counts_and_holds_to_max() -> None:
    before = RunData(
        run_id="b",
        run_status="succeeded",
        entries=[entry("a", "1"), entry("b", "2"), entry("c", "3")],
    )
    after = RunData(
        run_id="t",
        run_status="succeeded",
        entries=[
            entry("a", "1"),
            entry("b", "9", parse="failed"),
            entry("d", "4", decision="flag"),
        ],
    )
    change = data_change(after.entries, before.entries)
    assert change["counts"] == {
        "added": 1,
        "removed": 1,
        "changed": 1,
        "unparseable": 1,
        "flagged": 1,
    }
    rule = check(type="data_change", max={"removed": 0})
    result = evaluate_data_change(rule, after, before)
    assert (result.result, result.detail["over"]) == ("fail", ["removed"])
    assert evaluate_data_change(check(type="data_change"), after, before).result == "pass"
    assert evaluate_data_change(rule, after, None).result == "skipped"
    assert evaluate_data_change(rule, RunData(), before).code == "run_missing"


def test_the_revision_guard() -> None:
    def data(revision: str) -> RunData:
        return RunData(
            run_id="r", embedding={"component": "embed/gateway", "model": "m", "revision": revision}
        )

    guard = check(allow_change=False)
    assert guard.kind == "revision_guard"
    assert evaluate_revision(guard, data("1"), data("1")).result == "pass"
    changed = evaluate_revision(guard, data("2"), data("1"))
    assert (changed.result, changed.code, changed.detail["changed"]) == (
        "fail",
        "reindex_required",
        ["revision"],
    )
    allowed = evaluate_revision(check(allow_change=True), data("2"), data("1"))
    assert (allowed.result, allowed.code) == ("pass", "reindex_planned")
    assert evaluate_revision(guard, data("1"), None).result == "skipped"
    assert evaluate_revision(guard, RunData(), data("1")).code == "embedding_unknown"


# --- the record and reports -----------------------------------------------------------------------


def record_of(results: list[CheckResult], error: str | None = None) -> TestRunRecord:
    summary = Summary.of(results)
    started = datetime(2026, 10, 31, 14, 2, 12, tzinfo=UTC)
    return TestRunRecord(
        test_run=TestRunInfo(
            id="t1",
            suite=SuiteRef(id="s1", name="policies", version=3, document_hash="0" * 64),
            trigger="ci",
            status=summary.outcome if error is None else "error",
            error_code=error,
            created_by="u",
            created_at=started,
            started_at=started,
            duration_ms=1234.5,
        ),
        runs={"pipeline_run": "r1", "eval_run": None},
        summary=summary,
        outcome=summary.outcome if error is None else None,
        checks=results,
    )


def mixed_results() -> list[CheckResult]:
    data = eval_data(0.9)
    return [
        evaluate_metric(check(metric="recall@5", min=0.85), data),
        evaluate_metric(check(metric="recall@5", min=0.95), data),
        evaluate_metric(check(metric="ndcg@3", min=0.5), data),
        evaluate_revision(check(allow_change=False), data, None),
    ]


def test_the_suite_fails_on_a_failed_or_errored_check() -> None:
    results = mixed_results()
    assert Summary.of(results).model_dump() == {
        "checks": 4,
        "passed": 1,
        "failed": 1,
        "errored": 1,
        "skipped": 1,
    }
    assert Summary.of([results[0], results[3]]).outcome == "passed"
    assert Summary.of([results[0], results[2]]).outcome == "failed"


def test_the_junit_output_validates_and_has_one_case_per_check() -> None:
    document = etree.fromstring(junit_xml(record_of(mixed_results())))
    JUNIT_XSD.assertValid(document)
    suite = document.find("testsuite")
    assert suite is not None
    assert (
        suite.get("tests"),
        suite.get("failures"),
        suite.get("errors"),
        suite.get("skipped"),
    ) == (
        "4",
        "1",
        "1",
        "1",
    )
    cases = suite.findall("testcase")
    assert [case.get("name") for case in cases] == ["c", "c", "c", "c"]
    assert [child.tag for case in cases for child in case] == ["failure", "error", "skipped"]
    assert suite.get("timestamp") == "2026-10-31T14:02:12"


def test_a_run_that_couldnt_finish_is_one_junit_error() -> None:
    document = etree.fromstring(junit_xml(record_of([], error="corpus_not_found")))
    JUNIT_XSD.assertValid(document)
    errors = document.findall("testsuite/testcase/error")
    assert [error.get("type") for error in errors] == ["corpus_not_found"]
    assert document.get("errors") == "1"


def test_the_schema_refuses_what_ci_would_misread() -> None:
    bad = (
        b'<testsuites><testsuite name="s" tests="x" failures="0" errors="0" skipped="0" '
        b'time="0" timestamp="2026-10-31T14:02:12" hostname="h"/></testsuites>'
    )
    assert not JUNIT_XSD.validate(etree.fromstring(bad))


def test_markdown_and_messages_hold_codes_numbers_and_ids() -> None:
    results = mixed_results()
    text = markdown_summary(record_of(results), link_base="https://lab.example/stratafi")
    assert text.startswith("## policies: failed")
    assert "| c | metric | **fail** | 0.9 | >= 0.95 |  | below_min |" in text
    assert "(https://lab.example/stratafi/eval-runs/e1?query=q3)" in text
    assert result_message(results[1]) == "c: fail (below_min) value 0.9 threshold >= 0.95"


def test_the_record_round_trips_as_its_contract() -> None:
    record = record_of(mixed_results())
    again = TestRunRecord.model_validate(json.loads(record.model_dump_json(by_alias=True)))
    assert again.fingerprint() == record.fingerprint()
    assert again.model_dump(by_alias=True)["schema"] == "stratafi.test-run.v1"
