"""Custom checks in a suite: their code is stored and hashed with the suite, the
sandbox's output is read as untrusted, and the record and reports name the code by hash."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from stratafi_core.settings import SuiteSettings
from stratafi_core.specs import SpecInvalid
from stratafi_core.suites import (
    MAX_CODE_BYTES,
    MAX_CODE_FILES,
    EvidenceLink,
    SuiteRef,
    Summary,
    TestRunInfo,
    TestRunRecord,
    code_manifest,
    custom_result,
    dump_suite,
    markdown_summary,
    parse_suite,
    suite_hash,
)
from stratafi_testing.canary import new_canary

SOURCE = (
    "from stratafi.checks import CheckResult\n\ndef whole(ctx):\n    return CheckResult.passing()\n"
)


def suite(code: dict[str, str] | None = None, **check: Any) -> dict[str, Any]:
    return {
        "schema": "stratafi.test-suite.v1",
        "name": "tables",
        "target": {"pipeline": "p"},
        "corpus": "c",
        "checks": [{"id": "whole", "python": "checks/tables.py:whole", **check}],
        "code": {"checks/tables.py": SOURCE} if code is None else code,
    }


def custom_check() -> Any:
    return parse_suite(suite()).checks[0]


# --- the code -------------------------------------------------------------------------------------


def test_the_code_is_kept_with_the_suite_and_its_hash_covers_it() -> None:
    parsed = parse_suite(suite(code={"checks/tables.py": SOURCE, "checks/helpers.py": "X = 1\n"}))
    check = parsed.checks[0]
    assert (check.kind, check.path, check.function, check.timeout_s) == (
        "custom",
        "checks/tables.py",
        "whole",
        60,
    )
    assert parsed.custom_checks == [check]
    assert not parsed.needs_eval_run
    dumped = dump_suite(parsed)
    assert list(dumped["code"]) == ["checks/helpers.py", "checks/tables.py"]
    assert dump_suite(parse_suite(dumped)) == dumped
    changed = parse_suite(suite(code={"checks/tables.py": SOURCE + "# a comment\n"}))
    assert suite_hash(changed) != suite_hash(parse_suite(suite()))


def test_the_manifest_names_each_file_by_hash() -> None:
    [item] = code_manifest(parse_suite(suite()))
    assert item["path"] == "checks/tables.py"
    assert item["bytes"] == len(SOURCE.encode())
    assert len(item["sha256"]) == 64


@pytest.mark.parametrize(
    ("code", "path", "error"),
    [
        ({}, "(root)", "code_missing"),
        ({"checks/other.py": SOURCE}, "(root)", "code_missing"),
        ({"checks/tables.py": SOURCE, "../up.py": ""}, "code", "code_path_invalid"),
        ({"checks/tables.py": SOURCE, "/abs.py": ""}, "code", "code_path_invalid"),
        ({"checks/tables.py": SOURCE, "a//b.py": ""}, "code", "code_path_invalid"),
        ({"checks/tables.py": SOURCE, "a/./b.py": ""}, "code", "code_path_invalid"),
        ({"checks/tables.py": SOURCE, "notes.txt": ""}, "code", "code_path_invalid"),
        ({"checks/tables.py": "x" * (MAX_CODE_BYTES + 1)}, "code", "code_too_large"),
        ({"checks/tables.py": "nul\x00"}, "code", "code_not_text"),
        ({"checks/tables.py": "half \ud800 a pair"}, "code", "code_not_text"),
        (
            {"checks/tables.py": SOURCE}
            | {f"more/m{i}.py": "x = 1\n" for i in range(MAX_CODE_FILES)},
            "code",
            "too_many_code_files",
        ),
        (
            {"checks/tables.py": SOURCE}
            | {f"more/m{i}.py": "x" * (MAX_CODE_BYTES - 1) for i in range(4)},
            "code",
            "code_too_large",
        ),
    ],
)
def test_the_code_is_checked(code: dict[str, str], path: str, error: str) -> None:
    with pytest.raises(SpecInvalid) as raised:
        parse_suite(suite(code=code))
    assert (path, error) in [(issue.path, issue.code) for issue in raised.value.issues]


@pytest.mark.parametrize(
    "reference",
    ["checks/tables.py", "tables.py:", "../tables.py:whole", "/t.py:whole", "t.py:1x", "t.sh:f"],
)
def test_a_reference_names_a_relative_file_and_a_function(reference: str) -> None:
    with pytest.raises(SpecInvalid) as raised:
        parse_suite(suite(python=reference))
    assert raised.value.paths == ["checks.0.python"]


def test_the_code_never_appears_in_a_validation_problem() -> None:
    canary = new_canary()
    with pytest.raises(SpecInvalid) as raised:
        parse_suite(suite(code={"checks/tables.py": canary * 4000}))
    assert canary not in json.dumps([vars(issue) for issue in raised.value.issues])


# --- the sandbox's output ------------------------------------------------------------------------


def test_a_verdict_passes_fails_or_skips() -> None:
    check = custom_check()
    evidence = [EvidenceLink(kind="run", id="r1")]
    passed = custom_result(
        check,
        {"result": {"passed": True, "value": 2, "code": None, "detail": {"chunks": ["c1"]}}},
        evidence=evidence,
    )
    assert (passed.result, passed.value, passed.code, passed.detail) == (
        "pass",
        2.0,
        None,
        {"chunks": ["c1"]},
    )
    assert passed.type == "custom"
    assert passed.evidence == evidence
    failed = custom_result(check, {"result": {"passed": False, "code": "tables_split"}})
    assert (failed.result, failed.code) == ("fail", "tables_split")
    assert custom_result(check, {"result": {"passed": False}}).code == "assertion_failed"
    skipped = custom_result(check, {"result": {"passed": None, "code": "no_tables"}})
    assert (skipped.result, skipped.code) == ("skipped", "no_tables")


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"error": "check_raised", "exception": "ZeroDivisionError"}, "check_raised"),
        ({"error": "function_missing"}, "function_missing"),
        ({"error": "made_up"}, "output_invalid"),
        ({"error": 5}, "output_invalid"),
        ({"result": {"passed": "yes"}}, "output_invalid"),
        ({"result": {"passed": True, "value": "high"}}, "output_invalid"),
        ({"result": {"passed": True, "value": True}}, "output_invalid"),
        ({"result": {"passed": True, "value": float("nan")}}, "output_invalid"),
        ({"result": [1]}, "output_invalid"),
        ({}, "output_invalid"),
        ([], "output_invalid"),
    ],
)
def test_output_that_isnt_a_verdict_is_an_error(payload: Any, code: str) -> None:
    result = custom_result(custom_check(), payload)
    assert (result.result, result.code) == ("error", code)


def test_the_output_keeps_only_metadata() -> None:
    canary = new_canary()
    result = custom_result(
        custom_check(),
        {
            "error": "check_raised",
            "exception": f"Error {canary}",
        },
    )
    assert result.detail == {"exception": None}
    result = custom_result(
        custom_check(),
        {
            "result": {
                "passed": False,
                "code": f"Code {canary}",
                "detail": {
                    "chunks": ["c1", f"the chunk said {canary}"],
                    "text": canary,
                    "query_hash": "ab12",
                    f"key {canary}": 1,
                    "n": 2**60,
                    "deep": [[[[[["bottom"]]]]]],
                },
            }
        },
    )
    assert canary not in result.model_dump_json()
    assert result.code == "check_code_invalid"
    assert result.detail == {
        "chunks": ["c1", None],
        "query_hash": "ab12",
        "n": None,
        "deep": [[[[None]]]],
    }
    big = custom_result(
        custom_check(),
        {"result": {"passed": True, "detail": {f"k{i}": ["x" * 199] * 50 for i in range(50)}}},
    )
    assert big.detail == {"truncated": True}


# --- the record and reports ----------------------------------------------------------------------


def test_the_summary_names_the_code_by_hash() -> None:
    check = custom_check()
    results = [custom_result(check, {"result": {"passed": True}})]
    now = datetime(2026, 10, 31, tzinfo=UTC)
    record = TestRunRecord(
        test_run=TestRunInfo(
            id="t1",
            suite=SuiteRef(id="s1", name="tables", version=1, document_hash="h"),
            trigger="api",
            status="passed",
            created_by="u",
            created_at=now,
        ),
        summary=Summary.of(results),
        outcome="passed",
        checks=results,
        code=code_manifest(parse_suite(suite())),
    )
    text = markdown_summary(record)
    assert "Custom check code:" in text
    assert f"`checks/tables.py`: SHA-256 `{record.code[0]['sha256']}`" in text
    assert SOURCE not in text


def test_custom_checks_need_the_sandbox_and_its_review() -> None:
    assert SuiteSettings().custom_checks_blocked() == "sandbox_not_configured"
    configured = SuiteSettings(sandbox={"backend": "docker"})
    assert configured.custom_checks_blocked() == "sandbox_not_reviewed"
    blank = SuiteSettings(sandbox={"backend": "docker", "review": "  "})
    assert blank.custom_checks_blocked() == "sandbox_not_reviewed"
    reviewed = SuiteSettings(sandbox={"backend": "docker", "review": "SEC-1"})
    assert reviewed.custom_checks_blocked() is None
