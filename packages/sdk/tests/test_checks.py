"""Custom checks' API: ``CheckContext`` over a bundle, ``CheckResult`` and what the lab keeps
of it, and the sandbox's entry point (docker/sandbox/run_check.py), run here without its
container."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from stratafi.checks import (
    MAX_DETAIL_BYTES,
    RESULT_MARKER,
    CheckContext,
    CheckResult,
    is_text_field,
)
from stratafi_testing.canary import new_canary

ROOT = Path(__file__).resolve().parents[3]


def rate_tables_not_split(ctx: CheckContext) -> CheckResult:
    """An example check: no rate table is split across chunks."""
    split = [
        c.chunk_id
        for c in ctx.chunks()
        if c.metadata.get("table_part") and "rate" in (c.heading_path[-1:] or [""])[0].lower()
    ]
    return CheckResult.assert_true(not split, value=len(split), detail={"chunks": split[:20]})


def chunk(chunk_id: str, heading: str, **metadata: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_id": "d1",
        "ordinal": int(chunk_id[-1]),
        "text": f"{heading} text",
        "heading_path": ["Rates", heading],
        "metadata": metadata,
        "tokens": {"chars/4": 12},
        "a_field_from_a_later_lab": True,
    }


CONTEXT = {
    "check_id": "rate-tables-kept-whole",
    "suite": {"name": "tables", "version": 3},
    "runs": {"pipeline_run": "r1", "eval_run": "e1"},
    "metrics": {
        "metrics": {
            "knowledge-hit-v1": {"final": {"recall@5": {"mean": 0.8, "low": 0.7, "high": 0.9}}}
        }
    },
}


def test_the_plan_example_passes_when_rate_tables_stay_whole(tmp_path: Path) -> None:
    ctx = CheckContext.from_records(
        tmp_path / "bundle",
        chunks=[chunk("c1", "Savings rate table"), chunk("c2", "Fees", table_part=1)],
        context=CONTEXT,
    )
    result = rate_tables_not_split(ctx)
    assert result.to_dict() == {
        "passed": True,
        "value": 0.0,
        "code": None,
        "detail": {"chunks": []},
    }


def test_the_plan_example_fails_naming_the_split_chunks(tmp_path: Path) -> None:
    ctx = CheckContext.from_records(
        tmp_path,
        chunks=[
            chunk("c1", "Loan rate table", table_part=1),
            chunk("c2", "Loan rate table", table_part=2),
            chunk("c3", "Fees", table_part=1),
        ],
    )
    assert rate_tables_not_split(ctx).to_dict() == {
        "passed": False,
        "value": 2.0,
        "code": "assertion_failed",
        "detail": {"chunks": ["c1", "c2"]},
    }


def test_the_context_reads_the_run_its_documents_and_queries(tmp_path: Path) -> None:
    ctx = CheckContext.from_records(
        tmp_path,
        chunks=[chunk("c1", "Rates")],
        documents=[
            {"document_id": "d1", "name": "rates.pdf", "media_type": "application/pdf"},
            {"document_id": "d2", "name": "old.doc", "state": "stopped", "scan_outcome": "error"},
        ],
        queries=[
            {
                "item_id": "q1",
                "has_evidence": True,
                "tags": ["rates"],
                "file_types": ["application/pdf"],
                "metrics": {"knowledge-hit-v1": {"final": {"recall@5": 1}}},
            }
        ],
        context=CONTEXT,
    )
    assert ctx.check_id == "rate-tables-kept-whole"
    assert ctx.suite == {"name": "tables", "version": 3}
    assert ctx.runs["eval_run"] == "e1"
    assert ctx.metric("recall@5") == 0.8
    assert ctx.metric("recall@10") is None
    assert ctx.metric("recall@5", rule="doc-v1") is None
    [first] = ctx.chunks()
    assert (first.chunk_id, first.heading_path, first.tokens) == (
        "c1",
        ["Rates", "Rates"],
        {"chars/4": 12},
    )
    assert [document.name for document in ctx.documents()] == ["rates.pdf", "old.doc"]
    assert next(iter(ctx.documents())).state is None
    [query] = ctx.queries()
    assert query.value("recall@5") == 1.0
    assert query.value("mrr@10") is None
    assert query.tags == ["rates"]


def test_an_empty_bundle_reads_as_an_empty_run(tmp_path: Path) -> None:
    ctx = CheckContext(tmp_path, scratch=tmp_path / "scratch")
    assert (ctx.check_id, ctx.suite, ctx.runs, ctx.metrics) == (None, {}, {}, {})
    assert list(ctx.chunks()) == list(ctx.documents()) == list(ctx.queries()) == []
    assert ctx.scratch == tmp_path / "scratch"


def test_results_say_pass_fail_or_skip() -> None:
    assert CheckResult.passing(value=3).to_dict() == {
        "passed": True,
        "value": 3.0,
        "code": None,
        "detail": {},
    }
    assert CheckResult.failing("tables_split", value=2).to_dict()["code"] == "tables_split"
    assert CheckResult.assert_true(False, code="too_few").to_dict()["code"] == "too_few"
    assert CheckResult.assert_true(1, value=1).to_dict()["passed"] is True
    assert CheckResult.skip("no_tables").to_dict() == {
        "passed": None,
        "value": None,
        "code": "no_tables",
        "detail": {},
    }


def test_the_lab_keeps_ids_codes_and_numbers_never_text() -> None:
    canary = new_canary()
    result = CheckResult.failing(
        "Not A Code",
        value=float("nan"),
        detail={
            "chunks": ["c1", f"a sentence with {canary}"],
            "text": canary,
            "query_text": canary,
            "chunk_count": 2,
            "bad key!": 1,
            7: "seven",
            "nested": {"deep": {"deeper": {"deepest": {"bottom": {"floor": 1}}}}},
            "share": float("inf"),
            "big": 2**60,
            "flags": (True, None),
            "object": object(),
        },
    )
    kept = result.to_dict()
    assert canary not in json.dumps(kept)
    assert kept["code"] == "check_code_invalid"
    assert kept["value"] is None
    assert kept["detail"] == {
        "chunks": ["c1", None],
        "chunk_count": 2,
        "nested": {"deep": {"deeper": {"deepest": {"bottom": None}}}},
        "share": None,
        "big": None,
        "flags": [True, None],
        "object": None,
    }


def test_a_large_detail_is_truncated() -> None:
    detail = {f"k{i}": ["x" * 199] * 50 for i in range(50)}
    assert CheckResult.passing(detail=detail).to_dict()["detail"] == {"truncated": True}
    assert len(json.dumps(CheckResult.passing(detail={"n": 1}).to_dict())) < MAX_DETAIL_BYTES


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("text", True),
        ("query_text", True),
        ("Passage", True),
        ("chunk_id", False),
        ("chunk_ids", False),
        ("chunks", False),
        ("document_count", False),
        ("value", False),
    ],
)
def test_text_fields_follow_the_labs_rule(name: str, text: bool) -> None:
    assert is_text_field(name) is text


# --- the entry point ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_check() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_check", ROOT / "docker" / "sandbox" / "run_check.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry(
    run_check: Any,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    source: str,
    function: str = "check",
    path: str = "checks/custom.py",
) -> tuple[int, dict[str, Any], str]:
    code = tmp_path / "code"
    (code / path).parent.mkdir(parents=True, exist_ok=True)
    (code / path).write_text(source)
    (code / "checks" / "helpers.py").write_text("LIMIT = 2\n")
    bundle = CheckContext.from_records(
        tmp_path / "bundle", chunks=[chunk("c1", "Rates")], context=CONTEXT
    ).root
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    status = run_check.main(
        [str(code / path), function],
        bundle=str(bundle),
        scratch=str(scratch),
        code_root=str(code),
    )
    output = capsys.readouterr().out
    marked = [line for line in output.splitlines() if line.startswith(RESULT_MARKER)]
    return status, json.loads(marked[-1][len(RESULT_MARKER) :]), output


def test_the_entry_point_prints_the_verdict(
    run_check: Any, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    source = (
        "from stratafi.checks import CheckResult\n"
        "from checks.helpers import LIMIT\n"
        "def check(ctx):\n"
        "    print('a line the check printed')\n"
        "    (ctx.scratch / 'notes').write_text('scratch is writable')\n"
        "    n = sum(1 for _ in ctx.chunks())\n"
        "    return CheckResult.assert_true(n <= LIMIT, value=n, detail={'check': ctx.check_id})\n"
    )
    status, payload, output = entry(run_check, capsys, tmp_path, source)
    assert status == 0
    assert payload == {
        "result": {
            "passed": True,
            "value": 1.0,
            "code": None,
            "detail": {"check": "rate-tables-kept-whole"},
        }
    }
    assert "a line the check printed" in output


@pytest.mark.parametrize(
    ("source", "function", "error"),
    [
        ("def check(ctx):\n    return 1\n", "check", "not_a_check_result"),
        ("def other(ctx):\n    return None\n", "check", "function_missing"),
        ("check = 5\n", "check", "function_missing"),
        ("def check(ctx:\n", "check", "check_raised"),
        ("import sys\ndef check(ctx):\n    sys.exit(0)\n", "check", "check_raised"),
    ],
)
def test_the_entry_point_reports_why_there_is_no_verdict(
    run_check: Any,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    source: str,
    function: str,
    error: str,
) -> None:
    status, payload, _ = entry(run_check, capsys, tmp_path, source, function)
    assert status == 1
    assert payload["error"] == error


def test_a_raised_exception_reports_its_type_never_its_message(
    run_check: Any, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    canary = new_canary()
    source = f"def check(ctx):\n    raise ValueError({canary!r})\n"
    status, payload, output = entry(run_check, capsys, tmp_path, source)
    assert (status, payload) == (1, {"error": "check_raised", "exception": "ValueError"})
    assert canary not in output


def test_the_entry_point_needs_a_file_and_a_function(
    run_check: Any, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert run_check.main([]) == 2
    assert run_check.main([str(tmp_path / "missing.py"), "check"]) == 1
    output = capsys.readouterr().out
    assert '{"error": "usage"}' in output
    assert '{"error": "code_unreadable"}' in output
