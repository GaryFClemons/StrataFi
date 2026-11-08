"""Comparing eval runs against Postgres (integration and browser tiers).

Two retrieval configs over the same run and eval set version: hybrid-v1 with the
gateway's reranker, and the same without a reranker. The comparison pairs the 88 queries, and
says which configuration change made the difference.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import (
    HALCYON_RETRIEVAL,
    WEB,
    Lab,
    evaluate,
    hybrid_retrieval,
    prepare_halcyon,
    running_lab,
    serve_web,
)

pytestmark = pytest.mark.integration


def two_runs(lab: Lab) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    inputs = prepare_halcyon(lab)
    first = evaluate(lab, inputs, split="all", confirm_test_read=True)
    unranked = lab.save_retrieval(hybrid_retrieval("no-rerank", rerank="rerank/none"))
    second = evaluate(
        lab, inputs | {"retrieval_config_id": unranked}, split="all", confirm_test_read=True
    )
    assert (first["status"], second["status"]) == ("succeeded", "succeeded")
    return first, second, inputs


def test_two_eval_runs_compare_query_by_query(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        first, second, inputs = two_runs(lab)
        compared = lab.get("/eval-comparisons", a=first["id"], b=second["id"])
        assert compared["paired"] == 67
        assert [row["metric"] for row in compared["metrics"]] == [
            "hit@5", "recall@5", "mrr@10", "ndcg@10"
        ]  # fmt: skip
        for row in compared["metrics"]:
            assert row["n"] == 67
            assert row["low"] <= row["delta"] <= row["high"]
            assert 0 < row["p_value"] <= 1
            assert row["p_holm"] >= row["p_value"]
            assert row["wins"] + row["losses"] + row["ties"] == 67
        assert len(compared["queries"]) == 67
        assert compared["retrieval_diff"] == [
            {
                "path": "rerank.component",
                "op": "changed",
                "before": "rerank/gateway",
                "after": "rerank/none",
            },
            {
                "path": "rerank.params.alias",
                "op": "removed",
                "before": "rerank-default",
                "after": None,
            },
        ]
        assert compared["pipeline_diff"] == []
        assert compared["warnings"] == []

        # A run against itself: no difference, a p-value of 1.
        same = lab.get("/eval-comparisons", a=first["id"], b=first["id"], metric=["recall@5"])
        (row,) = same["metrics"]
        assert (row["delta"], row["low"], row["high"], row["p_value"]) == (0.0, 0.0, 0.0, 1.0)

        # Only runs of one eval set version pair; a different split pairs what it shares.
        dev = evaluate(lab, inputs, split="dev")
        partial = lab.get("/eval-comparisons", a=first["id"], b=dev["id"])
        assert partial["paired"] == 47
        # Fewer than 50 paired queries: the comparison warns that its intervals are wide.
        assert partial["warnings"] == ["few_queries", "different_splits"]
        other = lab.import_eval_set(
            "halcyon-other", HALCYON_RETRIEVAL.read_text(encoding="utf-8"), seed=1
        )
        elsewhere = evaluate(
            lab, inputs | {"eval_set_version_id": other["version"]["id"]}, split="dev"
        )
        refused = lab.client.get(
            "/api/v1/eval-comparisons",
            params={"a": first["id"], "b": elsewhere["id"]},
            headers=lab.editor,
        )
        assert (refused.status_code, refused.json()["code"]) == (409, "not_paired")
        bad = lab.client.get(
            "/api/v1/eval-comparisons",
            params={"a": first["id"], "b": second["id"], "metric": "f1@5"},
            headers=lab.editor,
        )
        assert (bad.status_code, bad.json()["code"]) == (400, "unknown_metric")


@pytest.mark.browser
def test_the_compare_screen_in_a_browser(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """The compare screen shows deltas with CIs (web/e2e/compare-flow.mjs)."""
    node = shutil.which("node")
    if node is None:
        pytest.fail("the browser flow needs Node 24 and Playwright")
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        first, second, _ = two_runs(lab)
        url = serve_web(lab, serve_app)
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [node, str(WEB / "e2e" / "compare-flow.mjs"), url, first["id"], second["id"]],
            capture_output=True,
            text=True,
            timeout=240,
            cwd=WEB,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("ok"), result.stdout
