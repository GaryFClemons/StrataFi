"""Eval runs end to end against Postgres (integration tier).

The Halcyon demo corpus (``fixtures/demo-corpus``) is imported by its manifest and run through
the default components with the fake gateway's models; ``hybrid-v1`` (BM25 and exact dense
search, RRF over 50, the gateway's reranker, top 8) is evaluated on all 88 of its queries.

- every query is scored, with aggregates and intervals, slices, latency and findability, and
  the ACL and version traps hold (no must-not-return or avoid violations)
- an identical eval run reuses the results: no model calls, and the same numbers
- the test split is locked: reading it needs a confirmation, which is counted
- a missed query's detail shows its passages at every checkpoint and where it was lost
- query text stays out of logs, events and audit events
"""

import shutil
import subprocess
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
import yaml

from stratafi_core.evalruns import EvalRunManifest
from stratafi_core.relevance import DOC, EVIDENCE_COVERAGE, KNOWLEDGE_HIT
from stratafi_testing.canary import CanaryCheck
from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import (
    HALCYON_RETRIEVAL,
    WEB,
    Lab,
    baseline_pipeline,
    evaluate,
    hybrid_retrieval,
    prepare_halcyon,
    running_lab,
    serve_web,
)

pytestmark = pytest.mark.integration

RETRIEVAL_YAML = HALCYON_RETRIEVAL.read_text(encoding="utf-8")
# A question answered only by Halcyon Rewards v4, which takes effect on 2027-01-01: on the eval
# set's day (2026-11-01) no retriever may return its evidence.
FUTURE_JSONL = (
    '{"as_of": "2026-11-01", "askers": {"staff": {"roles": ["staff"], "groups": ["all-staff"]}}}\n'
    '{"id": "future-reusable-cup", "query": "Do Rewards guests get bonus points for bringing '
    'their own reusable cup?", "asker": "staff", "expected": [{"doc_key": "loyalty", '
    '"version": "4", "section": "2 Earning points", "evidence": "5 bonus points on any visit '
    'when they bring their own reusable cup"}]}\n'
)


def calls(lab: Lab) -> Counter[str]:
    return Counter(record.endpoint for record in lab.gateway.records)


def test_the_baseline_evaluates_all_88_halcyon_queries(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """The baseline pipeline with hybrid-v1 (fake models) evaluates all 88 Halcyon
    queries, and an identical eval run reuses the cached results."""
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs = prepare_halcyon(lab)
        before = calls(lab)
        first = evaluate(lab, inputs, split="all", confirm_test_read=True)
        assert first["status"] == "succeeded", first
        assert first["counts"]["queries"] == 88
        assert first["results_cached"] is False
        metrics = first["metrics"]
        assert (metrics["queries"], metrics["with_evidence"], metrics["abstention"]) == (88, 67, 21)
        for rule in (KNOWLEDGE_HIT, EVIDENCE_COVERAGE, DOC):
            for checkpoint in ("dense", "keyword", "fused", "candidates", "reranked", "final"):
                recall = metrics["metrics"][rule][checkpoint]["recall@5"]
                assert recall["low"] <= recall["mean"] <= recall["high"]
                assert recall["n"] == 67
        # The ACL and the version traps hold for every query.
        assert metrics["violations"]["must_not_return"] == {"count": 0, "of": 88}
        assert metrics["violations"]["avoid"]["count"] == 0
        lost = metrics["where_lost"]
        assert lost["found"] + lost["parse"] + lost["retrieval"] + lost["rerank"] == 67
        # Every expected version is in effect on as_of and readable by its asker.
        assert lost["filtered"] == 0
        assert metrics["findability"]["rate"] == 1.0  # Markdown keeps every evidence item
        assert set(metrics["findability"]["by_file_type"]) == {"text/markdown"}
        assert "parse/markdown" in metrics["findability"]["by_parser"]
        assert "pto-leave" in metrics["slices"]["tag"]
        assert set(metrics["slices"]["file_type"]) == {"text/markdown"}  # abstentions have none
        assert metrics["latency_ms"]["total_ms"]["p95"] >= metrics["latency_ms"]["total_ms"]["p50"]
        assert metrics["cost"]["providers"]["model_gateway"]["calls"] > 0
        made = calls(lab) - before
        assert made["rerank"] > 0
        # One embedding per distinct query text: the "--as-staff" copies reuse their twin's.
        distinct = {item["query"] for item in yaml.safe_load(RETRIEVAL_YAML)["queries"]}
        assert len(distinct) == 81
        assert made["embeddings"] == 81

        rows = lab.get(f"/eval-runs/{first['id']}/queries", limit=200)["items"]
        assert len(rows) == 88
        assert {row["split"] for row in rows} == {"dev", "test"}
        abstentions = [row for row in rows if not row["has_evidence"]]
        assert len(abstentions) == 21
        assert all(row["hit"] is None for row in abstentions)
        assert all(row["score"]["near_misses"] for row in abstentions)
        misses = lab.get(f"/eval-runs/{first['id']}/queries", outcome="miss", limit=200)["items"]
        assert all(row["lost_at"] in ("retrieval", "rerank") for row in misses)
        # The reusable-cup bonus is only in Rewards v4, not in effect on 2026-11-01: an
        # abstention, and v4 never comes back for it.
        cup = next(row for row in rows if row["item_id"] == "version-reusable-cup-bonus")
        assert (cup["has_evidence"], cup["hit"]) == (False, None)
        assert all(not any(row["score"]["hidden"]) for row in rows)

        manifest = EvalRunManifest.model_validate(lab.get(f"/eval-runs/{first['id']}/manifest"))
        assert manifest.configuration.retrieval_ref["name"] == "hybrid-v1"
        assert manifest.inputs.eval_set["items"] == 88
        assert {model.role for model in manifest.models} == {"embed", "rerank"}
        rerank = next(model for model in manifest.models if model.role == "rerank")
        assert rerank.revision == lab.gateway.aliases["rerank-default"].revision
        assert len(manifest.execution.shards) == 4  # 88 queries, 25 a shard
        assert [egress.provider for egress in manifest.egress] == ["model_gateway"]

        # An identical eval run reuses everything: no model calls, the same numbers.
        before = calls(lab)
        second = evaluate(lab, inputs, split="all", confirm_test_read=True)
        assert second["status"] == "succeeded"
        assert second["results_cached"] is True
        assert calls(lab) - before == Counter({"models": 1})  # the reranker's revision
        assert second["metrics"]["metrics"] == metrics["metrics"]
        again = lab.get(f"/eval-runs/{second['id']}/queries", limit=200)["items"]
        assert [row["score"]["metrics"] for row in again] == [
            row["score"]["metrics"] for row in rows
        ]

        # Only the dev split: no confirmation needed, and fewer queries.
        dev = evaluate(lab, inputs, split="dev")
        assert dev["status"] == "succeeded"
        assert dev["test_read"] is False
        assert dev["counts"]["queries"] == 62

        # The test split is locked: a read needs confirm_test_read, and is counted.
        version = f"/eval-sets/{inputs['eval_set_id']}/versions/1"
        reads = lab.get(version)["test_reads"]
        assert reads == 2  # the two runs over "all"
        refused = lab.client.post(
            "/api/v1/eval-runs",
            json={k: v for k, v in inputs.items() if k != "eval_set_id"} | {"split": "test"},
            headers=lab.editor,
        )
        assert (refused.status_code, refused.json()["code"]) == (409, "test_split_locked")
        assert lab.get(version)["test_reads"] == 2
        test = evaluate(lab, inputs, split="test", confirm_test_read=True)
        assert test["counts"]["queries"] == 26
        assert lab.get(version)["test_reads"] == 3

        # A missed query's detail: its passages at every checkpoint, evidence marked.
        missed = misses[0]
        detail = lab.get(f"/eval-runs/{first['id']}/queries/{missed['item_id']}")
        assert detail["result"]["lost_at"] == missed["lost_at"]
        assert detail["item"]["query"]
        assert set(detail["checkpoints"]) == {
            "dense", "keyword", "fused", "candidates", "reranked", "final"
        }  # fmt: skip
        assert len(detail["checkpoints"]["final"]) <= 8
        hit = next(row for row in rows if row["hit"])
        found = lab.get(f"/eval-runs/{first['id']}/queries/{hit['item_id']}")
        relevant = [p for p in found["checkpoints"]["final"] if p["relevant"][KNOWLEDGE_HIT]]
        assert relevant
        marked = [p for p in relevant if p["highlights"]]
        for passage in marked:
            for start, end in passage["highlights"]:
                assert 0 <= start < end <= len(passage["text"])

        # A search as the asker, as of the eval set's day, through the same engine: the
        # executives-only memo stays out of staff's results and leads an executive's.
        memo = "executive/expansion-2027-v1.md"
        question = "When will the Harbor Point café close, and when will its staff be told?"
        searched = lab.post(
            "/search",
            {
                "run_id": inputs["run_id"],
                "query": question,
                "retrieval_config_id": inputs["retrieval_config_id"],
                "asker": {"name": "staff", "roles": ["staff"], "groups": ["all-staff"]},
                "as_of": "2026-11-01",
            },
        )
        assert all(hit["entry"] != memo for hit in searched["results"])
        as_exec = lab.post(
            "/search",
            {
                "run_id": inputs["run_id"],
                "query": question,
                "retrieval_config_id": inputs["retrieval_config_id"],
                "asker": {
                    "name": "exec",
                    "roles": ["exec"],
                    "groups": ["all-staff", "managers", "executives"],
                },
                "as_of": "2026-11-01",
            },
        )
        assert as_exec["results"][0]["entry"] == memo
        assert set(as_exec["checkpoints"]) == {
            "dense",
            "keyword",
            "fused",
            "candidates",
            "reranked",
        }

        # Evidence in a version not yet in effect: the filter hides it, and the miss says so.
        future = lab.import_eval_set("halcyon-future", FUTURE_JSONL, format="jsonl")
        hidden = evaluate(
            lab,
            inputs,
            eval_set_version_id=future["version"]["id"],
            split="all",
            confirm_test_read=True,
        )
        assert hidden["status"] == "succeeded", hidden
        assert hidden["metrics"]["where_lost"]["filtered"] == 1
        (row,) = lab.get(f"/eval-runs/{hidden['id']}/queries")["items"]
        assert (row["hit"], row["lost_at"]) == (False, "retrieval")
        assert row["score"]["hidden"] == [True]


def test_eval_runs_check_their_inputs_and_keep_query_text_out_of_logs(
    migrated_engine: sa.Engine,
    dev_identity: DevIdentity,
    tmp_path: Path,
    serve_app: Any,
    canary: CanaryCheck,
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        corpus_version = lab.import_manifest("halcyon", "demo-corpus/corpus")
        spec_id = lab.save_spec(baseline_pipeline())
        until_chunk = lab.wait_run(lab.start_run(spec_id, corpus_version, until="chunk")["id"])
        config = lab.save_retrieval(hybrid_retrieval())
        jsonl = (
            '{"askers": {"staff": {"roles": ["staff"], "groups": ["all-staff"]}}}\n'
            '{"id": "q1", "query": "' + canary.value + ' tip jar counts", "asker": '
            '"staff", "expected": [{"doc_key": "tips", "version": "3"}]}\n'
        )
        draft = lab.import_eval_set("canary", jsonl, format="jsonl", freeze=False)
        body = {
            "run_id": until_chunk["id"],
            "retrieval_config_id": config,
            "eval_set_version_id": draft["version"]["id"],
        }
        not_indexed = lab.client.post("/api/v1/eval-runs", json=body, headers=lab.editor)
        assert (not_indexed.status_code, not_indexed.json()["code"]) == (409, "run_not_indexed")
        indexed = lab.wait_run(lab.start_run(spec_id, corpus_version)["id"])
        body["run_id"] = indexed["id"]
        not_frozen = lab.client.post("/api/v1/eval-runs", json=body, headers=lab.editor)
        assert (not_frozen.status_code, not_frozen.json()["code"]) == (409, "eval_set_not_frozen")
        lab.post(f"/eval-sets/{draft['eval_set']['id']}/versions/1/freeze")
        viewer = dev_identity.headers(role="viewer")
        assert lab.client.post("/api/v1/eval-runs", json=body, headers=viewer).status_code == 403
        started = lab.post("/eval-runs", body | {"split": "all", "confirm_test_read": True})
        finished = lab.wait_eval(started["id"])
        assert finished["status"] == "succeeded", finished
        audit = lab.client.get("/api/v1/audit-events", headers=dev_identity.headers(role="owner"))
        canary.assert_absent(audit.text, str(finished))
        canary.assert_not_logged()


def test_cancelling_an_eval_run_stops_its_shards(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs = prepare_halcyon(lab)
        started = lab.post(
            "/eval-runs",
            {k: v for k, v in inputs.items() if k != "eval_set_id"}
            | {"split": "dev", "options": {"shard_size": 1}},
            expect=(202,),
        )
        cancelled = lab.post(f"/eval-runs/{started['id']}/cancel")
        assert cancelled["status"] in ("cancelling", "cancelled")
        final = lab.wait_eval(started["id"])
        assert final["status"] == "cancelled"
        with migrated_engine.connect() as conn:
            active = conn.execute(
                sa.text(
                    "SELECT count(*) FROM stratafi.eval_shards WHERE eval_run_id = :id "
                    "AND status IN ('queued', 'running')"
                ),
                {"id": uuid.UUID(started["id"])},
            ).scalar_one()
        assert active == 0


@pytest.mark.browser
def test_the_results_screen_in_a_browser(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Open the results, drill into a missed query, and see the checkpoint where its
    evidence was lost (web/e2e/eval-flow.mjs)."""
    node = shutil.which("node")
    if node is None:
        pytest.fail("the browser flow needs Node 24 and Playwright")
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs = prepare_halcyon(lab)
        finished = evaluate(lab, inputs, split="all", confirm_test_read=True)
        assert finished["status"] == "succeeded"
        url = serve_web(lab, serve_app)
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [node, str(WEB / "e2e" / "eval-flow.mjs"), url, finished["id"]],
            capture_output=True,
            text=True,
            timeout=240,
            cwd=WEB,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("ok"), result.stdout
