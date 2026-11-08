"""Tracking and comparison in the SDK and the CLI: annotations, the search language, comparing
many eval runs, traces, model comparison, aliases and maps."""

import json
import uuid
from typing import Any

import httpx2
import pytest

from stratafi import Client
from stratafi.cli import main

EVAL_A = str(uuid.UUID(int=11))
EVAL_B = str(uuid.UUID(int=21))
MODEL_ID = "00000000-0000-0000-0000-0000000000c3"
NOW = "2026-11-05T10:00:00Z"


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATAFI_URL", "http://stratafi.test")
    monkeypatch.setenv("STRATAFI_TOKEN", "token")
    monkeypatch.delenv("STRATAFI_DEV_SIGN_IN", raising=False)


def annotation(**changes: Any) -> dict[str, Any]:
    return {
        "title": None,
        "tags": {},
        "note": None,
        "archived": False,
        "archived_at": None,
        "updated_at": NOW,
        **changes,
    }


def tracked(identifier: str, label: str, recall: float, **tags: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "label": label,
        "status": "succeeded",
        "split": "dev",
        "title": None,
        "tags": tags,
        "archived": False,
        "created_by": "u-tenant0-editor",
        "created_at": NOW,
        "finished_at": NOW,
        "run_id": str(uuid.UUID(int=1)),
        "retrieval_config_id": str(uuid.UUID(int=12)),
        "pipeline_spec_id": str(uuid.UUID(int=2)),
        "eval_set_version_id": str(uuid.UUID(int=13)),
        "attributes": {"status": "succeeded"},
        "headline": {"hit@5": 0.9, "recall@5": recall, "mrr@10": 0.6, "ndcg@10": 0.65},
        "latency_p95_ms": 21.5,
        "values": {"params.rerank.component": "rerank/none"},
    }


def search_page(items: list[dict[str, Any]], cursor: str | None = None) -> dict[str, Any]:
    return {
        "items": items,
        "next_cursor": cursor,
        "total": 3,
        "scanned": 3,
        "keys": {"metrics": ["recall@5"], "params": ["rerank.component"], "tags": ["team"]},
    }


def test_search_posts_the_language_and_follows_cursors(api: Any) -> None:
    api.on(
        "POST",
        "/eval-runs/search",
        search_page([tracked(EVAL_A, "hybrid", 0.8), tracked(EVAL_B, "no rerank", 0.7)], "c1"),
        search_page([tracked(str(uuid.UUID(int=31)), "dense", 0.6)]),
    )
    with Client(token="t", url="http://stratafi.test", transport=api.transport) as client:
        found = client.evals.search(
            "metrics.recall@5 > 0.5",
            order_by=["metrics.recall@5 DESC"],
            columns=["params.rerank.component"],
        )
    assert [str(item.id) for item in found][:2] == [EVAL_A, EVAL_B]
    assert len(found) == 3
    first, second = (
        json.loads(request.content) for request in api.sent("POST", "/eval-runs/search")
    )
    assert first["filter"] == "metrics.recall@5 > 0.5"
    assert first["order_by"] == ["metrics.recall@5 DESC"]
    assert first["cursor"] is None
    assert second["cursor"] == "c1"


def test_annotating_sends_only_what_changes(api: Any) -> None:
    api.on("GET", f"/eval-runs/{EVAL_A}", api.eval_run("succeeded"))
    api.on(
        "PATCH",
        f"/eval-runs/{EVAL_A}/annotation",
        annotation(title="Hybrid", tags={"team": "rag"}),
        annotation(archived=True),
    )
    with Client(token="t", url="http://stratafi.test", transport=api.transport) as client:
        run = client.evals.get(EVAL_A)
        found = run.annotate(title="Hybrid", tags={"team": "rag"}, delete_tags=["old"])
        assert found.tags == {"team": "rag"}
        assert run.archive().archived is True
    first, second = (
        json.loads(request.content)
        for request in api.sent("PATCH", f"/eval-runs/{EVAL_A}/annotation")
    )
    assert first == {"title": "Hybrid", "set_tags": {"team": "rag"}, "delete_tags": ["old"]}
    assert second == {"archived": True}


def compare_many_record() -> dict[str, Any]:
    cell = {"mean": 0.7, "low": 0.6, "high": 0.8, "n": 52, "group": "leader", "behind_best": 0.0}
    other = {
        **cell,
        "mean": 0.6,
        "delta": -0.1,
        "delta_low": -0.2,
        "delta_high": 0.0,
        "p_value": 0.04,
        "p_holm": 0.04,
        "wins": 1,
        "losses": 6,
        "ties": 45,
        "group": "worse",
        "behind_best": -0.1,
    }
    run = {
        "id": EVAL_A,
        "label": "hybrid",
        "title": None,
        "tags": {},
        "split": "dev",
        "created_at": NOW,
        "run_id": str(uuid.UUID(int=1)),
        "retrieval_config": {"name": "hybrid-v1", "version": 1},
        "pipeline_spec": {"name": "baseline-v1", "version": 1},
        "corpus": {"name": "halcyon", "version": 1},
        "k_eval": 5,
        "models": {"rerank": "rerank-default"},
    }
    return {
        "runs": [run, {**run, "id": EVAL_B, "label": "no rerank"}],
        "baseline": 0,
        "rule": "knowledge-hit-v1",
        "checkpoint": "final",
        "metrics": ["recall@5"],
        "paired": 52,
        "resamples": 2000,
        "seed": 0,
        "table": [{"metric": "recall@5", "n": 52, "best": 0, "cells": [cell, other]}],
        "curves": [],
        "funnel": {"checkpoints": [], "hit": [[], []], "recall": [[], []]},
        "latency": {"steps": [], "p50": [[], []], "p95": [[], []], "mean": [[], []]},
        "where_lost": {"buckets": [], "counts": [[], []]},
        "violations": [{}, {}],
        "findability": [None, None],
        "slices": [],
        "params": [{"key": "rerank.component", "values": ["rerank/gateway", "rerank/none"]}],
        "queries": [],
        "warnings": [],
    }


def test_compare_many_sends_every_run_and_prints_a_table(
    api: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    api.on("GET", "/eval-comparisons/many", compare_many_record())
    assert (
        main(["eval", "compare", EVAL_A, EVAL_B, "--metric", "recall@5"], transport=api.transport)
        == 0
    )
    (request,) = api.sent("GET", "/eval-comparisons/many")
    assert request.url.params.get_list("ids") == [EVAL_A, EVAL_B]
    assert request.url.params.get_list("metric") == ["recall@5"]
    out = capsys.readouterr().out
    assert "52 paired queries" in out
    assert "0.700 [0.600, 0.800] leader" in out
    assert "Δ -0.100 (p 0.040) worse" in out
    assert "| rerank.component | rerank/gateway | rerank/none |" in out


def trace_record() -> dict[str, Any]:
    span = {
        "parent_id": "search",
        "attributes": {},
        "lists": [],
    }
    return {
        "total_ms": 10.0,
        "spans": [
            {**span, "id": "search", "parent_id": None, "name": "Search", "kind": "search",
             "start_ms": 0.0, "duration_ms": 10.0},
            {**span, "id": "keyword", "name": "Keyword search", "kind": "retriever",
             "start_ms": 0.0, "duration_ms": 4.0,
             "lists": [{"checkpoint": "keyword", "length": 50, "first_relevant_rank": 2,
                        "found": True, "top": []}]},
            {**span, "id": "rerank", "name": "Rerank", "kind": "reranker",
             "start_ms": 4.0, "duration_ms": 6.0},
        ],
        "path": [],
        "lost_at_checkpoint": "final",
        "eval_run_id": EVAL_A,
        "item_id": "q1",
        "rule": "knowledge-hit-v1",
        "has_evidence": True,
        "hit": False,
        "lost_at": "rerank",
        "embedding_revision": "r1",
        "rerank_revision": None,
        "visible_chunks": 120,
    }  # fmt: skip


def test_a_trace_prints_as_a_timeline(api: Any, capsys: pytest.CaptureFixture[str]) -> None:
    api.on("GET", f"/eval-runs/{EVAL_A}", api.eval_run("succeeded"))
    api.on("GET", f"/eval-runs/{EVAL_A}/queries/q1/trace", trace_record())
    assert main(["eval", "trace", EVAL_A, "q1"], transport=api.transport) == 0
    out = capsys.readouterr().out
    assert "| Keyword search | 0.0 | 4.0 |" in out
    assert "keyword 50 (first relevant 2)" in out
    assert "the evidence was dropped at: final" in out


def test_tagging_and_searching_from_the_command_line(
    api: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    api.on("GET", f"/eval-runs/{EVAL_A}", api.eval_run("succeeded"))
    api.on("PATCH", f"/eval-runs/{EVAL_A}/annotation", annotation(title="T", tags={"team": "rag"}))
    assert (
        main(
            ["eval", "tag", EVAL_A, "team=rag", "--delete", "old", "--title", "T"],
            transport=api.transport,
        )
        == 0
    )
    assert api.body("PATCH", f"/eval-runs/{EVAL_A}/annotation") == {
        "title": "T",
        "set_tags": {"team": "rag"},
        "delete_tags": ["old"],
    }
    assert "tags: team=rag" in capsys.readouterr().out
    api.on("POST", "/eval-runs/search", search_page([tracked(EVAL_A, "hybrid", 0.8, team="rag")]))
    code = main(
        ["eval", "search", "tags.team = 'rag'", "--column", "params.rerank.component"],
        transport=api.transport,
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "| hybrid | succeeded | dev | 0.800 | 0.600 | 21.5 | team=rag | rerank/none |" in out


def test_model_comparison_and_aliases(api: Any, capsys: pytest.CaptureFixture[str]) -> None:
    version = {
        "id": str(uuid.UUID(int=13)),
        "eval_set_id": str(uuid.UUID(int=14)),
        "version": 1,
    }
    api.on("GET", f"/eval-sets/versions/{version['id']}", version)
    api.on(
        "GET",
        "/model-comparisons",
        {
            "eval_set_version_id": version["id"],
            "role": "rerank",
            "metric": "recall@5",
            "rule": "knowledge-hit-v1",
            "checkpoint": "final",
            "split": None,
            "eval_runs": 2,
            "models": [
                {
                    "key": "rerank|rerank/gateway|rerank-default|",
                    "label": "rerank-default",
                    "role": "rerank",
                    "component": "rerank/gateway",
                    "name": "rerank-default",
                    "revision": None,
                    "runs": 1,
                    "best_eval_run_id": EVAL_A,
                    "best": {"mean": 0.8, "low": 0.7, "high": 0.9, "n": 52},
                    "latency_p95_ms": 30.0,
                    "egress": "model_gateway",
                }
            ],
            "pairs": [],
            "matrix": [[None]],
            "warnings": ["no_controlled_pairs"],
        },
    )
    assert (
        main(
            ["model", "compare", "--eval-set", version["id"], "--role", "rerank"],
            transport=api.transport,
        )
        == 0
    )
    out = capsys.readouterr().out
    assert (
        "| rerank-default | rerank/gateway | 1 | 0.800 | [0.700, 0.900] | 30.0 | model_gateway |"
        in out
    )
    assert "warnings: no_controlled_pairs" in out
    api.on(
        "PUT",
        f"/trained-models/{MODEL_ID}/aliases/champion",
        {
            "kind": "trained_model",
            "name": "halcyon-adapter",
            "alias": "champion",
            "target_id": MODEL_ID,
            "version": 2,
            "moved_from": str(uuid.UUID(int=99)),
        },
    )
    assert main(["model", "alias", MODEL_ID, "champion"], transport=api.transport) == 0
    assert "halcyon-adapter@champion is version 2, moved from" in capsys.readouterr().out
    api.on("DELETE", f"/trained-models/{MODEL_ID}/aliases/champion", httpx2.Response(204))
    assert main(["model", "alias", MODEL_ID, "champion", "--delete"], transport=api.transport) == 0
    assert api.sent("DELETE", f"/trained-models/{MODEL_ID}/aliases/champion")


def test_a_map_prints_its_mermaid(api: Any, capsys: pytest.CaptureFixture[str]) -> None:
    api.on(
        "GET",
        f"/eval-runs/{EVAL_A}/map",
        {
            "schema": "stratafi.pipeline-map.v1",
            "pipeline": None,
            "retrieval": None,
            "nodes": [],
            "edges": [],
            "summary": {},
            "mermaid": "flowchart LR\n  n_a --> n_b\n",
            "run_id": None,
            "eval_run_id": EVAL_A,
        },
    )
    assert main(["map", "--eval-run", EVAL_A], transport=api.transport) == 0
    assert capsys.readouterr().out == "flowchart LR\n  n_a --> n_b\n"
    assert main(["map"], transport=api.transport) == 2
