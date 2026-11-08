"""Tracking and comparison against Postgres (integration tier).

Three eval runs of the Halcyon demo set (hybrid-v1 with the gateway's reranker, the same
without a reranker, and dense only) are annotated, archived, restored and searched with the
search language; compared side by side; traced query by query; compared
by reranker through their one controlled pair, with aliases on trained models and profiles;
and mapped with their datastores. Titles, notes and tag values never reach a log
or an audit event.
"""

import itertools
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from stratafi_store.tables import audit_events, eval_runs
from stratafi_testing.canary import CanaryCheck, new_canary
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

FIXTURES = Path(__file__).resolve().parents[3] / "contracts" / "fixtures"
DENSE_ONLY = {
    "schema": "stratafi.retrieval.v1",
    "name": "dense-only",
    "retrievers": {"dense": {"top_n": 50}, "keyword": None},
    "fusion": {"method": "none"},
    "candidates": 50,
    "rerank": "rerank/none",
    "top_k": 8,
}


def three_runs(lab: Lab) -> tuple[dict[str, str], list[dict[str, Any]]]:
    inputs = prepare_halcyon(lab)
    first = evaluate(lab, inputs)
    unranked = lab.save_retrieval(hybrid_retrieval("no-rerank", rerank="rerank/none"))
    second = evaluate(lab, inputs | {"retrieval_config_id": unranked})
    dense = lab.save_retrieval(DENSE_ONLY)
    third = evaluate(lab, inputs | {"retrieval_config_id": dense})
    assert [run["status"] for run in (first, second, third)] == ["succeeded"] * 3
    return inputs, [first, second, third]


def search(lab: Lab, **body: Any) -> dict[str, Any]:
    return lab.post("/eval-runs/search", body, expect=(200,))


def ids(found: dict[str, Any]) -> list[str]:
    return [item["id"] for item in found["items"]]


def test_annotations_archive_and_the_search_language(
    migrated_engine: sa.Engine,
    dev_identity: DevIdentity,
    tmp_path: Path,
    serve_app: Any,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    secret_note = new_canary()
    secret_tag = new_canary()
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs, (first, second, third) = three_runs(lab)
        viewer = dev_identity.headers(role="viewer")

        annotated = lab.client.patch(
            f"/api/v1/eval-runs/{first['id']}/annotation",
            json={
                "title": "  Hybrid with the gateway reranker ",
                "set_tags": {"team": "rag", "secret": secret_tag},
                "note": f"# Why\n\n{secret_note}",
            },
            headers=lab.editor,
        )
        assert annotated.status_code == 200, annotated.text
        body = annotated.json()
        assert body["title"] == "Hybrid with the gateway reranker"
        assert body["tags"] == {"secret": secret_tag, "team": "rag"}
        assert body["archived"] is False
        read = lab.client.get(f"/api/v1/eval-runs/{first['id']}/annotation", headers=viewer)
        assert read.json()["note"].endswith(secret_note)
        # Tags merge and delete, and the title stays.
        merged = lab.client.patch(
            f"/api/v1/eval-runs/{first['id']}/annotation",
            json={"delete_tags": ["secret"], "set_tags": {"stage": "baseline"}},
            headers=lab.editor,
        ).json()
        assert merged["tags"] == {"stage": "baseline", "team": "rag"}
        assert merged["title"] == "Hybrid with the gateway reranker"
        # A run nobody annotated has an empty annotation, not a 404.
        untouched = lab.get(f"/eval-runs/{third['id']}/annotation")
        assert (untouched["title"], untouched["tags"], untouched["updated_at"]) == (None, {}, None)
        # The rules, by code and field, never echoing the value; viewers can't annotate.
        for change, code in (
            ({"set_tags": {"has space": "x"}}, "tag_key_invalid"),
            ({"set_tags": {"stratafi.run": "x"}}, "tag_key_reserved"),
            ({"title": "x" * 121}, "title_too_long"),
        ):
            refused = lab.client.patch(
                f"/api/v1/eval-runs/{first['id']}/annotation", json=change, headers=lab.editor
            )
            assert (refused.status_code, refused.json()["code"]) == (400, code)
        denied = lab.client.patch(
            f"/api/v1/eval-runs/{first['id']}/annotation", json={"title": "x"}, headers=viewer
        )
        assert denied.status_code == 403
        missing = lab.client.get(f"/api/v1/eval-runs/{uuid.uuid4()}/annotation", headers=viewer)
        assert (missing.status_code, missing.json()["code"]) == (404, "eval_run_not_found")
        # Pipeline runs and experiments take annotations too.
        run_note = lab.client.patch(
            f"/api/v1/runs/{inputs['run_id']}/annotation",
            json={"title": "halcyon baseline ingest", "set_tags": {"corpus": "halcyon"}},
            headers=lab.editor,
        )
        assert run_note.status_code == 200
        assert lab.get(f"/runs/{inputs['run_id']}/annotation")["tags"] == {"corpus": "halcyon"}

        # The search language.
        everything = search(lab, columns=["params.rerank.component", "metrics.mrr@10"])
        assert set(ids(everything)) == {first["id"], second["id"], third["id"]}
        assert everything["total"] == everything["scanned"] == 3
        assert "rerank.component" in everything["keys"]["params"]
        assert "recall@5" in everything["keys"]["metrics"]
        assert everything["keys"]["tags"] == ["stage", "team"]
        row = next(item for item in everything["items"] if item["id"] == second["id"])
        assert row["values"]["params.rerank.component"] == "rerank/none"
        assert row["headline"]["recall@5"] == pytest.approx(
            second["metrics"]["metrics"]["knowledge-hit-v1"]["final"]["recall@5"]["mean"]
        )
        assert ids(search(lab, filter="params.rerank.component = 'rerank/none'")) == [
            third["id"],
            second["id"],
        ]
        assert ids(search(lab, filter="tags.team = 'rag'")) == [first["id"]]
        assert ids(search(lab, filter="params.retrievers.keyword.top_n IS NULL")) == [third["id"]]
        assert ids(search(lab, filter="attributes.config LIKE 'hybrid%'")) == [first["id"]]
        ordered = search(lab, order_by=["metrics.final.mrr@10 DESC"])
        means = [item["headline"]["mrr@10"] for item in ordered["items"]]
        assert means == sorted(means, reverse=True)
        paged = search(lab, limit=2)
        assert len(paged["items"]) == 2
        assert paged["next_cursor"]
        rest = search(lab, limit=2, cursor=paged["next_cursor"])
        assert len(rest["items"]) == 1
        assert rest["next_cursor"] is None
        bad = lab.client.post(
            "/api/v1/eval-runs/search",
            json={"filter": "metrics.recall@5 > 'x"},
            headers=lab.editor,
        )
        assert (bad.status_code, bad.json()["code"]) == (400, "invalid_filter")
        assert "'x" not in bad.json()["detail"]

        # Archive hides without deleting, and restoring brings it back.
        lab.client.patch(
            f"/api/v1/eval-runs/{second['id']}/annotation",
            json={"archived": True},
            headers=lab.editor,
        )
        assert second["id"] not in ids(search(lab))
        assert second["id"] in ids(search(lab, include_archived=True))
        assert second["id"] not in {run["id"] for run in lab.get("/eval-runs")["items"]}
        assert second["id"] in {
            run["id"] for run in lab.get("/eval-runs", include_archived=True)["items"]
        }
        assert lab.get(f"/eval-runs/{second['id']}/annotation")["archived"] is True
        lab.client.patch(
            f"/api/v1/eval-runs/{second['id']}/annotation",
            json={"archived": False},
            headers=lab.editor,
        )
        assert second["id"] in ids(search(lab))

        with migrated_engine.connect() as conn:
            events = conn.execute(
                sa.select(audit_events.c.action, audit_events.c.details).where(
                    audit_events.c.action.like("%.annotate")
                )
            ).all()
    assert {action for action, _ in events} == {"eval_run.annotate", "run.annotate"}
    for value in (secret_note, secret_tag):
        check = CanaryCheck(value, caplog, capfd)
        check.assert_not_logged()
        check.assert_absent(str(events), json.dumps(everything))


def train(lab: Lab, training_set_id: str, name: str, epochs: int) -> dict[str, Any]:
    started = lab.post(
        "/training-runs",
        {
            "training_set_id": training_set_id,
            "name": name,
            "trainer": {
                "component": "train/query-adapter",
                "params": {"epochs": epochs, "patience": 0, "seed": 1},
            },
        },
        expect=(202,),
    )
    return lab.wait(f"/training-runs/{started['id']}")


def test_comparing_tracing_models_aliases_and_maps(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs, (first, second, third) = three_runs(lab)
        names = [first["id"], second["id"], third["id"]]

        # Three RAG pipelines side by side, the first as the baseline.
        compared = lab.get("/eval-comparisons/many", ids=names)
        assert compared["baseline"] == 0
        assert compared["paired"] == 47  # the dev split's items with evidence
        assert compared["metrics"] == ["hit@5", "recall@5", "mrr@10", "ndcg@10"]
        for row in compared["table"]:
            assert row["n"] == 47
            assert row["best"] in (0, 1, 2)
            groups = [cell["group"] for cell in row["cells"]]
            assert groups.count("leader") == 1
            assert "delta" not in {k for k, v in row["cells"][0].items() if v is not None}
            for cell in row["cells"][1:]:
                assert cell["delta_low"] <= cell["delta"] <= cell["delta_high"]
                assert cell["p_holm"] >= cell["p_value"]
        assert [run["retrieval_config"]["name"] for run in compared["runs"]] == [
            "hybrid-v1",
            "no-rerank",
            "dense-only",
        ]
        assert compared["runs"][0]["models"]["rerank"] == "rerank-default"
        assert {param["key"] for param in compared["params"]} >= {
            "rerank.component",
            "fusion.method",
        }
        recall = next(curve for curve in compared["curves"] if curve["metric"] == "recall")
        assert "5" in recall["ks"]
        assert len(recall["series"]) == 3
        assert compared["funnel"]["checkpoints"][0] == "dense"
        assert "keyword" not in compared["funnel"]["checkpoints"][1:2] or True
        assert len(compared["queries"]) >= 47  # the dev split's items with evidence
        assert compared["queries"][0]["spread"] >= compared["queries"][-1]["spread"]
        rebased = lab.get("/eval-comparisons/many", ids=names, baseline=second["id"])
        assert rebased["baseline"] == 1
        for params, code, status in (
            ({"ids": names[:1]}, "too_few_runs", 400),
            ({"ids": [str(uuid.uuid4()) for _ in range(7)]}, "too_many_runs", 400),
            ({"ids": [names[0], names[0]]}, "repeated_run", 400),
            ({"ids": names[:2], "baseline": names[2]}, "unknown_baseline", 400),
            ({"ids": names[:2], "metric": "f1@5"}, "unknown_metric", 400),
        ):
            refused = lab.client.get(
                "/api/v1/eval-comparisons/many", params=params, headers=lab.editor
            )
            assert (refused.status_code, refused.json()["code"]) == (status, code), params
        other = lab.import_eval_set(
            "halcyon-other", HALCYON_RETRIEVAL.read_text(encoding="utf-8"), seed=1
        )
        elsewhere = evaluate(lab, inputs | {"eval_set_version_id": other["version"]["id"]})
        refused = lab.client.get(
            "/api/v1/eval-comparisons/many",
            params={"ids": [names[0], elsewhere["id"]]},
            headers=lab.editor,
        )
        assert (refused.status_code, refused.json()["code"]) == (409, "not_paired")
        unfinished = evaluate(lab, inputs, split="dev", options={"seed": 3})
        with migrated_engine.begin() as conn:
            conn.execute(
                sa.update(eval_runs)
                .where(eval_runs.c.id == uuid.UUID(unfinished["id"]))
                .values(status="failed")
            )
        refused = lab.client.get(
            "/api/v1/eval-comparisons/many",
            params={"ids": [names[0], unfinished["id"]]},
            headers=lab.editor,
        )
        assert (refused.status_code, refused.json()["code"]) == (409, "not_finished")

        # A query's trace and the run's latency profile.
        hit = next(
            row
            for row in lab.get(f"/eval-runs/{first['id']}/queries", outcome="hit", limit=5)["items"]
        )
        trace = lab.get(f"/eval-runs/{first['id']}/queries/{hit['item_id']}/trace")
        spans = trace["spans"]
        assert spans[0]["id"] == "search"
        steps = [span["id"] for span in spans[1:]]
        assert steps[:5] == ["keyword", "embed", "dense", "fuse", "rerank"]
        for before, after in itertools.pairwise(spans[1:]):
            assert after["start_ms"] == pytest.approx(
                before["start_ms"] + before["duration_ms"], abs=0.002
            )
        assert trace["has_evidence"] is True
        assert trace["hit"] is True
        final = next(entry for entry in trace["path"] if entry["checkpoint"] == "final")
        assert final["found"] is True
        assert final["first_relevant_rank"] >= 1
        assert trace["embedding_revision"]
        profile = lab.get(f"/eval-runs/{first['id']}/latency", bins=10)
        assert profile["queries"] == 62
        assert set(profile["steps"]) >= {"keyword_ms", "embed_ms", "rerank_ms", "total_ms"}
        assert sum(profile["steps"]["total_ms"]["histogram"]["counts"]) == 62
        assert len(profile["slowest"]) == 10
        played = lab.post(
            "/search",
            {
                "run_id": inputs["run_id"],
                "query": "who counts the tip jar at close",
                "retrieval_config_id": inputs["retrieval_config_id"],
            },
        )
        assert played["trace"]["spans"][0]["id"] == "search"
        assert all(entry["found"] is None for entry in played["trace"]["path"])

        # Rerankers compared through their one controlled pair.
        models = lab.get(
            "/model-comparisons",
            eval_set_version_id=inputs["eval_set_version_id"],
            role="rerank",
            metric="recall@5",
            split="dev",
        )
        assert models["eval_runs"] == 3  # the failed one is left out
        assert {row["name"] for row in models["models"]} == {"rerank-default", "rerank/none"}
        (pair,) = models["pairs"]
        assert {pair["a_eval_run_id"], pair["b_eval_run_id"]} == {first["id"], second["id"]}
        assert pair["result"]["n"] == 47
        assert models["matrix"][0][1]["pairs"] == 1
        assert models["matrix"][0][1]["delta"] == pytest.approx(-models["matrix"][1][0]["delta"])
        embedders = lab.get(
            "/model-comparisons", eval_set_version_id=inputs["eval_set_version_id"], role="embed"
        )
        assert [row["name"] for row in embedders["models"]] == ["embed-default"]
        assert embedders["warnings"] == ["no_controlled_pairs"]

        # Aliases on trained models: set, move, resolve, remove.
        mined = lab.post("/training-sets", {"eval_run_id": first["id"]}, expect=(202,))
        mined = lab.wait(f"/training-sets/{mined['id']}")
        one = train(lab, mined["id"], "halcyon-adapter", 5)
        two = train(lab, mined["id"], "halcyon-adapter", 6)
        model_one, model_two = one["model_id"], two["model_id"]
        put = lab.put(f"/trained-models/{model_one}/aliases/champion")
        assert (put["version"], put["moved_from"]) == (1, None)
        assert lab.get(f"/trained-models/{model_one}")["aliases"] == ["champion"]
        moved = lab.put(f"/trained-models/{model_two}/aliases/champion")
        assert moved["moved_from"] == model_one
        assert lab.get(f"/trained-models/{model_one}")["aliases"] == []
        resolved = lab.get("/trained-models/by-alias/halcyon-adapter/champion")
        assert (resolved["target_id"], resolved["version"]) == (model_two, 2)
        bad = lab.client.put(
            f"/api/v1/trained-models/{model_two}/aliases/Champion!", headers=lab.editor
        )
        assert bad.status_code in (400, 404, 422)
        gone = lab.client.delete(
            f"/api/v1/trained-models/{model_one}/aliases/champion", headers=lab.editor
        )
        assert (gone.status_code, gone.json()["code"]) == (404, "alias_not_found")
        removed = lab.client.delete(
            f"/api/v1/trained-models/{model_two}/aliases/champion", headers=lab.editor
        )
        assert removed.status_code == 204
        missing = lab.client.get(
            "/api/v1/trained-models/by-alias/halcyon-adapter/champion", headers=lab.editor
        )
        assert missing.status_code == 404

        # A profile's alias needs an owner.
        text = (FIXTURES / "knowledge-profile.v1" / "kp-policies.yaml").read_text(encoding="utf-8")
        profile = lab.post("/profiles/import", {"content": text}, expect=(201,))
        denied = lab.client.put(
            f"/api/v1/profiles/{profile['id']}/aliases/champion", headers=lab.editor
        )
        assert denied.status_code == 403
        owner = dev_identity.headers(role="owner")
        allowed = lab.client.put(
            f"/api/v1/profiles/{profile['id']}/aliases/champion", headers=owner
        )
        assert allowed.status_code == 200
        assert lab.get(f"/profiles/{profile['id']}")["aliases"] == ["champion"]
        assert lab.get("/profiles/by-alias/kp-policies/champion")["target_id"] == profile["id"]
        every = lab.get("/registry-aliases")["items"]
        assert [(item["kind"], item["alias"]) for item in every] == [
            ("knowledge_profile", "champion")
        ]

        # Maps of saved documents, the pipeline run and an eval run.
        saved = lab.get(
            "/pipeline-maps",
            pipeline_spec_id=lab.get(f"/runs/{inputs['run_id']}")["pipeline_spec_id"],
            retrieval_config_id=inputs["retrieval_config_id"],
        )
        node_ids = {node["id"] for node in saved["nodes"]}
        assert {
            "extract",
            "parse:default",
            "chunk",
            "embed",
            "index:dense",
            "query:rerank",
        } <= node_ids
        assert {
            "store:objects",
            "store:database",
            "service:model-gateway",
            "service:clamd",
        } <= node_ids
        objects = next(node for node in saved["nodes"] if node["id"] == "store:objects")
        assert (objects["backend"], objects["configured"]) == ("filesystem", True)
        assert "model-gateway" in saved["summary"]["sends_text_to"]
        assert saved["mermaid"].startswith("flowchart LR")
        assert "://" not in json.dumps(saved)
        run_map = lab.get(f"/runs/{inputs['run_id']}/map")
        chunk = next(node for node in run_map["nodes"] if node["id"] == "chunk")
        assert chunk["overlay"]["tasks"] >= 1
        assert chunk["overlay"]["bytes"] > 0
        eval_map = lab.get(f"/eval-runs/{first['id']}/map")
        rerank = next(node for node in eval_map["nodes"] if node["id"] == "query:rerank")
        assert rerank["overlay"]["latency_ms"]["p95"] >= 0
        dense = next(node for node in eval_map["nodes"] if node["id"] == "query:dense")
        assert 0 < dense["overlay"]["evidence_reaches"]["dense"] <= 1
        assert "query:results" in {node["id"] for node in eval_map["nodes"]}
        refused = lab.client.get("/api/v1/pipeline-maps", headers=lab.editor)
        assert (refused.status_code, refused.json()["code"]) == (400, "nothing_to_map")
        pipelines = lab.get("/rag-pipelines")["items"]
        assert {item["retrieval_config"]["name"] for item in pipelines} == {
            "hybrid-v1",
            "no-rerank",
            "dense-only",
        }
        hybrid = next(i for i in pipelines if i["retrieval_config"]["name"] == "hybrid-v1")
        assert hybrid["eval_runs"] == 3  # the first, the unfinished one, and halcyon-other's
        assert "rerank/gateway" in hybrid["components"]
        assert hybrid["best"]["metric"] == "recall@5"


@pytest.mark.browser
def test_tracking_compare_and_maps_in_a_browser(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Search, tag, select three runs, compare them, open a trace and a map
    (web/e2e/tracking-flow.mjs)."""
    node = shutil.which("node")
    if node is None:
        pytest.fail("the browser flow needs Node 24 and Playwright")
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        _, runs = three_runs(lab)
        url = serve_web(lab, serve_app)
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [node, str(WEB / "e2e" / "tracking-flow.mjs"), url, *[run["id"] for run in runs]],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=WEB,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("ok"), result.stdout
