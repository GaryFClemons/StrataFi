"""Experiments against Postgres, with the fakes (integration tier).

A 12-trial grid over the Halcyon demo corpus: three chunkings, two embedding aliases and two
rerankers.
Trials that differ only in their reranker share a pipeline run, so six pipeline runs do the
work; extract, scan, parse and clean are shared by all of them. The plan's unique work is what
the runs execute, and planning the same experiment again finds everything cached. Budgets stop
an experiment cleanly: trials over ``max_trials`` are skipped and the admitted ones stay valid;
a cost over ``max_cost_usd`` stops it after the run that crossed it.
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
    baseline_pipeline,
    hybrid_retrieval,
    running_lab,
    serve_web,
)

pytestmark = pytest.mark.integration

DOCUMENT_STAGES = ("extract", "scan", "parse", "clean", "chunk", "enrich", "embed")


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


def grid(**changes: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": "stratafi.experiment.v1",
        "name": "chunking-and-models",
        "corpus": "halcyon",
        "eval_set": {"name": "halcyon-retrieval"},
        "base": {"pipeline": "baseline-v1", "retrieval": "hybrid-v1"},
        "objective": {"metric": "recall@5"},
        "constraints": [
            {"metric": "must_not_return_violations", "max": 0},
            {"metric": "latency_ms.p95", "max": 5000},
        ],
        "factors": {
            "chunk": [
                {"component": "chunk/sections"},
                {"component": "chunk/fixed-tokens", "params": {"size": 256, "overlap": 32}},
                {"component": "chunk/fixed-tokens", "params": {"size": 512, "overlap": 64}},
            ],
            "embed.params.alias": ["embed-default", "embed-local"],
            "rerank": ["rerank/none", {"component": "rerank/gateway"}],
        },
        "execution": {"max_parallel_runs": 1, "max_parallel_evals": 4},
    }
    document.update(changes)
    return document


def run_experiment(lab: Lab, document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    created = lab.post("/experiments", {"document": document}, expect=(201,))
    assert created["status"] == "planned"
    started = lab.post(f"/experiments/{created['id']}/start", {"confirm": True}, expect=(202,))
    assert started["status"] == "running"
    return created, lab.wait_experiment(created["id"])


def executed(work: dict[str, Any]) -> dict[str, int]:
    return {stage: counts.get("succeeded", 0) for stage, counts in work["done"].items()}


def test_a_12_trial_experiment_does_the_planned_work_once(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        planned = lab.post("/experiments/plan", {"document": grid()})
        assert planned["valid"], planned["issues"]
        plan = planned["plan"]
        assert (plan["trials"], plan["grid"], plan["work"]["pipelines"]) == (12, 12, 6)
        documents = plan["corpus"]["documents"]
        stages = plan["work"]["stages"]
        # Shared prefixes: one extract, scan, parse and clean per document; three chunkings;
        # six embeddings; six dense indexes and three keyword indexes.
        assert {stage: stages[stage]["unique"] for stage in stages} == {
            "extract": documents,
            "scan": documents,
            "parse": documents,
            "clean": documents,
            "chunk": 3 * documents,
            "enrich": 3 * documents,
            "embed": 6 * documents,
            "index": 9,
        }
        assert all(stages[stage]["cached"] == 0 for stage in stages)
        assert plan["estimate"]["eval"]["to_run"] == 12
        assert plan["eval_set"]["items"] == 62
        assert plan["warnings"] == []
        assert [trial["values"]["rerank"] for trial in planned["trials"][:2]] == [
            "rerank/none",
            "rerank/gateway",
        ]

        created, finished = run_experiment(lab, grid())
        assert finished["status"] == "succeeded", finished
        assert finished["counts"]["trials"] == {"succeeded": 12}
        assert finished["spent"]["total_usd"] > 0
        work = lab.get(f"/experiments/{created['id']}/work")
        # Every task the plan said would run ran exactly once; every other one was cached.
        assert executed(work) == {stage: stages[stage]["to_run"] for stage in stages}
        assert work["done"]["parse"].get("cached") == 5 * documents
        trials = lab.get(f"/experiments/{created['id']}/trials")["items"]
        assert [trial["ordinal"] for trial in trials] == list(range(12))
        assert len({trial["run_id"] for trial in trials}) == 6
        assert len({trial["eval_run_id"] for trial in trials}) == 12
        for trial in trials:
            assert trial["status"] == "succeeded"
            objective = trial["objective"]
            assert objective["low"] <= objective["mean"] <= objective["high"]
            assert trial["measures"]["recall@5"] == objective["mean"]
            assert [row["metric"] for row in trial["constraints"]] == [
                "must_not_return_violations",
                "latency_ms.p95",
            ]
            assert trial["feasible"] is all(row["passed"] for row in trial["constraints"])
        # The results views, from the finished trials.
        results = lab.get(f"/experiments/{created['id']}/results")
        assert results["trials"] == 12
        assert results["queries"] > 0
        assert results["objective"] == {
            "metric": "recall@5",
            "rule": "knowledge-hit-v1",
            "checkpoint": "final",
            "goal": "maximize",
        }
        board = results["leaderboard"]
        assert [row["rank"] for row in board] == list(range(1, 13))
        means = [row["objective"]["mean"] for row in board]
        assert means == sorted(means, reverse=True)
        assert [effect["path"] for effect in results["effects"]] == [
            "chunk",
            "embed.params.alias",
            "rerank",
        ]
        chunk_levels = results["effects"][0]["levels"]
        assert [row["pairs"] for row in chunk_levels] == [4, 4]  # 2 aliases x 2 rerankers
        for row in chunk_levels:
            assert row["low"] <= row["delta"] <= row["high"]
        assert len(results["interactions"]) == 3
        assert len(results["pareto"]["latency"]) == 12
        assert any(point["frontier"] for point in results["pareto"]["latency"])
        assert [row["file_type"] for row in results["file_types"]["types"]] == ["text/markdown"]
        assert results["file_types"]["routing"] == []  # no factor on the parse stage
        assert len(results["where_lost"]["trials"]) == 12
        mrr = lab.get(f"/experiments/{created['id']}/results", metric="mrr@10")
        assert mrr["objective"]["metric"] == "mrr@10"

        # Trials that differ only in the reranker share their pipeline run and its variant.
        assert trials[0]["run_id"] == trials[1]["run_id"]
        assert trials[0]["pipeline_spec_id"] == trials[1]["pipeline_spec_id"]
        # The first trial is the base itself; the other five pipelines are saved variants.
        names = [item["name"] for item in lab.get("/pipeline-specs", limit=100)["items"]]
        assert sorted(name.startswith("baseline-v1.x") for name in names) == [False] + [True] * 5

        # The same experiment again: the plan finds everything made, and the run reuses it.
        again = lab.post("/experiments/plan", {"document": grid()})
        assert all(
            counts["to_run"] == 0 and counts["cached"] == counts["unique"]
            for counts in again["plan"]["work"]["stages"].values()
        )
        assert again["plan"]["estimate"]["eval"]["to_run"] == 0
        assert all(trial["evaluated"] for trial in again["trials"])
        second, rerun = run_experiment(lab, grid())
        assert rerun["status"] == "succeeded"
        assert set(executed(lab.get(f"/experiments/{second['id']}/work")).values()) == {0}
        assert rerun["spent"]["total_usd"] == 0
        evals = [
            lab.get(f"/eval-runs/{trial['eval_run_id']}")
            for trial in lab.get(f"/experiments/{second['id']}/trials")["items"]
        ]
        assert all(evaluation["results_cached"] for evaluation in evals)


def test_successive_halving_scores_nested_samples_and_prunes(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Twelve retrieval-only trials share one pipeline run. Successive halving scores
    them on 10 dev queries, the survivors on 30, then on all 62; it prunes only trials clearly
    behind the rung's leader, and each rung reuses the shards of the rungs before it."""
    halving = {"method": "successive-halving", "eta": 3, "min_queries": 10, "seed": 3}
    document = grid(
        name="halving",
        factors={
            "top_k": [1, 3, 8],
            "rerank": ["rerank/none", {"component": "rerank/gateway"}],
            "candidates": [10, 50],
        },
        strategy=halving,
    )
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        planned = lab.post("/experiments/plan", {"document": document})
        assert planned["valid"], planned["issues"]
        plan = planned["plan"]
        assert (plan["strategy"], plan["trials"], plan["work"]["pipelines"]) == (
            "successive-halving",
            12,
            1,
        )
        search = plan["search"]
        assert search["rungs"] == [
            {"queries": 10, "keep": 4},
            {"queries": 30, "keep": 2},
            {"queries": 62, "keep": 2},
        ]
        assert search["cost"] == 12 * 10 + 4 * 20 + 2 * 32
        assert plan["estimate"]["eval"]["query_share"] == search["share"] < 0.5

        created, finished = run_experiment(lab, document)
        assert finished["status"] == "succeeded", finished
        counts = finished["counts"]["trials"]
        assert set(counts) == {"succeeded", "pruned"}, counts
        trials = lab.get(f"/experiments/{created['id']}/trials")["items"]
        assert len({trial["run_id"] for trial in trials}) == 1
        sizes = [10, 30, 62]
        scored = 0
        for trial in trials:
            rungs = trial["rungs"]
            assert [rung["queries"] for rung in rungs] == sizes[: len(rungs)]
            scored += rungs[-1]["queries"]
            for index, rung in enumerate(rungs):
                evaluation = lab.get(f"/eval-runs/{rung['eval_run_id']}")
                expected = {"seed": 3} if index == 2 else {"seed": 3, "size": sizes[index]}
                assert evaluation["options"]["sample"] == expected
                assert evaluation["options"]["shard_size"] == 10
                assert evaluation["counts"]["queries"] == sizes[index]
                # The rung before's shards are this sample's first ones, already made.
                assert evaluation["counts"]["shards"].get("cached", 0) == (
                    sizes[index - 1] // 10 if index else 0
                )
            if trial["status"] == "pruned":
                last = rungs[-1]
                assert last["kept"] is False
                assert last["against_leader"]["high"] < 0
                assert trial["objective"] == last["objective"]
            else:
                assert len(rungs) == 3
                assert all(rung["kept"] for rung in rungs)
                assert trial["eval_run_id"] == rungs[-1]["eval_run_id"]
                assert trial["objective"] == rungs[-1]["objective"]
        # Each (trial, query) pair was scored once, far fewer than the grid's.
        assert scored < 12 * 62
        survivors = counts["succeeded"]
        results = lab.get(f"/experiments/{created['id']}/results")
        assert results["trials"] == survivors
        assert len(results["leaderboard"]) == survivors
        # A pruned trial was never scored on every query: it can't be confirmed on test.
        pruned = next(trial for trial in trials if trial["status"] == "pruned")
        refused = lab.post(
            f"/experiments/{created['id']}/trials/{pruned['id']}/confirm",
            {"confirm_test_read": True},
            expect=(409,),
        )
        assert refused["code"] == "trial_pruned"


def test_budgets_stop_an_experiment_cleanly(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        # Trials over max_trials never start; the admitted ones finish and stay valid.
        created, stopped = run_experiment(lab, grid(budget={"max_trials": 5}))
        assert (stopped["status"], stopped["stop_reason"]) == ("stopped", "max_trials")
        assert stopped["counts"]["trials"] == {"succeeded": 5, "skipped": 7}
        trials = lab.get(f"/experiments/{created['id']}/trials")["items"]
        assert [trial["status"] for trial in trials[:5]] == ["succeeded"] * 5
        assert {trial["error_code"] for trial in trials[5:]} == {"max_trials"}
        assert all(trial["run_id"] is None for trial in trials[5:])
        assert len({trial["run_id"] for trial in trials[:5]}) == 3

        # A cost budget: the first pipeline run embeds through the gateway at a price, which
        # is over a budget of nothing, so the experiment stops after it.
        costly = grid(
            name="costly",
            factors={"chunk.params.max_tokens": [600, 700], "rerank": ["rerank/none"]},
            budget={"max_cost_usd": 0},
        )
        created, stopped = run_experiment(lab, costly)
        assert (stopped["status"], stopped["stop_reason"]) == ("stopped", "max_cost_usd")
        assert stopped["spent"]["ingest_usd"] > 0
        trials = lab.get(f"/experiments/{created['id']}/trials")["items"]
        assert trials[0]["status"] == "cancelled"
        assert (trials[1]["status"], trials[1]["error_code"]) == ("skipped", "max_cost_usd")
        assert trials[1]["run_id"] is None
        runs = {trial["run_id"] for trial in trials if trial["run_id"]}
        assert len(runs) == 1


def test_plans_report_problems_and_starting_needs_confirmation(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        bad = lab.post(
            "/experiments/plan",
            {
                "document": grid(
                    corpus="missing",
                    factors={"nowhere": [1], "chunk.params.max_tokens": [1, 1]},
                )
            },
        )
        assert not bad["valid"]
        assert {"path": "corpus", "code": "not_found"} in bad["issues"]
        refused = lab.client.post(
            "/api/v1/experiments", json={"document": grid(corpus="missing")}, headers=lab.editor
        )
        assert (refused.status_code, refused.json()["code"]) == (400, "invalid_experiment")
        yaml_plan = lab.post(
            "/experiments/plan",
            {
                "content": "schema: stratafi.experiment.v1\n"
                "name: yaml\ncorpus: halcyon\neval_set: {name: halcyon-retrieval}\n"
                "base: {pipeline: baseline-v1, retrieval: hybrid-v1}\n"
                "objective: {metric: mrr@10}\n"
                "factors:\n  top_k: [5, 8]\n"
            },
        )
        assert yaml_plan["valid"], yaml_plan["issues"]
        assert yaml_plan["plan"]["work"]["pipelines"] == 1

        created = lab.post("/experiments", {"document": grid(name="later")}, expect=(201,))
        unconfirmed = lab.client.post(
            f"/api/v1/experiments/{created['id']}/start", json={}, headers=lab.editor
        )
        assert unconfirmed.status_code == 400
        assert lab.get(f"/experiments/{created['id']}")["status"] == "planned"
        cancelled = lab.post(f"/experiments/{created['id']}/cancel")
        assert cancelled["status"] == "cancelled"
        again = lab.client.post(
            f"/api/v1/experiments/{created['id']}/start",
            json={"confirm": True},
            headers=lab.editor,
        )
        assert (again.status_code, again.json()["code"]) == (409, "not_planned")
        listed = lab.get("/experiments")["items"]
        assert [item["name"] for item in listed] == ["later"]


def test_a_trial_is_confirmed_once_on_the_test_split(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """The guardrails, and the one-shot, logged test-split confirmation; then a confirmed
    trial exported as a knowledge profile, reviewed, downloaded and imported."""
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        small = grid(
            name="confirm",
            factors={
                "chunk.params.max_tokens": [600, 800],
                "rerank": ["rerank/none", {"component": "rerank/gateway"}],
            },
        )
        created, finished = run_experiment(lab, small)
        assert finished["status"] == "succeeded"
        experiment = created["id"]
        results = lab.get(f"/experiments/{experiment}/results")
        guard = results["guardrails"]
        assert guard["leader"] == results["leaderboard"][0]["ordinal"]
        assert guard["trials"][0]["group"] == "leader"
        assert {row["group"] for row in guard["trials"][1:]} <= {"tied", "worse"}
        assert guard["recommended"] in {row["ordinal"] for row in guard["trials"]}
        assert guard["sample_size"]["queries"] == results["queries"]
        assert guard["sample_size"]["min_detectable_difference"] > 0

        trials = lab.get(f"/experiments/{experiment}/trials")["items"]
        chosen = next(trial for trial in trials if trial["ordinal"] == guard["recommended"])
        version = finished["eval_set_version_id"]

        def reads() -> int:
            sets = lab.get("/eval-sets")["items"]
            latest = next(
                item["latest_version"] for item in sets if item["name"] == "halcyon-retrieval"
            )
            assert latest["id"] == version
            return latest["test_reads"]

        assert reads() == 0
        first = lab.post(
            f"/experiments/{experiment}/trials/{chosen['id']}/confirm",
            {"confirm_test_read": True},
            expect=(202,),
        )
        assert (first["ordinal"], first["reason"]) == (1, None)
        assert reads() == 1
        evaluation = lab.wait_eval(first["eval_run_id"])
        assert (evaluation["status"], evaluation["split"], evaluation["test_read"]) == (
            "succeeded",
            "test",
            True,
        )
        # A second read without a reason is refused, and nothing is read or counted.
        other = next(trial for trial in trials if trial["id"] != chosen["id"])
        path = f"/api/v1/experiments/{experiment}/trials/{other['id']}/confirm"
        refused = lab.client.post(path, json={"confirm_test_read": True}, headers=lab.editor)
        assert (refused.status_code, refused.json()["code"]) == (409, "test_split_already_read")
        assert reads() == 1
        unconfirmed = lab.client.post(path, json={}, headers=lab.editor)
        assert unconfirmed.status_code == 400
        # With a reason it's a logged second read.
        second = lab.post(
            f"/experiments/{experiment}/trials/{other['id']}/confirm",
            {"confirm_test_read": True, "reason": "candidate_changed", "note": "kept here"},
            expect=(202,),
        )
        assert (second["ordinal"], second["reason"]) == (2, "candidate_changed")
        assert reads() == 2
        lab.wait_eval(second["eval_run_id"])
        listed = lab.get(f"/experiments/{experiment}/confirmations")
        assert listed["test_reads"] == 2
        first_read = listed["items"][0]
        assert first_read["dev"] == chosen["objective"]
        assert first_read["test"]["n"] > 0
        assert first_read["status"] == "succeeded"
        owner = dev_identity.headers(role="owner")
        events = lab.client.get("/api/v1/audit-events", headers=owner).json()["items"]
        confirms = [event for event in events if event["action"] == "experiment.confirm"]
        assert [event["details"]["reason"] for event in confirms] == ["candidate_changed", None]
        assert all("kept here" not in str(event["details"]) for event in events)
        assert sum(event["action"] == "eval_set.test_read" for event in events) == 2

        # The confirmed trials become a reviewed knowledge profile.
        assert all("compatibility" not in row for row in results["leaderboard"])
        unconfirmed = next(
            trial for trial in trials if trial["id"] not in (chosen["id"], other["id"])
        )
        refused = lab.client.post(
            "/api/v1/profiles",
            json={"experiment_id": experiment, "trial_id": unconfirmed["id"], "name": "kp-halcyon"},
            headers=lab.editor,
        )
        assert (refused.status_code, refused.json()["code"]) == (409, "not_confirmed")
        exported = lab.post(
            "/profiles",
            {"experiment_id": experiment, "trial_id": chosen["id"], "name": "kp-halcyon"},
            expect=(201,),
        )
        assert (exported["version"], exported["status"]) == (1, "draft")
        profile = exported["document"]
        assert profile["schema"] == "stratafi.knowledge-profile.v1"
        assert profile["evidence"]["dev"]["mean"] == chosen["objective"]["mean"]
        assert profile["evidence"]["test"]["mean"] == first_read["test"]["mean"]
        assert profile["evidence"]["test_read"] == 1
        assert profile["evidence"]["guardrails"]["recommended"] is True
        assert profile["pipeline"]["stages"]["chunk"]["params"]["max_tokens"] in (600, 800)
        assert profile["retrieval"]["name"]
        assert profile["models"]["embed"]["alias"] == "embed-default"
        assert profile["models"]["embed"]["revision"]
        again = lab.post(
            "/profiles",
            {"experiment_id": experiment, "trial_id": chosen["id"], "name": "kp-halcyon"},
            expect=(200,),
        )
        assert (again["id"], again["created"]) == (exported["id"], False)
        second_version = lab.post(
            "/profiles",
            {"experiment_id": experiment, "trial_id": other["id"], "name": "kp-halcyon"},
            expect=(201,),
        )
        assert second_version["version"] == 2
        diff = lab.get(f"/profiles/{second_version['id']}/diff")
        assert diff["against"]["version"] == 1
        assert diff["changes"]
        assert lab.get(f"/profiles/{exported['id']}/diff")["against"] is None

        # Review: an owner decides; an editor can't.
        decision = f"/api/v1/profiles/{exported['id']}/approvals"
        by_editor = lab.client.post(decision, json={"decision": "approved"}, headers=lab.editor)
        assert by_editor.status_code == 403
        approved = lab.client.post(decision, json={"decision": "approved"}, headers=owner)
        assert approved.status_code == 200
        assert approved.json()["status"] == "approved"
        assert approved.json()["approvals"][0]["decided_by"] == "u-tenant0-owner"

        # The export carries the review and imports back as the same profile.
        download = lab.client.get(f"/api/v1/profiles/{exported['id']}/export", headers=lab.editor)
        assert download.headers["content-type"].startswith("application/yaml")
        assert download.text.startswith("# stratafi.knowledge-profile.v1: kp-halcyon\n")
        assert "status: approved" in download.text
        # Like a spec, a name's versions dedupe against the latest: v1 again is v3.
        imported = lab.post("/profiles/import", {"content": download.text}, expect=(201,))
        assert (imported["version"], imported["source"]) == (3, "import")
        assert imported["document_hash"] == exported["document_hash"]
        same = lab.post("/profiles/import", {"content": download.text}, expect=(200,))
        assert (same["id"], same["created"]) == (imported["id"], False)
        renamed = download.text.replace("name: kp-halcyon\n", "name: kp-halcyon-copy\n", 1)
        copy = lab.client.post("/api/v1/profiles/import", json={"content": renamed}, headers=owner)
        assert copy.status_code == 201
        assert (copy.json()["source"], copy.json()["status"]) == ("import", "draft")
        # Its author may decide it too, and the record says so.
        own = f"/api/v1/profiles/{copy.json()['id']}/approvals"
        mine = lab.client.post(own, json={"decision": "approved"}, headers=owner)
        assert mine.status_code == 200
        assert mine.json()["approvals"][-1]["self_review"] is True
        assert approved.json()["approvals"][0]["self_review"] is False
        listed = lab.get("/profiles")["items"]
        assert {item["name"] for item in listed} == {"kp-halcyon", "kp-halcyon-copy"}


BROWSER_EXPERIMENT = """schema: stratafi.experiment.v1
name: browser-flow
corpus: halcyon
eval_set: {name: halcyon-retrieval}
base: {pipeline: baseline-v1, retrieval: hybrid-v1}
objective: {metric: recall@5}
constraints:
  - {metric: must_not_return_violations, max: 0}
factors:
  chunk.params.max_tokens: [600, 800]
  rerank: [rerank/none, {component: rerank/gateway}]
"""


@pytest.mark.browser
def test_the_experiment_screens_in_a_browser(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Write, plan, start and follow an experiment; every results view renders
    (web/e2e/experiment-flow.mjs)."""
    node = shutil.which("node")
    if node is None:
        pytest.fail("the browser flow needs Node 24 and Playwright")
    document = tmp_path / "experiment.yaml"
    document.write_text(BROWSER_EXPERIMENT, encoding="utf-8")
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        prepare(lab)
        url = serve_web(lab, serve_app)
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [node, str(WEB / "e2e" / "experiment-flow.mjs"), url, str(document), "4"],
            capture_output=True,
            text=True,
            timeout=420,
            cwd=WEB,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("ok"), result.stdout
