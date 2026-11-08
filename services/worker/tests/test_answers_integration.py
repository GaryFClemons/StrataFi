"""Answer-level evaluation and the question review queue against Postgres, with the fakes
(integration tier).

An answer run answers the Halcyon demo set's dev queries from an eval run's final passages with
``generate/fake``, which also judges; the scores, the labels and the judges' agreement with
them come back through the API, and an identical run reuses the answers. ``generate/gateway``
fails while the gateway refuses chat, and writes the answers once it doesn't. Drafted questions
wait in the review queue, and only the accepted ones, edits included, reach a new version of the
eval set.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from stratafi_core.answers import citations_in
from stratafi_store.tables import audit_events
from stratafi_testing.canary import CanaryCheck
from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import WEB, Lab, evaluate, prepare_halcyon, running_lab, serve_web

pytestmark = pytest.mark.integration

FAKE = {"component": "generate/fake"}


def answer(lab: Lab, eval_run_id: str, **body: Any) -> dict[str, Any]:
    started = lab.post(
        "/answer-runs",
        {"eval_run_id": eval_run_id, "generator": FAKE, "judge": FAKE, **body},
        expect=(202,),
    )
    assert started["status"] == "queued"
    return lab.wait(f"/answer-runs/{started['id']}")


def test_answers_are_graded_judged_and_calibrated(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs = prepare_halcyon(lab)
        evaluation = evaluate(lab, inputs)
        assert evaluation["status"] == "succeeded"
        run = answer(lab, evaluation["id"])
        assert run["status"] == "succeeded", run
        assert run["results_cached"] is False
        assert set(run["config"]["prompts"]) == {
            "grounded-qa-v2",
            "judge-faithfulness-v2",
            "judge-correctness-v2",
            "draft-questions-v2",
        }
        metrics = run["metrics"]
        counts = metrics["counts"]
        assert counts["queries"] == evaluation["eval_set"]["items"]
        assert 0 < counts["with_evidence"] < counts["queries"]  # the set has abstention items
        assert counts["judged"] == counts["queries"]
        for name in ("abstention_accuracy", "answered", "citation_precision", "faithfulness_pass"):
            estimate = metrics[name]
            assert estimate["low"] <= estimate["mean"] <= estimate["high"], name
        usage = run["usage"]
        assert usage["by_prompt"]["grounded-qa-v2"] == counts["queries"]
        assert usage["by_prompt"]["judge-faithfulness-v2"] == counts["queries"]
        assert usage["egress"] == ["none"]
        assert usage["revisions"] == {"generator": "fake-generate-v1", "judge": "fake-generate-v1"}

        rows = lab.get(f"/answer-runs/{run['id']}/queries")["items"]
        assert len(rows) == counts["queries"]
        answered = next(row for row in rows if not row["score"]["abstained"])
        abstained = next(row for row in rows if row["score"]["abstained"])
        detail = lab.get(f"/answer-runs/{run['id']}/queries/{answered['item_id']}")
        sources = {passage["source_id"] for passage in detail["passages"]}
        assert citations_in(detail["answer"])
        assert set(citations_in(detail["answer"])) <= sources
        assert all(passage["text"] for passage in detail["passages"])
        assert detail["score"]["faithfulness"] == 5  # the fake quotes a source sentence

        # Calibration: people's labels, against the judge and the abstention detector.
        path = f"/answer-runs/{run['id']}/labels"
        lab.put(path, {"item_id": answered["item_id"], "dimension": "faithful", "label": True})
        for row, label in ((answered, False), (abstained, True), (rows[2], True)):
            lab.put(path, {"item_id": row["item_id"], "dimension": "abstained", "label": label})
        # A later label replaces the person's earlier one.
        lab.put(path, {"item_id": rows[2]["item_id"], "dimension": "abstained", "label": False})
        agreement = lab.get(f"/answer-runs/{run['id']}/agreement")["dimensions"]
        assert agreement["faithful"]["n"] == 1
        assert agreement["faithful"]["agreement"]["mean"] == 1.0
        abstention = agreement["abstained"]
        assert abstention["n"] == 3
        expected = (2 + (rows[2]["score"]["abstained"] is False)) / 3
        assert abstention["agreement"]["mean"] == pytest.approx(expected, abs=1e-6)
        assert agreement["correct"]["n"] == 0
        labelled = lab.get(f"/answer-runs/{run['id']}/queries")["items"]
        mine = next(row for row in labelled if row["item_id"] == answered["item_id"])
        assert {(label["dimension"], label["label"]) for label in mine["labels"]} == {
            ("faithful", True),
            ("abstained", False),
        }
        lab.put(path, {"item_id": "no-such-item", "dimension": "faithful", "label": True},
                expect=(404,))  # fmt: skip

        # The same configuration again reuses the answers, and their labels follow them.
        again = answer(lab, evaluation["id"])
        assert again["results_cached"] is True
        assert again["metrics"] == run["metrics"]
        repeated = lab.get(f"/answer-runs/{again['id']}/agreement")["dimensions"]
        assert repeated["abstained"]["n"] == 3
        listed = lab.get("/answer-runs", eval_run_id=evaluation["id"])["items"]
        assert [item["id"] for item in listed] == [again["id"], run["id"]]

        # A gateway that refuses chat fails the run...
        gateway = {"generator": {"component": "generate/gateway"}, "judge": None}
        lab.gateway.refuse_chat = True
        refused = answer(lab, evaluation["id"], **gateway)
        assert refused["status"] == "failed"
        assert refused["error_code"] == "gateway_refused_403"
        # ...and once it answers, the gateway's model writes the answers.
        lab.gateway.refuse_chat = False
        allowed = answer(lab, evaluation["id"], **gateway)
        assert allowed["status"] == "succeeded", allowed
        assert allowed["usage"]["egress"] == ["model_gateway"]
        assert allowed["usage"]["revisions"]["generator"] == (
            lab.gateway.aliases["chat-default"].revision
        )
        lab.post(
            "/answer-runs",
            {"eval_run_id": evaluation["id"], "generator": {"component": "rerank/none"}},
            expect=(400,),
        )

        with migrated_engine.connect() as conn:
            actions = conn.execute(
                sa.select(audit_events.c.action, audit_events.c.details).where(
                    audit_events.c.action.in_(("answer_run.start", "answer.label"))
                )
            ).all()
        assert {action for action, _ in actions} == {"answer_run.start", "answer.label"}
        assert all(set(details) <= {"eval_run", "generator", "judge", "dimension"}
                   for _, details in actions)  # fmt: skip


def test_drafted_questions_wait_for_review_before_a_version(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs = prepare_halcyon(lab)
        eval_set = inputs["eval_set_id"]
        before = lab.get(f"/eval-sets/{eval_set}")["latest_version"]
        batch = lab.post(
            f"/eval-sets/{eval_set}/question-batches",
            {"run_id": inputs["run_id"], "generator": FAKE, "count": 5, "seed": 1},
            expect=(202,),
        )
        finished = lab.wait(f"/eval-sets/{eval_set}/question-batches/{batch['id']}")
        assert finished["status"] == "succeeded", finished
        assert finished["counts"]["drafted"] == 5
        assert finished["config"]["prompt"]["id"] == "draft-questions-v2"
        proposed = lab.get(f"/eval-sets/{eval_set}/candidates", status="proposed")["items"]
        assert len(proposed) == 5
        for candidate in proposed:
            assert candidate["query"] == candidate["drafted_query"]
            assert candidate["evidence"]
            assert candidate["doc_key"]
            assert candidate["split"] == "dev"
        # Nothing reaches a version unreviewed.
        refused = lab.post(f"/eval-sets/{eval_set}/candidates/add", expect=(409,))
        assert refused["code"] == "nothing_accepted"

        first, second, third = proposed[:3]
        decide = f"/eval-sets/{eval_set}/candidates"
        edited = lab.post(
            f"{decide}/{first['id']}/decision",
            {"decision": "accepted", "query": "How long can batch brew sit in its server?"},
        )
        assert (edited["status"], edited["edited"]) == ("accepted", True)
        lab.post(f"{decide}/{second['id']}/decision", {"decision": "rejected"})
        lab.post(
            f"{decide}/{third['id']}/decision",
            {"decision": "accepted", "split": "test", "tags": ["brew"]},
        )
        added = lab.post(f"/eval-sets/{eval_set}/candidates/add", expect=(201,))
        version = added["version"]
        assert version["version"] == before["version"] + 1
        assert version["item_count"] == before["item_count"] + 2
        assert version["source"]["format"] == "generated"
        assert {candidate["status"] for candidate in added["added"]} == {"added"}
        items = lab.get(f"/eval-sets/{eval_set}/versions/{version['version']}/items", limit=200)[
            "items"
        ]
        new = {item["id"]: item for item in items if item["id"].startswith("gen-")}
        assert len(new) == 2
        mine = new[next(c["item_id"] for c in added["added"] if c["id"] == first["id"])]
        assert mine["query"] == "How long can batch brew sit in its server?"
        assert mine["expected"][0]["evidence"] == first["evidence"]
        assert mine["expected"][0]["doc_key"] == first["doc_key"]
        assert "generated" in mine["tags"]
        theirs = new[next(c["item_id"] for c in added["added"] if c["id"] == third["id"])]
        assert theirs["split"] == "test"
        assert theirs["tags"] == ["brew", "generated"]
        # Added is settled; the queue keeps the rest.
        settled = lab.post(
            f"{decide}/{first['id']}/decision", {"decision": "rejected"}, expect=(409,)
        )
        assert settled["code"] == "already_added"
        left = lab.get(f"/eval-sets/{eval_set}/candidates", status="proposed")["items"]
        assert len(left) == 2
        with migrated_engine.connect() as conn:
            details = conn.execute(
                sa.select(audit_events.c.details).where(
                    audit_events.c.action.like("eval_set.question%")
                )
            ).scalars()
            assert all("query" not in found and "evidence" not in found for found in details)


@pytest.mark.browser
def test_the_review_queue_and_answers_in_a_browser(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """The review queue flow works: draft questions, edit and accept one, accept one,
    reject one, add the accepted ones to a new version; then answer an eval run's queries and
    label an answer (web/e2e/review-flow.mjs)."""
    node = shutil.which("node")
    if node is None:
        pytest.fail("the browser flow needs Node 24 and Playwright")
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs = prepare_halcyon(lab)
        evaluation = evaluate(lab, inputs)
        url = serve_web(lab, serve_app)
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [
                node,
                str(WEB / "e2e" / "review-flow.mjs"),
                url,
                "halcyon-retrieval",
                evaluation["id"],
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=WEB,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("ok"), result.stdout


def test_answers_and_drafted_questions_keep_text_out_of_logs(
    migrated_engine: sa.Engine,
    dev_identity: DevIdentity,
    tmp_path: Path,
    serve_app: Any,
    canary: CanaryCheck,
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs = prepare_halcyon(lab)
        jsonl = (
            '{"askers": {"staff": {"roles": ["staff"], "groups": ["all-staff"]}}}\n'
            '{"id": "q1", "query": "' + canary.value + ' tip jar counts", "asker": '
            '"staff", "expected": [{"doc_key": "tips", "version": "3"}]}\n'
        )
        draft = lab.import_eval_set("canary", jsonl, format="jsonl", freeze=False)
        lab.post(f"/eval-sets/{draft['eval_set']['id']}/versions/1/freeze")
        evaluation = evaluate(
            lab,
            inputs,
            eval_set_version_id=draft["version"]["id"],
            split="all",
            confirm_test_read=True,
        )
        assert evaluation["status"] == "succeeded", evaluation
        run = answer(lab, evaluation["id"])
        assert run["status"] == "succeeded", run
        eval_set = inputs["eval_set_id"]
        batch = lab.post(
            f"/eval-sets/{eval_set}/question-batches",
            {"run_id": inputs["run_id"], "generator": FAKE, "count": 1},
            expect=(202,),
        )
        lab.wait(f"/eval-sets/{eval_set}/question-batches/{batch['id']}")
        candidate = lab.get(f"/eval-sets/{eval_set}/candidates")["items"][0]
        lab.post(
            f"/eval-sets/{eval_set}/candidates/{candidate['id']}/decision",
            {"decision": "accepted", "query": f"{canary.value} and a question?"},
        )
        lab.post(f"/eval-sets/{eval_set}/candidates/add", expect=(201,))
        audit = lab.client.get("/api/v1/audit-events", headers=dev_identity.headers(role="owner"))
        records = lab.get(f"/answer-runs/{run['id']}")
        canary.assert_absent(audit.text, str(records), str(lab.get(f"/eval-sets/{eval_set}")))
        canary.assert_not_logged()
