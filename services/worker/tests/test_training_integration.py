"""Post-training end to end against Postgres, with the fakes (integration tier).

The Halcyon demo set's dev eval run is mined into a training set (the test split is refused), a
query adapter is trained on it over the fake gateway's embedder, registered with its card,
and measured: an eval run with the adapter in its retrieval config, paired against the base on
the dev split, then one confirmation on the test split. An identical run reuses the model, a
cancelled run stops, a withdrawn model fails new runs, and purging a document purges the
training set and the model built from it. Neither queries nor passages reach logs or audit
events.
"""

import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from stratafi.cli import main as cli
from stratafi_store.tables import audit_events
from stratafi_testing.canary import CanaryCheck
from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import (
    WEB,
    Lab,
    evaluate,
    hybrid_retrieval,
    prepare_halcyon,
    running_lab,
    serve_web,
)

pytestmark = pytest.mark.integration

ADAPTER = {
    "component": "train/query-adapter",
    "params": {"epochs": 40, "learning_rate": 0.002, "patience": 0, "seed": 1},
}


def recall(run: dict[str, Any], checkpoint: str = "dense", metric: str = "recall@5") -> float:
    return float(run["metrics"]["metrics"]["knowledge-hit-v1"][checkpoint][metric]["mean"])


def mine(lab: Lab, eval_run_id: str, **body: Any) -> dict[str, Any]:
    started = lab.post("/training-sets", {"eval_run_id": eval_run_id, **body}, expect=(202,))
    assert started["status"] == "queued"
    return lab.wait(f"/training-sets/{started['id']}")


def train(lab: Lab, training_set_id: str, name: str, trainer: dict[str, Any]) -> dict[str, Any]:
    started = lab.post(
        "/training-runs",
        {"training_set_id": training_set_id, "name": name, "trainer": trainer},
        expect=(202,),
    )
    return lab.wait(f"/training-runs/{started['id']}")


def test_an_adapter_is_mined_trained_registered_and_measured(
    migrated_engine: sa.Engine,
    dev_identity: DevIdentity,
    tmp_path: Path,
    serve_app: Any,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs = prepare_halcyon(lab)
        base = evaluate(lab, inputs)
        assert base["status"] == "succeeded"
        assert base["split"] == "dev"

        # The test split is locked: an eval run that read it can't be mined.
        whole = evaluate(lab, inputs, split="all", confirm_test_read=True)
        lab.post("/training-sets", {"eval_run_id": whole["id"]}, expect=(409,))

        mined = mine(lab, base["id"])
        assert mined["status"] == "succeeded", mined
        assert mined["cached"] is False
        stats = mined["stats"]
        dev_items = base["eval_set"]["items"]
        assert stats["items"] == dev_items
        assert stats["records"] + sum(stats["dropped"].values()) == dev_items
        assert stats["pairs"] > 30
        assert stats["abstentions"] > 0
        assert stats["negatives"] > stats["pairs"]
        assert stats["validation"] == round(stats["records"] * 0.2)
        assert stats["leak_guard"]["test_items"] > 0
        assert mined["context"]["dense"]["dims"] == 384
        assert mined["context"]["eval_set"]["split"] == "dev"

        page = lab.get(f"/training-sets/{mined['id']}/records", limit=200)
        assert page["total"] == stats["records"]
        pairs = [record for record in page["items"] if record["positives"]]
        assert all(p["reason"] == "evidence" for r in pairs for p in r["positives"])
        assert {n["reason"] for r in page["items"] for n in r["negatives"]} >= {
            "retrieved",
            "abstention",
        }
        exported = lab.client.get(
            f"/api/v1/training-sets/{mined['id']}/export",
            params={"format": "sentence-transformers"},
            headers=lab.editor,
        )
        assert exported.status_code == 200
        rows = [json.loads(line) for line in exported.text.splitlines()]
        assert len(rows) == stats["positives"]
        assert set(rows[0]) == {"item_id", "split", "anchor", "positive", "negatives"}
        chat = lab.client.get(
            f"/api/v1/training-sets/{mined['id']}/export",
            params={"format": "chat"},
            headers=lab.editor,
        )
        # The Halcyon set has no reference answers: only the abstention items have targets.
        assert len(chat.text.splitlines()) == stats["abstentions"]

        # The same mining again reuses the records.
        again = mine(lab, base["id"])
        assert again["cached"] is True
        assert again["cache_key"] == mined["cache_key"]
        assert again["stats"] == mined["stats"]

        # Train the adapter; a trainer that isn't a trainer, or bad parameters, is refused.
        lab.post(
            "/training-runs",
            {"training_set_id": mined["id"], "name": "x", "trainer": {"component": "rerank/none"}},
            expect=(400,),
        )
        lab.post(
            "/training-runs",
            {
                "training_set_id": mined["id"],
                "name": "x",
                "trainer": {"component": "train/query-adapter", "params": {"epochs": 0}},
            },
            expect=(400,),
        )
        run = train(lab, mined["id"], "halcyon-adapter", ADAPTER)
        assert run["status"] == "succeeded", run
        assert run["kind"] == "query-adapter"
        assert run["model_cached"] is False
        assert run["base"]["model"] == "embed-default"
        assert run["base"]["dims"] == 384
        metrics = run["metrics"]
        epochs = metrics["epochs"]
        assert len(epochs) == 40
        assert epochs[-1]["train_loss"] < epochs[0]["train_loss"]
        assert metrics["baseline"]["mrr@10"] is not None
        assert 0 <= metrics["best_epoch"] <= 40
        assert run["progress"]["epoch"] == 40

        record = lab.get(f"/training-runs/{run['id']}/manifest")
        assert record["schema"] == "stratafi.training-run.v1"
        assert record["configuration"]["trainer"]["component"] == "train/query-adapter"
        assert record["inputs"]["training_set"]["id"] == mined["id"]
        assert record["outputs"][0]["cache_key"] == run["model_key"]

        model = lab.get(f"/trained-models/{run['model_id']}")
        assert (model["name"], model["version"], model["status"]) == ("halcyon-adapter", 1, "ready")
        assert model["stored"] is True
        card = model["card"]
        assert card["schema"] == "stratafi.model-card.v1"
        assert card["use"]["how"].endswith(model["id"])
        assert card["training_set"]["eval_set"]["split"] == "dev"
        assert card["limits"]

        download = lab.client.get(
            f"/api/v1/trained-models/{model['id']}/download", headers=lab.editor
        )
        assert download.status_code == 200
        archive = zipfile.ZipFile(io.BytesIO(download.content))
        assert sorted(archive.namelist()) == [
            "manifest.json",
            "model-card.json",
            "model/adapter.npy",
        ]
        # Every file is there with the checksum the manifest gives it, and the card is the
        # registry's.
        listed = json.loads(archive.read("manifest.json"))
        for file in listed["files"]:
            data = archive.read(f"model/{file['path']}")
            assert hashlib.sha256(data).hexdigest() == file["sha256"]
            assert len(data) == file["size"]
        assert json.loads(archive.read("model-card.json")) == model["card"]

        # The identical run reuses the model, and registers a new version of the name.
        repeat = train(lab, mined["id"], "halcyon-adapter", ADAPTER)
        assert repeat["status"] == "succeeded"
        assert repeat["model_cached"] is True
        assert repeat["model_key"] == run["model_key"]
        assert repeat["metrics"] == run["metrics"]
        second = lab.get(f"/trained-models/{repeat['model_id']}")
        assert second["version"] == 2

        # Measure an adapter: in the retrieval config, paired against the base on dev. The
        # Halcyon set's held-out items ask about other documents than the ones it trains on, so
        # early stopping may keep the identity; trained without a validation split, an adapter
        # keeps its last epoch, fitted to the dev items.
        whole = mine(lab, base["id"], options={"validation_share": 0})
        assert whole["status"] == "succeeded", whole
        assert whole["stats"]["validation"] == 0
        fitted = train(lab, whole["id"], "halcyon-adapter-fitted", ADAPTER)
        assert fitted["status"] == "succeeded", fitted
        assert fitted["metrics"]["best_epoch"] == 40
        adapted_config = lab.save_retrieval(
            hybrid_retrieval(
                "halcyon-adapted",
                retrievers={
                    "dense": {"top_n": 50, "adapter": fitted["model_id"]},
                    "keyword": {"top_n": 50},
                },
            )
        )
        adapted = evaluate(lab, {**inputs, "retrieval_config_id": adapted_config})
        assert adapted["status"] == "succeeded", adapted
        assert adapted["results_cached"] is False
        compared = lab.get("/eval-comparisons", a=base["id"], b=adapted["id"], checkpoint="dense")
        assert compared["paired"] > 30
        assert [change["path"] for change in compared["retrieval_diff"]] == [
            "retrievers.dense.adapter"
        ]
        manifest = lab.get(f"/eval-runs/{adapted['id']}/manifest")
        roles = {model_["role"] for model_ in manifest["models"]}
        assert roles == {"adapter", "embed", "rerank"}
        # On the items it was trained on, the adapter lifts dense retrieval.
        assert recall(adapted) > recall(base)

        # One confirmation on the locked test split.
        confirmed = evaluate(
            lab,
            {**inputs, "retrieval_config_id": adapted_config},
            split="test",
            confirm_test_read=True,
        )
        assert confirmed["status"] == "succeeded"
        assert confirmed["test_read"] is True

        # A withdrawn model fails new eval runs that name it.
        withdrawn = lab.post(f"/trained-models/{second['id']}/withdraw")
        assert withdrawn["status"] == "withdrawn"
        stale = lab.save_retrieval(
            hybrid_retrieval(
                "halcyon-withdrawn",
                retrievers={
                    "dense": {"top_n": 50, "adapter": second["id"]},
                    "keyword": {"top_n": 50},
                },
            )
        )
        failed = evaluate(lab, {**inputs, "retrieval_config_id": stale})
        assert failed["status"] == "failed"
        assert failed["error_code"] == "model_withdrawn"

        # The search playground applies the adapter too.
        searched = lab.post(
            "/search",
            {
                "run_id": inputs["run_id"],
                "query": "how fast does PTO accrue for hourly staff",
                "retrieval_config_id": adapted_config,
            },
        )
        assert searched["results"]

        listed = lab.get("/trained-models", name="halcyon-adapter")["items"]
        assert [item["version"] for item in listed] == [2, 1]

        with migrated_engine.connect() as conn:
            actions = conn.execute(
                sa.select(audit_events.c.action, audit_events.c.details).where(
                    audit_events.c.action.like("train%")
                )
            ).all()
        assert {action for action, _ in actions} >= {
            "training_set.start",
            "training_set.export",
            "training_run.start",
            "trained_model.withdraw",
            "trained_model.download",
        }
    # Training sets quote queries and passages: neither reaches a log or an audit event.
    first = pairs[0]
    for text in (first["query"], first["positives"][0]["text"]):
        check = CanaryCheck(text, caplog, capfd)
        check.assert_not_logged()
        check.assert_absent(str(actions), json.dumps(mined), json.dumps(model))


def test_cancelling_a_training_run_and_purging_what_it_was_trained_on(
    migrated_engine: sa.Engine,
    dev_identity: DevIdentity,
    tmp_path: Path,
    serve_app: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs = prepare_halcyon(lab)
        base = evaluate(lab, inputs)
        mined = mine(lab, base["id"])
        assert mined["status"] == "succeeded"

        # A queued run cancels at once; a long one stops at its next step.
        long = {
            "component": "train/query-adapter",
            "params": {"epochs": 1000, "batch_size": 2, "patience": 0},
        }
        started = lab.post(
            "/training-runs",
            {"training_set_id": mined["id"], "name": "halcyon-long", "trainer": long},
            expect=(202,),
        )
        cancelled = lab.post(f"/training-runs/{started['id']}/cancel")
        assert cancelled["status"] in ("cancelled", "cancelling")
        final = lab.wait(f"/training-runs/{started['id']}")
        assert final["status"] == "cancelled"
        assert final["model_id"] is None

        run = train(lab, mined["id"], "halcyon-adapter", ADAPTER)
        assert run["status"] == "succeeded"

        # The command line follows the run and downloads the model.
        monkeypatch.setenv("STRATAFI_URL", serve_app(lab.app))
        monkeypatch.setenv("STRATAFI_DEV_SIGN_IN", "tenant0:editor")
        monkeypatch.delenv("STRATAFI_TOKEN", raising=False)
        assert cli(["train", "show", run["id"]]) == 0
        saved = tmp_path / "halcyon-adapter.zip"
        assert cli(["model", "download", run["model_id"], "--output", str(saved)]) == 0
        with zipfile.ZipFile(saved) as archive:
            assert "model/adapter.npy" in archive.namelist()

        # Purging a document the training set quotes purges the set's records and the model.
        records = lab.get(f"/training-sets/{mined['id']}/records", limit=200)["items"]
        positive = next(r["positives"][0] for r in records if r["positives"])
        documents = lab.get("/documents", limit=200)["items"]
        stem = f"{positive['doc_key']}-v{positive['version']}"
        document = next(d for d in documents if stem in d["name"])
        owner = lab.identity.headers(role="owner")
        response = lab.client.delete(f"/api/v1/documents/{document['id']}", headers=owner)
        assert response.status_code == 200, response.text
        assert response.json()["models"] == 1
        model = lab.get(f"/trained-models/{run['model_id']}")
        assert model["status"] == "withdrawn"
        assert model["stored"] is False
        assert model["withdrawn_by"] == "system:purge"
        gone = lab.client.get(f"/api/v1/training-sets/{mined['id']}/records", headers=lab.editor)
        assert gone.status_code == 409
        assert gone.json()["code"] == "artifact_missing"


@pytest.mark.browser
def test_mining_training_and_evaluating_an_adapter_in_a_browser(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Mine a training set, train an adapter, follow it, open its card and evaluate it on
    dev (web/e2e/training-flow.mjs)."""
    node = shutil.which("node")
    if node is None:
        pytest.fail("the browser flow needs Node 24 and Playwright")
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        inputs = prepare_halcyon(lab)
        evaluation = evaluate(lab, inputs)
        assert evaluation["status"] == "succeeded"
        query = lab.get(f"/eval-runs/{evaluation['id']}/queries", limit=1)["items"][0]
        detail = lab.get(f"/eval-runs/{evaluation['id']}/queries/{query['item_id']}")
        url = serve_web(lab, serve_app)
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [node, str(WEB / "e2e" / "training-flow.mjs"), url, detail["item"]["query"]],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=WEB,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("ok"), result.stdout
