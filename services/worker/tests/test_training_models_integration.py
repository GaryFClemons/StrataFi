"""Fine-tuned embedders and rerankers, measured in the lab.

The Halcyon demo corpus through an in-process MiniLM embedder; a training set mined from its dev
eval run; ``train/sentence-transformers`` fine-tunes the embedder and ``train/cross-encoder`` the
reranker; each trained model is then used as ``trained/<model ID>`` (a pipeline run that
re-embeds with the fine-tuned embedder, a retrieval config that reranks with the fine-tuned
cross-encoder) and compared with its base on the dev split, then confirmed once on the test
split. It downloads two small models from Hugging Face and trains on CPU in a few minutes, so it
runs only with the ``sentence-transformers`` extra and ``STRATAFI_TEST_HF_MODELS=1``.
"""

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import (
    HALCYON_RETRIEVAL,
    Lab,
    baseline_pipeline,
    hybrid_retrieval,
    running_lab,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(3600),
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None
        or importlib.util.find_spec("sentence_transformers") is None
        or os.environ.get("STRATAFI_TEST_HF_MODELS") != "1",
        reason="needs the sentence-transformers extra and STRATAFI_TEST_HF_MODELS=1 (downloads)",
    ),
]

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def pipeline(model: str) -> dict[str, Any]:
    spec = baseline_pipeline()
    spec["name"] = "halcyon-minilm"
    spec["stages"]["embed"] = {
        "component": "embed/sentence-transformers",
        "params": {"model": model},
    }
    return spec


def fresh(lab: Lab) -> None:
    """Mint the lab's sign-in again: tokens last five minutes, and this test runs longer."""
    lab.editor.update(lab.identity.headers(role="editor"))


def wait(lab: Lab, path: str, *, timeout: float) -> dict[str, Any]:
    """``lab.wait`` with a fresh token on every poll."""
    deadline = time.monotonic() + timeout
    while True:
        fresh(lab)
        found = lab.get(path)
        if found["status"] in ("succeeded", "failed", "cancelled"):
            return found
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} didn't finish")
        time.sleep(1.0)


def evaluate(lab: Lab, run_id: str, config_id: str, version_id: str, **body: Any) -> dict[str, Any]:
    fresh(lab)
    started = lab.post(
        "/eval-runs",
        {
            "run_id": run_id,
            "retrieval_config_id": config_id,
            "eval_set_version_id": version_id,
            **body,
        },
        expect=(202,),
    )
    found = wait(lab, f"/eval-runs/{started['id']}", timeout=900)
    assert found["status"] == "succeeded", found
    return found


def score(run: dict[str, Any], checkpoint: str, metric: str) -> float:
    return float(run["metrics"]["metrics"]["knowledge-hit-v1"][checkpoint][metric]["mean"])


def train(lab: Lab, training_set_id: str, name: str, trainer: dict[str, Any]) -> dict[str, Any]:
    fresh(lab)
    started = lab.post(
        "/training-runs",
        {"training_set_id": training_set_id, "name": name, "trainer": trainer},
        expect=(202,),
    )
    run = wait(lab, f"/training-runs/{started['id']}", timeout=1800)
    assert run["status"] == "succeeded", run
    return run


def compare(lab: Lab, a: dict[str, Any], b: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    fresh(lab)
    found = lab.get("/eval-comparisons", a=a["id"], b=b["id"], checkpoint=checkpoint)
    return {
        item["metric"]: [round(item[k], 3) for k in ("mean_a", "mean_b", "delta", "low", "high")]
        + [item["p_value"]]
        for item in found["metrics"]
        if item.get("delta") is not None
    }


def test_fine_tuned_models_against_their_bases(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    slots = {"cpu": 2, "io": 3, "gpu": 1}
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app, slots=slots) as lab:
        version = lab.import_manifest("halcyon", "demo-corpus/corpus")
        base_run = wait(
            lab,
            f"/runs/{lab.start_run(lab.save_spec(pipeline(EMBEDDER)), version)['id']}",
            timeout=900,
        )
        assert base_run["status"] == "succeeded", base_run
        imported = lab.import_eval_set(
            "halcyon-retrieval", HALCYON_RETRIEVAL.read_text(encoding="utf-8"), seed=0
        )
        eval_set = imported["version"]["id"]
        dense_only = lab.save_retrieval(
            hybrid_retrieval(
                "halcyon-dense",
                retrievers={"dense": {"top_n": 50}, "keyword": None},
                fusion={"method": "none"},
                rerank={"component": "rerank/none"},
            )
        )
        reranked = lab.save_retrieval(
            hybrid_retrieval(
                "halcyon-cross-encoder",
                rerank={"component": "rerank/cross-encoder", "params": {"model": RERANKER}},
            )
        )
        base_dev = evaluate(lab, base_run["id"], dense_only, eval_set)
        started = lab.post("/training-sets", {"eval_run_id": base_dev["id"]}, expect=(202,))
        mined = wait(lab, f"/training-sets/{started['id']}", timeout=600)
        assert mined["status"] == "succeeded", mined

        # The embedder, fine-tuned and re-embedding the corpus.
        embedder = train(
            lab,
            mined["id"],
            "halcyon-minilm",
            {
                "component": "train/sentence-transformers",
                "params": {"model": EMBEDDER, "epochs": 3, "batch_size": 16, "device": "cpu"},
            },
        )
        assert embedder["kind"] == "embedder"
        fresh(lab)
        card = lab.get(f"/trained-models/{embedder['model_id']}")["card"]
        assert card["use"]["compatibility"] == "C"
        spec = lab.save_spec(pipeline(f"trained/{embedder['model_id']}"))
        trained_run = wait(lab, f"/runs/{lab.start_run(spec, version)['id']}", timeout=900)
        assert trained_run["status"] == "succeeded", trained_run
        tuned_dev = evaluate(lab, trained_run["id"], dense_only, eval_set)

        # The reranker, fine-tuned on a set mined from the hybrid search's candidates.
        base_rerank_dev = evaluate(lab, base_run["id"], reranked, eval_set)
        fresh(lab)
        started = lab.post("/training-sets", {"eval_run_id": base_rerank_dev["id"]}, expect=(202,))
        candidates = wait(lab, f"/training-sets/{started['id']}", timeout=600)
        assert candidates["status"] == "succeeded", candidates
        reranker = train(
            lab,
            candidates["id"],
            "halcyon-cross-encoder",
            {
                "component": "train/cross-encoder",
                "params": {"model": RERANKER, "epochs": 2, "batch_size": 16, "device": "cpu"},
            },
        )
        fresh(lab)
        tuned_rerank = lab.save_retrieval(
            hybrid_retrieval(
                "halcyon-cross-encoder-tuned",
                rerank={
                    "component": "rerank/cross-encoder",
                    "params": {"model": f"trained/{reranker['model_id']}"},
                },
            )
        )
        tuned_rerank_dev = evaluate(lab, base_run["id"], tuned_rerank, eval_set)

        # One confirmation each on the locked test split.
        confirm = {"split": "test", "confirm_test_read": True}
        base_test = evaluate(lab, base_run["id"], dense_only, eval_set, **confirm)
        tuned_test = evaluate(lab, trained_run["id"], dense_only, eval_set, **confirm)
        base_rerank_test = evaluate(lab, base_run["id"], reranked, eval_set, **confirm)
        tuned_rerank_test = evaluate(lab, base_run["id"], tuned_rerank, eval_set, **confirm)

        report = {
            "training_sets": {"dense": mined["stats"], "candidates": candidates["stats"]},
            "embedder": {
                "training": embedder["metrics"],
                "dev": compare(lab, base_dev, tuned_dev, "dense"),
                "test": compare(lab, base_test, tuned_test, "dense"),
            },
            "reranker": {
                "training": reranker["metrics"],
                "dev": compare(lab, base_rerank_dev, tuned_rerank_dev, "final"),
                "test": compare(lab, base_rerank_test, tuned_rerank_test, "final"),
            },
        }
        print("FINE-TUNING REPORT", json.dumps(report))
        # Trained on most of the dev split's items, neither model loses ground there.
        assert score(tuned_dev, "dense", "mrr@10") >= score(base_dev, "dense", "mrr@10") - 0.02
        assert (
            score(tuned_rerank_dev, "final", "mrr@10")
            >= score(base_rerank_dev, "final", "mrr@10") - 0.02
        )
