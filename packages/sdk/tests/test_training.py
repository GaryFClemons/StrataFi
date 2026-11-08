"""Post-training in the SDK and the CLI: mining a training set, training a
model, exports, the model registry and a model's zip."""

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from stratafi import Client
from stratafi.cli import build_parser, main
from stratafi.errors import ConfigError

SET_ID = "00000000-0000-0000-0000-0000000000a1"
RUN_ID = "00000000-0000-0000-0000-0000000000b2"
MODEL_ID = "00000000-0000-0000-0000-0000000000c3"
EVAL_ID = "00000000-0000-0000-0000-0000000000d4"
NOW = "2026-11-05T10:00:00Z"


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATAFI_URL", "http://stratafi.test")
    monkeypatch.setenv("STRATAFI_TOKEN", "token")
    monkeypatch.delenv("STRATAFI_DEV_SIGN_IN", raising=False)


def training_set(status: str = "succeeded") -> dict[str, Any]:
    return {
        "id": SET_ID,
        "name": "halcyon-retrieval",
        "eval_run_id": EVAL_ID,
        "status": status,
        "options": {},
        "cache_key": "a" * 64 if status == "succeeded" else None,
        "cached": False,
        "context": {},
        "stats": {
            "records": 63,
            "pairs": 52,
            "abstentions": 11,
            "positives": 63,
            "negatives": 498,
            "validation": 13,
            "leak_guard": {"near_duplicates": 0},
        }
        if status == "succeeded"
        else None,
        "error_code": None,
        "created_by": "dev",
        "created_at": NOW,
        "started_at": None,
        "finished_at": None,
    }


def training_run(status: str = "succeeded") -> dict[str, Any]:
    done = status == "succeeded"
    return {
        "id": RUN_ID,
        "training_set_id": SET_ID,
        "name": "halcyon-adapter",
        "kind": "query-adapter",
        "trainer": "train/query-adapter",
        "trainer_version": "1.0.0",
        "params": {"epochs": 2},
        "base": {"model": "embed-default"},
        "status": status,
        "progress": {"epoch": 2, "epochs": 2},
        "metrics": {
            "baseline": {"recall@10": 0.6, "mrr@10": 0.4},
            "epochs": [
                {"epoch": 1, "train_loss": 2.1, "recall@10": 0.7, "mrr@10": 0.5},
                {"epoch": 2, "train_loss": 1.4, "recall@10": 0.8, "mrr@10": 0.6},
            ],
            "best_epoch": 2,
        }
        if done
        else None,
        "manifest_hash": None,
        "model_key": "b" * 64 if done else None,
        "model_cached": False,
        "model_id": MODEL_ID if done else None,
        "error_code": None,
        "created_by": "dev",
        "created_at": NOW,
        "started_at": None,
        "finished_at": None,
    }


def model(status: str = "ready") -> dict[str, Any]:
    return {
        "id": MODEL_ID,
        "name": "halcyon-adapter",
        "version": 1,
        "kind": "query-adapter",
        "training_run_id": RUN_ID,
        "model_key": "b" * 64,
        "base": {"model": "embed-default", "revision": "r1"},
        "status": status,
        "stored": True,
        "card": {"use": {"how": f"retrievers.dense.adapter: {MODEL_ID}"}},
        "withdrawn_by": None,
        "withdrawn_at": None,
        "created_by": "dev",
        "created_at": NOW,
    }


@pytest.mark.parametrize(
    "argv",
    [
        f"train mine {EVAL_ID} --max-negatives 4 --validation-share 0.25 --wait",
        f"train export {SET_ID} --as chat --split train --output out.jsonl",
        f"train start {SET_ID} --name hc-adapter --param epochs=40 --param in_batch=false --wait",
        f"train start {SET_ID} --name halcyon-minilm --trainer train/sentence-transformers",
        f"train show {RUN_ID}",
        f"train wait {RUN_ID} --timeout 60",
        f"train cancel {RUN_ID}",
        "model list --kind query-adapter",
        f"model show {MODEL_ID}",
        f"model withdraw {MODEL_ID}",
        f"model download {MODEL_ID} --output halcyon-adapter.zip",
    ],
)
def test_the_training_commands_parse(argv: str) -> None:
    args = build_parser().parse_args(argv.split())
    assert callable(args.handler)


def test_mining_and_training_through_the_sdk(api: Any) -> None:
    api.on("POST", "/training-sets", training_set("queued"))
    api.on("GET", f"/training-sets/{SET_ID}", training_set())
    api.on("POST", "/training-runs", training_run("queued"))
    api.on("GET", f"/training-runs/{RUN_ID}", training_run())
    with Client("http://stratafi.test", token="token", transport=api.transport) as client:
        mined = client.training.mine(EVAL_ID, options={"max_negatives": 4}, wait=True)
        assert mined.succeeded
        sent = json.loads(api.requests[0].content)
        assert sent == {"eval_run_id": EVAL_ID, "options": {"max_negatives": 4}}
        run = client.training.train(mined, name="halcyon-adapter", params={"epochs": 2}, wait=True)
        assert run.succeeded
        assert run.model_id == MODEL_ID
        body = json.loads(
            next(r for r in api.requests if r.url.path.endswith("/training-runs")).content
        )
        assert body["trainer"] == {"component": "train/query-adapter", "params": {"epochs": 2}}
        with pytest.raises(ConfigError):
            client.training.train(mined, name="Not A Name")
        with pytest.raises(ConfigError):
            mined.cancel()


def test_records_exports_and_the_registry(api: Any, tmp_path: Path) -> None:
    api.on("GET", f"/training-sets/{SET_ID}", training_set())
    records = [{"item_id": f"q{n}", "split": "train"} for n in range(3)]
    api.on(
        "GET",
        f"/training-sets/{SET_ID}/records",
        {"items": records[:2], "total": 3, "offset": 0},
        {"items": records[2:], "total": 3, "offset": 2},
    )
    api.on(
        "GET",
        f"/training-sets/{SET_ID}/export",
        httpx2.Response(200, text='{"anchor":"a"}\n{"anchor":"b"}\n'),
    )
    api.on("GET", "/trained-models", {"items": [model()]})
    api.on("GET", f"/trained-models/{MODEL_ID}", model())
    api.on("POST", f"/trained-models/{MODEL_ID}/withdraw", model("withdrawn"))
    api.on("GET", f"/trained-models/{MODEL_ID}/download", httpx2.Response(200, content=b"PK\x03"))
    with Client("http://stratafi.test", token="token", transport=api.transport) as client:
        found = client.training.get_set(SET_ID)
        assert [record["item_id"] for record in found.records()] == ["q0", "q1", "q2"]
        assert found.export("cross-encoder").splitlines() == ['{"anchor":"a"}', '{"anchor":"b"}']
        export = next(r for r in api.requests if r.url.path.endswith("/export"))
        assert export.url.params["format"] == "cross-encoder"
        assert [m.name for m in client.models.list(kind="query-adapter")] == ["halcyon-adapter"]
        assert client.models.withdraw(MODEL_ID).status == "withdrawn"
        target = client.models.download(MODEL_ID, tmp_path / "model.zip")
        assert target.read_bytes() == b"PK\x03"


def test_the_cli_trains_and_exits_by_outcome(
    api: Any, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    api.on("POST", "/training-runs", training_run("queued"))
    api.on("GET", f"/training-runs/{RUN_ID}", training_run())
    argv = ["train", "start", SET_ID, "--name", "halcyon-adapter", "--param", "epochs=2", "--wait"]
    assert main(argv, transport=api.transport) == 0
    out = capsys.readouterr().out
    assert "| 2 (kept) | 1.4 | 0.8 | 0.6 |" in out
    assert f"model: {MODEL_ID}" in out
    body = json.loads(next(r for r in api.requests if r.method == "POST").content)
    assert body["trainer"]["params"] == {"epochs": 2}  # YAML read the number as a number
    api.on("GET", f"/training-runs/{RUN_ID}", training_run("failed"))
    assert main(["train", "wait", RUN_ID], transport=api.transport) == 1
    api.on("GET", f"/trained-models/{MODEL_ID}/download", httpx2.Response(200, content=b"zip"))
    output = tmp_path / "m.zip"
    assert (
        main(["model", "download", MODEL_ID, "--output", str(output)], transport=api.transport) == 0
    )
    assert output.read_bytes() == b"zip"
    assert main(["train", "start", SET_ID, "--name", "x", "--param", "nope"]) == 2
