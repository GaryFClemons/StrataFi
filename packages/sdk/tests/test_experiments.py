"""Experiments and knowledge profiles in the SDK and the command: plan,
create, start with confirmation, follow to the end; export, download and import a profile."""

import uuid
from pathlib import Path
from typing import Any

import httpx2
import pytest

from stratafi import Client
from stratafi.cli import build_parser, main

EXPERIMENT_ID = str(uuid.UUID(int=40))
NOW = "2026-10-31T12:00:00Z"
DOCUMENT = {
    "schema": "stratafi.experiment.v1",
    "name": "chunking",
    "corpus": "halcyon",
    "eval_set": {"name": "halcyon-retrieval"},
    "base": {"pipeline": "baseline-v1", "retrieval": "hybrid-v1"},
    "objective": {"metric": "recall@5"},
    "factors": {"chunk.params.max_tokens": [400, 800]},
}
PLAN = {
    "factors": [],
    "strategy": "grid",
    "grid": 2,
    "excluded": 0,
    "trials": 2,
    "work": {
        "pipelines": 2,
        "stages": {"chunk": {"tasks": 42, "unique": 42, "cached": 0, "to_run": 42}},
        "stopped": {},
    },
    "estimate": {
        "wall_seconds": 12.5,
        "cost_usd": {"ingest": 0.01, "eval": 0.0, "total": 0.01},
        "eval": {"trials": 2, "to_run": 2, "queries": 63, "shards": 6},
    },
    "budget": {"trials": {"planned": 2, "max": 60, "over": False}},
    "warnings": [],
}


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATAFI_URL", "http://stratafi.test")
    monkeypatch.setenv("STRATAFI_TOKEN", "token")
    monkeypatch.delenv("STRATAFI_DEV_SIGN_IN", raising=False)


def experiment(status: str = "planned", **changes: Any) -> dict[str, Any]:
    record = {
        "id": EXPERIMENT_ID,
        "name": "chunking",
        "status": status,
        "document": DOCUMENT,
        "document_hash": "a" * 64,
        "corpus_version_id": str(uuid.UUID(int=3)),
        "eval_set_version_id": str(uuid.UUID(int=4)),
        "pipeline_spec_id": str(uuid.UUID(int=2)),
        "retrieval_config_id": str(uuid.UUID(int=5)),
        "plan": PLAN,
        "spent": {"total_usd": 0.0},
        "counts": {"trials": {"pending": 2}},
        "stop_reason": None,
        "error_code": None,
        "created_by": "u-tenant0-editor",
        "started_by": None,
        "created_at": NOW,
        "started_at": None,
        "finished_at": None,
    }
    record.update(changes)
    return record


def planned(valid: bool = True) -> dict[str, Any]:
    issues = [] if valid else [{"path": "corpus", "code": "not_found"}]
    return {
        "valid": valid,
        "issues": issues,
        "document": DOCUMENT if valid else None,
        "plan": PLAN if valid else None,
        "trials": [],
    }


@pytest.fixture
def experiment_file(tmp_path: Path) -> str:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "schema: stratafi.experiment.v1\nname: chunking\ncorpus: halcyon\n"
        "eval_set: {name: halcyon-retrieval}\n"
        "base: {pipeline: baseline-v1, retrieval: hybrid-v1}\n"
        "objective: {metric: recall@5}\nfactors:\n  chunk.params.max_tokens: [400, 800]\n",
        encoding="utf-8",
    )
    return str(path)


def test_the_sdk_plans_creates_and_starts_an_experiment(api: Any, experiment_file: str) -> None:
    api.on("POST", "/experiments/plan", planned())
    api.on("POST", "/experiments", experiment(), status=201)
    api.on("POST", f"/experiments/{EXPERIMENT_ID}/start", experiment("running"), status=202)
    api.on("GET", f"/experiments/{EXPERIMENT_ID}/trials", {"items": []})
    with Client("http://stratafi.test", token="t", transport=api.transport) as client:
        plan = client.experiments.plan(experiment_file)
        assert plan.valid
        # A file's YAML is read here, in order: the factors keep the order they were written.
        assert api.body("POST", "/experiments/plan") == {"document": DOCUMENT}
        created = client.experiments.create(DOCUMENT)
        assert created.status == "planned"
        assert created.plan["trials"] == 2
        started = created.start()
        assert started.status == "running"
        assert api.body("POST", f"/experiments/{EXPERIMENT_ID}/start") == {"confirm": True}
        assert list(started.trials()) == []
        # YAML text goes to the API as it is.
        client.experiments.plan("schema: stratafi.experiment.v1\nname: x\n")
        assert api.body("POST", "/experiments/plan") == {
            "content": "schema: stratafi.experiment.v1\nname: x\n"
        }


def test_experiment_commands_parse() -> None:
    parser = build_parser()
    for argv in (
        "experiment plan experiment.yaml",
        "experiment start experiment.yaml --confirm --wait --timeout 60",
        f"experiment show {EXPERIMENT_ID}",
        f"experiment wait {EXPERIMENT_ID}",
        f"experiment cancel {EXPERIMENT_ID}",
    ):
        assert callable(parser.parse_args(argv.split()).handler)


def test_plan_prints_the_unique_work(
    api: Any, experiment_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    api.on("POST", "/experiments/plan", planned())
    assert main(["experiment", "plan", experiment_file], transport=api.transport) == 0
    out = capsys.readouterr().out
    assert "2 trials (grid; 2 combinations, 0 excluded) over 2 pipeline runs" in out
    assert "| chunk | 42 | 42 | 0 | 42 |" in out
    api.on("POST", "/experiments/plan", planned(valid=False))
    assert main(["experiment", "plan", experiment_file], transport=api.transport) == 1
    assert "corpus: not_found" in capsys.readouterr().err
    halving = planned()
    halving["plan"] = {
        **PLAN,
        "strategy": "successive-halving",
        "search": {"eta": 3, "rungs": [{"queries": 10}, {"queries": 63}], "share": 0.4},
    }
    api.on("POST", "/experiments/plan", halving)
    assert main(["experiment", "plan", experiment_file], transport=api.transport) == 0
    out = capsys.readouterr().out
    assert "Successive halving (eta 3): 10 queries -> 63 queries; nominally 40%" in out


def test_starting_needs_confirmation_and_waits_for_the_end(
    api: Any, experiment_file: str, sse_response: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["experiment", "start", experiment_file], transport=api.transport) == 2
    assert not api.sent("POST", "/experiments")
    api.on("POST", "/experiments", experiment(), status=201)
    api.on("POST", f"/experiments/{EXPERIMENT_ID}/start", experiment("running"), status=202)
    stopped = experiment("stopped", stop_reason="max_trials", counts={"trials": {"succeeded": 1}})
    api.on("GET", f"/experiments/{EXPERIMENT_ID}", experiment("running"), stopped)
    api.on("GET", "/events", sse_response(("experiment", {"run_id": EXPERIMENT_ID})))
    argv = ["experiment", "start", experiment_file, "--confirm", "--wait"]
    assert main(argv, transport=api.transport) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"experiment {EXPERIMENT_ID} stopped (trials: 1 succeeded)")
    assert "[stopped by max_trials]" in out
    failed = experiment("failed", error_code="x")
    api.on("GET", f"/experiments/{EXPERIMENT_ID}", failed)
    assert main(["experiment", "wait", EXPERIMENT_ID], transport=api.transport) == 1


PROFILE_ID = str(uuid.UUID(int=50))
TRIAL_ID = str(uuid.UUID(int=41))


def stored_profile(**changes: Any) -> dict[str, Any]:
    record = {
        "id": PROFILE_ID,
        "name": "kp-policies",
        "version": 2,
        "status": "draft",
        "source": "trial",
        "document": {},
        "document_hash": "b" * 64,
        "experiment_id": EXPERIMENT_ID,
        "trial_id": TRIAL_ID,
        "approvals": [],
        "created": True,
        "created_by": "u-tenant0-editor",
        "created_at": NOW,
    }
    record.update(changes)
    return record


def test_profiles_export_download_and_import(
    api: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api.on("POST", "/profiles", stored_profile(), status=201)
    argv = ["profile", "export", TRIAL_ID, "--experiment", EXPERIMENT_ID, "--name", "kp-policies"]
    assert main(argv, transport=api.transport) == 0
    assert api.body("POST", "/profiles") == {
        "experiment_id": EXPERIMENT_ID,
        "trial_id": TRIAL_ID,
        "name": "kp-policies",
        "description": "",
    }
    assert capsys.readouterr().out.startswith("profile kp-policies v2 draft")
    text = "schema: stratafi.knowledge-profile.v1\nname: kp-policies\n"
    api.on(
        "GET",
        f"/profiles/{PROFILE_ID}/export",
        httpx2.Response(200, text=text, headers={"Content-Type": "application/yaml"}),
    )
    assert main(["profile", "download", PROFILE_ID], transport=api.transport) == 0
    assert capsys.readouterr().out == text
    path = tmp_path / "profile.yaml"
    path.write_text(text, encoding="utf-8")
    api.on("POST", "/profiles/import", stored_profile(created=False), status=200)
    assert main(["profile", "import", str(path)], transport=api.transport) == 0
    assert api.body("POST", "/profiles/import") == {"content": text}
    assert "the latest version holds this profile already" in capsys.readouterr().out
