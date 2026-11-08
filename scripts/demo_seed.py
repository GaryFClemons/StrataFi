#!/usr/bin/env python3
"""Seed a lab with no models: the Halcyon Coffee demo corpus (``fixtures/demo-corpus``) through a
pipeline that needs no model gateway, its eval set, and three eval runs, so the Evals table,
Compare, Maps and Models have something to show on first sign-in. By default it seeds compose's
app profile (``docker compose --profile app up -d --wait``).

    uv run python scripts/demo_seed.py
    uv run python scripts/demo_seed.py --url http://127.0.0.1:8000    # another API, and --web

The pipeline uses the default components (the default parse route and ``chunk/sections``), embeds
with ``embed/fake-hash``, and one config reranks with ``rerank/fake-overlap``: the lab-only
stand-ins for real models, in process. Their numbers show the screens, not a model's quality.
``--pipeline`` seeds with another spec instead, such as one that embeds through the model
gateway (``scripts/local.py seed`` does that with Ollama's models).

It can run again, after a fix or for more data: the corpus, the specs and the eval set stay one
version each, and each run adds a pipeline run (its finished work cached) and three eval runs.
It signs in as the local owner through the API's development route, as the web app does, and
prints names, IDs, codes and numbers only. Exit codes: 0 seeded, 1 the run indexed nothing or an
eval run didn't succeed, 2 a usage or configuration error, 3 the API is unreachable or failed.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from stratafi import ApiError, Client, ConfigError, StratafiError

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "demo-corpus"
MANIFEST = FIXTURES / "corpus" / "manifest.yaml"
EVAL_SET = FIXTURES / "retrieval.yaml"
CORPUS = "halcyon"
EVAL_SET_NAME = "halcyon-retrieval"
TIMEOUT = 1800.0

PIPELINE: dict[str, Any] = {
    "schema": "stratafi.pipeline.v1",
    "name": "demo-baseline",
    "description": "The default components with the lab's fake-hash embedder: no gateway needed",
    "stages": {
        # ClamAV is optional on one machine: add scan/clamav where a clamd runs.
        "scan": {"checks": [{"component": "scan/content-safety"}, {"component": "scan/dedupe"}]},
        "clean": [{"component": "clean/whitespace"}],
        "chunk": {"component": "chunk/sections"},
        "embed": {"component": "embed/fake-hash"},
    },
}


def _retrieval(name: str, **changes: Any) -> dict[str, Any]:
    return {
        "schema": "stratafi.retrieval.v1",
        "name": name,
        "retrievers": {"dense": {"top_n": 50}, "keyword": {"top_n": 50}},
        "fusion": {"method": "rrf", "k": 60},
        "candidates": 50,
        "filters": {"current_only": True},
        "rerank": {"component": "rerank/fake-overlap"},
        "top_k": 8,
        **changes,
    }


# Two configs that differ only in the reranker make a controlled pair for the Models screen.
RETRIEVAL: list[dict[str, Any]] = [
    _retrieval("demo-hybrid"),
    _retrieval("demo-hybrid-no-rerank", rerank="rerank/none"),
    _retrieval(
        "demo-dense-only",
        retrievers={"dense": {"top_n": 50}, "keyword": None},
        fusion={"method": "none"},
        rerank="rerank/none",
    ),
]


def load_pipeline(path: Path | None) -> dict[str, Any]:
    if path is None:
        return PIPELINE
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        raise ConfigError(f"can't read the pipeline spec {path}") from None
    if not isinstance(document, dict):
        raise ConfigError(f"{path} isn't a pipeline spec")
    return document


def why_nothing_was_indexed(client: Client, run_id: str) -> list[str]:
    """Why a run's documents went no further: its failed tasks' error codes, and the reasons its
    scan checks gave (from a few scan reports). Codes only, never a document's text."""
    reasons = {
        f"{task['stage']}: {task.get('error_code') or 'failed'}"
        for task in client.get(f"/runs/{run_id}/tasks", status="failed", limit=200)["items"]
    }
    scans = client.get(f"/runs/{run_id}/tasks", stage="scan", limit=200)["items"]
    for key in list(dict.fromkeys(task["cache_key"] for task in scans if task["cache_key"]))[:3]:
        report = client.get(f"/artifacts/{key}/content")
        for check in report.get("checks", []) if isinstance(report, dict) else []:
            if check.get("outcome") not in ("clean", "skipped"):
                reasons.add(f"{check.get('component')}: {check.get('reason') or check['outcome']}")
    return sorted(reasons)


def seed(client: Client, *, pipeline: dict[str, Any], web: str) -> int:
    try:
        client.corpora.create(CORPUS, description="Halcyon Coffee Co.'s invented handbook")
    except ApiError as error:
        if error.status != 409:  # 409: an earlier seed made it; uploading again changes nothing
            raise
    client.corpora.upload(CORPUS, manifest=MANIFEST)
    version = client.corpora.freeze(CORPUS)
    print(f"corpus {CORPUS} v{version.version}: {version.document_count} documents")

    saved = client.pipelines.push(pipeline)
    run = client.runs.start(saved, corpus=version, wait=True, timeout=TIMEOUT)
    entries = run.record.counts.entries or {}
    indexed = entries.get("done", 0)
    total, label = sum(entries.values()), f"{saved.name} v{saved.version}"
    print(f"run {run.id} ({label}): {run.status}, {indexed} of {total} documents indexed")
    if run.status != "succeeded" or not indexed:
        for reason in why_nothing_was_indexed(client, run.id):
            print(f"  {reason}")
            if reason.endswith("signatures_stale"):
                print("  ClamAV's signatures are over 72 hours old: freshclam must reach its")
                print("  mirror.")
        return 1

    imported = client.eval_sets.import_file(EVAL_SET, name=EVAL_SET_NAME, test_fraction=0.3)
    print(f"eval set {EVAL_SET_NAME} v{imported.version.version}: {imported.report.items} items")
    eval_runs = []
    for document in RETRIEVAL:
        eval_run = client.evals.run(
            run, retrieval=document, eval_set=imported.version, wait=True, timeout=TIMEOUT
        )
        headline = eval_run.metrics.headline() if eval_run.status == "succeeded" else {}
        numbers = ", ".join(
            f"{name} {value:.3f}" for name, value in headline.items() if value is not None
        )
        print(f"eval run {eval_run.id} ({document['name']}): {eval_run.status}; {numbers}")
        if eval_run.status != "succeeded":
            return 1
        eval_runs.append(eval_run.id)
    print(f"Compare them: {web}/compare/{'/'.join(eval_runs)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8100", help="the lab's API")
    parser.add_argument("--web", default="http://127.0.0.1:4100/stratafi", help="its web app")
    parser.add_argument("--sign-in", default="local:owner", help="the synthetic user, tenant:role")
    parser.add_argument("--pipeline", type=Path, help="a pipeline spec to seed with instead")
    args = parser.parse_args(argv)
    try:
        client = Client(args.url, dev_sign_in=args.sign_in)
        return seed(client, pipeline=load_pipeline(args.pipeline), web=args.web.rstrip("/"))
    except ConfigError as error:
        print(f"demo_seed: {error}", file=sys.stderr)
        return 2
    except ApiError as error:
        code = error.code or error.title
        print(f"demo_seed: the API answered {error.status} {code}", file=sys.stderr)
        return 2 if 400 <= error.status < 500 and error.status != 429 else 3
    except StratafiError as error:  # unreachable, a wait that timed out
        print(f"demo_seed: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
