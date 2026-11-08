"""The load test: a synthetic corpus through the lab, then a 1,000-query evaluation.

    STRATAFI_TEST_PG_ADMIN_DSN=postgresql://... uv run python scripts/load_test.py \\
        --documents 1000 --chunks 100 --queries 1000 --cpu-slots 4 --out load.json

The full scale is ``--documents 10000 --chunks 100`` (10,000 documents, a million chunks), for
a lab host. The test runs the API and a worker in this process against a new
database (through ``STRATAFI_TEST_PG_ADMIN_DSN``, a superuser, as the integration tier does)
and a local artifact store, with the fakes for clamd and the model gateway, so it measures
Stratafi rather than a model:

1. **Corpus.** ``--documents`` synthetic Markdown documents, each with ``--chunks`` sections of
   ``--chunk-chars`` characters and a phrase found nowhere else, written with a manifest.
2. **Ingest.** The folder imported through the manifest, frozen, and run through scan, parse,
   chunk (``chunk/sections``, sized so each section is a chunk), embed (the fake gateway's 384
   dimensions) and index (exact dense and Postgres full-text). The queue is sampled every
   second through the metrics' collector: its depth and the oldest job's age.
3. **Evaluation.** ``--queries`` queries, each a document's unique phrase, as one eval set, run
   on CPU with hybrid retrieval (BM25 and exact dense, RRF), no reranker.

It prints, and with ``--out`` writes, JSON: throughput (documents and chunks a second), stage
durations, the queue's peaks, and the evaluation's wall time, latency percentiles and recall.
Every document is invented; no text reaches the output.
"""

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

WORDS = [
    f"{a}{b}{c}"
    for a in ("ka", "lo", "mi", "ne", "pu", "ra", "so", "ti", "va", "ze")
    for b in ("ber", "dan", "fel", "gor", "hal", "jun", "kir", "lem", "mos", "nul")
    for c in ("a", "e", "i", "o", "u", "ar", "en", "is", "on", "um")
]


def sentence(rng: random.Random, words: int = 12) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(words)).capitalize() + "."


def write_corpus(folder: Path, documents: int, chunks: int, chunk_chars: int) -> list[str]:
    """The documents and their manifest; returns each document's unique phrase."""
    rng = random.Random(7)  # noqa: S311 - synthetic text, not secrets
    phrases = []
    entries = []
    for number in range(documents):
        phrase = f"anchor{number:06d} {rng.choice(WORDS)} {rng.choice(WORDS)}"
        phrases.append(phrase)
        lines = [f"# Synthetic policy {number}", ""]
        for section in range(chunks):
            body: list[str] = []
            if section == 0:
                body.append(f"This policy covers {phrase}.")
            while sum(len(part) + 1 for part in body) < chunk_chars - 80:
                body.append(sentence(rng))
            lines += [f"## Section {section}", "", " ".join(body), ""]
        path = f"doc-{number:06d}.md"
        (folder / path).write_text("\n".join(lines), encoding="utf-8")
        entries.append(
            f"  - {{path: {path}, collection: load, doc_key: doc-{number:06d}, version: '1', "
            f"effective_from: 2026-01-01, title: Synthetic policy {number}, "
            f"acl: {{groups: [all-staff], users: []}}, "
            f"status: published, expect: indexed}}"
        )
    manifest = ["tenant: tenant0", "documents:", *entries, ""]
    (folder / "manifest.yaml").write_text("\n".join(manifest), encoding="utf-8")
    return phrases


def pipeline(chunk_chars: int) -> dict[str, Any]:
    return {
        "schema": "stratafi.pipeline.v1",
        "name": "load",
        "stages": {
            "scan": {
                "checks": [
                    {"component": "scan/content-safety"},
                    {"component": "scan/clamav", "params": {"max_signature_age_hours": 72}},
                    {"component": "scan/dedupe"},
                ]
            },
            "parse": {"route": {"default": {"component": "parse/markdown"}}},
            "clean": [],
            "chunk": {
                # A section and its heading fit in one chunk, so each section is a chunk.
                "component": "chunk/sections",
                "params": {
                    "max_tokens": max(16, chunk_chars // 4 + 32),
                    "min_tokens": 0,
                    "max_chunks": 100_000,
                },
                "index_text": "{text}",
            },
            "embed": {"component": "embed/gateway", "params": {"alias": "embed-default"}},
            "index": {
                "dense": {"component": "index/exact"},
                "keyword": {"component": "index/postgres-fts"},
            },
        },
    }


RETRIEVAL = {
    "schema": "stratafi.retrieval.v1",
    "name": "load-hybrid",
    "retrievers": {"dense": {"top_n": 50}, "keyword": {"top_n": 50}},
    "fusion": {"method": "rrf", "k": 60},
    "candidates": 50,
    "top_k": 5,
}


def eval_set(phrases: list[str], queries: int) -> str:
    rng = random.Random(11)  # noqa: S311 - a sample, not a secret
    chosen = rng.sample(range(len(phrases)), min(queries, len(phrases)))
    while len(chosen) < queries:  # more queries than documents: ask again
        chosen.append(rng.randrange(len(phrases)))
    lines = []
    for index, number in enumerate(chosen):
        item = {
            "id": f"q{index:05d}",
            "query": phrases[number],
            "asker": "staff",
            "expected": [{"doc_key": f"doc-{number:06d}", "version": "1"}],
        }
        lines.append(json.dumps(item))
    return "\n".join(lines) + "\n"


def headline(metrics: dict[str, Any], name: str) -> Any:
    """A metric's estimate after the final checkpoint, under the first match rule."""
    for checkpoints in (metrics.get("metrics") or {}).values():
        value = ((checkpoints or {}).get("final") or {}).get(name)
        if value is not None:
            return value
    return None


def sample_queue(engine: Any) -> tuple[int, float]:
    """Active jobs and the oldest runnable job's age, from the metrics' collector."""
    from stratafi_store.jobs import QueueCollector

    depth, oldest = 0, 0.0
    for family in QueueCollector(engine).collect():
        for sample in family.samples:
            if family.name == "stratafi_queue_jobs":
                depth += int(sample.value)
            else:
                oldest = max(oldest, float(sample.value))
    return depth, oldest


def stage_seconds() -> dict[str, dict[str, float]]:
    from stratafi_core.observability import REGISTRY

    stages: dict[str, dict[str, float]] = {}
    for family in REGISTRY.collect():
        if family.name != "stratafi_stage_task_duration_seconds":
            continue
        for sample in family.samples:
            stage = sample.labels["stage"]
            if sample.name.endswith("_sum"):
                stages.setdefault(stage, {})["seconds"] = round(sample.value, 2)
            elif sample.name.endswith("_count"):
                stages.setdefault(stage, {})["tasks"] = int(sample.value)
    return stages


def run(args: argparse.Namespace) -> dict[str, Any]:
    import os

    from stratafi_store.db import create_engine
    from stratafi_store.migrations import upgrade
    from stratafi_testing.databases import DatabaseFactory
    from stratafi_testing.identity import DevIdentity
    from stratafi_testing.lab import running_lab
    from stratafi_testing.servers import ServerThread

    admin = os.environ.get("STRATAFI_TEST_PG_ADMIN_DSN")
    if not admin:
        raise SystemExit("load_test: set STRATAFI_TEST_PG_ADMIN_DSN to a Postgres 16 superuser")
    factory = DatabaseFactory(admin)
    servers: list[ServerThread] = []

    def serve(app: Any) -> str:
        server = ServerThread(app).start()
        servers.append(server)
        return server.url

    report: dict[str, Any] = {
        "documents": args.documents,
        "chunks_per_document": args.chunks,
        "chunk_chars": args.chunk_chars,
        "queries": args.queries,
        "slots": {"cpu": args.cpu_slots, "io": args.io_slots},
    }
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            started = time.monotonic()
            phrases = write_corpus(corpus, args.documents, args.chunks, args.chunk_chars)
            report["generate_seconds"] = round(time.monotonic() - started, 1)
            engine = create_engine(factory.create(), pool_size=20, max_overflow=10)
            upgrade(engine)
            identity = DevIdentity.create(root / "keys")
            slots = {"cpu": args.cpu_slots, "io": args.io_slots}
            with running_lab(
                engine, identity, root / "lab", serve, folder_roots={"load": corpus}, slots=slots
            ) as lab:
                created = lab.post("/corpora", {"name": "load"})
                started = time.monotonic()
                lab.post(
                    "/documents/import-manifest",
                    {"root": "load", "path": ".", "corpus_id": created["id"]},
                )
                report["import_seconds"] = round(time.monotonic() - started, 1)
                version = lab.post(f"/corpora/{created['id']}/versions")["id"]
                spec = lab.save_spec(pipeline(args.chunk_chars))
                started = time.monotonic()
                run_id = lab.start_run(spec, version)["id"]
                peak_depth, peak_age = 0, 0.0
                while True:
                    # Development tokens live minutes; a long run outlives several.
                    lab.editor = identity.headers(role="editor")
                    state = lab.get(f"/runs/{run_id}")
                    depth, age = sample_queue(engine)
                    peak_depth, peak_age = max(peak_depth, depth), max(peak_age, age)
                    if state["status"] in ("succeeded", "failed", "cancelled"):
                        break
                    time.sleep(1.0)
                seconds = time.monotonic() - started
                chunks = (
                    sum(
                        int(task["summary"].get("chunks", 0)) for task in lab.tasks(run_id, "chunk")
                    )
                    if args.documents <= 200
                    else None
                )
                report["pipeline"] = {
                    "status": state["status"],
                    "seconds": round(seconds, 1),
                    "documents_per_second": round(args.documents / seconds, 2),
                    "chunks_per_second": round(args.documents * args.chunks / seconds, 1),
                    "chunks_counted": chunks,
                    "stages": stage_seconds(),
                    "queue_peak_jobs": peak_depth,
                    "queue_peak_oldest_seconds": round(peak_age, 1),
                }
                if state["status"] != "succeeded":
                    return report
                lab.editor = identity.headers(role="editor")
                # Every query in the dev split: a locked test split would hold back 30%.
                imported = lab.import_eval_set(
                    "load", eval_set(phrases, args.queries), format="jsonl", test_fraction=0
                )
                config = lab.save_retrieval(RETRIEVAL)
                started = time.monotonic()
                evaluation = lab.post(
                    "/eval-runs",
                    {
                        "run_id": run_id,
                        "retrieval_config_id": config,
                        "eval_set_version_id": imported["version"]["id"],
                    },
                    expect=(202,),
                )
                finished = None
                deadline = time.monotonic() + args.eval_timeout
                while time.monotonic() < deadline:
                    lab.editor = identity.headers(role="editor")
                    finished = lab.get(f"/eval-runs/{evaluation['id']}")
                    if finished["status"] in ("succeeded", "failed", "cancelled"):
                        break
                    time.sleep(1.0)
                seconds = time.monotonic() - started
                metrics = finished.get("metrics") or {}
                report["evaluation"] = {
                    "status": finished["status"],
                    "seconds": round(seconds, 1),
                    "queries_per_second": round(args.queries / seconds, 1),
                    "queries_evaluated": metrics.get("queries"),
                    "recall@5": headline(metrics, "recall@5"),
                    "mrr@10": headline(metrics, "mrr@10"),
                    "latency_ms": metrics.get("latency_ms") or metrics.get("latency"),
                }
            engine.dispose()
    finally:
        for server in servers:
            server.stop()
        factory.drop_all()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--documents", type=int, default=1000)
    parser.add_argument("--chunks", type=int, default=100, help="chunks per document")
    parser.add_argument("--chunk-chars", type=int, default=256)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--cpu-slots", type=int, default=4)
    parser.add_argument("--io-slots", type=int, default=4)
    parser.add_argument("--eval-timeout", type=float, default=3600.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = run(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    ok = report.get("pipeline", {}).get("status") == "succeeded"
    return 0 if ok and report.get("evaluation", {}).get("status") == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
