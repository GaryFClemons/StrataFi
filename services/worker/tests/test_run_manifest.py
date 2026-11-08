"""The run manifest has every applicable field of the run record (unit tier).

``start_manifest`` (written when a run starts) and ``build_manifest`` (when it ends) are pure, so
this builds a manifest for a synthetic run, one entry of which failed, and checks each group of
``RUN_RECORD`` field by field. Fields that don't apply to a pipeline run are listed with why.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from stratafi_components import default_registry
from stratafi_core.manifest import RunManifest
from stratafi_core.planner import EntryInput
from stratafi_core.specs import resolve_pipeline
from stratafi_store.runs import Run, RunEntry, StoredArtifact, Task, build_manifest
from stratafi_worker import environment
from stratafi_worker.runs import CorpusEntries, embed_model, select_sample, start_manifest

# What every run records, group by group: the manifest paths that hold each recorded item. ``[]``
# means every item of a list, ``{}`` every value of a mapping.
RUN_RECORD: dict[str, list[str]] = {
    "Configuration": [
        "configuration.pipeline",
        "configuration.pipeline_hash",
        "configuration.components[].id",
        "configuration.components[].version",
        "configuration.components[].params",
        "configuration.components[].params_hash",
    ],
    "Inputs": [
        "inputs.corpus.corpus_version_id",
        "inputs.corpus.manifest_hash",
        "inputs.documents[].sha256",
        "inputs.documents[].media_type",
        "inputs.documents[].traits",
    ],
    "Models": [
        "models[].alias",
        "models[].served_model",
        "models[].revision",
        "models[].dimensions",
        "models[].dtype",
        "models[].prefixes",
        "models[].tokenizer",
    ],
    "Environment": [
        "environment.stratafi_version",
        "environment.commit",
        "environment.python",
        "environment.libraries",
        "environment.os",
        "environment.container_image",
    ],
    "Hardware": [
        "hardware.cpu_model",
        "hardware.cpu_cores",
        "hardware.ram_bytes",
        "hardware.gpus[].model",
        "hardware.gpus[].memory_bytes",
        "hardware.gpus[].driver",
        "hardware.resource_classes",
    ],
    "Execution": [
        "execution.status",
        "execution.started_at",
        "execution.finished_at",
        "execution.duration_ms",
        "execution.stages[].duration_ms",
        "execution.tasks[].stage",
        "execution.tasks[].status",
        "execution.tasks[].attempts",
        "execution.tasks[].started_at",
        "execution.tasks[].finished_at",
        "execution.tasks[].duration_ms",
    ],
    "Cost": [
        "cost.providers{}.calls",
        "cost.providers{}.tokens",
        "cost.estimated",
        "cost.price_table",
    ],
    "Egress": ["egress[].provider", "egress[].documents"],
    "Outputs": ["outputs[].cache_key", "outputs[].size_bytes", "outputs[].summary"],
}
# A component without parameters records an empty mapping.
MAY_BE_EMPTY = {"configuration.components[].params"}
# Recorded where they apply, so they're checked on the tasks they apply to, below.
WHERE_THEY_APPLY = (
    "the document: every task but the corpus-level index tasks",
    "peak memory: tasks that run in their own process (parse, in the sandbox)",
    "throughput: stages that executed tasks",
    "cache hit or miss: every task's cache_hit",
    "errors with codes: failed tasks' error_code",
)
# Not part of a pipeline run: an evaluation run records them.
NOT_APPLICABLE = (
    "configuration.retrieval",
    "configuration.retrieval_hash",
    "inputs.eval_set",
)

SPEC = {
    "schema": "stratafi.pipeline.v1",
    "name": "baseline",
    "stages": {
        "scan": {
            "checks": [
                {"component": "scan/content-safety"},
                {"component": "scan/clamav"},
                {"component": "scan/pii"},
                {"component": "scan/dedupe"},
            ]
        },
        "parse": {"route": {"default": {"component": "parse/markdown"}}},
        "clean": [{"component": "clean/whitespace"}],
        "chunk": {"component": "chunk/sections"},
        "enrich": {"component": "enrich/none"},
        "embed": {"component": "embed/gateway", "params": {"alias": "embed-local"}},
        "index": {"dense": {"component": "index/exact"}, "keyword": {"component": "index/bm25"}},
    },
}
T0 = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)


def values(document: Any, path: str) -> list[Any]:
    """Every value at ``path`` in ``document``."""
    found = [document]
    for part in path.split("."):
        name, every = part.rstrip("[]{}"), part[len(part.rstrip("[]{}")) :]
        found = [item[name] for item in found]
        if every == "[]":
            found = [item for items in found for item in items]
        elif every == "{}":
            found = [item for items in found for item in items.values()]
    return found


def key(number: int) -> str:
    return f"{number:064x}"


def run_record(status: str, manifest: dict[str, Any] | None = None) -> Run:
    return Run(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        kind="pipeline",
        mode="interactive",
        status=status,
        pipeline_spec_id=uuid.uuid4(),
        corpus_version_id=uuid.uuid4(),
        until_stage=None,
        sample={"size": 2, "seed": 7},
        rerun_of=None,
        rerun_from=None,
        spec_hash="0" * 64,
        model_env={"embed": {"revision": "rev-1"}},
        manifest=manifest,
        manifest_hash=None,
        counts={},
        error_code="tasks_failed" if status == "failed" else None,
        created_by="user0",
        created_at=T0,
        updated_at=T0,
        started_at=T0,
        finished_at=T0 + timedelta(seconds=30) if status != "running" else None,
    )


def task(
    run: Run,
    ordinal: int,
    stage: str,
    number: int,
    *,
    status: str = "succeeded",
    metrics: dict[str, float] | None = None,
    error_code: str | None = None,
) -> Task:
    return Task(
        id=uuid.uuid4(),
        workspace_id=run.workspace_id,
        run_id=run.id,
        ordinal=ordinal,
        stage=stage,
        component_id=f"{stage}/component",
        component_version="1.0.0",
        cache_key=key(number),
        inputs=[],
        params={},
        env={},
        route=None,
        resource_class="cpu",
        status=status,
        cache_hit=status == "cached",
        job_id=None if status == "cached" else uuid.uuid4(),
        attempts=0 if status == "cached" else 1,
        error_code=error_code,
        metrics=metrics or {},
        created_at=T0,
        started_at=T0 + timedelta(seconds=number),
        finished_at=T0 + timedelta(seconds=number + 1),
        duration_ms=0.0 if status == "cached" else 1000.0,
    )


@pytest.fixture
def manifest(monkeypatch: pytest.MonkeyPatch) -> RunManifest:
    monkeypatch.setenv("STRATAFI_COMMIT", "0123456789ab")
    monkeypatch.setenv("STRATAFI_IMAGE", "stratafi/app:dev")
    monkeypatch.setattr(
        environment,
        "_gpus",
        lambda: [{"model": "NVIDIA L4", "memory_bytes": 24 * 1024**3, "driver": "550.54"}],
    )
    registry = default_registry()
    spec = resolve_pipeline(SPEC, registry)
    documents = [
        {
            "entry": f"doc{index}.pdf",
            "document_id": str(uuid.uuid4()),
            "sha256": key(1000 + index),
            "size_bytes": 2048,
            "media_type": "application/pdf",
            "traits": {"pages": 2, "scanned": False},
        }
        for index in range(3)
    ]
    entries = [
        EntryInput(row["entry"], row["document_id"], row["sha256"], ".pdf") for row in documents
    ]
    corpus = CorpusEntries(
        entries,
        {
            "corpus_id": str(uuid.uuid4()),
            "corpus_version_id": str(uuid.uuid4()),
            "version": 3,
            "manifest_hash": key(99),
        },
        documents,
        frozenset(),
    )
    selected = select_sample(entries, {"size": 2, "seed": 7})
    started = run_record("running")
    start = start_manifest(
        started,
        spec,
        {"id": str(started.pipeline_spec_id), "name": "baseline", "version": 1},
        corpus,
        selected,
        [embed_model(spec, registry, "rev-1")],
        registry,
        price_table={"model_gateway": 0.02},
        resource_classes=["cpu", "io"],
    )
    run = run_record("failed", start)
    first, second = (
        RunEntry(
            run.workspace_id,
            run.id,
            ordinal,
            entry.name,
            uuid.UUID(entry.document_id),
            entry.sha256,
            False,
            entry.label,
            state,
            code,
        )
        for ordinal, entry, state, code in (
            (0, selected[0], "done", "done"),
            (1, selected[1], "failed", "sandbox_timeout"),
        )
    )
    gateway = {"chunks": 4, "tokens": 900, "gateway.calls": 1}
    tasks = [
        task(run, 0, "extract", 1, status="cached"),
        task(run, 0, "scan", 2),
        task(run, 0, "parse", 3, metrics={"elements": 12, "peak_memory_bytes": 52_428_800}),
        task(run, 0, "clean", 4, metrics={"edits": 3}),
        task(run, 0, "chunk", 5, metrics={"chunks": 4}),
        task(run, 0, "enrich", 6),
        task(run, 0, "embed", 7, metrics=gateway),
        task(run, 1, "extract", 8),
        task(run, 1, "scan", 9),
        task(run, 1, "parse", 10, status="failed", error_code="sandbox_timeout"),
        task(run, -1, "index", 11, metrics={"vectors": 4}),
        task(run, -1, "index", 12, metrics={"chunks": 4}),
    ]
    made = {
        t.cache_key: StoredArtifact(
            "vectors" if t.stage == "embed" else t.stage,
            4096,
            {"dims": 768, "served_model": "nomic-embed-text"} if t.stage == "embed" else {"n": 1},
        )
        for t in tasks
        if t.status in ("succeeded", "cached")
    }
    return build_manifest(run, [first, second], tasks, made)


@pytest.mark.parametrize("area", list(RUN_RECORD))
def test_every_applicable_field_of_the_run_record_is_recorded(
    manifest: RunManifest, area: str
) -> None:
    document = manifest.model_dump(mode="json", by_alias=True)
    for path in RUN_RECORD[area]:
        found = values(document, path)
        assert found, path
        empty = (None,) if path in MAY_BE_EMPTY else (None, "", [], {})
        assert all(value not in empty for value in found), path


def test_fields_recorded_where_they_apply(manifest: RunManifest) -> None:
    tasks = {(task.entry, task.stage): task for task in manifest.execution.tasks}
    assert all((task.entry is None) == (task.stage == "index") for task in tasks.values())
    names = sorted({task.entry for task in manifest.execution.tasks if task.entry})
    parse = tasks[(names[0], "parse")]
    assert parse.peak_memory_bytes == 52_428_800
    failed = tasks[(names[1], "parse")]
    assert (failed.status, failed.error_code) == ("failed", "sandbox_timeout")
    assert tasks[(names[0], "extract")].cache_hit is True
    assert tasks[(names[0], "embed")].cache_hit is False
    stages = {stage.stage: stage for stage in manifest.execution.stages}
    assert stages["parse"].throughput_per_second == 1.0
    assert stages["extract"].cached == 1
    assert stages["parse"].failed == 1


def test_cost_and_egress_count_only_work_the_run_did(manifest: RunManifest) -> None:
    gateway = manifest.cost.providers["model_gateway"]
    assert (gateway.calls, gateway.tokens) == (1, 900)
    assert manifest.cost.estimated == pytest.approx(900 / 1000 * 0.02)
    egress = {item.provider: item.documents for item in manifest.egress}
    # Only the gateway received anything: the scan ran in process.
    assert list(egress) == ["model_gateway"]
    assert len(egress["model_gateway"]) == 1


def test_fields_that_dont_apply_to_a_pipeline_run(manifest: RunManifest) -> None:
    document = manifest.model_dump(mode="json", by_alias=True)
    for path in NOT_APPLICABLE:
        assert values(document, path) == [None], path


def test_the_manifest_is_its_fingerprint(manifest: RunManifest) -> None:
    again = RunManifest.model_validate(manifest.model_dump(mode="json", by_alias=True))
    assert again.fingerprint() == manifest.fingerprint()
    assert manifest.execution.status == "failed"
    assert manifest.run.mode == "interactive"
    assert manifest.inputs.sample is not None
    assert manifest.inputs.sample.size == 2


def test_samples_are_reproducible() -> None:
    entries = [EntryInput(f"doc{i:02}.md", str(i), key(i), ".md") for i in range(20)]
    chosen = select_sample(entries, {"size": 5, "seed": 3})
    assert chosen == select_sample(entries, {"size": 5, "seed": 3})
    assert [entry.name for entry in chosen] == sorted(entry.name for entry in chosen)
    assert chosen != select_sample(entries, {"size": 5, "seed": 4})
    assert select_sample(entries, {"entries": ["doc03.md", "doc01.md"]}) == [
        entries[1],
        entries[3],
    ]
    assert select_sample(entries, {"size": 50}) == entries
    assert select_sample(entries, None) == entries
