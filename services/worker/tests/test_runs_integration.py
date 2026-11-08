"""Runs end to end against Postgres (integration tier).

The API and a worker run in this process, with fakes for clamd and the model gateway (each a
real server on a free port) and the local artifact store. The corpus is six of
the format fixtures (Markdown, plain text, DOCX and PDF through the default parse route) plus a
copy of one of them, which the scan's dedupe check excludes.

- a run produces every artifact, from raw to the indexes, and a complete manifest
- an identical second run is all cache hits and finishes in under 2 seconds
- changing the chunker re-runs only the chunk step (chunk and enrich), embed and index
- cancelling mid-run leaves a consistent state, and a later run reuses what finished
- document text stays out of logs, events, summaries, manifests and audit events
"""

import shutil
import subprocess
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, SqlWorkspaceDirectory
from stratafi_components import default_registry
from stratafi_components.embed import FakeHashEmbed, FakeHashParams, Vectors
from stratafi_core.artifacts import Chunk
from stratafi_core.components import ComponentInfo, ComponentRegistry, StageContext
from stratafi_core.manifest import RunManifest
from stratafi_core.settings import OfflineSettings
from stratafi_store.artifacts import LocalArtifactStore, artifact_key, artifact_part_key
from stratafi_store.jobs import JobQueue
from stratafi_store.lineage import ancestors
from stratafi_store.tables import artifacts, jobs, stage_tasks
from stratafi_testing import fake_gateway
from stratafi_testing.canary import CanaryCheck
from stratafi_testing.clamd import fake_clamd
from stratafi_testing.identity import DevIdentity
from stratafi_worker.handlers import builtin_handlers
from stratafi_worker.runs import Services, register_run_handlers
from stratafi_worker.worker import Worker

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "formats"
DOCUMENTS = [
    FIXTURES / name
    for name in (
        "food-safety.docx",
        "food-safety.pdf",
        "tips.md",
        "tips.pdf",
        "wholesale.docx",
        "wholesale.txt",
    )
]
# A copy of wholesale.txt that sorts after it, so it's the copy: dedupe keeps a group's first
# name.
COPIED = "wholesale.txt"
DUPLICATE = "wholesale_copy.txt"
FINISHED = ("succeeded", "failed", "cancelled")
DOCUMENT_STAGES = ("extract", "scan", "parse", "clean", "chunk", "enrich", "embed")


def baseline_spec(**chunk_params: Any) -> dict[str, Any]:
    """The default components, with the fake gateway's 384-dimension alias."""
    return {
        "schema": "stratafi.pipeline.v1",
        "name": "baseline-test",
        "stages": {
            "extract": {"unpack": ["unpack/zip"]},
            "scan": {
                "checks": [
                    {"component": "scan/content-safety"},
                    {"component": "scan/clamav", "params": {"max_signature_age_hours": 72}},
                    {"component": "scan/pii"},
                    {"component": "scan/dedupe"},
                ],
            },
            "clean": [{"component": "clean/whitespace"}],
            "chunk": {
                "component": "chunk/sections",
                "params": {"max_tokens": 400, "min_tokens": 200, **chunk_params},
            },
            "enrich": {"component": "enrich/none"},
            "embed": {"component": "embed/gateway", "params": {"alias": "embed-default"}},
            "index": {
                "dense": {"component": "index/exact"},
                "keyword": {"component": "index/bm25"},
            },
        },
    }


class SlowParams(FakeHashParams):
    steps: int = Field(default=20, ge=1, le=1000)
    step_seconds: float = Field(default=0.05, ge=0, le=1)


class SlowEmbed(FakeHashEmbed):
    """The fake embedder, slowed down, checking for cancellation between steps."""

    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="embed/slow-fake",
        stage="embed",
        version="1.0.0",
        title="Slow fake",
        description="The hashed-words embedder, slowed down for cancellation tests.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = SlowParams

    def embed(
        self,
        chunks: list[Chunk],
        params: BaseModel,
        *,
        expected_revision: str | None,
        context: StageContext,
    ) -> Vectors:
        slow = cast(SlowParams, params)
        for _ in range(slow.steps):
            context.checkpoint()
            time.sleep(slow.step_seconds)
        return super().embed(chunks, params, expected_revision=expected_revision, context=context)


def registry() -> ComponentRegistry:
    components = default_registry()
    components.register(SlowEmbed)
    return components


@dataclass
class Lab:
    client: TestClient
    engine: sa.Engine
    store: LocalArtifactStore
    gateway: fake_gateway.FakeGateway
    editor: dict[str, str]
    workspace_id: uuid.UUID
    corpus_version_id: str
    documents: dict[str, str]  # entry name → document ID
    app: Any  # the API, for serving it over HTTP

    def save_spec(self, document: dict[str, Any]) -> str:
        response = self.client.post("/api/v1/pipeline-specs", json=document, headers=self.editor)
        assert response.status_code in (200, 201), response.text
        return response.json()["id"]

    def start(self, spec_id: str, **body: Any) -> dict[str, Any]:
        response = self.client.post(
            "/api/v1/runs",
            json={"pipeline_spec_id": spec_id, "corpus_version_id": self.corpus_version_id} | body,
            headers=self.editor,
        )
        assert response.status_code == 202, response.text
        return response.json()

    def run(self, run_id: str) -> dict[str, Any]:
        return self.client.get(f"/api/v1/runs/{run_id}", headers=self.editor).json()

    def wait(self, run_id: str, *, timeout: float = 60.0, poll: float = 0.02) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.run(run_id)
            if run["status"] in FINISHED:
                return run
            time.sleep(poll)
        raise AssertionError(f"run {run_id} didn't finish: {self.run(run_id)['status']}")

    def tasks(self, run_id: str, **params: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor = None
        while True:
            query = {"limit": 200, **params, **({"cursor": cursor} if cursor else {})}
            page = self.client.get(
                f"/api/v1/runs/{run_id}/tasks", params=query, headers=self.editor
            ).json()
            items += page["items"]
            cursor = page["next_cursor"]
            if cursor is None:
                return items

    def entries(self, run_id: str) -> list[dict[str, Any]]:
        page = self.client.get(
            f"/api/v1/runs/{run_id}/entries", params={"limit": 200}, headers=self.editor
        ).json()
        return page["items"]

    def manifest(self, run_id: str) -> RunManifest:
        response = self.client.get(f"/api/v1/runs/{run_id}/manifest", headers=self.editor)
        assert response.status_code == 200, response.text
        return RunManifest.model_validate(response.json())


@contextmanager
def running_lab(
    engine: sa.Engine,
    dev_identity: DevIdentity,
    tmp_path: Path,
    serve_app: Any,
    *,
    seed: bool = True,
) -> Iterator[Lab]:
    """The API (with development sign-in), a worker and the fakes. ``seed`` uploads the
    fixtures into a corpus and freezes its first version."""
    migrated_engine = engine
    gateway = fake_gateway.FakeGateway()
    gateway_url = serve_app(fake_gateway.create_app(gateway))
    components = registry()
    store = LocalArtifactStore(tmp_path / "store")
    with fake_clamd() as (clamd_host, clamd_port):
        settings = OfflineSettings(
            intake={"spool_dir": tmp_path / "spool"},
            identity={
                "service_signing_key": dev_identity.private_key_path,
                "dev_sign_in_key": dev_identity.private_key_path,
            },
            services={"clamd": f"{clamd_host}:{clamd_port}"},
            models={"gateway_url": gateway_url, "prices": {"model_gateway": 0.02}},
            worker={"max_attempts": 2, "slots": {"cpu": 2, "io": 2, "gpu": 0}},
        )
        app = create_app(
            settings,
            identity=DevIdentityProvider(dev_identity.verifier),
            workspaces=SqlWorkspaceDirectory(migrated_engine, auto_provision=True),
            engine=migrated_engine,
            store=store,
            registry=components,
            connect=False,
        )
        queue = JobQueue(
            migrated_engine,
            lease_seconds=30,
            backoff_base_seconds=0.05,
            backoff_max_seconds=0.1,
            jitter=False,
        )
        handlers = builtin_handlers()
        services = Services(migrated_engine, settings)
        orchestrator = register_run_handlers(
            handlers,
            engine=migrated_engine,
            queue=queue,
            store=store,
            registry=components,
            services=services,
            settings=settings,
        )
        worker = Worker(
            queue,
            handlers,
            slots={"cpu": 2, "io": 2},
            poll_seconds=0.02,
            heartbeat_seconds=0.1,
            shutdown_grace_seconds=1.0,
            maintenance=[orchestrator.reconcile],
        )
        client = TestClient(app)
        editor = dev_identity.headers(role="editor")
        workspace = client.get("/api/v1/me", headers=editor).json()["workspace"]["id"]
        version_id, documents = "", {}
        if seed:
            corpus = client.post(
                "/api/v1/corpora", json={"name": "fixtures"}, headers=editor
            ).json()
            files = [(path.name, path.read_bytes()) for path in DOCUMENTS]
            files.append((DUPLICATE, (FIXTURES / COPIED).read_bytes()))
            uploaded = client.post(
                "/api/v1/documents",
                files=[("file", (name, data, "application/octet-stream")) for name, data in files],
                params={"corpus_id": corpus["id"]},
                headers=editor,
            )
            assert uploaded.status_code == 201, uploaded.text
            documents = {item["name"]: item["document"]["id"] for item in uploaded.json()["files"]}
            version = client.post(f"/api/v1/corpora/{corpus['id']}/versions", headers=editor)
            version_id = version.json()["id"]
        worker.start()
        try:
            yield Lab(
                client,
                migrated_engine,
                store,
                gateway,
                editor,
                uuid.UUID(workspace),
                version_id,
                documents,
                app,
            )
        finally:
            worker.stop(grace_seconds=1.0)
            services.close()


@pytest.fixture
def lab(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> Iterator[Lab]:
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as running:
        yield running


def stage_statuses(tasks: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    statuses: dict[str, Counter[str]] = {}
    for task in tasks:
        statuses.setdefault(task["stage"], Counter())[task["status"]] += 1
    return statuses


def test_a_run_produces_every_artifact_and_a_complete_manifest(lab: Lab) -> None:
    spec_id = lab.save_spec(baseline_spec())
    started = lab.start(spec_id, mode="interactive")
    assert started["status"] == "queued"
    run = lab.wait(started["id"])
    assert run["status"] == "succeeded", run
    assert run["error_code"] is None

    entries = {entry["name"]: entry for entry in lab.entries(run["id"])}
    assert set(entries) == {path.name for path in DOCUMENTS} | {DUPLICATE}
    assert entries[DUPLICATE]["duplicate"] is True
    assert (entries[DUPLICATE]["state"], entries[DUPLICATE]["stop_code"]) == (
        "excluded",
        "excluded",
    )
    done = [name for name, entry in entries.items() if entry["state"] == "done"]
    assert sorted(done) == sorted(path.name for path in DOCUMENTS), [
        (
            name,
            entry["state"],
            entry["stop_code"],
            [
                (t["stage"], t["status"], t["error_code"], t["attempts"])
                for t in lab.tasks(run["id"], ordinal=entry["ordinal"])
            ],
        )
        for name, entry in entries.items()
        if entry["state"] != "done"
    ]

    tasks = lab.tasks(run["id"])
    by_entry: dict[str | None, list[str]] = {}
    for task in tasks:
        assert task["status"] == "succeeded", task
        by_entry.setdefault(task["entry"], []).append(task["stage"])
    for name in done:
        assert by_entry[name] == list(DOCUMENT_STAGES)
    assert by_entry[DUPLICATE] == ["extract", "scan"]
    assert by_entry[None] == ["index", "index"]

    # Every task's artifact is stored and recorded, with its lineage.
    kinds = Counter()
    with lab.engine.connect() as conn:
        recorded = {
            row.cache_key: row
            for row in conn.execute(
                sa.select(artifacts).where(artifacts.c.workspace_id == lab.workspace_id)
            )
        }
        for task in tasks:
            row = recorded[task["cache_key"]]
            kinds[row.kind] += 1
            assert lab.store.exists(artifact_key(lab.workspace_id, task["cache_key"]))
        dense = next(t for t in tasks if t["component_id"] == "index/exact")
        lineage = ancestors(conn, lab.workspace_id, dense["cache_key"])
    assert kinds == {
        # The copy's raw artifact is the original's: same bytes, same extension, same key.
        "raw": 7,
        "scan_report": 7,
        "parsed": 6,
        "cleaned": 6,
        "chunks": 6,
        "enriched": 6,
        "vectors": 6,
        "index": 2,
    }
    for task in tasks:
        if task["stage"] == "embed":
            sidecar = artifact_part_key(lab.workspace_id, task["cache_key"], "sidecar")
            assert lab.store.exists(sidecar)
    # The dense index descends from every embedded document.
    for name in done:
        assert entries[name]["sha256"] in lineage

    manifest = lab.manifest(run["id"])
    assert run["manifest_hash"] == manifest.fingerprint()
    assert manifest.execution.status == "succeeded"
    assert manifest.configuration.pipeline_hash == run["spec_hash"]
    assert {component.path for component in manifest.configuration.components} >= {
        "stages.extract.component",
        "stages.chunk",
        "stages.embed",
        "stages.index.dense",
        "stages.index.keyword",
    }
    assert {document.entry for document in manifest.inputs.documents} == set(entries)
    (model,) = manifest.models
    alias = lab.gateway.aliases["embed-default"]
    assert (model.alias, model.revision, model.dimensions) == ("embed-default", alias.revision, 384)
    assert model.served_model == alias.served_name
    assert model.tokenizer == "chars/4"
    assert manifest.environment.python
    assert manifest.environment.libraries
    assert manifest.hardware.cpu_cores
    assert manifest.hardware.ram_bytes
    assert manifest.hardware.resource_classes == ["cpu", "io"]
    stages = {stage.stage: stage for stage in manifest.execution.stages}
    assert list(stages) == [*DOCUMENT_STAGES, "index"]
    assert all(stage.succeeded == stage.tasks for stage in stages.values())
    assert stages["parse"].throughput_per_second
    parse_tasks = [task for task in manifest.execution.tasks if task.stage == "parse"]
    assert all(task.peak_memory_bytes for task in parse_tasks)
    usage = manifest.cost.providers["model_gateway"]
    assert usage.calls == 6
    assert usage.tokens > 0
    assert manifest.cost.estimated == pytest.approx(round(usage.tokens / 1000 * 0.02, 6))
    egress = {item.provider: set(item.documents) for item in manifest.egress}
    # The PII scan runs in process: only the gateway received anything.
    assert set(egress) == {"model_gateway"}
    assert egress["model_gateway"] == {lab.documents[name] for name in done}
    assert {output.cache_key for output in manifest.outputs} == {t["cache_key"] for t in tasks}

    # The worker's service identity reached the gateway.
    assert all(call.identity for call in lab.gateway.records)


def test_an_identical_second_run_is_all_cache_hits_in_under_two_seconds(lab: Lab) -> None:
    spec_id = lab.save_spec(baseline_spec())
    first = lab.wait(lab.start(spec_id)["id"])
    assert first["status"] == "succeeded"
    embeddings = sum(1 for call in lab.gateway.records if call.endpoint == "embeddings")

    began = time.monotonic()
    second = lab.wait(lab.start(spec_id)["id"], poll=0.005)
    elapsed = time.monotonic() - began
    assert second["status"] == "succeeded"
    assert elapsed < 2.0, elapsed

    tasks = lab.tasks(second["id"])
    assert len(tasks) == len(lab.tasks(first["id"]))
    assert all(task["status"] == "cached" and task["cache_hit"] for task in tasks)
    assert all(task["job_id"] is None for task in tasks)
    assert second["counts"]["tasks"] == {"cached": len(tasks)}
    with lab.engine.connect() as conn:
        stage_jobs = conn.execute(
            sa.select(sa.func.count())
            .select_from(jobs)
            .where(jobs.c.kind == "stage.run", jobs.c.payload["run_id"].astext == second["id"])
        ).scalar_one()
    assert stage_jobs == 0
    # Nothing was embedded again; the run only asked for the model's revision.
    assert sum(1 for c in lab.gateway.records if c.endpoint == "embeddings") == embeddings
    manifest = lab.manifest(second["id"])
    assert manifest.cost.providers == {}
    assert manifest.egress == []
    assert {o.cache_key for o in manifest.outputs} == {t["cache_key"] for t in tasks}
    print(f"identical second run: {elapsed:.3f} s for {len(tasks)} tasks")


def test_changing_the_chunker_reruns_only_chunk_embed_and_index(lab: Lab) -> None:
    first = lab.wait(lab.start(lab.save_spec(baseline_spec()))["id"])
    assert first["status"] == "succeeded"
    changed = lab.save_spec(baseline_spec(max_tokens=300, min_tokens=100))
    second = lab.wait(lab.start(changed)["id"])
    assert second["status"] == "succeeded"

    statuses = stage_statuses(lab.tasks(second["id"]))
    for stage in ("extract", "scan", "parse", "clean"):
        assert set(statuses[stage]) == {"cached"}, (stage, statuses[stage])
    # Enrich belongs to the chunk step: it reads the new chunks.
    for stage in ("chunk", "enrich", "embed", "index"):
        assert set(statuses[stage]) == {"succeeded"}, (stage, statuses[stage])


def test_a_rerun_from_a_stage_runs_that_stage_and_later_ones_again(lab: Lab) -> None:
    first = lab.wait(lab.start(lab.save_spec(baseline_spec()), until="clean")["id"])
    assert first["status"] == "succeeded"
    assert {task["stage"] for task in lab.tasks(first["id"])} == {
        "extract",
        "scan",
        "parse",
        "clean",
    }
    response = lab.client.post(
        f"/api/v1/runs/{first['id']}/rerun", json={"from_stage": "parse"}, headers=lab.editor
    )
    assert response.status_code == 202, response.text
    rerun = lab.wait(response.json()["id"])
    assert (rerun["rerun_of"], rerun["rerun_from"], rerun["until"]) == (
        first["id"],
        "parse",
        "clean",
    )
    statuses = stage_statuses(lab.tasks(rerun["id"]))
    assert set(statuses["extract"]) == set(statuses["scan"]) == {"cached"}
    assert set(statuses["parse"]) == set(statuses["clean"]) == {"succeeded"}
    beyond = lab.client.post(
        f"/api/v1/runs/{first['id']}/rerun", json={"from_stage": "embed"}, headers=lab.editor
    )
    assert beyond.status_code == 400
    assert beyond.json()["code"] == "stage_after_until"


def test_cancelling_mid_run_leaves_a_consistent_state(lab: Lab) -> None:
    spec = baseline_spec()
    spec["stages"]["embed"] = {"component": "embed/slow-fake", "params": {"dimensions": 64}}
    spec_id = lab.save_spec(spec)
    run_id = lab.start(spec_id)["id"]
    deadline = time.monotonic() + 60
    while not lab.tasks(run_id, stage="embed", status="running"):
        assert time.monotonic() < deadline, "no embed task started"
        time.sleep(0.02)

    response = lab.client.post(f"/api/v1/runs/{run_id}/cancel", headers=lab.editor)
    assert response.status_code == 200, response.text
    assert response.json()["status"] in ("cancelling", "cancelled")
    run = lab.wait(run_id)
    assert run["status"] == "cancelled"
    assert run["finished_at"] is not None
    assert run["manifest_hash"] is not None

    tasks = lab.tasks(run_id)
    statuses = Counter(task["status"] for task in tasks)
    assert statuses["cancelled"] >= 1
    assert not set(statuses) & {"pending", "queued", "running"}
    assert all(entry["state"] != "active" for entry in lab.entries(run_id))
    with lab.engine.connect() as conn:
        open_jobs = conn.execute(
            sa.select(sa.func.count())
            .select_from(jobs)
            .where(
                jobs.c.payload["run_id"].astext == run_id,
                jobs.c.status.in_(("queued", "running")),
            )
        ).scalar_one()
        recorded = set(
            conn.execute(
                sa.select(artifacts.c.cache_key).where(artifacts.c.workspace_id == lab.workspace_id)
            ).scalars()
        )
        task_rows = conn.execute(
            sa.select(stage_tasks.c.status, stage_tasks.c.cache_key).where(
                stage_tasks.c.run_id == uuid.UUID(run_id)
            )
        ).all()
    assert open_jobs == 0
    # What finished is recorded; what was cancelled made nothing (unless another entry with
    # the same bytes made the same artifact).
    succeeded = {key for status, key in task_rows if status == "succeeded"}
    cancelled = {key for status, key in task_rows if status == "cancelled"}
    assert succeeded <= recorded
    assert not (cancelled - succeeded) & recorded
    manifest = lab.manifest(run_id)
    assert manifest.execution.status == "cancelled"
    assert manifest.fingerprint() == run["manifest_hash"]

    # A later run of the same spec reuses everything that finished.
    again = lab.wait(lab.start(spec_id)["id"])
    assert again["status"] == "succeeded"
    reused = {task["cache_key"] for task in lab.tasks(again["id"]) if task["cache_hit"]}
    assert reused == succeeded


def test_document_text_stays_out_of_logs_events_and_records(
    lab: Lab, canary: CanaryCheck, serve_app: Any
) -> None:
    corpus = lab.client.post("/api/v1/corpora", json={"name": "canary"}, headers=lab.editor).json()
    text = f"# Canary\n\nThe phrase {canary.value} must never leave the artifacts.\n"
    uploaded = lab.client.post(
        "/api/v1/documents",
        files=[("file", ("canary.md", text.encode(), "text/markdown"))],
        params={"corpus_id": corpus["id"]},
        headers=lab.editor,
    )
    assert uploaded.status_code == 201, uploaded.text
    version = lab.client.post(f"/api/v1/corpora/{corpus['id']}/versions", headers=lab.editor)
    run = lab.start(lab.save_spec(baseline_spec()), corpus_version_id=version.json()["id"])
    finished = lab.wait(run["id"])
    assert finished["status"] == "succeeded"

    responses = [
        lab.client.get(f"/api/v1/runs/{run['id']}{path}", headers=lab.editor).text
        for path in ("", "/entries", "/tasks", "/manifest")
    ]
    with lab.engine.connect() as conn:
        summaries = [
            str(row.summary)
            for row in conn.execute(
                sa.select(artifacts.c.summary).where(artifacts.c.workspace_id == lab.workspace_id)
            )
        ]
        audit = [
            str(row.details)
            for row in conn.execute(sa.text("SELECT details FROM stratafi.audit_events"))
        ]
    canary.assert_absent(*responses, *summaries, *audit)
    canary.assert_not_logged()
    # The text is in the artifacts, where it belongs.
    tasks = lab.tasks(run["id"], stage="chunk")
    chunks = lab.store.get(artifact_key(lab.workspace_id, tasks[0]["cache_key"]))
    assert canary.value.encode() in chunks


def test_inspectors_read_artifacts_and_search_finds_a_fixture_passage(lab: Lab) -> None:
    run = lab.wait(lab.start(lab.save_spec(baseline_spec()))["id"])
    assert run["status"] == "succeeded"
    tasks = lab.tasks(run["id"])
    tips = [task for task in tasks if task["entry"] == "tips.pdf"]
    by_stage = {task["stage"]: task for task in tips}
    # The tasks carry their artifacts' summaries for the inspectors' tables.
    assert by_stage["extract"]["summary"]["media_type"] == "application/pdf"
    assert by_stage["scan"]["summary"]["decision"] == "continue"
    assert by_stage["chunk"]["summary"]["chunk_count"] >= 1

    def get(path: str, **params: Any):
        response = lab.client.get(f"/api/v1{path}", params=params, headers=lab.editor)
        assert response.status_code == 200, response.text
        return response

    parsed_key = by_stage["parse"]["cache_key"]
    record = get(f"/artifacts/{parsed_key}").json()
    assert (record["kind"], record["inputs"][0]["key"]) == (
        "parsed",
        by_stage["extract"]["cache_key"],
    )
    parsed = get(f"/artifacts/{parsed_key}/content").json()
    assert "two people count the tip jar together" in " ".join(parsed["text"].split())
    report = get(f"/artifacts/{by_stage['scan']['cache_key']}/content").json()
    assert [check["component"] for check in report["checks"]] == [
        "scan/content-safety",
        "scan/clamav",
        "scan/pii",
        "scan/dedupe",
    ]
    pii = next(check for check in report["checks"] if check["component"] == "scan/pii")
    assert (pii["outcome"], pii["findings"]) == ("clean", {})
    page = get(f"/artifacts/{by_stage['chunk']['cache_key']}/chunks", limit=1).json()
    assert len(page["items"]) == 1
    assert page["items"][0]["char_end"] > page["items"][0]["char_start"]
    vectors = lab.client.get(
        f"/api/v1/artifacts/{by_stage['embed']['cache_key']}/content", headers=lab.editor
    )
    assert vectors.headers["content-disposition"].startswith("attachment")
    other = lab.client.get(f"/api/v1/artifacts/{'f' * 64}", headers=lab.editor)
    assert other.status_code == 404

    query = "Who counts the tip jar at close?"
    for mode in ("keyword", "dense", "hybrid"):
        response = lab.client.post(
            "/api/v1/search",
            json={"run_id": run["id"], "query": query, "mode": mode, "top_k": 3},
            headers=lab.editor,
        )
        assert response.status_code == 200, response.text
        found = response.json()
        assert found["results"][0]["entry"] in ("tips.md", "tips.pdf"), (mode, found["results"])
        assert "tip jar" in " ".join(found["results"][0]["text"].split())
        expected = {"keyword", "dense", "fused"} if mode == "hybrid" else {mode}
        assert set(found["checkpoints"]) == expected
    assert found["revision"] == lab.gateway.aliases["embed-default"].revision

    until_parse = lab.wait(lab.start(lab.save_spec(baseline_spec()), until="parse")["id"])
    missing = lab.client.post(
        "/api/v1/search",
        json={"run_id": until_parse["id"], "query": query},
        headers=lab.editor,
    )
    assert missing.status_code == 409
    assert missing.json()["code"] == "index_missing"


WEB = Path(__file__).resolve().parents[3] / "web"
FLOW = WEB / "e2e" / "pipeline-flow.mjs"


class AppFiles(StaticFiles):
    """The built web app: its files, and index.html for every lab route."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as error:
            if error.status_code != 404:
                raise
        return await super().get_response("index.html", scope)


@pytest.mark.browser
def test_the_five_steps_in_a_browser(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    """Upload three fixture documents, run to Parse on a sample, see the parsed text,
    run to Embed, and search returns a fixture passage (web/e2e/pipeline-flow.mjs)."""
    dist = WEB / "dist"
    if not (dist / "index.html").is_file():
        pytest.fail("build the web app first: cd web && npm run build")
    node = shutil.which("node")
    if node is None:
        pytest.fail("the browser flow needs Node 24 and Playwright")
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app, seed=False) as lab:
        root = Starlette(
            routes=[
                Mount("/stratafi", app=AppFiles(directory=dist, html=True)),
                Mount("/", app=lab.app),
            ]
        )
        url = serve_app(root)
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [node, str(FLOW), f"{url}/stratafi/", str(FIXTURES)],
            capture_output=True,
            text=True,
            timeout=240,
            cwd=WEB,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("ok"), result.stdout
