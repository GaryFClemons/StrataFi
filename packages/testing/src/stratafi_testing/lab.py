"""A running lab for integration tests: the API (with development sign-in) and a worker in this
process, against Postgres and a local artifact store, with fakes for clamd and the model
gateway, each a real server on a free port. Test tools only, never deployed.

    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        corpus_version = lab.import_manifest("halcyon", "demo-corpus/corpus")
        run = lab.wait_run(lab.start_run(lab.save_spec(spec), corpus_version)["id"])

The worker handles pipeline runs and eval runs. ``folder_roots`` defaults to the repository's
``fixtures/`` as ``fixtures``. The API and worker packages are imported when a lab starts, so
this module costs nothing to the packages that don't use it.

``prepare_halcyon`` and ``evaluate`` run the Halcyon Coffee demo corpus (``fixtures/demo-corpus``)
through a lab: its 24 documents by their manifest through the default components
(``baseline_pipeline``), ``hybrid-v1`` (``hybrid_retrieval``), and its 88 queries as an eval set.
"""

import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stratafi_testing import fake_gateway
from stratafi_testing.clamd import fake_clamd
from stratafi_testing.identity import DevIdentity

FIXTURES = Path(__file__).resolve().parents[4] / "fixtures"
FINISHED = ("succeeded", "failed", "cancelled")
EXPERIMENT_FINISHED = ("succeeded", "stopped", "failed", "cancelled")
TEST_RUN_FINISHED = ("passed", "failed", "error", "cancelled")


@dataclass
class Lab:
    client: Any  # fastapi.testclient.TestClient
    app: Any
    engine: Any
    store: Any
    gateway: fake_gateway.FakeGateway
    identity: DevIdentity
    editor: dict[str, str]
    workspace_id: uuid.UUID

    def post(self, path: str, body: Any = None, *, expect: tuple[int, ...] = (200, 201, 202)):
        response = self.client.post(f"/api/v1{path}", json=body, headers=self.editor)
        if response.status_code not in expect:
            raise AssertionError(response.text)
        return response.json()

    def put(self, path: str, body: Any = None, *, expect: tuple[int, ...] = (200,)):
        response = self.client.put(f"/api/v1{path}", json=body, headers=self.editor)
        if response.status_code not in expect:
            raise AssertionError(response.text)
        return response.json()

    def get(self, path: str, **params: Any) -> Any:
        response = self.client.get(f"/api/v1{path}", params=params, headers=self.editor)
        if response.status_code != 200:
            raise AssertionError(response.text)
        return response.json()

    def save_spec(self, document: Mapping[str, Any]) -> str:
        return self.post("/pipeline-specs", dict(document))["id"]

    def save_retrieval(self, document: Mapping[str, Any]) -> str:
        return self.post("/retrieval-configs", dict(document))["id"]

    def import_manifest(self, corpus: str, path: str) -> str:
        """A corpus imported from ``fixtures/<path>`` by its manifest; its first version's ID."""
        created = self.post("/corpora", {"name": corpus})
        self.post(
            "/documents/import-manifest",
            {"root": "fixtures", "path": path, "corpus_id": created["id"]},
        )
        return self.post(f"/corpora/{created['id']}/versions")["id"]

    def upload_corpus(self, corpus: str, files: Mapping[str, bytes]) -> str:
        """A corpus of ``files`` (name → bytes), uploaded and frozen; its first version's ID."""
        created = self.post("/corpora", {"name": corpus})
        response = self.client.post(
            "/api/v1/documents",
            params={"corpus_id": created["id"]},
            files=[
                ("file", (name, data, "application/octet-stream")) for name, data in files.items()
            ],
            headers=self.editor,
        )
        if response.status_code != 201:
            raise AssertionError(response.text)
        return self.post(f"/corpora/{created['id']}/versions")["id"]

    def tasks(self, run_id: str, stage: str) -> list[dict[str, Any]]:
        return self.get(f"/runs/{run_id}/tasks", stage=stage, limit=200)["items"]

    def artifact(self, cache_key: str) -> bytes:
        response = self.client.get(f"/api/v1/artifacts/{cache_key}/content", headers=self.editor)
        if response.status_code != 200:
            raise AssertionError(response.text)
        return response.content

    def import_eval_set(self, name: str, content: str, **body: Any) -> dict[str, Any]:
        return self.post("/eval-sets/import", {"name": name, "content": content, **body})

    def start_run(self, spec_id: str, corpus_version_id: str, **body: Any) -> dict[str, Any]:
        return self.post(
            "/runs",
            {"pipeline_spec_id": spec_id, "corpus_version_id": corpus_version_id, **body},
            expect=(202,),
        )

    def wait(self, path: str, *, timeout: float = 120.0, poll: float = 0.05) -> dict[str, Any]:
        """Poll ``path`` (a run or an eval run) until it finishes."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = self.get(path)
            if found["status"] in FINISHED:
                return found
            time.sleep(poll)
        raise AssertionError(f"{path} didn't finish: {self.get(path)['status']}")

    def wait_run(self, run_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.wait(f"/runs/{run_id}", **kwargs)

    def wait_eval(self, eval_run_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.wait(f"/eval-runs/{eval_run_id}", **kwargs)

    def wait_test_run(
        self, test_run_id: str, *, timeout: float = 300.0, poll: float = 0.1
    ) -> dict[str, Any]:
        """Poll a test run until it ends: passed, failed, error or cancelled."""
        deadline = time.monotonic() + timeout
        path = f"/test-runs/{test_run_id}"
        while time.monotonic() < deadline:
            found = self.get(path)
            if found["status"] in TEST_RUN_FINISHED:
                return found
            time.sleep(poll)
        raise AssertionError(f"{path} didn't finish: {self.get(path)['status']}")

    def wait_experiment(
        self, experiment_id: str, *, timeout: float = 300.0, poll: float = 0.1
    ) -> dict[str, Any]:
        """Poll an experiment until it ends: succeeded, stopped, failed or cancelled."""
        deadline = time.monotonic() + timeout
        path = f"/experiments/{experiment_id}"
        while time.monotonic() < deadline:
            found = self.get(path)
            if found["status"] in EXPERIMENT_FINISHED:
                return found
            time.sleep(poll)
        raise AssertionError(f"{path} didn't finish: {self.get(path)['status']}")


@contextmanager
def running_lab(
    engine: Any,
    dev_identity: DevIdentity,
    tmp_path: Path,
    serve_app: Any,
    *,
    registry: Any = None,
    folder_roots: Mapping[str, Path] | None = None,
    suites: Mapping[str, Any] | None = None,
    slots: Mapping[str, int] | None = None,
) -> Iterator[Lab]:
    from fastapi.testclient import TestClient

    from stratafi_api.app import create_app
    from stratafi_api.identity import DevIdentityProvider, SqlWorkspaceDirectory
    from stratafi_components import default_registry
    from stratafi_core.settings import OfflineSettings
    from stratafi_store.artifacts import LocalArtifactStore
    from stratafi_store.jobs import JobQueue
    from stratafi_worker.answers import register_answer_handlers
    from stratafi_worker.automation import register_automation_handlers
    from stratafi_worker.evals import register_eval_handlers
    from stratafi_worker.experiments import register_experiment_handlers
    from stratafi_worker.handlers import builtin_handlers
    from stratafi_worker.runs import Services, register_run_handlers
    from stratafi_worker.testruns import register_test_handlers
    from stratafi_worker.training import register_training_handlers
    from stratafi_worker.worker import Worker

    gateway = fake_gateway.FakeGateway()
    gateway_url = serve_app(fake_gateway.create_app(gateway))
    components = registry if registry is not None else default_registry()
    store = LocalArtifactStore(tmp_path / "store")
    with fake_clamd() as (clamd_host, clamd_port):
        settings = OfflineSettings(
            intake={
                "spool_dir": tmp_path / "spool",
                "folder_roots": dict(folder_roots or {"fixtures": FIXTURES}),
            },
            identity={
                "service_signing_key": dev_identity.private_key_path,
                "dev_sign_in_key": dev_identity.private_key_path,
            },
            services={"clamd": f"{clamd_host}:{clamd_port}"},
            models={"gateway_url": gateway_url, "prices": {"model_gateway": 0.02}},
            worker={"max_attempts": 2, "slots": {"cpu": 2, "io": 3, "gpu": 0}},
            # Webhook receivers in tests listen on this machine, on ports the OS picks.
            suites={"webhook_allow_private": True, "webhook_ports": None, **(suites or {})},
        )
        app = create_app(
            settings,
            identity=DevIdentityProvider(dev_identity.verifier),
            workspaces=SqlWorkspaceDirectory(engine, auto_provision=True),
            engine=engine,
            store=store,
            registry=components,
            connect=False,
        )
        queue = JobQueue(
            engine,
            lease_seconds=30,
            backoff_base_seconds=0.05,
            backoff_max_seconds=0.1,
            jitter=False,
        )
        handlers = builtin_handlers()
        services = Services(engine, settings, store)
        runs = register_run_handlers(
            handlers,
            engine=engine,
            queue=queue,
            store=store,
            registry=components,
            services=services,
            settings=settings,
        )
        evals = register_eval_handlers(
            handlers,
            engine=engine,
            queue=queue,
            store=store,
            registry=components,
            services=services,
            settings=settings,
        )
        experiments = register_experiment_handlers(
            handlers, engine=engine, queue=queue, runs=runs, evals=evals, settings=settings
        )
        register_answer_handlers(
            handlers,
            engine=engine,
            store=store,
            registry=components,
            services=services,
            settings=settings,
        )
        tests = register_test_handlers(
            handlers,
            engine=engine,
            queue=queue,
            store=store,
            registry=components,
            services=services,
            settings=settings,
        )
        register_training_handlers(
            handlers,
            engine=engine,
            store=store,
            registry=components,
            services=services,
            settings=settings,
        )
        scheduler = register_automation_handlers(
            handlers, engine=engine, queue=queue, settings=settings
        )
        worker = Worker(
            queue,
            handlers,
            slots=dict(slots or {"cpu": 2, "io": 3}),
            poll_seconds=0.02,
            heartbeat_seconds=0.1,
            shutdown_grace_seconds=1.0,
            maintenance=[
                runs.reconcile,
                evals.reconcile,
                experiments.reconcile,
                tests.reconcile,
                scheduler.tick,
            ],
        )
        client = TestClient(app)
        editor = dev_identity.headers(role="editor")
        workspace = client.get("/api/v1/me", headers=editor).json()["workspace"]["id"]
        worker.start()
        try:
            yield Lab(
                client,
                app,
                engine,
                store,
                gateway,
                dev_identity,
                editor,
                uuid.UUID(workspace),
            )
        finally:
            worker.stop(grace_seconds=1.0)
            services.close()


WEB = Path(__file__).resolve().parents[4] / "web"


def serve_web(lab: Lab, serve_app: Any) -> str:
    """Serve the built web app at ``/stratafi/`` beside the lab's API; the app's URL. Build it
    first (``npm run build`` in ``web/``)."""
    from starlette.applications import Starlette
    from starlette.exceptions import HTTPException
    from starlette.responses import Response
    from starlette.routing import Mount
    from starlette.staticfiles import StaticFiles
    from starlette.types import Scope

    class AppFiles(StaticFiles):
        """The built app's files, and index.html for every lab route."""

        async def get_response(self, path: str, scope: Scope) -> Response:
            try:
                return await super().get_response(path, scope)
            except HTTPException as error:
                if error.status_code != 404:
                    raise
            return await super().get_response("index.html", scope)

    dist = WEB / "dist"
    if not (dist / "index.html").is_file():
        raise AssertionError("build the web app first: cd web && npm run build")
    root = Starlette(
        routes=[
            Mount("/stratafi", app=AppFiles(directory=dist, html=True)),
            Mount("/", app=lab.app),
        ]
    )
    return f"{serve_app(root)}/stratafi/"


# --- the Halcyon demo corpus in a lab ----------------------------------------------------

HALCYON = FIXTURES / "demo-corpus"
HALCYON_RETRIEVAL = HALCYON / "retrieval.yaml"


def baseline_pipeline(name: str = "baseline-v1") -> dict[str, Any]:
    """The default components (the default parse route, ``chunk/sections``, exact vectors and
    BM25) with the fake gateway's 384-dimension alias, and ClamAV and the PII scan (its default
    categories) in the scan."""
    return {
        "schema": "stratafi.pipeline.v1",
        "name": name,
        "stages": {
            "scan": {
                "checks": [
                    {"component": "scan/content-safety"},
                    {"component": "scan/clamav", "params": {"max_signature_age_hours": 72}},
                    {"component": "scan/pii"},
                    {"component": "scan/dedupe"},
                ]
            },
            "clean": [{"component": "clean/whitespace"}],
            "chunk": {"component": "chunk/sections"},
            "embed": {"component": "embed/gateway", "params": {"alias": "embed-default"}},
        },
    }


def hybrid_retrieval(name: str = "hybrid-v1", **changes: Any) -> dict[str, Any]:
    """Hybrid search: BM25 and exact dense, RRF over the best 50 of each, the gateway's
    reranker, top 8, current versions only."""
    return {
        "schema": "stratafi.retrieval.v1",
        "name": name,
        "retrievers": {"dense": {"top_n": 50}, "keyword": {"top_n": 50}},
        "fusion": {"method": "rrf", "k": 60},
        "candidates": 50,
        "filters": {"current_only": True},
        "rerank": {"component": "rerank/gateway", "params": {"alias": "rerank-default"}},
        "top_k": 8,
        **changes,
    }


def prepare_halcyon(lab: Lab) -> dict[str, str]:
    """The Halcyon corpus run to its indexes, hybrid-v1, and the Halcyon eval set: their IDs."""
    corpus_version = lab.import_manifest("halcyon", "demo-corpus/corpus")
    run = lab.wait_run(lab.start_run(lab.save_spec(baseline_pipeline()), corpus_version)["id"])
    if run["status"] != "succeeded":
        raise AssertionError(f"the Halcyon run {run['status']}")
    imported = lab.import_eval_set(
        "halcyon-retrieval", HALCYON_RETRIEVAL.read_text(encoding="utf-8"), seed=0
    )
    return {
        "run_id": run["id"],
        "retrieval_config_id": lab.save_retrieval(hybrid_retrieval()),
        "eval_set_version_id": imported["version"]["id"],
        "eval_set_id": imported["eval_set"]["id"],
    }


def evaluate(lab: Lab, inputs: Mapping[str, str], **body: Any) -> dict[str, Any]:
    """Start an eval run of ``inputs`` (``prepare_halcyon``'s) and wait for it."""
    started = lab.post(
        "/eval-runs",
        {
            "run_id": inputs["run_id"],
            "retrieval_config_id": inputs["retrieval_config_id"],
            "eval_set_version_id": inputs["eval_set_version_id"],
            **body,
        },
        expect=(202,),
    )
    return lab.wait_eval(started["id"])
