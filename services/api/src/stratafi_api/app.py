"""The API app factory.

``create_app(settings)`` builds the FastAPI app without connecting to anything: the database
engine and the artifact store connect on first use, so the app can also be built offline to
export the OpenAPI contract (``scripts/export_openapi.py``). Tests pass their own ``identity``,
``engine`` and ``store``.

Every route lives under ``/api/v1``. Errors are RFC 9457 problems that never echo input;
every response carries ``X-Request-Id``.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from sqlalchemy.engine import Engine

from stratafi_api import __version__
from stratafi_api.events import EventHub
from stratafi_api.identity import (
    DevIdentityProvider,
    IdentityProvider,
    IdentityService,
    SqlTokens,
    SqlWorkspaceDirectory,
    StaticWorkspaces,
    WorkspaceDirectory,
)
from stratafi_api.middleware import RequestIdMiddleware
from stratafi_api.problems import PROBLEM_MEDIA_TYPE, install_problem_handlers
from stratafi_api.routes import (
    answers,
    artifacts,
    audit,
    auth,
    chat,
    comparisons,
    components,
    corpora,
    dev,
    documents,
    embeddings,
    evalruns,
    evalsets,
    events,
    experiments,
    features,
    health,
    jobs,
    maps,
    me,
    metrics,
    profiles,
    questions,
    registries,
    runs,
    search,
    specs,
    testsuites,
    tokens,
    traces,
    tracking,
    training,
    webhooks,
)
from stratafi_components import default_registry
from stratafi_components.retrieval import RetrievalCache
from stratafi_core.components import ComponentRegistry
from stratafi_core.identity import IdentityConfigError, TokenVerifier, service_tokens
from stratafi_core.settings import Settings
from stratafi_store.artifacts import ArtifactStore, artifact_store_from_settings
from stratafi_store.components import sync_components
from stratafi_store.db import create_engine, libpq_url
from stratafi_store.evalruns import EvalOrchestrator
from stratafi_store.experiments import ExperimentOrchestrator
from stratafi_store.hooks import finish_hooks
from stratafi_store.jobs import JobQueue
from stratafi_store.runs import Orchestrator
from stratafi_store.scope import api_engine
from stratafi_store.testruns import TestOrchestrator
from stratafi_store.training import ModelFiles

logger = logging.getLogger("stratafi.api")

API_PREFIX = "/api/v1"
TITLE = "Stratafi API"


def identity_provider_from_settings(settings: Settings) -> IdentityProvider:
    """Identity tokens verified with the development key's public key."""
    identity = settings.identity
    if identity.public_key is None:
        raise IdentityConfigError("identity.public_key is not set")
    return DevIdentityProvider(TokenVerifier.from_file(identity.public_key))


def contract(app: FastAPI) -> dict[str, Any]:
    """The OpenAPI document, corrected to what the API really sends: validation errors are 400
    problems, never FastAPI's default 422 body, and problems are ``application/problem+json``.

    Paths are relative to the API base, named in ``servers``, so a client joins them to whatever
    base its host gives it (``StratafiHost.apiBase``).
    """
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=[{"url": API_PREFIX}],
    )
    paths = schema.get("paths", {})
    if any(not path.startswith(f"{API_PREFIX}/") for path in paths):
        raise RuntimeError(f"every route must live under {API_PREFIX}")
    schema["paths"] = {path.removeprefix(API_PREFIX): ops for path, ops in paths.items()}
    schemas = schema.setdefault("components", {}).setdefault("schemas", {})
    for name in ("HTTPValidationError", "ValidationError"):
        schemas.pop(name, None)
    problem = {"$ref": "#/components/schemas/ProblemDetails"}
    for operations in schema.get("paths", {}).values():
        for operation in operations.values():
            responses = operation.setdefault("responses", {})
            if responses.pop("422", None) is not None:
                responses.setdefault(
                    "400",
                    {
                        "description": "Invalid request",
                        "content": {PROBLEM_MEDIA_TYPE: {"schema": problem}},
                    },
                )
            for response in responses.values():
                content = response.get("content", {})
                if PROBLEM_MEDIA_TYPE in content:
                    content.pop("application/json", None)
            operation["responses"] = dict(sorted(responses.items()))
    return schema


def create_app(
    settings: Settings,
    *,
    identity: IdentityProvider | None = None,
    workspaces: WorkspaceDirectory | None = None,
    engine: Engine | None = None,
    store: ArtifactStore | None = None,
    registry: ComponentRegistry | None = None,
    connect: bool = True,
) -> FastAPI:
    """Build the API. With ``connect=False`` nothing is configured from settings that isn't
    passed in: for exporting the contract and for unit tests. ``registry`` defaults to the
    built-in components; when the app starts with a database, it records the registry in the
    ``components`` table."""
    if connect:
        if engine is None and settings.database.url is not None:
            engine = create_engine(
                settings.database.url.get_secret_value(),
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                application_name="stratafi-api",
            )
        if store is None:
            store = artifact_store_from_settings(settings.storage)
        if identity is None:
            identity = identity_provider_from_settings(settings)
    if engine is not None:
        # Row-level security: a request sees only its principal's workspace, and a
        # transaction outside any scope sees nothing.
        engine = api_engine(engine)
    if workspaces is None:
        auto = settings.identity.auto_provision_workspaces
        workspaces = (
            SqlWorkspaceDirectory(engine, auto_provision=auto)
            if engine is not None
            else StaticWorkspaces(auto_provision=auto)
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.engine is not None and connect:
            with app.state.engine.begin() as conn:
                count = sync_components(conn, app.state.registry.descriptions())
            logger.info("component registry recorded", extra={"components": count})
        yield
        if app.state.events is not None:
            app.state.events.stop()
        if app.state.engine is not None:
            app.state.engine.dispose()

    app = FastAPI(
        title=TITLE,
        version=__version__,
        description="Stratafi's REST API. Errors are RFC 9457 problem details.",
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        lifespan=lifespan,
        # Operation IDs are the handler names (`health`, `audit_events`): short, stable, and
        # independent of the path, so generated clients don't churn.
        generate_unique_id_function=lambda route: route.name,
    )
    app.state.settings = settings
    app.state.registry = registry if registry is not None else default_registry()
    app.state.engine = engine
    app.state.store = store
    # Trained models' files, for adapters in the search playground.
    app.state.model_files = (
        ModelFiles(engine, store) if engine is not None and store is not None else None
    )
    app.state.jobs = (
        JobQueue(engine, lease_seconds=settings.worker.lease_seconds)
        if engine is not None
        else None
    )
    on_finish = (
        finish_hooks(app.state.jobs, max_attempts=settings.worker.max_attempts)
        if engine is not None
        else None
    )
    app.state.runs = (
        Orchestrator(
            engine,
            app.state.jobs,
            app.state.registry,
            max_attempts=settings.worker.max_attempts,
            on_finish=on_finish,
        )
        if engine is not None
        else None
    )
    app.state.evals = (
        EvalOrchestrator(
            engine,
            app.state.jobs,
            max_attempts=settings.worker.max_attempts,
            on_finish=on_finish,
        )
        if engine is not None
        else None
    )
    app.state.experiments = (
        ExperimentOrchestrator(
            engine,
            app.state.jobs,
            app.state.runs,
            app.state.evals,
            max_attempts=settings.worker.max_attempts,
        )
        if engine is not None
        else None
    )
    app.state.tests = (
        TestOrchestrator(
            engine,
            app.state.jobs,
            app.state.registry,
            max_attempts=settings.worker.max_attempts,
            webhook_attempts=settings.suites.webhook_attempts,
        )
        if engine is not None
        else None
    )
    app.state.events = EventHub(libpq_url(engine)) if engine is not None else None
    app.state.indexes = search.IndexCache()
    app.state.retrieval_cache = RetrievalCache()
    app.state.identity = (
        IdentityService(
            identity,
            workspaces,
            settings.identity.roles_for(),
            tokens=SqlTokens(engine) if engine is not None else None,
            group_map=dict(settings.identity.group_map),
        )
        if identity is not None
        else None
    )
    app.state.dev_sign_in = False
    # The API's own calls to the model gateway for callers who sent no identity token (API
    # tokens).
    app.state.service_tokens = None
    if connect:
        try:
            app.state.service_tokens = service_tokens(
                signing_key=settings.identity.service_signing_key,
                name=settings.identity.service_name,
            )
        except (IdentityConfigError, OSError):
            logger.warning("the API's service identity is off: its key can't be read")
    install_problem_handlers(app)

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            app.openapi_schema = contract(app)
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]
    app.add_middleware(RequestIdMiddleware)
    for router in (
        health.router,
        auth.router,
        me.router,
        features.router,
        components.router,
        specs.pipeline_router,
        specs.retrieval_router,
        documents.router,
        corpora.router,
        evalsets.router,
        questions.router,
        tracking.router,
        comparisons.router,
        traces.router,
        maps.router,
        evalruns.router,
        answers.router,
        training.router,
        experiments.router,
        profiles.router,
        registries.router,
        testsuites.router,
        webhooks.router,
        tokens.router,
        jobs.router,
        runs.router,
        artifacts.router,
        search.router,
        chat.router,
        embeddings.router,
        events.router,
        audit.router,
        metrics.router,
    ):
        app.include_router(router, prefix=API_PREFIX)
    if dev.enabled(settings) and settings.identity.dev_sign_in_key is not None:
        try:
            dev_router = dev.build_router(settings.identity.dev_sign_in_key)
        except (OSError, ValueError, IdentityConfigError):
            logger.warning("development sign-in is off: its key can't be read")
        else:
            app.include_router(dev_router, prefix=API_PREFIX)
            app.state.dev_sign_in = True
    return app
