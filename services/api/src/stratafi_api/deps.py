"""What routes need from the app: the database engine and the artifact store, or a 503."""

from fastapi import Request
from sqlalchemy.engine import Engine

from stratafi_api.problems import Problem
from stratafi_store.artifacts import ArtifactStore


def engine_of(request: Request) -> Engine:
    engine = request.app.state.engine
    if engine is None:
        raise Problem("unavailable", code="database_not_configured")
    return engine


def store_of(request: Request) -> ArtifactStore:
    store = request.app.state.store
    if store is None:
        raise Problem("unavailable", code="storage_not_configured")
    return store
