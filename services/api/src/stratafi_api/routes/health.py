"""Liveness and readiness."""

import logging
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from stratafi_api import __version__
from stratafi_store.artifacts import ArtifactStoreError
from stratafi_store.db import ping

logger = logging.getLogger("stratafi.api.health")
router = APIRouter(tags=["health"])

CheckStatus = Literal["ok", "failed", "not_configured"]


class Check(BaseModel):
    status: CheckStatus
    code: str | None = None


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    version: str
    checks: dict[str, Check]


class Liveness(BaseModel):
    status: Literal["ok"]


@router.get("/health/live", response_model=Liveness, summary="The process is up")
def live() -> Liveness:
    return Liveness(status="ok")


@router.get(
    "/health",
    response_model=Health,
    summary="Ready: the database and the artifact store answer",
    responses={503: {"model": Health, "description": "A dependency is unavailable"}},
)
def health(request: Request) -> JSONResponse:
    checks = {"database": _database(request), "storage": _storage(request)}
    ready = all(check.status == "ok" for check in checks.values())
    body = Health(status="ok" if ready else "unavailable", version=__version__, checks=checks)
    return JSONResponse(body.model_dump(mode="json", exclude_none=True), 200 if ready else 503)


def _database(request: Request) -> Check:
    engine = request.app.state.engine
    if engine is None:
        return Check(status="not_configured")
    try:
        ping(engine)
    except Exception as exc:
        logger.warning("database check failed", extra={"exc_kind": type(exc).__name__})
        return Check(status="failed", code=type(exc).__name__)
    return Check(status="ok")


def _storage(request: Request) -> Check:
    store = request.app.state.store
    if store is None:
        return Check(status="not_configured")
    try:
        store.check()
    except ArtifactStoreError as exc:
        logger.warning("storage check failed", extra={"exc_kind": type(exc).__name__})
        return Check(status="failed", code=type(exc).__name__)
    return Check(status="ok")
