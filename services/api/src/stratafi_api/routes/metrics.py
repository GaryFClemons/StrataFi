"""Prometheus metrics (``stratafi_core.observability``): ``GET /api/v1/metrics``.

The process's own counters (requests by route template), and the queue read from Postgres for
this scrape. With ``observability.metrics_token`` set, the scrape sends it as a bearer token; a
prod deployment must set one. Outside the OpenAPI contract: it's for Prometheus, not clients.
"""

import hmac

from fastapi import APIRouter, Request
from fastapi.responses import Response
from prometheus_client import CollectorRegistry, generate_latest

from stratafi_api.problems import Problem
from stratafi_core.observability import CONTENT_TYPE, exposition
from stratafi_store.jobs import QueueCollector

router = APIRouter(tags=["metrics"])


@router.get("/metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    settings = request.app.state.settings.observability
    if not settings.metrics:
        raise Problem("not_found", code="metrics_off")
    if settings.metrics_token is not None:
        scheme, _, value = request.headers.get("authorization", "").partition(" ")
        expected = settings.metrics_token.get_secret_value()
        if scheme.lower() != "bearer" or not hmac.compare_digest(value.strip(), expected):
            raise Problem("unauthorized", code="metrics_token")
    body = exposition()
    engine = request.app.state.engine
    if engine is not None:
        # A registry for this scrape alone: apps (and tests) don't share the queue's collector.
        queue = CollectorRegistry(auto_describe=True)
        queue.register(QueueCollector(engine))
        body += generate_latest(queue)
    return Response(body, media_type=CONTENT_TYPE)
