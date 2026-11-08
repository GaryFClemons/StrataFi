"""Request IDs and one metadata-only log line per request.

A request keeps a well-formed ``X-Request-Id`` it arrives with, or gets a new one. The ID goes
into the log context (``stratafi_core.logs.log_context``), every problem body, and the response's
``X-Request-Id`` header. The request log line names the route template, never the raw path or
query string, which can carry text (a search query).

An exception nothing else handled becomes the fixed 500 problem here, inside the request's
context, so even that response carries its request ID. (Starlette's own catch-all runs in its
outermost middleware, outside this one.)
"""

import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from stratafi_api.problems import problem_response
from stratafi_core.ids import is_valid_request_id, new_request_id
from stratafi_core.logs import log_context
from stratafi_core.observability import HTTP_DURATION, HTTP_REQUESTS

REQUEST_ID_HEADER = "X-Request-Id"
_HEADER = REQUEST_ID_HEADER.lower().encode("latin-1")

logger = logging.getLogger("stratafi.api.requests")


def _incoming(scope: Scope) -> str | None:
    values = [value for key, value in scope.get("headers", ()) if key.lower() == _HEADER]
    if len(values) == 1:
        candidate = values[0].decode("latin-1")
        if is_valid_request_id(candidate):
            return candidate
    return None


def _route_template(scope: Scope) -> str:
    # Routing records the matched route in the scope. Its path is the template, with no IDs,
    # relative to the API base, as contracts/openapi.json names it.
    return getattr(scope.get("route"), "path", "unmatched")


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = _incoming(scope) or new_request_id()
        status = 500
        started = time.perf_counter()
        response_started = False

        async def send_with_id(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != _HEADER
                ]
                headers.append((_HEADER, request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        with log_context(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_id)
            except Exception as exc:
                if response_started:
                    raise
                # The JSON formatter records the exception's type and frames, not its message.
                logger.error("unhandled exception", exc_info=exc)
                await problem_response("internal")(scope, receive, send_with_id)
            finally:
                method, route = scope.get("method", ""), _route_template(scope)
                seconds = time.perf_counter() - started
                # Prometheus: the route's template, never its path or query.
                HTTP_REQUESTS.labels(method, route, str(status)).inc()
                HTTP_DURATION.labels(method, route).observe(seconds)
                logger.info(
                    "request",
                    extra={
                        "method": method,
                        "route": route,
                        "status": status,
                        "duration_ms": round(seconds * 1000, 1),
                    },
                )
