"""Live events as server-sent events: job and run progress."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses

router = APIRouter(tags=["events"])

KEEPALIVE_SECONDS = 15.0
RETRY_MILLISECONDS = 3000


@router.get(
    "/events",
    summary="Live events for the workspace, as server-sent events",
    description=(
        "A `text/event-stream` of JSON `data` lines. `event: job`: `workspace_id`, `job_id`, "
        "`kind`, `status`, `attempts`, and `progress` or `error_code` when present. "
        "`event: run`: `workspace_id`, `run_id`, `status`, and `counts` (tasks by status, by "
        "stage, and entries by state) when present. `event: eval_run`: the same, with the eval "
        "run's ID as `run_id` and `counts` of shards by status and queries. `event: experiment` "
        "and `event: test_run`: the same, with the experiment's or the test run's ID as "
        "`run_id`. Comment lines keep "
        "the connection alive. "
        "Events are hints: read state from the API after reconnecting."
    ),
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}},
        **problem_responses(400, 401, 403, 503),
    },
)
async def events(
    request: Request,
    job_id: Annotated[uuid.UUID | None, Query(description="Only this job's events")] = None,
    run_id: Annotated[uuid.UUID | None, Query(description="Only this run's events")] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> StreamingResponse:
    hub = request.app.state.events
    if hub is None:
        raise Problem("unavailable", code="events_not_configured")
    subscription = hub.subscribe(principal.workspace.id)

    async def stream() -> AsyncIterator[str]:
        sequence = 0
        try:
            yield f"retry: {RETRY_MILLISECONDS}\n: connected\n\n"
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(
                        subscription.queue.get(), timeout=KEEPALIVE_SECONDS
                    )
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if job_id is not None and event.get("job_id") != str(job_id):
                    continue
                if run_id is not None and event.get("run_id") != str(run_id):
                    continue
                if subscription.lagged:
                    subscription.lagged = False
                    yield "event: lagged\ndata: {}\n\n"
                sequence += 1
                data = json.dumps(event, separators=(",", ":"))
                yield f"id: {sequence}\nevent: {event.get('type', 'message')}\ndata: {data}\n\n"
        finally:
            hub.unsubscribe(subscription)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
