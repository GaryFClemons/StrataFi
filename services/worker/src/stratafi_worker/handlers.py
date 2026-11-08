"""Built-in job handlers. Runs' handlers (``run.start``, ``stage.run``) are in
``stratafi_worker.runs``, eval runs' (``eval.start``, ``eval.shard``, ``eval.finish``) in
``stratafi_worker.evals``; sweep and test handlers arrive with their steps.

``system.selftest`` checks the queue end to end: it runs ``steps`` steps of ``step_seconds``
each, reporting progress and checking for cancellation between them. Enqueue one to see a
worker, its heartbeats and the live events working.
"""

import time
from collections.abc import Mapping
from typing import Any

from stratafi_worker.worker import HandlerRegistry, JobContext, PermanentJobError

MAX_SELFTEST_STEPS = 1000
MAX_SELFTEST_STEP_SECONDS = 10.0


def selftest(context: JobContext) -> Mapping[str, Any]:
    try:
        steps = int(context.payload.get("steps", 3))
        step_seconds = float(context.payload.get("step_seconds", 0.1))
    except (TypeError, ValueError):
        raise PermanentJobError("invalid_payload") from None
    if not 0 <= steps <= MAX_SELFTEST_STEPS or not 0 <= step_seconds <= MAX_SELFTEST_STEP_SECONDS:
        raise PermanentJobError("invalid_payload")
    for done in range(steps):
        context.checkpoint()
        time.sleep(step_seconds)
        context.progress(done=done + 1, total=steps, force=done + 1 == steps)
    return {"steps": steps}


def builtin_handlers() -> HandlerRegistry:
    registry = HandlerRegistry()
    registry.register("system.selftest", selftest)
    return registry
