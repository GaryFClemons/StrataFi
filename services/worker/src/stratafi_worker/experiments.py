"""The worker's experiment job: ``experiment.advance`` moves a running
experiment on, a step at a time (``stratafi_store.experiments.ExperimentOrchestrator``).

Its pipeline runs and eval runs are ordinary runs: their own jobs do the work, and each one's
end queues the next ``experiment.advance``. The orchestrator's ``reconcile`` runs with the
worker's heartbeats, for an experiment that missed one or whose time budget ran out.
"""

import uuid
from collections.abc import Mapping
from typing import Any

from sqlalchemy.engine import Engine

from stratafi_core.settings import Settings
from stratafi_store.evalruns import EvalOrchestrator
from stratafi_store.experiments import ADVANCE_JOB, ExperimentOrchestrator
from stratafi_store.jobs import JobQueue
from stratafi_store.runs import Orchestrator
from stratafi_worker.worker import HandlerRegistry, JobContext


def register_experiment_handlers(
    handlers: HandlerRegistry,
    *,
    engine: Engine,
    queue: JobQueue,
    runs: Orchestrator,
    evals: EvalOrchestrator,
    settings: Settings,
) -> ExperimentOrchestrator:
    """Register ``experiment.advance``; returns the orchestrator, whose ``reconcile`` the worker
    runs with its heartbeats."""
    orchestrator = ExperimentOrchestrator(
        engine, queue, runs, evals, max_attempts=settings.worker.max_attempts
    )

    def advance(job: JobContext) -> Mapping[str, Any]:
        experiment = orchestrator.advance(uuid.UUID(str(job.payload["experiment_id"])))
        if experiment is None:
            return {"skipped": True}
        return {"status": experiment.status, "counts": experiment.counts}

    handlers.register(ADVANCE_JOB, advance)
    return orchestrator
