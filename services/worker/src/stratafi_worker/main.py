"""``stratafi-worker``: claim and run jobs until stopped.

    uv run --env-file .env stratafi-worker

Settings come from ``config/stratafi.yaml`` and the environment: ``worker.slots`` per resource
class, leases, backoff, and ``providers.limits``; ``storage`` for artifacts; ``services`` and
``models`` for the clients stage components use. SIGTERM or Ctrl-C stops it gracefully: no new
claims, running jobs get ``worker.shutdown_grace_seconds``, then go back to the queue.
"""

import os
import signal
import sys
import threading
from pathlib import Path

from stratafi_components import default_registry
from stratafi_components.confine import harden_process
from stratafi_core.logs import configure_logging
from stratafi_core.observability import start_metrics_server
from stratafi_core.settings import Settings
from stratafi_store.artifacts import artifact_store_from_settings
from stratafi_store.db import create_engine
from stratafi_store.jobs import JobQueue
from stratafi_store.retention import RetentionSweeper
from stratafi_worker.answers import register_answer_handlers
from stratafi_worker.automation import register_automation_handlers
from stratafi_worker.evals import register_eval_handlers
from stratafi_worker.experiments import register_experiment_handlers
from stratafi_worker.handlers import builtin_handlers
from stratafi_worker.limits import ProviderLimiter, SharedProviderLimiter
from stratafi_worker.runs import Services, register_run_handlers
from stratafi_worker.testruns import register_test_handlers
from stratafi_worker.training import register_training_handlers
from stratafi_worker.worker import Worker, default_owner

HEALTH_FILE_ENV = "STRATAFI_WORKER_HEALTH_FILE"


def main() -> int:
    settings = Settings()
    configure_logging("stratafi-worker", level=settings.log_level)
    if settings.database.url is None:
        print("stratafi-worker: database.url is not set", file=sys.stderr)
        return 2
    # A prod configuration must confine parsers and guard its metrics.
    problems = settings.deployment_problems()
    for problem in problems:
        print(f"stratafi-worker: {problem}", file=sys.stderr)
    if problems:
        return 2
    # A parser runs as this process's user: it mustn't read this process's memory or
    # environment, which hold the database's and the store's credentials.
    harden_process()
    worker_settings = settings.worker
    slots = sum(worker_settings.slots.values())
    engine = create_engine(
        settings.database.url.get_secret_value(),
        pool_size=max(2, slots + 1),
        max_overflow=2,
        application_name="stratafi-worker",
    )
    queue = JobQueue(
        engine,
        lease_seconds=worker_settings.lease_seconds,
        backoff_base_seconds=worker_settings.backoff_base_seconds,
        backoff_max_seconds=worker_settings.backoff_max_seconds,
    )
    handlers = builtin_handlers()
    store = artifact_store_from_settings(settings.storage)
    services = Services(engine, settings, store)
    registry = default_registry()
    orchestrator = register_run_handlers(
        handlers,
        engine=engine,
        queue=queue,
        store=store,
        registry=registry,
        services=services,
        settings=settings,
    )
    evals = register_eval_handlers(
        handlers,
        engine=engine,
        queue=queue,
        store=store,
        registry=registry,
        services=services,
        settings=settings,
    )
    experiments = register_experiment_handlers(
        handlers, engine=engine, queue=queue, runs=orchestrator, evals=evals, settings=settings
    )
    register_answer_handlers(
        handlers,
        engine=engine,
        store=store,
        registry=registry,
        services=services,
        settings=settings,
    )
    register_training_handlers(
        handlers,
        engine=engine,
        store=store,
        registry=registry,
        services=services,
        settings=settings,
    )
    tests = register_test_handlers(
        handlers,
        engine=engine,
        queue=queue,
        store=store,
        registry=registry,
        services=services,
        settings=settings,
    )
    scheduler = register_automation_handlers(
        handlers, engine=engine, queue=queue, settings=settings
    )
    # Retention: one worker at a time purges what expired.
    retention = RetentionSweeper(engine, store, settings.retention)
    health_file = os.environ.get(HEALTH_FILE_ENV)
    owner = default_owner()
    worker = Worker(
        queue,
        handlers,
        slots=worker_settings.slots,
        limiter=(
            SharedProviderLimiter(engine, settings.providers.limits, holder=owner)
            if settings.providers.shared_limits
            else ProviderLimiter(settings.providers.limits)
        ),
        owner=owner,
        poll_seconds=worker_settings.poll_seconds,
        heartbeat_seconds=worker_settings.heartbeat_seconds,
        shutdown_grace_seconds=worker_settings.shutdown_grace_seconds,
        health_file=Path(health_file) if health_file else None,
        maintenance=[
            orchestrator.reconcile,
            evals.reconcile,
            experiments.reconcile,
            tests.reconcile,
            scheduler.tick,
            retention.tick,
        ],
    )
    stopped = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    observability = settings.observability
    if observability.metrics and observability.worker_metrics_port is not None:
        # Prometheus scrapes the worker here: jobs, stages and providers.
        start_metrics_server(observability.worker_metrics_port, observability.worker_metrics_host)
    worker.start()
    try:
        while not stopped.wait(1.0):
            pass
    finally:
        worker.stop()
        services.close()
        engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
