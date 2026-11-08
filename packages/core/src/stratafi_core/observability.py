"""Prometheus metrics: what the API and the worker count.

One registry per process, ``REGISTRY``, Stratafi's own. Every label is a route template, a job
kind, a stage, an outcome or error code, a resource class or a provider's configured name:
never a document, chunk or query's text, a name or an ID.

- **The API** (``stratafi_api.middleware``): requests and their durations by route template,
  method and status, and at ``GET /api/v1/metrics`` the queue, read from Postgres when scraped
  (``stratafi_store.jobs.QueueCollector``): jobs by status and resource class, and the age of
  the oldest queued job per class.
- **The worker** (``worker.metrics_port``): jobs by kind and outcome with their durations,
  running jobs by class, stage tasks and their durations by stage and outcome (throughput),
  provider waits and errors.

Run records keep each run's own timings, counts and costs: experiment data, not metrics.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry(auto_describe=True)
CONTENT_TYPE = CONTENT_TYPE_LATEST

# Seconds: from a cache hit to a long parse or an eval shard.
_DURATIONS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 900)

HTTP_REQUESTS = Counter(
    "stratafi_http_requests_total",
    "API requests, by route template, method and status",
    ["method", "route", "status"],
    registry=REGISTRY,
)
HTTP_DURATION = Histogram(
    "stratafi_http_request_duration_seconds",
    "API request durations, by route template and method",
    ["method", "route"],
    buckets=_DURATIONS,
    registry=REGISTRY,
)
JOBS_FINISHED = Counter(
    "stratafi_jobs_finished_total",
    "Jobs a worker finished, by kind and outcome",
    ["kind", "outcome"],
    registry=REGISTRY,
)
JOB_DURATION = Histogram(
    "stratafi_job_duration_seconds",
    "How long a worker ran each job, by kind",
    ["kind"],
    buckets=_DURATIONS,
    registry=REGISTRY,
)
JOBS_RUNNING = Gauge(
    "stratafi_jobs_running",
    "Jobs this worker is running, by resource class",
    ["resource_class"],
    registry=REGISTRY,
)
STAGE_TASKS = Counter(
    "stratafi_stage_tasks_total",
    "Pipeline stage tasks finished, by stage and outcome: each is one document's stage",
    ["stage", "outcome"],
    registry=REGISTRY,
)
STAGE_DURATION = Histogram(
    "stratafi_stage_task_duration_seconds",
    "Stage task durations, by stage",
    ["stage"],
    buckets=_DURATIONS,
    registry=REGISTRY,
)
PROVIDER_WAIT = Histogram(
    "stratafi_provider_wait_seconds",
    "Time spent waiting for a provider's rate and concurrency limits",
    ["provider"],
    buckets=(0, 0.01, 0.05, 0.1, 0.5, 1, 5, 15, 60),
    registry=REGISTRY,
)
PROVIDER_ERRORS = Counter(
    "stratafi_provider_errors_total",
    "Failed calls to a provider, by its error code",
    ["provider", "code"],
    registry=REGISTRY,
)


def exposition() -> bytes:
    """The registry in Prometheus' text format."""
    return generate_latest(REGISTRY)


def start_metrics_server(port: int, host: str = "127.0.0.1") -> None:
    """Serve ``/metrics`` for Prometheus on its own thread (the worker's)."""
    from prometheus_client import start_http_server

    start_http_server(port, addr=host, registry=REGISTRY)
