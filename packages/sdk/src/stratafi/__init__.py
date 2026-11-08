"""The Stratafi Python SDK.

    from stratafi import Client

    client = Client.from_env()
    base = client.pipelines.get("baseline-v1")
    run = client.runs.start(base, corpus="halcyon@latest", wait=True)

``stratafi.models`` holds the API's bodies, generated from the OpenAPI contract; the ``stratafi``
command (``stratafi.cli``) wraps the same client.
"""

from stratafi.client import Client
from stratafi.errors import (
    ApiError,
    ConfigError,
    StratafiError,
    UnreachableError,
    WaitTimeout,
)
from stratafi.resources import EvalRun, Experiment, Pipeline, RetrievalConfig, Run
from stratafi.results import Comparison, Metrics, Records
from stratafi.version import __version__

__all__ = [
    "ApiError",
    "Client",
    "Comparison",
    "ConfigError",
    "EvalRun",
    "Experiment",
    "Metrics",
    "Pipeline",
    "Records",
    "RetrievalConfig",
    "Run",
    "StratafiError",
    "UnreachableError",
    "WaitTimeout",
    "__version__",
]
