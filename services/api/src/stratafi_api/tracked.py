"""Eval runs as tracked runs: a store search row flattened for the search language,
comparisons, model comparison and the RAG pipelines list. Shared by the tracking routes."""

from typing import Any

from stratafi_core.tracking import TrackedRun, flatten_metrics, flatten_params
from stratafi_store.tracking import SearchRow


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def tracked_run(row: SearchRow) -> TrackedRun:
    context = row.context or {}
    attributes: dict[str, Any] = {
        "id": str(row.id),
        "status": row.status,
        "split": row.split,
        "title": row.title,
        "created_by": row.created_by,
        "created_at": _iso(row.created_at),
        "finished_at": _iso(row.finished_at),
        "eval_set": row.eval_set_name,
        "eval_set_version": row.eval_set_version,
        "config": row.retrieval_name,
        "config_version": row.retrieval_version,
        "pipeline": row.pipeline_name,
        "pipeline_version": row.pipeline_version,
        "run_id": str(row.pipeline_run_id),
        "corpus": row.corpus_name,
        "corpus_version": row.corpus_version,
        "experiment": row.experiment_name,
        "trial": row.trial,
        "test_read": row.test_read,
        "results_cached": row.results_cached,
        "archived": row.archived_at is not None,
        "k_eval": context.get("k_eval"),
    }
    return TrackedRun(
        id=str(row.id),
        attributes=attributes,
        params=flatten_params(row.stages, row.retrieval, context.get("models")),
        metrics=flatten_metrics(row.metrics),
        tags={str(key): str(value) for key, value in (row.tags or {}).items()},
    )


def label_of(row: SearchRow) -> str:
    """A short name for a run in charts and tables: its title, or its config and pipeline."""
    if row.title:
        return row.title
    config = f"{row.retrieval_name} v{row.retrieval_version}" if row.retrieval_name else "config"
    pipeline = f"{row.pipeline_name} v{row.pipeline_version}" if row.pipeline_name else "pipeline"
    return f"{config} · {pipeline}"
