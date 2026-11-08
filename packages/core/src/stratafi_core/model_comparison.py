"""Comparing models across eval runs, by controlled pairs.

A **model** is a role (``embed``, ``rerank``, ``adapter``), a component, a model name and a
revision, as eval runs record them in their plan's ``models``. For one eval set version and a
role, ``compare_models`` gives:

- **the leaderboard:** each model's eval runs, its best run on the metric (with that run's
  interval), the median of its runs' p95 latency, and its egress
- **controlled pairs:** two eval runs of the same split and corpus version whose flattened
  parameters differ only in the role's keys (``ROLE_KEYS``), and whose models differ. Their
  paired difference on shared queries isolates the model, where best-run rankings mix it with
  whatever else differs
- **the matrix:** for each ordered pair of models, the mean of their controlled pairs' deltas
  (the column's model minus the row's), and how many pairs there are

The caller gives each run's flattened ``TrackedRun`` and, for the pairs it wants measured,
their per-query scores. Numbers, IDs and model names only.
"""

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from stratafi_core.comparison import score_value
from stratafi_core.stats import DEFAULT_RESAMPLES, compare_paired
from stratafi_core.tracking import TrackedRun, differing_params

Role = Literal["embed", "rerank", "adapter"]
ROLES: tuple[Role, ...] = ("embed", "rerank", "adapter")
# The parameter keys that belong to each role: a pair may differ only in these.
ROLE_KEYS: dict[str, tuple[str, ...]] = {
    "embed": ("embed.", "models.embed."),
    "rerank": ("rerank.", "models.rerank."),
    "adapter": ("retrievers.dense.adapter", "models.adapter."),
}
MAX_PAIRS = 40


@dataclass(frozen=True)
class ModelKey:
    role: str
    component: str
    name: str
    revision: str | None

    @property
    def key(self) -> str:
        return f"{self.role}|{self.component}|{self.name}|{self.revision or ''}"

    @property
    def label(self) -> str:
        revision = f" ({self.revision[:12]})" if self.revision else ""
        return f"{self.name}{revision}"


def model_of(run: TrackedRun, role: str) -> ModelKey | None:
    """The model a run used in ``role``, from its flattened parameters; None if it used none
    (no dense retriever for an embedder, no adapter)."""
    params = run.params
    component = params.get(f"models.{role}.component")
    if not isinstance(component, str):
        return None
    name = (
        params.get(f"models.{role}.alias")
        or params.get(f"models.{role}.params.alias")
        or params.get(f"models.{role}.params.model")
        or component
    )
    revision = params.get(f"models.{role}.revision")
    return ModelKey(
        role, component, str(name), str(revision) if revision not in (None, "") else None
    )


def in_role(key: str, role: str) -> bool:
    return any(
        key == prefix or (prefix.endswith(".") and key.startswith(prefix))
        for prefix in ROLE_KEYS[role]
    )


def controlled(a: TrackedRun, b: TrackedRun, role: str) -> bool:
    """Whether two runs differ only in ``role``'s model: the same split and corpus version,
    different models, and every differing parameter the role's."""
    for name in ("split", "corpus", "corpus_version", "eval_set", "eval_set_version"):
        if a.attributes.get(name) != b.attributes.get(name):
            return False
    model_a, model_b = model_of(a, role), model_of(b, role)
    if model_a is None or model_b is None or model_a.key == model_b.key:
        return False
    differing = differing_params([a, b])
    return bool(differing) and all(in_role(key, role) for key in differing)


def controlled_pairs(runs: Sequence[TrackedRun], role: str) -> list[tuple[int, int]]:
    """Index pairs (earlier, later in ``runs``) that differ only in the role's model."""
    pairs = [
        (i, j)
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
        if controlled(runs[i], runs[j], role)
    ]
    return pairs[:MAX_PAIRS]


Estimate = Mapping[str, Any]  # mean, low, high, n


def leaderboard(
    runs: Sequence[TrackedRun],
    role: str,
    metric: str,
    *,
    estimates: Mapping[str, Estimate | None],
    egress: Callable[[TrackedRun], str | None] = lambda run: None,
) -> list[dict[str, Any]]:
    """Each model with its runs, best first by its best run's mean on ``metric``."""
    groups: dict[str, list[TrackedRun]] = {}
    keys: dict[str, ModelKey] = {}
    for run in runs:
        model = model_of(run, role)
        if model is None:
            continue
        groups.setdefault(model.key, []).append(run)
        keys[model.key] = model
    rows = []
    for key, members in groups.items():
        scored = [
            (estimate, run)
            for run in members
            if (estimate := estimates.get(run.id)) is not None
            and isinstance(estimate.get("mean"), int | float)
        ]
        best = max(scored, key=lambda pair: (pair[0]["mean"], pair[1].id), default=None)
        latencies = [
            float(value)
            for run in members
            if isinstance(value := run.metrics.get("latency.total_ms.p95"), int | float)
        ]
        model = keys[key]
        rows.append(
            {
                "key": key,
                "label": model.label,
                "role": model.role,
                "component": model.component,
                "name": model.name,
                "revision": model.revision,
                "runs": len(members),
                "best_eval_run_id": best[1].id if best else None,
                "best": dict(best[0]) if best else None,
                "latency_p95_ms": round(statistics.median(latencies), 3) if latencies else None,
                "egress": egress(members[0]),
            }
        )
    rows.sort(key=lambda row: (-(row["best"] or {}).get("mean", -1.0), row["key"]))
    return rows


def pair_result(
    a: Mapping[str, Mapping[str, Any]],
    b: Mapping[str, Mapping[str, Any]],
    *,
    rule: str,
    checkpoint: str,
    metric: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, Any] | None:
    """B against A on the queries with evidence both scored."""
    shared = sorted(
        item
        for item in set(a) & set(b)
        if a[item].get("has_evidence") and b[item].get("has_evidence")
    )
    values = [
        (
            score_value(a[item], rule, checkpoint, metric),
            score_value(b[item], rule, checkpoint, metric),
        )
        for item in shared
    ]
    kept = [(x, y) for x, y in values if x is not None and y is not None]
    result = compare_paired(
        [x for x, _ in kept], [y for _, y in kept], resamples=resamples, seed=seed
    )
    return result.as_dict() if result is not None else None


def matrix(
    models: Sequence[str], pairs: Sequence[Mapping[str, Any]]
) -> list[list[dict[str, Any] | None]]:
    """``cells[i][j]``: the mean delta of model j over model i across their controlled pairs,
    and how many; None where no pair measured them."""
    index = {key: position for position, key in enumerate(models)}
    sums: dict[tuple[int, int], list[float]] = {}
    for pair in pairs:
        result = pair.get("result")
        if not result or pair["a_model"] not in index or pair["b_model"] not in index:
            continue
        i, j = index[pair["a_model"]], index[pair["b_model"]]
        sums.setdefault((i, j), []).append(float(result["delta"]))
        sums.setdefault((j, i), []).append(-float(result["delta"]))
    return [
        [
            {"delta": round(sum(found) / len(found), 6), "pairs": len(found)}
            if (found := sums.get((i, j)))
            else None
            for j in range(len(models))
        ]
        for i in range(len(models))
    ]
