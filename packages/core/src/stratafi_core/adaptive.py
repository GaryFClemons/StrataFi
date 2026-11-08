"""Adaptive search: successive halving.

A grid evaluates every trial on every dev query. Successive halving evaluates every trial on a
small subset of the queries first, keeps the most promising ``1/eta`` of them, evaluates those
on a subset ``eta`` times larger, and so on; the last rung evaluates the survivors on every
query. The subsets are prefixes of one seeded order of the queries, so each rung's subset holds
the one before it. Eval runs cut their queries into shards with content-addressed keys
(``stratafi_core.evalruns``); successive halving cuts them ``min_queries`` at a time, so a rung
reuses the earlier rungs' shards and a trial pays for each query once, however far it goes.

Pruning respects uncertainty. A rung ranks its trials by mean and keeps the best
``keep``. Every other trial is compared with the rung's leader on the same queries, with a
paired bootstrap of the difference as comparisons are everywhere else, and it's dropped only
when the whole interval is behind the leader. A trial that might still be the best stays, so
noise on a small subset doesn't prune the winner, and a rung can keep more than ``keep``.

``halving_schedule`` plans the rungs, ``prune`` decides one, and ``successive_halving`` runs the
whole search against a scoring function, as the unit benchmark does; the experiment
orchestrator (``stratafi_store.experiments``) runs it with eval runs. Random search is a seeded
draw of the grid (``Strategy``: ``random`` with ``n``); TPE through Optuna would be an optional
extra and isn't built.
"""

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from stratafi_core.stats import DEFAULT_RESAMPLES, Estimate, bootstrap_means

DEFAULT_ETA = 3
DEFAULT_MIN_QUERIES = 10


@dataclass(frozen=True)
class Rung:
    queries: int  # the subset's size; the last rung's is every query
    keep: int  # trials kept by mean, before uncertainty keeps more; the last rung keeps all

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def halving_schedule(
    trials: int, queries: int, *, eta: int = DEFAULT_ETA, min_queries: int = DEFAULT_MIN_QUERIES
) -> list[Rung]:
    """The rungs: subsets growing by ``eta`` from ``min_queries``, each keeping ``1/eta`` of its
    trials, while a subset is smaller than every query and more than one trial is left; then
    the last rung, on every query."""
    if trials <= 0 or queries <= 0:
        return []
    rungs: list[Rung] = []
    count, alive = min_queries, trials
    while count < queries and alive > 1:
        keep = max(1, math.ceil(alive / eta))
        rungs.append(Rung(count, keep))
        count, alive = count * eta, keep
    rungs.append(Rung(queries, alive))
    return rungs


def nominal_cost(schedule: Sequence[Rung], trials: int) -> dict[str, Any]:
    """The queries a schedule scores when each rung keeps exactly ``keep``: each (trial, query)
    once, as nested shards pay; every rung's whole subset, without that reuse; and the grid."""
    alive, before, cost, evaluations = trials, 0, 0, 0
    for rung in schedule:
        cost += alive * (rung.queries - before)
        evaluations += alive * rung.queries
        before, alive = rung.queries, rung.keep
    grid = trials * schedule[-1].queries if schedule else 0
    return {
        "cost": cost,
        "evaluations": evaluations,
        "grid": grid,
        "share": round(cost / grid, 4) if grid else 0.0,
    }


def query_order(queries: Sequence[str], seed: int) -> list[str]:
    """One seeded order of the queries: every rung's subset is a prefix of it."""
    ordered = sorted(queries)
    random.Random(seed).shuffle(ordered)  # noqa: S311 - a seeded order, not a secret
    return ordered


@dataclass(frozen=True)
class Verdict:
    mean: float
    kept: bool
    # Outside the top ``keep``: the trial minus the leader, on the same queries, with its
    # interval.
    against_leader: Estimate | None = None

    def as_dict(self) -> dict[str, Any]:
        found = asdict(self)
        found["against_leader"] = self.against_leader.as_dict() if self.against_leader else None
        return found


@dataclass(frozen=True)
class RungDecision:
    ranked: list[int]  # best first
    verdicts: dict[int, Verdict]

    @property
    def kept(self) -> list[int]:
        return [trial for trial in self.ranked if self.verdicts[trial].kept]

    @property
    def dropped(self) -> list[int]:
        return [trial for trial in self.ranked if not self.verdicts[trial].kept]


def prune(
    values: Mapping[int, Sequence[float]],
    *,
    keep: int,
    goal: str = "maximize",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> RungDecision:
    """Which trials a rung keeps (the module docstring says how). ``values`` holds each trial's
    per-query values on the same queries, in the same order."""
    sizes = {len(found) for found in values.values()}
    if len(sizes) > 1:
        raise ValueError("paired values come in pairs")
    sign = 1.0 if goal == "maximize" else -1.0
    means = {trial: float(np.mean(found)) if len(found) else 0.0 for trial, found in values.items()}
    ranked = sorted(values, key=lambda trial: (-sign * means[trial], trial))
    if not ranked:
        return RungDecision([], {})
    top, rest = ranked[: max(1, keep)], ranked[max(1, keep) :]
    verdicts = {trial: Verdict(round(means[trial], 6), True) for trial in top}
    if rest and sizes.pop():
        base = np.asarray(values[ranked[0]], dtype=float)
        differences = np.column_stack(
            [np.asarray(values[trial], dtype=float) - base for trial in rest]
        )
        intervals = bootstrap_means(differences, resamples=resamples, seed=seed)
        for trial, interval in zip(rest, intervals, strict=True):
            behind = interval is not None and (interval.high < 0 if sign > 0 else interval.low > 0)
            verdicts[trial] = Verdict(round(means[trial], 6), not behind, interval)
    else:  # no queries to go on: nothing is clearly behind
        verdicts.update({trial: Verdict(round(means[trial], 6), True) for trial in rest})
    return RungDecision(ranked, verdicts)


@dataclass(frozen=True)
class RungResult:
    queries: int
    evaluated: list[int]
    decision: RungDecision


@dataclass
class HalvingResult:
    winner: int | None
    rungs: list[RungResult] = field(default_factory=list)
    cost: int = 0  # queries scored, each (trial, query) once: what nested shards pay
    evaluations: int = 0  # every rung's whole subset, without that reuse
    grid_cost: int = 0  # every trial on every query

    @property
    def share(self) -> float:
        return self.cost / self.grid_cost if self.grid_cost else 0.0

    @property
    def share_without_reuse(self) -> float:
        return self.evaluations / self.grid_cost if self.grid_cost else 0.0


Scorer = Callable[[Sequence[int], Sequence[str]], Mapping[int, Sequence[float]]]


def successive_halving(
    trials: Sequence[int],
    queries: Sequence[str],
    score: Scorer,
    *,
    eta: int = DEFAULT_ETA,
    min_queries: int = DEFAULT_MIN_QUERIES,
    goal: str = "maximize",
    seed: int = 0,
    resamples: int = DEFAULT_RESAMPLES,
) -> HalvingResult:
    """Run the search: ``score(trials, queries)`` gives each trial's values on those queries,
    in their order. The winner is the best mean on every query among the survivors."""
    order = query_order(queries, seed)
    schedule = halving_schedule(len(trials), len(order), eta=eta, min_queries=min_queries)
    result = HalvingResult(winner=None, grid_cost=len(trials) * len(order))
    alive, before = sorted(trials), 0
    for index, rung in enumerate(schedule):
        subset = order[: rung.queries]
        values = score(alive, subset)
        result.cost += len(alive) * (len(subset) - before)
        result.evaluations += len(alive) * len(subset)
        last = index == len(schedule) - 1
        decision = prune(
            {trial: values[trial] for trial in alive},
            keep=len(alive) if last else rung.keep,
            goal=goal,
            resamples=resamples,
            seed=seed,
        )
        result.rungs.append(RungResult(len(subset), alive, decision))
        if last:
            result.winner = decision.ranked[0] if decision.ranked else None
        alive, before = sorted(decision.kept), len(subset)
    return result
