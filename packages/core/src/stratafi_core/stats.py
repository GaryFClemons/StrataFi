"""Statistics that show uncertainty.

Eval sets are small (88 queries in the Halcyon demo set), so a gap of a few points is often
noise. Every aggregate therefore comes with a 95% confidence interval from a bootstrap over
queries: B resamples (2,000 by default) drawn by one generator seeded with a recorded seed, so
the same seed gives the same interval.

``bootstrap_means`` resamples once for every column of a queries x metrics matrix: each resample
is a vector of counts (how often each query was drawn, a multinomial draw), and the resampled
means are one matrix product, ``counts @ values / n``. Intervals are the percentile interval of
those means.

Comparing two runs is paired: the same queries under both configurations. The interval
of the mean difference comes from the same bootstrap over queries; a sign-flip randomization
test gives its p-value; and Holm's method adjusts p-values when many comparisons are made.

Guardrails: ``min_detectable_difference`` is the smallest paired difference an eval set
of ``n`` queries detects with the usual power, from the spread of per-query differences; and
``repeat_summary`` says how much a non-deterministic stage's repeats disagree.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np

DEFAULT_RESAMPLES = 2000
DEFAULT_LEVEL = 0.95
ROUND = 6


@dataclass(frozen=True)
class Estimate:
    """A mean over queries with its bootstrap interval."""

    mean: float
    low: float
    high: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def resample_counts(n: int, *, resamples: int, seed: int) -> np.ndarray:
    """``resamples`` x ``n`` draw counts: row b says how often each query is in resample b."""
    generator = np.random.default_rng(seed)
    return generator.multinomial(n, np.full(n, 1.0 / n), size=resamples)


def bootstrap_means(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    level: float = DEFAULT_LEVEL,
) -> list[Estimate | None]:
    """Each column's mean over rows (queries), with a percentile bootstrap interval at
    ``level``. None for a column with no rows."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.size == 0 or matrix.shape[0] == 0:
        columns = matrix.shape[1] if matrix.ndim == 2 else 0
        return [None] * columns
    n = matrix.shape[0]
    means = matrix.mean(axis=0)
    if n == 1:
        return [Estimate(round(float(m), ROUND), round(float(m), ROUND), round(float(m), ROUND), 1)
                for m in means]  # fmt: skip
    boot = resample_counts(n, resamples=resamples, seed=seed) @ matrix / n
    tail = (1 - level) / 2
    lows, highs = np.quantile(boot, [tail, 1 - tail], axis=0)
    return [
        Estimate(round(float(mean), ROUND), round(float(low), ROUND), round(float(high), ROUND), n)
        for mean, low, high in zip(means, lows, highs, strict=True)
    ]


def bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    level: float = DEFAULT_LEVEL,
) -> Estimate | None:
    return bootstrap_means(
        np.asarray(values, dtype=float).reshape(-1, 1),
        resamples=resamples,
        seed=seed,
        level=level,
    )[0]


def percentile(values: Sequence[float], q: float) -> float | None:
    """The ``q``-th percentile (0 to 100) of ``values``, linear between points; None if empty."""
    if not values:
        return None
    result = float(np.percentile(np.asarray(values, dtype=float), q))
    return round(result, 3) if math.isfinite(result) else None


# --- comparing two runs ------------------------------------------------------------


@dataclass(frozen=True)
class PairedComparison:
    """B against A on the same queries: B's mean minus A's, its bootstrap interval, a paired
    randomization test's two-sided p-value, and per-query wins, losses and ties (for B)."""

    n: int
    mean_a: float
    mean_b: float
    delta: float
    low: float
    high: float
    p_value: float
    wins: int
    losses: int
    ties: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def paired_randomization(
    a: Sequence[float], b: Sequence[float], *, resamples: int = DEFAULT_RESAMPLES, seed: int = 0
) -> float:
    """The two-sided p-value of a paired sign-flip randomization test of the mean difference:
    under no difference, each query's difference is as likely to have either sign. Counts the
    observed assignment, so p is never 0: (1 + extreme) / (1 + resamples)."""
    differences = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = differences.size
    if n == 0 or not np.any(differences):
        return 1.0
    observed = abs(differences.mean())
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(resamples, n))
    flipped = np.abs(signs @ differences) / n
    extreme = int(np.count_nonzero(flipped >= observed - 1e-12))
    return round((1 + extreme) / (1 + resamples), ROUND)


def compare_paired(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    level: float = DEFAULT_LEVEL,
    tie: float = 1e-9,
) -> PairedComparison | None:
    """B against A on paired values (the same queries in the same order). None if empty."""
    first, second = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if first.shape != second.shape:
        raise ValueError("paired values come in pairs")
    if first.size == 0:
        return None
    differences = second - first
    interval = bootstrap_mean(differences, resamples=resamples, seed=seed, level=level)
    if interval is None:  # pragma: no cover - there is at least one pair
        return None
    return PairedComparison(
        n=int(first.size),
        mean_a=round(float(first.mean()), ROUND),
        mean_b=round(float(second.mean()), ROUND),
        delta=interval.mean,
        low=interval.low,
        high=interval.high,
        p_value=paired_randomization(first, second, resamples=resamples, seed=seed),
        wins=int(np.count_nonzero(differences > tie)),
        losses=int(np.count_nonzero(differences < -tie)),
        ties=int(np.count_nonzero(np.abs(differences) <= tie)),
    )


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjustment for several comparisons, in the input order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = round(running, ROUND)
    return adjusted


# --- guardrails ----------------------------------------------------------------------

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_MARGIN = 0.02  # practical equivalence on the objective
FEW_QUERIES = 50


def min_detectable_difference(
    sd_difference: float, n: int, *, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER
) -> float | None:
    """The smallest mean paired difference a two-sided test at ``alpha`` detects with
    ``power``, on ``n`` queries whose differences spread with ``sd_difference``. None for
    fewer than two queries."""
    if n < 2:
        return None
    normal = NormalDist()
    z = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)
    return round(z * sd_difference / math.sqrt(n), ROUND)


@dataclass(frozen=True)
class RepeatSummary:
    """Repeats of one measurement: their mean, the standard deviation between them, and the
    range."""

    repeats: int
    mean: float
    sd: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def repeat_summary(values: Sequence[float]) -> RepeatSummary | None:
    """How a non-deterministic stage's repeats agree. None without a repeat."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return RepeatSummary(
        repeats=int(array.size),
        mean=round(float(array.mean()), ROUND),
        sd=round(sd, ROUND),
        low=round(float(array.min()), ROUND),
        high=round(float(array.max()), ROUND),
    )
