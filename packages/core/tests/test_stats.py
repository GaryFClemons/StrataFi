"""Bootstrap intervals: seeded, reproducible, and sensible."""

import numpy as np
import pytest

from stratafi_core.stats import (
    bootstrap_mean,
    bootstrap_means,
    compare_paired,
    holm,
    paired_randomization,
    percentile,
    resample_counts,
)


def test_the_same_seed_gives_the_same_interval() -> None:
    values = np.random.default_rng(1).random(90)
    assert bootstrap_mean(values, seed=42) == bootstrap_mean(values, seed=42)
    assert bootstrap_mean(values, seed=42) != bootstrap_mean(values, seed=43)


def test_resample_counts_draw_n_queries_each_time() -> None:
    counts = resample_counts(90, resamples=50, seed=0)
    assert counts.shape == (50, 90)
    assert set(counts.sum(axis=1)) == {90}


def test_intervals_bracket_the_mean_and_shrink_with_more_queries() -> None:
    generator = np.random.default_rng(7)
    small = bootstrap_mean(generator.binomial(1, 0.6, 30).astype(float), seed=1)
    large = bootstrap_mean(generator.binomial(1, 0.6, 3000).astype(float), seed=1)
    assert small is not None
    assert large is not None
    assert small.low <= small.mean <= small.high
    assert large.high - large.low < small.high - small.low
    assert large.low < 0.6 < large.high


def test_columns_are_resampled_together() -> None:
    values = np.column_stack([np.arange(10, dtype=float), np.arange(10, dtype=float) * 2])
    first, second = bootstrap_means(values, seed=5)
    assert first is not None
    assert second is not None
    assert second.mean == pytest.approx(first.mean * 2)
    assert second.low == pytest.approx(first.low * 2)
    assert second.high == pytest.approx(first.high * 2)


def test_edge_cases() -> None:
    assert bootstrap_means(np.empty((0, 3))) == [None, None, None]
    single = bootstrap_mean([0.25])
    assert single is not None
    assert (single.mean, single.low, single.high, single.n) == (0.25, 0.25, 0.25, 1)
    constant = bootstrap_mean([1.0] * 20, seed=3)
    assert constant is not None
    assert (constant.low, constant.high) == (1.0, 1.0)
    assert percentile([], 95) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


# --- paired comparisons ------------------------------------------------------------


def test_identical_runs_give_an_interval_around_zero_and_a_large_p_value() -> None:
    """Identical runs give a CI containing 0 and a large p-value."""
    values = np.random.default_rng(3).binomial(1, 0.7, 90).astype(float)
    same = compare_paired(values, values, seed=1)
    assert same is not None
    assert same.low <= 0.0 <= same.high
    assert same.delta == 0.0
    assert same.p_value == 1.0
    assert (same.wins, same.losses, same.ties) == (0, 0, 90)


def test_noise_alone_is_not_called_a_difference() -> None:
    generator = np.random.default_rng(5)
    a = generator.binomial(1, 0.7, 90).astype(float)
    b = a.copy()
    flips = generator.choice(90, 6, replace=False)
    b[flips] = 1 - b[flips]  # a few queries change, both ways
    result = compare_paired(a, b, seed=2)
    assert result is not None
    assert result.p_value > 0.05
    assert result.low <= 0.0 <= result.high


def test_a_synthetic_shift_is_detected() -> None:
    """A synthetic shift is detected."""
    generator = np.random.default_rng(9)
    a = generator.uniform(0.2, 0.8, 90)
    b = np.clip(a + 0.1 + generator.normal(0, 0.03, 90), 0, 1)
    result = compare_paired(a, b, seed=4)
    assert result is not None
    assert result.p_value < 0.01
    assert 0.0 < result.low <= result.delta <= result.high
    assert result.wins > result.losses
    worse = compare_paired(b, a, seed=4)
    assert worse is not None
    assert worse.high < 0.0
    assert worse.losses == result.wins


def test_paired_results_reproduce_for_a_seed_and_pair_strictly() -> None:
    a, b = [0.0, 1.0, 1.0, 0.5], [1.0, 1.0, 0.0, 1.0]
    assert compare_paired(a, b, seed=7) == compare_paired(a, b, seed=7)
    assert compare_paired([], []) is None
    with pytest.raises(ValueError, match="pairs"):
        compare_paired([1.0], [1.0, 0.0])
    assert paired_randomization([0.0, 0.0], [0.0, 0.0]) == 1.0


def test_holm_adjusts_in_step_down_order() -> None:
    # Sorted 0.01, 0.02, 0.03, 0.04: 4 * 0.01, 3 * 0.02, 2 * 0.03, 1 * 0.04, made monotone.
    assert holm([0.04, 0.01, 0.03, 0.02]) == [0.06, 0.04, 0.06, 0.06]
    assert holm([0.5, 0.9]) == [1.0, 1.0]
    assert holm([]) == []
