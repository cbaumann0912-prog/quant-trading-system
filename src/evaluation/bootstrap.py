"""
Bootstrap confidence intervals, i.i.d. and block.

`block_bootstrap` exists because the i.i.d. bootstrap destroys serial
dependence. Financial returns are close to uncorrelated in the mean but
strongly dependent in higher moments (volatility clustering), so resampling
individual observations produces intervals that are too narrow -- it
manufactures independence the data does not have. Sampling contiguous
blocks preserves dependence up to the block length, at the cost of a
bias/variance tradeoff governed by that length.
"""
import numpy as np
from typing import Callable

from src.utils.random_state import get_rng

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn: Callable,
    n_bootstrap: int,
    confidence: float,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Compute a bootstrap percentile confidence interval for a statistic.

    Parameters
    ----------
    data
        1-D array of observed values to resample from.
    statistic_fn
        Function that takes a 1-D array and returns a scalar statistic.
    n_bootstrap
        Number of bootstrap resamples to draw.
    confidence
        Desired confidence level (e.g. 0.95).
    seed
        Random seed. None resolves to utils.random_state.DEFAULT_SEED.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) of the confidence interval.

    Notes
    -----
    This is the i.i.d. bootstrap: observations are resampled individually,
    which destroys serial dependence. On return series with volatility
    clustering the resulting interval is too narrow, because the procedure
    manufactures independence the data does not have. Use `block_bootstrap`
    for any dependent series.

    History: this function previously drew from the legacy process-global
    generator via `np.random.choice` and exposed no seed, so every interval
    it produced was unreproducible. It now routes through `get_rng` like the
    rest of the framework. Intervals computed before that change cannot be
    reproduced from the code and must be regenerated.
    """
    rng = get_rng(seed)
    bootstrap_stats = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        sample = rng.choice(data, len(data), replace=True)
        bootstrap_stats[b] = statistic_fn(sample)

    lower = np.percentile(bootstrap_stats, ((1 - confidence) / 2)*100)
    upper = np.percentile(bootstrap_stats, (1 - ((1 - confidence) / 2))*100)

    result = (lower, upper)

    return result


def block_bootstrap(
    series: np.ndarray,
    block_size: int,
    n_samples: int,
    statistic_fn: Callable,
    seed: int | None = None,
) -> np.ndarray:
    """
    Draw block bootstrap resamples from a time series and compute a statistic
    on each resample.

    Parameters
    ----------
    series
        1-D array of the original time series, assumed ordered.
    block_size
        Length of each contiguous block, ℓ. block_size=1 degenerates to
        standard i.i.d. bootstrap
    n_samples
        Number of bootstrap resamples to draw, B. This should be large
        (~1000-10000+) for a stable estimate of the sampling distribution.
    statistic_fn
        Function applied to each resampled series, e.g. np.mean, or a Sharpe
        ratio function. Must accept a 1-D array and return a scalar.
    seed
        Random seed. None resolves to utils.random_state.DEFAULT_SEED.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) — the bootstrap distribution of the
        statistic, i.e. [θ̂*_1, θ̂*_2, ..., θ̂*_B].
    """
    series = np.asarray(series)
    n = series.shape[0]

    if block_size >= n:
        raise ValueError(
            f"block_size ({block_size}) must be smaller than len(series) ({n})"
        )
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    rng = get_rng(seed)

    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size

    results = np.empty(n_samples, dtype=float)

    for b in range(n_samples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)

        resample = np.concatenate([series[s:s + block_size] for s in starts])[:n]

        results[b] = statistic_fn(resample)

    return results
