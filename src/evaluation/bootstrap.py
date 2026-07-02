import numpy as np
from typing import Callable

def bootstrap_confidence_interval(data: np.ndarray, statistic_fn: Callable, n_bootstrap: int, confidence: float) -> tuple[float, float]:
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

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) of the confidence interval.
    """

    bootstrap_stats = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
       sample = np.random.choice(data, len(data), True)
       bootstrap_stats[b] = statistic_fn(sample)

    lower = np.percentile(bootstrap_stats, ((1 - confidence) / 2)*100)
    upper = np.percentile(bootstrap_stats, (1 - ((1 - confidence) / 2))*100)

    result = (lower,upper)

    return result


def block_bootstrap(
    series: np.ndarray,
    block_size: int,
    n_samples: int,
    statistic_fn: Callable,
    seed: int = 42,
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
        Random seed for reproducibility

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

    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size  

    results = np.empty(n_samples, dtype=float)

    for b in range(n_samples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)

        resample = np.concatenate([series[s:s + block_size] for s in starts])[:n]

        results[b] = statistic_fn(resample)

    return results