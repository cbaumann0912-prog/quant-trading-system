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