import numpy as np

def bootstrap_confidence_interval(data, statistic_fn, n_bootstrap, confidence) -> tuple[float, float]:
    """Returns a tuple whose first entry represents the lower bound of the confidence interval and the second is the upper bound
    
    Parameters
    ----------
    data: np.ndarray
    statistic_fn: function
    n_bootstrap: int
    confidence: float
    """

    bootstrap_stats = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
       sample = np.random.choice(data, len(data), True)
       bootstrap_stats[b] = statistic_fn(sample)

    lower = np.percentile(bootstrap_stats, ((1 - confidence) / 2)*100)
    upper = np.percentile(bootstrap_stats, (1 - ((1 - confidence) / 2))*100)

    result = (lower,upper)

    return result