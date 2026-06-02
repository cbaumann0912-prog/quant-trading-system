import numpy as np
import pandas as pd
from scipy import stats


def t_test_mean(
    returns: pd.Series,
    null_mean: float,
    confidence: float,
) -> dict:
    """
    Test whether the mean of a return series is significantly different
    from a null value using a one-sample t-test.

    Parameters
    ----------
    returns : pd.Series
    null_mean : float
    confidence : float

    Returns
    -------
    dict with keys:
        t_stat          : float 
        p_value         : float
        reject_null     : bool
        confidence_interval : tuple[float, float]

    Raises
    ------
    ValueError
        If n < 2 (cannot compute sample std with fewer than 2 observations).
    """

    if len(returns) < 2:
        raise ValueError("cannot compute std with fewer than 2 observations")

    n = len(returns)
    x_bar = returns.mean()
    s = returns.std()
    se = (s/np.sqrt(n))

    W = (x_bar-null_mean) / se

    df = n-1

    p_value = 2 * stats.t.sf(np.abs(W), df)

    alpha = 1 - confidence
    reject_null = p_value < alpha

    t_crit = stats.t.ppf(1 - alpha/2, df)
    CI = (x_bar - t_crit * se, x_bar + t_crit * se)

    return {"t_stat": W, "p_value": p_value, "reject_null": reject_null, "confidence_interval": CI }


def p_value_interpretation(p: float, alpha: float) -> str:
    """Returns a precise statistical interpretation of a p-value given a significance level."""
    
    if p < alpha:
        return (
        f"Given the significance level {alpha} and that the null hypothesis is true, "
        f"there is a {p} probability that the observed value or more extrem to occur." 
        f"The observed value is statistically significant and we reject the null"
        )
    else:
        return (
        f"Given the significance level {alpha} and that the null hypothesis is true, "
        f"there is a {p} probability that the observed value or more extrem to occur. We fail to "
        f"reject the null" 
        )
   

def compute_effect_size_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Returns Cohen's d effect size measuring the standardized difference between two groups."""
    
    mean1 = group1.mean()
    mean2 = group2.mean()

    n1 = len(group1)
    n2 = len(group2)
    var1 = group1.var()
    var2 = group2.var()

    s_pooled = np.sqrt((((n1-1)*var1)+((n2-1)*var2)) / (n1 + n2 - 2))

    d = (mean1 - mean2) / s_pooled

    return d
    

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
    