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
        Series of observed returns.
    null_mean : float
        Hypothesised mean under H0 (e.g. 0.0).
    confidence : float
        Confidence level, e.g. 0.95 for 95%.

    Returns
    -------
    dict with keys:
        t_stat          : float  - the t-test statistic
        p_value         : float  - two-sided p-value
        reject_null     : bool   - True if H0 is rejected at given confidence
        confidence_interval : tuple[float, float] - (lower, upper) CI for the mean

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