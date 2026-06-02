import numpy as np
import pandas as pd
import math
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
        f"there is a {p} probability that the observed value or more extreme to occur." 
        f"The observed value is statistically significant and we reject the null"
        )
    else:
        return (
        f"Given the significance level {alpha} and that the null hypothesis is true, "
        f"there is a {p} probability that the observed value or more extreme to occur. We fail to "
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
    
def compute_required_sample_size(effect_size: float, alpha: float, power: float) -> int:
    """
    Compute the minimum sample size needed to achieve a given power level.
    Parameters
    ----------
    effect_size : float
    alpha : float
    power : float

    Returns
    -------
    n_req : int
    """

    z_half_alpha = stats.norm.ppf(1-(alpha/2))
    z_beta = stats.norm.ppf(power)
    
    n = ((z_half_alpha + z_beta) / effect_size)**2

    return math.ceil(n)

def compute_achieved_power(n: int, effect_size: float, alpha: float) -> float:
    """
    Compute the power achieved by a test given sample size and effect size.

    Parameters
    ----------
    n : int
    effect_size : float
    alpha : float

    Returns
    -------
    power : float
    """
    z_half_alpha = stats.norm.ppf(1-(alpha/2))

    power = stats.norm.cdf(((effect_size * math.sqrt(n)) - z_half_alpha))
    
    return power