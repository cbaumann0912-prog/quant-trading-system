import scipy.stats as stats
import math


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
    int
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
    float
    """
    z_half_alpha = stats.norm.ppf(1-(alpha/2))

    power = stats.norm.cdf(((effect_size * math.sqrt(n)) - z_half_alpha))
    
    return power