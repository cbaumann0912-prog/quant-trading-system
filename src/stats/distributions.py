"""
Analytic densities and return-path simulation.

Provides normal, lognormal, and Student-t densities plus a tail-mass
comparison. The comparison is the point of the module: the normal
assumption underlying most closed-form risk formulas assigns negligible
probability to moves that FX markets deliver regularly, and quantifying
that gap is a prerequisite for trusting any parametric VaR figure.
"""
import math
import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
from typing import Callable

from src.utils.random_state import get_rng

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Evaluate the normal probability density function.

    Parameters
    ----------
    x
        Point(s) at which to evaluate the PDF.
    mu
        Mean of the distribution.
    sigma
        Standard deviation of the distribution.

    Returns
    -------
    np.ndarray
        PDF value(s) at x.
    """
    return (1 / (sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*(x-mu)/sigma)**2


def normal_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Evaluate the normal cumulative distribution function.

    Parameters
    ----------
    x
        Point(s) at which to evaluate the CDF.
    mu
        Mean of the distribution.
    sigma
        Standard deviation of the distribution.

    Returns
    -------
    np.ndarray
        CDF value(s) at x, in [0, 1].
    """
    z = (x-mu) / (sigma*math.sqrt(2))
    return 0.5*(1+np.vectorize(math.erf)(z))


def lognormal_pdf(X: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Evaluate the log-normal probability density function.

    Parameters
    ----------
    X
        Point(s) at which to evaluate the PDF. Must be strictly positive.
    mu
        Mean of the underlying normal distribution (log-space).
    sigma
        Standard deviation of the underlying normal distribution (log-space).

    Returns
    -------
    np.ndarray
        PDF value(s) at X.

    Raises
    ------
    ValueError
        If any element of X is <= 0.
    """
    x = np.asarray(X, dtype=float)
    if np.any(X <= 0):
        raise ValueError("lognormal_pdf is only defined for x>0")
    return (1/(X*sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((np.log(x)-mu)/sigma)**2)


def simulate_log_returns(mu: float, sigma: float, n: int, seed: int | None = None) -> np.ndarray:
    """
    Simulate a series of normally distributed log returns.

    Parameters
    ----------
    mu
        Mean log return per period.
    sigma
        Standard deviation of log returns per period.
    n
        Number of return observations to generate.
    seed
        Random seed. None resolves to utils.random_state.DEFAULT_SEED.

    Returns
    -------
    np.ndarray
        Array of simulated log returns, shape (n,).
    """
    rng = get_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n)


def simulate_price_path(S0: float, mu: float, sigma: float, n: int, seed: int | None = None) -> np.ndarray:
    """
    Simulate a geometric Brownian motion price path.

    Parameters
    ----------
    S0
        Initial asset price.
    mu
        Mean log return per period.
    sigma
        Standard deviation of log returns per period.
    n
        Total number of price points (including S0).
    seed
        Random seed. None resolves to utils.random_state.DEFAULT_SEED.

    Returns
    -------
    np.ndarray
        Simulated price path of length n, starting at S0.
    """
    log_returns = simulate_log_returns(mu, sigma, n - 1, seed)
    prices = np.empty(n)
    prices[0] = S0
    prices[1:] = S0 * np.exp(np.cumsum(log_returns))
    return prices


def student_t_pdf(x: np.ndarray, df: float, mu: float, sigma: float) -> np.ndarray:
    """
    Evaluate the location-scale Student-t probability density function.

    Parameters
    ----------
    x
        Point(s) at which to evaluate the PDF.
    df
        Degrees of freedom. Must be > 0.
    mu
        Location parameter (mean when df > 1).
    sigma
        Scale parameter (proportional to std when df > 2).

    Returns
    -------
    np.ndarray
        PDF value(s) at x.
    """
    t = (x-mu)/sigma
    fsubt = (gamma((df+1)/2)/((np.sqrt(df*np.pi))*gamma(df/2)))*((1+(t**2)/df)**(-(df+1)/2))
    return (1/sigma)*fsubt


def tail_mass_comparison(dist1: Callable, dist2: Callable, threshold: float) -> dict:
    """
    Compare the right tail mass of two distributions above a threshold.

    Parameters
    ----------
    dist1
        Callable PDF of the first distribution, accepting a scalar x.
    dist2
        Callable PDF of the second distribution, accepting a scalar x.
    threshold
        Lower bound of the tail region to integrate over.

    Returns
    -------
    dict with keys: dist1_tail_mass, dist2_tail_mass, difference
    """
    result1 = quad(dist1, threshold, np.inf)[0]
    result2 = quad(dist2, threshold, np.inf)[0]
    result = {"dist1_tail_mass": result1, "dist2_tail_mass": result2, "difference": result2 - result1}
    return result
