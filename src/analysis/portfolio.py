import numpy as np
import pandas as pd

from src.analysis.portfolio_stats import compute_covariance_matrix, compute_portfolio_return, compute_portfolio_variance
from scipy.optimize import minimize
from scipy.stats import norm
from src.stats.optimization import constrained_optimize

def markowitz_sharpe(
    portfolio_return: float,
    portfolio_variance: float,
    ann_factor: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute the annualized Sharpe ratio of a Markowitz portfolio.

    Parameters
    ----------
    portfolio_return : float
        Expected portfolio return per observation period.
    portfolio_variance : float
        Portfolio variance per observation period.
    ann_factor : float
        Number of observations per year, computed from the return index of
        the series being annualized.
    risk_free_rate : float, default=0.0
        Annualized risk-free rate

    Returns
    -------
    float
        Annualized Sharpe ratio. Returns NaN if the portfolio variance
        is non-positive.
    """
    if portfolio_variance <= 0:
        return np.nan

    portfolio_vol = np.sqrt(portfolio_variance)
    rf_period = risk_free_rate / ann_factor

    return ((portfolio_return - rf_period) / portfolio_vol * np.sqrt(ann_factor))


def markowitz_weights(
    returns: pd.DataFrame,
    target_return: float,
    ann_factor: float,
    allow_short: bool = True,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Solve the minimum-variance Markowitz portfolio for a target return.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series, one column per asset.
    target_return : float
        Required expected portfolio return per observation period.
    ann_factor : float
        Number of observations per year used to annualize the Sharpe
        ratio.
    allow_short : bool, default=True
        Whether short selling is permitted.
    risk_free_rate : float, default=0.0
        Annualized risk-free rate passed through to
        ``markowitz_sharpe``.

    Returns
    -------
    dict
        Dictionary with keys:
        weights,
        portfolio_return,
        portfolio_variance,
        sharpe.
    """

    p_bar = returns.mean().to_numpy()
    sigma = compute_covariance_matrix(returns)
    n = returns.shape[1]

    SCALE = 1e6
    objective = lambda w: (w @ sigma @ w) * SCALE
    jac = lambda w: 2 * sigma @ w * SCALE

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ p_bar - target_return},
    ]
    bounds = [(-1.0, 1.0)] * n if allow_short else [(0, None)] * n
    options = {"ftol": 1e-9, "maxiter": 2000}

    try:
        result = constrained_optimize(
            objective, x0=np.ones(n) / n, constraints=constraints,
            bounds=bounds, jac=jac, options=options,
        )
    except RuntimeError:
        sigma_inv = np.linalg.inv(sigma)
        ones = np.ones(n)
        A = ones @ sigma_inv @ ones
        B = ones @ sigma_inv @ p_bar
        C = p_bar @ sigma_inv @ p_bar
        kkt_matrix = np.array([[C, B], [B, A]])
        rhs = np.array([2 * target_return, 2.0])
        lam, nu = np.linalg.solve(kkt_matrix, rhs)
        x0_fallback = 0.5 * sigma_inv @ (lam * p_bar + nu * ones)
        result = constrained_optimize(
            objective, x0=x0_fallback, constraints=constraints,
            bounds=bounds, jac=jac, options=options,
        )

    x = result.x

    portfolio_return = compute_portfolio_return(x, p_bar)
    portfolio_variance = compute_portfolio_variance(x, sigma)

    sharpe = markowitz_sharpe(
        portfolio_return=portfolio_return,
        portfolio_variance=portfolio_variance,
        ann_factor=ann_factor,
        risk_free_rate=risk_free_rate,
    )

    assert np.isclose(x.sum(), 1.0, atol=1e-10), (
        f"Weights sum to {x.sum()}, expected 1."
    )

    return {
        "weights": x,
        "portfolio_return": portfolio_return,
        "portfolio_variance": portfolio_variance,
        "sharpe": sharpe,
    }


def _markowitz_weights_closed_form(returns: pd.DataFrame, target_return: float) -> dict:
    """
    Closed-form (KKT, equality-only) solution to Markowitz mean-variance.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series, columns = assets.
    target_return : float
        Required expected portfolio return per observation period.

    Returns
    -------
    dict with keys:
        weights, portfolio_return, portfolio_variance
    """
    p_bar = returns.mean().to_numpy()
    sigma = compute_covariance_matrix(returns)
    n = returns.shape[1]

    ones = np.ones(n)
    sigma_inv = np.linalg.inv(sigma)

    A = ones @ sigma_inv @ ones
    B = ones @ sigma_inv @ p_bar
    C = p_bar @ sigma_inv @ p_bar

    kkt_matrix = np.array([
        [C, B],
        [B, A],
    ])
    rhs = np.array([2 * target_return, 2.0])

    lam, nu = np.linalg.solve(kkt_matrix, rhs)

    x = 0.5 * sigma_inv @ (lam * p_bar + nu * ones)

    assert np.isclose(x.sum(), 1.0, atol=1e-10), (
        f"Weights sum to {x.sum()}, expected 1."
    )

    portfolio_return = compute_portfolio_return(x, p_bar)
    portfolio_variance = compute_portfolio_variance(x, sigma)

    return {
        "weights": x,
        "portfolio_return": portfolio_return,
        "portfolio_variance": portfolio_variance,
    }


def _solve_affine_weight_coefficients(returns: pd.DataFrame) -> tuple:
    """
    Derive x(r) = a + b*r for the unconstrained Markowitz solution.

    Returns
    -------
    a, b : np.ndarray, shape (n_assets,)
        Coefficients such that weights(r) = a + b * r.
    """
    p_bar = returns.mean().to_numpy()
    sigma = compute_covariance_matrix(returns)
    sigma_inv = np.linalg.inv(sigma)
    ones = np.ones(returns.shape[1])

    A = p_bar @ sigma_inv @ p_bar
    B = p_bar @ sigma_inv @ ones
    C = ones @ sigma_inv @ ones
    denom = A * C - B ** 2

    lam1 = 2 * C / denom
    nu1 = 2 * B / denom
    lam0 = -2 * B / denom
    nu0 = -2 * A / denom

    a = 0.5 * sigma_inv @ (lam0 * p_bar - nu0 * ones)
    b = 0.5 * sigma_inv @ (lam1 * p_bar - nu1 * ones)

    return a, b


def leverage_bounded_return_range(
    returns: pd.DataFrame,
    leverage_caps: np.ndarray,
) -> tuple:
    """
    Compute the feasible target-return range [r_min, r_max] such that
    every asset's weight stays within its leverage cap: |x_i(r)| <= L_i.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series, columns = assets.
    leverage_caps : np.ndarray, shape (n_assets,)
        Max gross leverage per asset (e.g. 50.0 for 50:1).

    Returns
    -------
    r_min, r_max : float
        Feasible sweep range satisfying all per-asset leverage caps
        simultaneously (intersection of per-asset bounds).
    """
    a, b = _solve_affine_weight_coefficients(returns)
    leverage_caps = np.asarray(leverage_caps)

    lower_bounds = np.empty(len(a))
    upper_bounds = np.empty(len(a))

    for i in range(len(a)):
        L_i = leverage_caps[i]
        candidate_1 = (-L_i - a[i]) / b[i]
        candidate_2 = (L_i - a[i]) / b[i]
        lower_bounds[i] = min(candidate_1, candidate_2)
        upper_bounds[i] = max(candidate_1, candidate_2)

    r_min = lower_bounds.max()
    r_max = upper_bounds.min()

    if r_min > r_max:
        raise ValueError(
            f"Infeasible leverage caps: r_min={r_min} exceeds r_max={r_max}. "
            "No target return satisfies all per-asset leverage constraints."
        )

    return r_min, r_max


def efficient_frontier(
    returns: pd.DataFrame,
    ann_factor: float,
    n_points: int = 50,
    leverage_caps: np.ndarray = None,
) -> dict:
    """
    Sweep target returns across their feasible range and compute the
    minimum-variance portfolio at each target via markowitz_weights().

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series, columns = assets (EUR/USD, GBP/USD, USD/JPY).
    ann_factor : float
        Number of observations per year passed through to markowitz_weights
        for Sharpe annualization.
    n_points : int
        Number of target-return points to sweep across the frontier.
    leverage_caps : np.ndarray, shape (n_assets,), optional
        If provided, bounds the sweep range to [r_min, r_max] such that
        every asset's weight stays within its leverage cap

    Returns
    -------
    dict with keys:
        'returns'      : np.ndarray, shape (n_points,)
        'volatilities' : np.ndarray, shape (n_points,)
        'sharpes'      : np.ndarray, shape (n_points,)
        'weights'      : np.ndarray, shape (n_points, n_assets)
    """
    if leverage_caps is not None:
        r_min, r_max = leverage_bounded_return_range(returns, leverage_caps)
    else:
        p_bar = returns.mean().to_numpy()
        r_min, r_max = p_bar.min(), p_bar.max()

    target_returns = np.linspace(r_min, r_max, n_points)

    volatilities = np.empty(n_points)
    sharpes = np.empty(n_points)
    weights = np.empty((n_points, returns.shape[1]))

    for i, r in enumerate(target_returns):
        result = markowitz_weights(returns, target_return=r, ann_factor=ann_factor, allow_short=True)
        volatilities[i] = np.sqrt(result["portfolio_variance"])
        sharpes[i] = result["sharpe"]
        weights[i] = result["weights"]

    return {
        "returns": target_returns,
        "volatilities": volatilities,
        "sharpes": sharpes,
        "weights": weights,
    }
 
 
def minimum_variance_portfolio(returns: pd.DataFrame) -> dict:
    """
    Find the global minimum-variance portfolio directly via closed form
 
    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series, columns = assets.
 
    Returns
    -------
    dict with keys:
        'weights'  : np.ndarray, shape (n_assets,)
        'return'   : float  -- r*, the return at the minimum-variance point
        'variance' : float  -- sigma^2 at the minimum-variance point
    """
    sigma = compute_covariance_matrix(returns)
    ones = np.ones(len(sigma))
    sigma_inv = np.linalg.pinv(sigma)

    weights = sigma_inv @ ones
    weights /= ones @ sigma_inv @ ones
    
    portfolio_return = compute_portfolio_return(weights, returns.mean().to_numpy())        
    portfolio_variance = compute_portfolio_variance(weights, sigma) 

    return {
    "weights": weights,
    "return": portfolio_return,
    "variance": portfolio_variance
    }


def _risk_parity_objective(w, sigma):
    vol = np.sqrt(w @ sigma @ w)
    rc = w * (sigma @ w) / vol
    return np.sum((rc[:, None] - rc[None, :]) ** 2)


def risk_parity_weights(returns: pd.DataFrame, ann_factor: float) -> dict:
    """
    Compute equal risk contribution (ERC) portfolio weights via numerical optimization.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return time series. Rows = observations, columns = assets.
    ann_factor : float
        Number of observations per year used to annualize portfolio_vol.

    Returns
    -------
    dict with keys:
        weights : np.ndarray, shape (n,)
            ERC portfolio weights, sums to 1.
        risk_contributions : np.ndarray, shape (n,)
            Per-asset risk contributions, sums to portfolio_vol.
        portfolio_vol : float
            Annualized portfolio volatility at ERC weights.
    """
    sigma = compute_covariance_matrix(returns)
    n = sigma.shape[0]

    SCALE = 1e8

    result = minimize(
        lambda w, s: _risk_parity_objective(w, s) * SCALE,
        x0=np.ones(n) / n,
        args=(sigma,),
        method="SLSQP",
        bounds=[(-1.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"ftol": 1e-9, "maxiter": 2000},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    weights = result.x
    portfolio_vol = np.sqrt(weights @ sigma @ weights) * np.sqrt(ann_factor)
    risk_contributions = weights * (sigma @ weights) / portfolio_vol

    return {
        "weights": weights,
        "risk_contributions": risk_contributions,
        "portfolio_vol": portfolio_vol,
    }


def kelly_fraction(mu: float, sigma: float) -> float:
    """
    Compute the full Kelly fraction for continuous returns.

    Parameters
    ----------
    mu : float
        Expected (mean) return of the strategy, same period basis as sigma.
    sigma : float
        Standard deviation of returns, same period basis as mu.

    Returns
    -------
    float
        Full Kelly fraction f* = mu / sigma^2.

    Raises
    ------
    ValueError
        If sigma <= 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return mu / sigma**2


def fractional_kelly(mu: float, sigma: float, fraction: float = 0.5) -> float:
    """
    Compute a fractional Kelly position size.

    Parameters
    ----------
    mu : float
        Expected (mean) return of the strategy.
    sigma : float
        Standard deviation of returns.
    fraction : float, default 0.5
        Fraction of full Kelly to apply (e.g. 0.5 for half-Kelly, 0.25 for quarter-Kelly).
        Must be in (0, 1].

    Returns
    -------
    float
        fraction * kelly_fraction(mu, sigma).

    Raises
    ------
    ValueError
        If fraction is not in (0, 1].
    """
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return fraction * kelly_fraction(mu, sigma)


def var_historical(returns: pd.Series, confidence: float) -> float:
    """
    Compute historical (empirical simulation) Value-at-Risk.

    Parameters
    ----------
    returns : pd.Series
        Periodic strategy or asset returns (not prices).
    confidence : float
        Confidence level alpha, e.g. 0.95 or 0.99.

    Returns
    -------
    float
        VaR as a loss magnitude, in the same units as ``returns``.

    Raises
    ------
    ValueError
        If confidence is not in (0, 1).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    percentile = (1.0 - confidence) * 100.0
    return float(-np.percentile(returns, percentile))


def var_parametric(returns: pd.Series, confidence: float) -> float:
    """
    Compute parametric (variance-covariance) Value-at-Risk.

    Assumes returns ~ N(mu, sigma^2) and solves for the loss threshold
    such that P(loss > VaR) = 1 - confidence:

        VaR_alpha = sigma * Phi^-1(alpha) - mu

    where Phi^-1 is the inverse standard normal CDF.

    Parameters
    ----------
    returns : pd.Series
        Periodic strategy or asset returns (not prices).
    confidence : float
        Confidence level alpha, e.g. 0.95 or 0.99.

    Returns
    -------
    float
        VaR as a loss magnitude, in the same units as ``returns``.

    Raises
    ------
    ValueError
        If confidence is not in (0, 1).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(confidence)
    return float(z * sigma - mu)


def var_monte_carlo(
    returns: pd.Series,
    confidence: float,
    n_simulations: int = 100_000,
    seed: int = None,
) -> float:
    """
    Compute Monte Carlo Value-at-Risk.

    Fits mu and sigma from the historical sample, simulates
    ``n_simulations`` draws from N(mu, sigma^2), and reads the empirical
    (1 - confidence) percentile off the simulated distribution.

    Parameters
    ----------
    returns : pd.Series
        Periodic strategy or asset returns (not prices), used only to fit
        mu and sigma.
    confidence : float
        Confidence level alpha, e.g. 0.95 or 0.99.
    n_simulations : int, default 100_000
        Number of Monte Carlo draws.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    float
        VaR as a loss magnitude, in the same units as ``returns``.

    Raises
    ------
    ValueError
        If confidence is not in (0, 1), or n_simulations is not positive.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_simulations <= 0:
        raise ValueError(f"n_simulations must be positive, got {n_simulations}")

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(mu, sigma, n_simulations)

    percentile = (1.0 - confidence) * 100.0
    return float(-np.percentile(simulated_returns, percentile))


def cvar(returns: pd.Series, confidence: float) -> float:
    """
    Compute historical Conditional Value-at-Risk (Expected Shortfall).

    The expected loss conditional on breaching historical VaR:

        CVaR_alpha = -E[R | R <= -VaR_hist_alpha]

    Parameters
    ----------
    returns : pd.Series
        Periodic strategy or asset returns (not prices).
    confidence : float
        Confidence level alpha, e.g. 0.95 or 0.99.

    Returns
    -------
    float
        CVaR as a loss magnitude. By construction, cvar(returns,
        confidence) >= var_historical(returns, confidence) for the same
        series and confidence level.

    Raises
    ------
    ValueError
        If confidence is not in (0, 1).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    percentile = (1.0 - confidence) * 100.0
    threshold = np.percentile(returns, percentile)
    tail = returns[returns <= threshold]

    if len(tail) == 0:
        return float(-threshold)
    return float(-np.mean(tail))