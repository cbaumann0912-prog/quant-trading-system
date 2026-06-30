import numpy as np
import pandas as pd

from src.analysis.portfolio_stats import compute_covariance_matrix, compute_portfolio_return, compute_portfolio_variance
from scipy.optimize import minimize

def markowitz_sharpe(
    portfolio_return: float,
    portfolio_variance: float,
    ann_factor: float = 252.0,
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
    ann_factor : float, default=252.0
        Number of observations per year.
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
    allow_short: bool = True,
    ann_factor: float = 252.0,
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
    allow_short : bool, default=True
        Whether short selling is permitted.
    ann_factor : float, default=252.0
        Number of observations per year used to annualize the Sharpe
        ratio.
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
    if not allow_short:
        raise NotImplementedError(
            "Constrained Markowitz optimization requires inequality "
            "constraints and must be solved numerically via scipy SLSQP"
        )

    p_bar = returns.mean().to_numpy()
    sigma = compute_covariance_matrix(returns)

    n_assets = returns.shape[1]
    ones = np.ones(n_assets)

    sigma_inv = np.linalg.inv(sigma)

    A = p_bar @ sigma_inv @ p_bar
    B = p_bar @ sigma_inv @ ones
    C = ones @ sigma_inv @ ones

    kkt = 0.5 * np.array([
        [A, -B],
        [B, -C],
    ])
    rhs = np.array([target_return, 1.0])
    lam, nu = np.linalg.solve(kkt, rhs)
    x = 0.5 * sigma_inv @ (lam * p_bar - nu * ones)

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
        result = markowitz_weights(returns, target_return=r, allow_short=True)
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


def risk_parity_weights(returns: pd.DataFrame, ann_factor: float = 252.0) -> dict:
    """
    Compute equal risk contribution (ERC) portfolio weights via numerical optimization.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return time series. Rows = observations, columns = assets.
    ann_factor : float, default=252.0
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

    result = minimize(
    _risk_parity_objective,
    x0=np.ones(n) / n,
    args=(sigma,),
    method="SLSQP",
    bounds=[(-1.0, 1.0)] * n,
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
    options={"ftol": 1e-12, "maxiter": 1000},
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