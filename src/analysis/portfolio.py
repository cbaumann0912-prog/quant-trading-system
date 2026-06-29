import numpy as np
import pandas as pd

from src.analysis.portfolio_stats import compute_covariance_matrix, compute_portfolio_return, compute_portfolio_variance


def markowitz_sharpe(
    portfolio_return: float,
    portfolio_variance: float,
    ann_factor: float = 252.0,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute the annualized Sharpe ratio of a Markowitz portfolio.

    portfolio_return and portfolio_variance are assumed to be
    per-observation-period quantities, consistent with the sampling
    frequency implied by ``ann_factor``.

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
    allow_short: bool = False,
    ann_factor: float = 252.0,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Solve the minimum-variance Markowitz portfolio for a target return.

    Solves the equality-constrained quadratic program

        min xᵀΣx
        s.t. μᵀx = target_return
             1ᵀx = 1

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series, one column per asset.
    target_return : float
        Required expected portfolio return per observation period.
    allow_short : bool, default=False
        Whether short selling is permitted.

        If True, solves the unconstrained Markowitz problem using the
        closed-form KKT solution, which may produce negative weights.

        If False, raises NotImplementedError because long-only
        optimization requires inequality constraints (x >= 0) and a
        numerical optimizer (e.g. scipy.optimize.SLSQP).
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
            "Long-only Markowitz optimization requires inequality "
            "constraints (x >= 0) and must be solved numerically."
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