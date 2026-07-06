import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Dict
import scipy.stats
import scipy.optimize
from statsmodels.stats.diagnostic import acorr_ljungbox


def fit_ols(X: NDArray[np.float64], y: NDArray[np.float64], add_intercept: bool = True,) -> dict:
    """
    Fit OLS regression via the normal equations.

    Parameters
    ----------
    X
        Design matrix of shape (n, p). An intercept column is prepended if add_intercept=True.
    y
        Target vector of shape (n,).
    add_intercept
        If True, prepend a column of ones to X before fitting.

    Returns
    -------
    dict with keys: coefficients, residuals, r_squared, std_errors
    """
    if add_intercept:
        X = np.column_stack([np.ones(len(y)), X])

    AtA = X.T @ X
    Atb = X.T @ y

    beta = np.linalg.solve(AtA, Atb)

    y_hat = X @ beta
    residuals = y - y_hat

    R_squared = r_squared(y, y_hat)

    n, p = X.shape
    RSS = np.sum(residuals**2)
    sigma_squared = RSS / (n - p)

    var_beta = sigma_squared * np.linalg.inv(AtA)
    std_errors = np.sqrt(np.diag(var_beta))

    return {
    'coefficients': beta,
    'residuals':    residuals,
    'r_squared':    R_squared,
    'std_errors':   std_errors,
}


def r_squared(y: NDArray[np.float64],y_hat: NDArray[np.float64]) -> float:
    """
    Compute the coefficient of determination R².

    Parameters
    ----------
    y
        Observed response values.
    y_hat
        Fitted values from the model.

    Returns
    -------
    float
        R² in [0, 1] for well-specified OLS models.
    """
    RSS = np.sum((y-y_hat)**2)
    TSS = np.sum((y-np.mean(y))**2)
    r2 = 1 - (RSS / TSS)
    return r2


def adj_r_squared(y: NDArray[np.float64], y_hat: NDArray[np.float64], p: int) -> float:
    """
    Compute adjusted R² with degrees-of-freedom penalty.

    Parameters
    ----------
    y
        Observed response values.
    y_hat
        Fitted values from the model.
    p
        Number of predictors, excluding the intercept.

    Returns
    -------
    float
        Adjusted R².
    """

    r2 = r_squared(y, y_hat)
    n = y.shape[0]

    if n-p-1 <= 0:
        raise ValueError("n-p-1 must be > 0")
    else:
        adj_r_squared = 1 - ((1 - r2) * ((n-1) / (n-p-1)))

    return adj_r_squared


def residual_diagnostics(y: NDArray[np.float64], y_hat: NDArray[np.float64], lags: int = 20) -> Dict[str, float]:
    """Run a standard residual diagnostic suite on OLS residuals.

    Parameters
    ----------
    y
        Observed response values.
    y_hat
        Fitted values from the model.
    lags
        Number of lags for the Ljung-Box test. Default 20.

    Returns
    -------
    dict with keys:
        'mean', 'variance', 'excess_kurtosis',
        'lb_stat', 'lb_pvalue', 'lag1_autocorr'
    """
    residuals = y - y_hat

    mean = np.mean(residuals)
    variance = np.var(residuals,ddof=1)
    excess_kurtosis = scipy.stats.kurtosis(residuals)
    lb_result = acorr_ljungbox(residuals, lags=[lags])
    lb_stat = float(lb_result['lb_stat'].iloc[0])
    lb_pvalue = float(lb_result['lb_pvalue'].iloc[0])
    lag1_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0,1]

    return {
    'mean': mean,
    'variance': variance,
    'excess_kurtosis': excess_kurtosis,
    'lb_stat': lb_stat,
    'lb_pvalue': lb_pvalue,
    'lag1_autocorr': lag1_autocorr,
}


def ridge_fit(X: np.ndarray, y: np.ndarray, lambda_: float) -> dict:
    """
    Fit Ridge regression using the closed-form solution.

    Parameters
    ----------
    X
        Design matrix (standardized, no intercept column)
    y
        Target vector
    lambda_
        Regularization strength. lambda_=0 recovers OLS.

    Returns
    -------
    dict with keys: coefficients, intercept, lambda_
    """
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_c = X - X_mean
    y_c = y - y_mean

    AtA = X_c.T @ X_c
    Atb = X_c.T @ y_c
    lambda_I = lambda_ * np.identity(X_c.shape[1])

    beta = np.linalg.solve((AtA + lambda_I), Atb)

    intercept = y_mean - X_mean @ beta

    return {
    'coefficients': beta,
    'intercept': intercept,
    'lambda_': lambda_
}


def lasso_objective(beta: np.ndarray, X_c: np.ndarray, y_c: np.ndarray, lambda_: float) -> float:
    """
    Evaluate the Lasso objective (RSS + L1 penalty) at a given coefficient vector.

    Parameters
    ----------
    beta
        Coefficient vector of shape (p,).
    X_c
        Mean-centered design matrix of shape (n, p).
    y_c
        Mean-centered target vector of shape (n,).
    lambda_
        Regularization strength. Scales the L1 penalty term.

    Returns
    -------
    float
        Scalar objective value: RSS + lambda_ * sum(|beta|).
    """
    residuals = y_c - X_c @ beta
    rss = residuals @ residuals
    l1_penalty = lambda_ * np.sum(np.abs(beta))
    return rss + l1_penalty


def lasso_fit(X: np.ndarray, y: np.ndarray, lambda_: float) -> dict:
    """
    Fit Lasso regression using scipy.optimize.minimize with L1 penalty.

    Parameters
    ----------
    X
        Design matrix (standardized, no intercept column)
    y
        Target vector
    lambda_
        Regularization strength. lambda_=0 recovers OLS.

    Returns
    -------
    dict with keys: coefficients, intercept, lambda_, n_nonzero
    """
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_c = X - X_mean
    y_c = y - y_mean

    x0 = np.zeros(X_c.shape[1])

    result = scipy.optimize.minimize(
        lasso_objective,
        x0,
        args=(X_c, y_c, lambda_),
        method='L-BFGS-B'
    )

    beta = result.x

    intercept = y_mean - X_mean @ beta

    beta[np.abs(beta) < 1e-6] = 0.0

    return {
        'coefficients': beta,
        'intercept': intercept,
        'lambda_': lambda_,
        'n_nonzero': int(np.sum(beta != 0))
    }


def interaction_regression(
    y: pd.Series,
    x1: pd.Series,
    x2: pd.Series,
) -> dict:
    """
    Fit y = b0 + b1*x1 + b2*x2 + b3*(x1*x2) + epsilon via OLS, using the
    full matrix formulation so standard errors reflect the joint
    covariance structure of all four estimated coefficients rather than
    treating each term as if it were estimated in isolation.
    
    Parameters
    ----------
    y : pd.Series
        Response variable (e.g. forward return).
    x1 : pd.Series
        First predictor (e.g. PC2 signal level).
    x2 : pd.Series
        Second predictor (e.g. rolling volatility regime variable).

    Returns
    -------
    dict with keys:
        'coefficients' : dict {'intercept', 'x1', 'x2', 'interaction'}
        'std_errors' : dict, same keys as coefficients
        't_stats' : dict, same keys
        'p_values' : dict, same keys
        'r_squared' : float
        'adj_r_squared' : float
        'n_obs' : int
        'condition_number' : float -- condition number of X'X; flag
            values above ~1e10 as a sign of near-singular design matrix
            (severe collinearity between x1, x2, and x1*x2), which
            inflates all standard errors and can make coefficients
            numerically unstable even when the fit itself looks fine.
    """
    y_aligned, x1_aligned = y.align(x1, join="inner")
    y_aligned, x2_aligned = y_aligned.align(x2, join="inner")
    x1_aligned = x1_aligned.reindex(y_aligned.index)
    x2_aligned = x2_aligned.reindex(y_aligned.index)

    valid = (
        y_aligned.notna()
        & x1_aligned.notna()
        & x2_aligned.notna()
    )
    y_vals = y_aligned[valid].to_numpy()
    x1_vals = x1_aligned[valid].to_numpy()
    x2_vals = x2_aligned[valid].to_numpy()

    n = y_vals.shape[0]
    k = 4

    interaction_vals = x1_vals * x2_vals

    design_matrix = np.column_stack([
        np.ones(n),
        x1_vals,
        x2_vals,
        interaction_vals,
    ])

    xtx = design_matrix.T @ design_matrix
    condition_number = np.linalg.cond(xtx)

    xtx_inv = np.linalg.pinv(xtx)
    beta_hat = xtx_inv @ design_matrix.T @ y_vals

    fitted = design_matrix @ beta_hat
    residuals = y_vals - fitted

    ssr = np.sum(residuals ** 2)
    y_mean = y_vals.mean()
    sst = np.sum((y_vals - y_mean) ** 2)

    df = n - k

    if df <= 0:
        raise ValueError(
            f"Insufficient observations: n={n}, need more than k={k} "
            f"parameters to estimate degrees of freedom."
        )

    if sst < 1e-10:
        r_squared_val = np.nan
        adj_r_squared_val = np.nan
    else:
        r_squared_val = 1 - ssr / sst
        adj_r_squared_val = 1 - (1 - r_squared_val) * (n - 1) / df

    sigma_sq_hat = ssr / df
    cov_beta = sigma_sq_hat * xtx_inv

    se = np.sqrt(np.diag(cov_beta))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = np.where(se > 1e-10, beta_hat / se, np.nan)

    p_vals = np.array([
        2 * (1 - scipy.stats.t.cdf(np.abs(t), df)) if not np.isnan(t) else np.nan
        for t in t_vals
    ])

    labels = ["intercept", "x1", "x2", "interaction"]

    return {
        "coefficients": dict(zip(labels, beta_hat)),
        "std_errors": dict(zip(labels, se)),
        "t_stats": dict(zip(labels, t_vals)),
        "p_values": dict(zip(labels, p_vals)),
        "r_squared": r_squared_val,
        "adj_r_squared": adj_r_squared_val,
        "n_obs": n,
        "condition_number": condition_number,
    }