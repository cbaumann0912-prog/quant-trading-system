from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_SCALE = 100.0


def _neg_log_likelihood(params: np.ndarray, eps: np.ndarray) -> float:
    """Negative Gaussian log-likelihood of a GARCH(1,1) path, given demeaned
    shocks `eps`. Returns a large finite penalty instead of raising/inf for
    infeasible or numerically degenerate parameter draws, so the optimizer
    can step away from them instead of crashing.
    """
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10

    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = eps.var()
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]

    if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
        return 1e10

    log_lik = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
    if not np.isfinite(log_lik):
        return 1e10
    
    return -log_lik


def fit_garch(returns: pd.Series) -> dict:
    """Fit a GARCH(1,1) model to a return series via maximum likelihood.

    Model
    -----
    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    Parameters
    ----------
    returns : pd.Series
        Return series (e.g. daily log returns). NaNs are dropped before
        fitting. Internally rescaled by 100x -- gradient-based optimizers
        are numerically unstable when the objective's curvature spans many
        orders of magnitude, which happens when omega is ~1e-7 and alpha/beta
        are O(1) -- and every output below is converted back to the
        original scale.

    Returns
    -------
    dict with keys:
        omega           : float     -- baseline variance level, original scale
        alpha           : float     -- ARCH coefficient, reaction to shocks
        beta            : float     -- GARCH coefficient, variance persistence
        persistence     : float     -- alpha + beta; decay speed of shocks.
                                       Close to 1 => shocks decay slowly.
        long_run_vol    : float     -- sqrt(omega / (1 - persistence)),
                                       the unconditional volatility the
                                       process reverts to. NaN if
                                       persistence >= 1 (non-stationary
                                       variance, no finite long-run level).
        conditional_vol : pd.Series -- fitted sigma_t path, same index as
                                       the (NaN-dropped) input, original
                                       scale.
    """
    clean = returns.dropna()
    values = clean.to_numpy()
    eps = (values - values.mean()) * _SCALE

    sample_var = eps.var()
    x0 = np.array([0.05 * sample_var, 0.05, 0.90])
    bounds = [(1e-10, None), (1e-8, 1 - 1e-8), (1e-8, 1 - 1e-8)]

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(eps,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000},
    )

    omega_scaled, alpha, beta = result.x
    alpha = float(alpha)
    beta = float(beta)
    persistence = alpha + beta

    omega = float(omega_scaled / _SCALE**2)
    long_run_vol = (
        float(np.sqrt(omega / (1 - persistence))) if persistence < 1 else float("nan")
    )

    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = eps.var()
    for t in range(1, n):
        sigma2[t] = omega_scaled + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]

    conditional_vol = pd.Series(
        np.sqrt(sigma2) / _SCALE,
        index=clean.index,
        name="conditional_vol",
    )

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": float(persistence),
        "long_run_vol": long_run_vol,
        "conditional_vol": conditional_vol,
    }
