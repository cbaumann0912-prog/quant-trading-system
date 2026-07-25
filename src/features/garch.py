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


def _kmeans_1d(x: np.ndarray, k: int, n_init: int = 10, max_iter: int = 100) -> np.ndarray:
    """Lloyd's algorithm k-means, specialized to 1-D data.

    Parameters
    ----------
    x : np.ndarray
        1-D array of observations.
    k : int
        Number of clusters.
    n_init : int, default 10
        Number of random restarts.
    max_iter : int, default 100
        Max iterations per restart.

    Returns
    -------
    np.ndarray
        Integer cluster labels (0..k-1), same length as `x`.
    """
    rng = np.random.default_rng(55)
    n = len(x)
    best_labels = None
    best_inertia = np.inf

    for _ in range(n_init):
        init_idx = rng.choice(n, size=k, replace=False)
        centroids = x[init_idx].copy()

        for _ in range(max_iter):
            dist = np.abs(x[:, None] - centroids[None, :])
            labels = np.argmin(dist, axis=1)

            new_centroids = centroids.copy()
            for c in range(k):
                members = x[labels == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean()

            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        inertia = np.sum((x - centroids[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels

    return best_labels


def classify_vol_regime(conditional_vol: pd.Series, n_regimes: int = 2) -> pd.Series:
    """Classify each day's conditional volatility into a discrete regime via
    1-D k-means clustering.

    Parameters
    ----------
    conditional_vol : pd.Series
        Conditional volatility path, e.g. `fit_garch(returns)["conditional_vol"]`.
        NaNs are dropped before clustering.
    n_regimes : int, default 2
        Number of clusters. If 2, clusters are relabeled "low"/"high" by
        ascending centroid value. If not 2, clusters are relabeled
        "regime_0" (lowest centroid) through "regime_{n_regimes-1}" (highest).

    Returns
    -------
    pd.Series
        String-labeled regime per day, index = `conditional_vol.dropna()`.

    Raises
    ------
    ValueError
        If `n_regimes < 2` or exceeds the number of non-NaN observations.
    """
    clean = conditional_vol.dropna()
    x = clean.to_numpy()

    if n_regimes < 2:
        raise ValueError(f"n_regimes must be >= 2, got {n_regimes}.")
    if n_regimes > len(x):
        raise ValueError(
            f"n_regimes ({n_regimes}) cannot exceed the number of non-NaN "
            f"observations ({len(x)})."
        )

    raw_labels = _kmeans_1d(x, k=n_regimes)
    
    centroid_by_cluster = {c: x[raw_labels == c].mean() for c in np.unique(raw_labels)}
    rank_by_cluster = {
        c: rank for rank, c in enumerate(sorted(centroid_by_cluster, key=centroid_by_cluster.get))
    }

    if n_regimes == 2:
        name_by_rank = {0: "low", 1: "high"}
    else:
        name_by_rank = {r: f"regime_{r}" for r in range(n_regimes)}

    labels = np.array([name_by_rank[rank_by_cluster[c]] for c in raw_labels])

    return pd.Series(labels, index=clean.index, name="vol_regime")
