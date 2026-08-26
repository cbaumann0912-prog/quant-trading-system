"""
Factor decompositions of return panels: CAPM and PCA-based.

CAPM here is the single-index expected-return identity, not a test of the
model. `pca_factor_decomposition` is descriptive: principal components of
an FX return panel have no guaranteed economic meaning, and the usual
reading of PC1 as "dollar level" and PC2 as "carry" is an interpretation
imposed after the fact, not a result the decomposition establishes.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.features.pca import eigendecomposition

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def capm_expected_return(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    rf_rate: float = 0.0
) -> dict:
    """
    Estimate CAPM alpha and beta via OLS regression of excess asset returns
    on excess market returns, and report regression diagnostics.

    Model: (R_i - R_f) = alpha + beta * (R_m - R_f) + epsilon

    Parameters
    ----------
    asset_returns : pd.Series
        Period returns for the asset (e.g. EUR/USD).
    market_returns : pd.Series
        Period returns for the market proxy (e.g. DXY).
    rf_rate : float, default 0.0
        Risk-free rate for the same period frequency as the returns.

    Returns
    -------
    dict with keys:
        'alpha' : float
        'beta' : float
        'r_squared' : float
        'alpha_t_stat' : float
        'alpha_p_value' : float
    """
    asset_returns, market_returns = asset_returns.align(market_returns, join="inner")

    excess_asset = asset_returns.to_numpy() - rf_rate
    excess_market = market_returns.to_numpy() - rf_rate

    n = excess_asset.shape[0]

    market_mean = excess_market.mean()
    asset_mean = excess_asset.mean()

    market_dev = excess_market - market_mean
    asset_dev = excess_asset - asset_mean

    sxx = np.sum(market_dev ** 2)
    sxy = np.sum(market_dev * asset_dev)

    market_var = sxx / (n - 1)

    if market_var < 1e-10:
        beta = np.nan
    else:
        beta = sxy / sxx

    alpha = asset_mean - beta * market_mean

    fitted = alpha + beta * excess_market
    residuals = excess_asset - fitted

    ssr = np.sum(residuals ** 2)
    sst = np.sum(asset_dev ** 2)

    if sst < 1e-10:
        r_squared = 0.0
    else:
        r_squared = 1 - ssr / sst

    df = n - 2

    if df <= 0 or sxx < 1e-10:
        alpha_se = np.nan
        alpha_t_stat = np.nan
        alpha_p_value = np.nan
    else:
        residual_var = ssr / df
        alpha_se_sq = residual_var * (1.0 / n + (market_mean ** 2) / sxx)

        if alpha_se_sq < 1e-10:
            alpha_se = np.nan
            alpha_t_stat = np.nan
            alpha_p_value = np.nan
        else:
            alpha_se = np.sqrt(alpha_se_sq)
            alpha_t_stat = alpha / alpha_se
            alpha_p_value = 2 * (1 - stats.t.cdf(np.abs(alpha_t_stat), df))

    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "alpha_t_stat": alpha_t_stat,
        "alpha_p_value": alpha_p_value,
    }


def pca_factor_decomposition(
    returns: pd.DataFrame,
    n_factors: int
) -> dict:
    """
    Extract PCA-based statistical factors from a returns covariance matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Raw (not pre-centered) log returns, columns = asset/pair names,
        index = datetime. Mean-centering is performed internally.
    n_factors : int
        Number of principal components to retain (<= number of columns).

    Returns
    -------
    dict with keys:
        'loadings' : pd.DataFrame, shape (n_assets, n_factors)
            Eigenvector weight of each asset on each retained factor.
            Sign convention: largest-magnitude loading per component is
            forced positive for run-to-run reproducibility.
        'factor_returns' : pd.DataFrame, shape (n_obs, n_factors)
            Time series of each retained principal component, R_centered @ V_k.
        'explained_variance' : pd.Series, length n_factors
            Proportion of total covariance-matrix trace explained by each
            retained component, sorted descending.
        'residual_returns' : pd.DataFrame, shape (n_obs, n_assets)
            Mean-centered returns minus the reconstruction from the
            retained factors — variance not captured by the top n_factors
            components.

    Raises
    ------
    ValueError
        If n_factors exceeds the number of asset columns in returns.
    """

    if n_factors > returns.shape[1]:
        raise ValueError("n_factors cannot exceed number of assets")

    mean_centered = returns - returns.mean(axis=0)

    cov_matrix = mean_centered.cov()

    eigenvalues, eigenvectors = eigendecomposition(cov_matrix.values)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    for i in range(eigenvectors.shape[1]):
        max_idx = np.argmax(np.abs(eigenvectors[:, i]))
        if eigenvectors[max_idx, i] < 0:
            eigenvectors[:, i] *= -1

    V_k = eigenvectors[:, :n_factors]
    lambda_k = eigenvalues[:n_factors]

    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = lambda_k / total_variance

    factor_returns = mean_centered.values @ V_k

    reconstructed = factor_returns @ V_k.T

    residual_returns = mean_centered.values - reconstructed

    loadings_df = pd.DataFrame(
        V_k,
        index=returns.columns,
        columns=[f"PC{i+1}" for i in range(n_factors)]
    )

    factor_returns_df = pd.DataFrame(
        factor_returns,
        index=returns.index,
        columns=[f"PC{i+1}" for i in range(n_factors)]
    )

    residual_returns_df = pd.DataFrame(
        residual_returns,
        index=returns.index,
        columns=returns.columns
    )

    explained_variance_series = pd.Series(
        explained_variance_ratio,
        index=[f"PC{i+1}" for i in range(n_factors)]
    )

    return {
        "loadings": loadings_df,
        "factor_returns": factor_returns_df,
        "explained_variance": explained_variance_series,
        "residual_returns": residual_returns_df
    }
