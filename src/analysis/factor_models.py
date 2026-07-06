import numpy as np
import pandas as pd
from scipy import stats


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