import numpy as np
import pandas as pd

from src.stats.regression import fit_ols
from src.data.stationarity import adf_test

def engle_granger_test(y: pd.Series, x: pd.Series) -> dict:
    """
    Test for cointegration between two I(1) price series using the
    Engle-Granger two-step procedure.

    The null hypothesis is no cointegration (residuals are I(1)).
    Rejection of the null implies the spread is stationary and
    mean-reverting.

    Parameters
    ----------
    y : pd.Series
        Dependent price level series (e.g. EUR/USD). Must be I(1).
    x : pd.Series
        Independent price level series (e.g. GBP/USD). Must be I(1).

    Returns
    -------
    dict with keys:
        alpha          : float  — OLS intercept
        hedge_ratio    : float  — OLS slope coefficient (beta)
        residuals      : pd.Series — spread series (y - alpha - beta*x)
        adf_stat       : float  — ADF test statistic on residuals
        adf_p          : float  — ADF p-value on residuals
        is_cointegrated: bool   — True if adf_p < 0.05
    """
    A = np.column_stack([np.ones(len(x)), x.values])
    A = x.values.reshape(-1, 1)
    b = y.values
    
    ols = fit_ols(A,b)
    coefficients = ols['coefficients']
    alpha = coefficients[0]
    hedge_ratio = coefficients[1]

    residuals = y - alpha - hedge_ratio * x

    result = adf_test(residuals)
    adf_stat = result['adf_stat']
    adf_p = result['adf_p']

    is_cointegrated = adf_p < 0.05

    return {
        'alpha': alpha,
        'hedge_ratio': hedge_ratio,
        'residuals': residuals,
        'adf_stat': adf_stat,
        'adf_p': adf_p,
        'is_cointegrated': is_cointegrated
} 


def cointegration_spread(
    y: pd.Series,
    x: pd.Series,
    alpha: float,
    hedge_ratio: float,
) -> pd.Series:
    """
    Compute the cointegration spread given an estimated cointegrating vector.

    spread_t = y_t - alpha - hedge_ratio * x_t

    Parameters
    ----------
    y : pd.Series
        Dependent price level series.
    x : pd.Series
        Independent price level series.
    alpha : float
        Intercept from engle_granger_test.
    hedge_ratio : float
        Slope (beta) from engle_granger_test.

    Returns
    -------
    pd.Series
        Spread series with same index as y and x.
    """
    return y - alpha - hedge_ratio * x