import numpy as np
import pandas as pd

from src.stats.regression import fit_ols
from src.data.stationarity import adf_test
from statsmodels.tsa.vector_ar.vecm import coint_johansen

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


def johansen_test(
    data: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """
    Johansen cointegration test for multiple time series.
    
    Parameters
    ----------
    data : pd.DataFrame
        Columns = price/log-price series (e.g. EUR/USD, GBP/USD, USD/JPY).
        Rows = time-ordered observations.
    det_order : int
        Deterministic trend assumption (0 = no trend, constant in cointegrating eq).
    k_ar_diff : int
        Number of lagged differences in the VECM.

    Returns
    -------
    dict with:
        - trace_stat: array of trace test statistics per rank
        - trace_crit_vals: critical values (90/95/99%)
        - max_eig_stat: array of max eigenvalue statistics per rank
        - max_eig_crit_vals: critical values
        - eigenvalues: eigenvalues of the rank-reduction problem
        - eigenvectors: cointegrating vectors
        - rank_trace: inferred rank from trace test
        - rank_max_eig: inferred rank from max eigenvalue test
    """
    if data.isnull().any().any():
        raise ValueError("johansen_test: input data contains NaNs — drop or fill before testing")

    if data.shape[1] < 2:
        raise ValueError("johansen_test: requires at least 2 series")

    series_names = list(data.columns)
    result = coint_johansen(data.values, det_order, k_ar_diff)

    trace_stat = result.lr1
    trace_crit_vals = result.cvt
    max_eig_stat = result.lr2
    max_eig_crit_vals = result.cvm

    rank_trace = _infer_rank(trace_stat, trace_crit_vals, conf_col=1)
    rank_max_eig = _infer_rank(max_eig_stat, max_eig_crit_vals, conf_col=1)

    return {
        "trace_stat": trace_stat,
        "trace_crit_vals": trace_crit_vals,
        "max_eig_stat": max_eig_stat,
        "max_eig_crit_vals": max_eig_crit_vals,
        "eigenvalues": result.eig,
        "eigenvectors": result.evec,
        "rank_trace": rank_trace,
        "rank_max_eig": rank_max_eig,
        "series_names": series_names,
    }


def _infer_rank(stat: np.ndarray, crit_vals: np.ndarray, conf_col: int = 1) -> int:
    """
    Walk rank hypotheses and stop at the first r where the
    test statistic fails to reject. That r is the inferred cointegration rank.
    """
    for r in range(len(stat)):
        if stat[r] < crit_vals[r, conf_col]:
            return r
    return len(stat)