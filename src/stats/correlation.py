import pandas as pd
import numpy as np


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix from a DataFrame of returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Each column is a return series for one instrument.
        Rows are observations (days).

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix with ones on the diagonal.
        Shape: (n_assets, n_assets), indexed by column names.

    Raises
    ------
    ValueError
        If returns is empty or contains fewer than 2 columns.
    """

    if returns.empty:
        raise ValueError("returns must not be empty")
    if returns.shape[1] < 2:
        raise ValueError("there must be at least 2 columns in returns")
    
    return returns.corr()
    

def rolling_correlation(
    s1: pd.Series,
    s2: pd.Series,
    window: int,
) -> pd.Series:
    """
    Compute rolling Pearson correlation between two return series.

    Parameters
    ----------
    s1 : pd.Series
        First return series.
    s2 : pd.Series
        Second return series.
    window : int
        Rolling window size in periods.

    Returns
    -------
    pd.Series
        Rolling correlation values aligned to input index.
        First (window - 1) entries will be NaN.

    Raises
    ------
    ValueError
        If window < 2, series lengths differ, or window > len(s1).
    """
    if window < 2:
        raise ValueError("Window must be at least 2.")
    if len(s1) != len(s2):
        raise ValueError("s1 and s2 must be same length")
    if window > len(s1):
        raise ValueError("window must not be larger than len(s1)")
    return s1.rolling(window).corr(s2)
