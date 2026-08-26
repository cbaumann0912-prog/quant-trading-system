"""
Correlation matrices, rolling correlation, and regime-shift detection.

Pearson correlation measures linear association only, and is not robust to
the outliers that dominate FX return tails. A rolling correlation that
moves may reflect a genuine change in dependence or merely a single large
observation entering and leaving the window; the two are not
distinguishable from the correlation series alone.
"""
import pandas as pd
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix from a DataFrame of returns.

    Parameters
    ----------
    returns
        Each column is a return series for one instrument. Rows are observations (days).

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
    s1
        First return series.
    s2
        Second return series.
    window
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


def detect_correlation_regime_shifts(
    s1: pd.Series,
    s2: pd.Series,
    window: int = 60,
    threshold: float = 2.0,
    k: float = 0.5,
    stride: int | None = None,
) -> pd.Series:
    """
    Flag points where the rolling correlation between s1 and s2 has
    undergone a statistically significant, sustained shift from its
    baseline regime, using a Fisher z-transformed two-sided CUSUM
    evaluated on non-overlapping windows.

    Parameters
    ----------
    s1, s2 : pd.Series
        Two return series with identical index and length.
    window : int, default 60
        Rolling window. Must be > 3 (Fisher z variance is 1/(window-3)).
    threshold : float, default 2.0
        CUSUM decision boundary h, in standardized units.
    k : float, default 0.5
        CUSUM allowance/reference value, in standardized units.
    stride : int or None, default None
        Spacing, in bars, between correlation windows fed to the CUSUM.
        Defaults to `window`.

    Returns
    -------
    pd.Series
        Boolean flags, same length and index as s1. True only at the
        (non-overlapping) evaluation points where a regime shift is
        detected; False everywhere else.

    Raises
    ------
    ValueError
        If window <= 3 (Fisher z variance undefined), or if
        `rolling_correlation` rejects s1/s2.
    """
    if window <= 3:
        raise ValueError(
            "window must be > 3 for the Fisher z variance "
            "1/(window-3) to be defined."
        )
    stride = stride or window
    n_burn = 5

    rolling_corr = rolling_correlation(s1, s2, window=window)

    r_clipped = rolling_corr.clip(-0.999999, 0.999999)
    z = np.arctanh(r_clipped)

    sigma = 1.0 / np.sqrt(window - 3)

    flags = pd.Series(False, index=s1.index)

    valid_idx = z.dropna().index
    if len(valid_idx) == 0:
        return flags

    thinned_idx = valid_idx[::stride]

    pos = 0
    n_thinned = len(thinned_idx)
    while pos < n_thinned:
        n_b = min(n_burn, n_thinned - pos)
        if n_b < n_burn:
            break

        burn_idx = thinned_idx[pos:pos + n_b]
        mu0 = z.loc[burn_idx].mean()
        pos += n_b

        c_pos = 0.0
        c_neg = 0.0
        while pos < n_thinned:
            t = thinned_idx[pos]
            diff = (z.loc[t] - mu0) / sigma
            c_pos = max(0.0, c_pos + diff - k)
            c_neg = max(0.0, c_neg - diff - k)
            pos += 1

            if c_pos > threshold or c_neg > threshold:
                flags.loc[t] = True
                break

    return flags
