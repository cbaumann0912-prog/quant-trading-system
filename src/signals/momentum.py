"""
Cross-sectional and time-series momentum signal construction.
"""
import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def momentum_signal(data: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Time-series momentum signal: signal_t = sign(P_t / P_{t-lookback} - 1).

    Matches the turbulent-regime rule in
    `research/strategies/volatility_regime_breakout_mean_revert.md` (Section 4,
    Moskowitz, Ooi & Pedersen 2012 convention), re-evaluated every bar with no
    confirmation lag.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'price' column, indexed by datetime ascending.
    lookback : int
        Bars back to compare against. The strategy spec uses 78 trading days.

    Returns
    -------
    pd.Series
        Values in {-1.0, 0.0, 1.0} (0.0 when price is exactly unchanged over
        the window), NaN for the first `lookback` bars (insufficient history).
    """
    price = data["price"]
    return np.sign(price / price.shift(lookback) - 1)
