import numpy as np
import pandas as pd


def price_zscore_signal(data: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Price-level mean-reversion predictor: z_price_t = (P_t - mean_lookback(P))
    / std_lookback(P).

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'price' column, indexed by datetime ascending.
    lookback : int
        Rolling window for the mean/std estimate. The strategy spec uses 26
        trading days (`trading_days_per_year // 12`, one month at 312 FX
        trading days/year) -- deliberately shorter than the 78-day momentum
        lookback, since mean-reversion is expected to operate on a faster
        timescale than the momentum leg.

    Returns
    -------
    pd.Series
        Same index as `data`. NaN for the first `lookback - 1` bars
        (insufficient history for the rolling mean/std) and wherever the
        rolling std is exactly zero (undefined z-score, e.g. a flat price
        segment) rather than raising or silently producing +/-inf.
    """
    price = data["price"]
    rolling_mean = price.rolling(lookback).mean()
    rolling_std = price.rolling(lookback).std()

    z = (price - rolling_mean) / rolling_std
    return z.mask(rolling_std == 0)


def _ladder_step(
    z_i: float,
    rungs: int,
    direction: int,
    entry_idx: int | None,
    i: int,
    entry_thresholds: tuple[float, float, float],
    exit_z: float,
    time_stop: int,
) -> tuple[float, int, int, int | None]:
    """
    Single-bar state transition for the 3-rung mean-reversion ladder. Shared
    by `mean_reversion_ladder_signal` (ungated) and `regime_gated.py`'s
    combined signal_fn (calm-regime leg), so the ladder logic exists in
    exactly one place.

    Returns
    -------
    (exposure_i, rungs, direction, entry_idx)
    """
    if rungs == 0:
        if pd.isna(z_i):
            return float("nan"), rungs, direction, entry_idx

        if abs(z_i) > entry_thresholds[0]:
            direction = -1 if z_i > 0 else 1
            rungs = 1
            entry_idx = i
            return direction * rungs / 3, rungs, direction, entry_idx

        return 0.0, 0, 0, None

    if pd.isna(z_i):
        return direction * rungs / 3, rungs, direction, entry_idx

    if abs(z_i) < exit_z or (i - entry_idx) >= time_stop:
        return 0.0, 0, 0, None

    if rungs < 3 and abs(z_i) > entry_thresholds[rungs]:
        rungs += 1

    return direction * rungs / 3, rungs, direction, entry_idx


def mean_reversion_ladder_signal(
    data: pd.DataFrame,
    lookback: int,
    entry_thresholds: tuple[float, float, float] = (2.0, 2.5, 3.0),
    exit_z: float = 0.5,
    time_stop: int = 26,
) -> pd.Series:
    """
    Full stateful 3-rung mean-reversion ladder (ungated -- always active,
    regardless of regime).
    
    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'price' column, indexed by datetime ascending.
    lookback : int
        Rolling window passed to `price_zscore_signal`.
    entry_thresholds : tuple[float, float, float], default (2.0, 2.5, 3.0)
        Absolute z-score levels for rungs 1, 2, and 3 respectively.
    exit_z : float, default 0.5
        Absolute z-score level inside which an open position exits.
    time_stop : int, default 26
        Maximum bars an open position is held before a forced exit,
        regardless of z-score.

    Returns
    -------
    pd.Series
        Exposure in {-1, -2/3, -1/3, 0, 1/3, 2/3, 1}, same index as `data`.
        NaN wherever `price_zscore_signal` is NaN and no position is open
        (warmup); a position already open is held through a missing price
        rather than reset.
    """
    z = price_zscore_signal(data, lookback)
    n = len(data)
    exposure = np.full(n, np.nan)

    rungs, direction, entry_idx = 0, 0, None

    for i in range(n):
        exposure[i], rungs, direction, entry_idx = _ladder_step(
            z.iloc[i], rungs, direction, entry_idx, i, entry_thresholds, exit_z, time_stop
        )

    return pd.Series(exposure, index=data.index)
