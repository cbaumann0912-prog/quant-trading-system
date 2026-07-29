from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MA_WINDOW = 100
DEFAULT_VOL_WINDOW = 100
DEFAULT_ENTRY_THRESHOLD = 1.0
DEFAULT_REVERSION_X = 1.0
DEFAULT_CAP_MULTIPLIER = 3


def zscore_deviation(
    daily_prices: pd.Series,
    ma_window: int = DEFAULT_MA_WINDOW,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> pd.Series:
    """Deviation of price from its rolling mean, scaled by rolling deviation vol.

        z_t = (price_t - MA_t) / std(price - MA)_t

    Both windows are trailing, so the series is causal.

    Note the construction largely guarantees its own stationarity: differencing
    a price against its own trailing mean will tend to produce a mean-reverting
    series regardless of market behaviour. An ADF rejection on ``z`` is
    therefore weak evidence about the market and strong evidence about the
    transform. See ``research/strategies/s03_ou_halflife_mean_reversion/``.

    Parameters
    ----------
    daily_prices : pd.Series
        Daily closes indexed by date.
    ma_window, vol_window : int
        Trailing window lengths in observations.

    Returns
    -------
    pd.Series
        Z-scored deviation, ``ma_window + vol_window - 2`` observations shorter
        than the input.
    """
    ma = daily_prices.rolling(window=ma_window).mean()
    deviation = (daily_prices - ma).dropna()
    rolling_vol = deviation.rolling(window=vol_window).std()
    return (deviation / rolling_vol).dropna()


def half_life_from_theta(theta: float) -> float:
    """Convert an OU mean-reversion rate to a half-life in observations."""
    return float(np.log(2) / theta)


def extract_excursions(
    z_score: pd.Series,
    censoring_cap: float,
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
    reversion_x: float = DEFAULT_REVERSION_X,
) -> pd.DataFrame:
    """Identify threshold excursions and time how long each takes to revert.

    An excursion opens when ``|z|`` first reaches ``entry_threshold``, runs
    while ``z`` keeps its sign, and is characterised by its signed peak
    magnitude. Reversion is timed from the peak and counted when ``z`` either
    crosses zero or falls ``reversion_x`` below the peak.

    Excursions that have not reverted within ``censoring_cap`` observations are
    recorded at the cap and flagged, rather than dropped or left unbounded.

    Parameters
    ----------
    z_score : pd.Series
        Output of :func:`zscore_deviation`.
    censoring_cap : float
        Maximum reversion time to scan for, conventionally
        ``DEFAULT_CAP_MULTIPLIER`` times the fitted half-life.
    entry_threshold : float
        ``|z|`` level that opens an excursion.
    reversion_x : float
        Retracement from the peak, in z units, that counts as reverted.

    Returns
    -------
    pd.DataFrame
        Columns ``peak``, ``reversion_time``, ``censored``.

    Notes
    -----
    Censoring interacts with peak magnitude: large-peak excursions are likelier
    to hit the cap, which biases their mean reversion time downward. Any
    comparison of reversion time across peak-magnitude pools inherits that bias
    and should not be read as a market result without controlling for it.
    """
    z = z_score.values
    n = len(z)

    excursions = []
    i = 0
    while i < n:
        if abs(z[i]) >= entry_threshold:
            sign = 1 if z[i] > 0 else -1
            running_peak = z[i] * sign
            peak_idx = i

            j = i + 1
            while j < n and np.sign(z[j]) == sign:
                mag = z[j] * sign
                if mag > running_peak:
                    running_peak = mag
                    peak_idx = j
                j += 1
            excursion_end_idx = j - 1

            target = running_peak - reversion_x
            reversion_time = None
            scan_end = min(peak_idx + 1 + int(np.ceil(censoring_cap)), n)
            for k in range(peak_idx + 1, scan_end):
                if np.sign(z[k]) != sign:
                    reversion_time = k - peak_idx
                    break
                mag_k = z[k] * sign
                if mag_k <= target:
                    reversion_time = k - peak_idx
                    break

            if reversion_time is not None and reversion_time <= censoring_cap:
                excursions.append({
                    "peak": running_peak,
                    "reversion_time": reversion_time,
                    "censored": False,
                })
            else:
                excursions.append({
                    "peak": running_peak,
                    "reversion_time": censoring_cap,
                    "censored": True,
                })

            i = excursion_end_idx + 1
        else:
            i += 1

    return pd.DataFrame(excursions)


def split_pools(excursions: pd.DataFrame, pool_split: float) -> tuple[np.ndarray, np.ndarray]:
    """Split excursion reversion times into small- and large-peak pools."""
    large = excursions[excursions["peak"] >= pool_split]["reversion_time"].values
    small = excursions[excursions["peak"] < pool_split]["reversion_time"].values
    return small, large
