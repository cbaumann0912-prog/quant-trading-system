import pandas as pd
import numpy as np


def triple_barrier_labels(
    prices: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: tuple[float, float],
    max_holding: int,
    vol_lookback: int = 20,
) -> pd.Series:
    """
    Label events using the triple barrier method, in log-return space.

    Parameters
    ----------
    prices : pd.Series
        Full price series, indexed by datetime.
    events : pd.DatetimeIndex
        Timestamps at which to evaluate entry + barriers.
    pt_sl : tuple[float, float]
        (upper_mult, lower_mult) — volatility-scaled multiples applied to
        cumulative log return from entry:
            upper_barrier = entry_log_price + pt_sl[0] * vol
            lower_barrier = entry_log_price - pt_sl[1] * vol
        where `vol` is the rolling std of log returns over `vol_lookback`
        bars ending at the entry bar.
    max_holding : int
        Vertical barrier — max number of bars forward from entry.
    vol_lookback : int
        Window (in bars) for the rolling volatility estimate.

    Returns
    -------
    pd.Series
        Index = events, values in {-1, 0, +1} or NaN.
        NaN means "not evaluable" — either insufficient history for vol,
        or the forward window was cut short by the end of the price series
        (not by max_holding) with no barrier touched before the cutoff.
        This is distinct from a true 0, which means vertical expiry at the
        full max_holding horizon with a log return of exactly zero.
    """
    log_prices = np.log(prices)
    log_returns = log_prices.diff()
    rolling_vol = log_returns.rolling(vol_lookback).std()
    price_index = prices.index
    n = len(price_index)

    labels = pd.Series(index=events, dtype=float)

    for event_time in events:
        if event_time not in price_index:
            labels[event_time] = np.nan
            continue

        loc = price_index.get_loc(event_time)

        vol = rolling_vol.iloc[loc]
        if pd.isna(vol) or vol == 0:
            labels[event_time] = np.nan
            continue

        target_end_loc = loc + max_holding
        end_loc = min(target_end_loc, n - 1)
        truncated = end_loc < target_end_loc

        if end_loc <= loc:
            labels[event_time] = np.nan
            continue

        entry_log_price = log_prices.iloc[loc]
        upper_barrier = entry_log_price + pt_sl[0] * vol
        lower_barrier = entry_log_price - pt_sl[1] * vol

        window = log_prices.iloc[loc + 1: end_loc + 1]

        label = None
        for p in window:
            if p >= upper_barrier:
                label = 1
                break
            if p <= lower_barrier:
                label = -1
                break

        if label is None:
            if truncated:
                labels[event_time] = np.nan
                continue
            log_ret = window.iloc[-1] - entry_log_price
            label = int(np.sign(log_ret))

        labels[event_time] = label

    return labels