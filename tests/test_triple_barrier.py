import numpy as np
import pandas as pd
import pytest

from src.signals.triple_barrier import triple_barrier_labels


def _make_price_series(prices: list[float], start: str = "2024-01-01", freq: str = "1min") -> pd.Series:
    index = pd.date_range(start=start, periods=len(prices), freq=freq)
    return pd.Series(prices, index=index)


def test_upper_barrier_returns_positive_one():
    np.random.seed(28)
    prices = _make_price_series(
        list(np.random.normal(0, 1, 25) + 100) + [100.00, 100.80, 101.5, 102.10, 101.90, 101.70]
    )
    entry_time = prices.index[25]
    events = pd.DatetimeIndex([entry_time])

    labels = triple_barrier_labels(
        prices=prices,
        events=events,
        pt_sl=(2, 2),
        max_holding=5,
        vol_lookback=20,
    )

    assert labels[entry_time] == 1


def test_lower_barrier_returns_negative_one():
    np.random.seed(28)
    prices = _make_price_series(
        list(np.random.normal(0, 1, 25) + 100) + [100.00, 99.60, 99.10, 98.40, 98.90, 99.20]
    )
    entry_time = prices.index[25]
    events = pd.DatetimeIndex([entry_time])

    labels = triple_barrier_labels(
        prices=prices,
        events=events,
        pt_sl=(2, 2),
        max_holding=5,
        vol_lookback=20,
    )

    assert labels[entry_time] == -1


def test_time_expiry_returns_sign_of_return():
    np.random.seed(28)
    prices = _make_price_series(
        list(np.random.normal(0, 1, 25) + 75) + [75.00, 75.20, 74.80, 75.60, 74.90, 75.10]
    )
    entry_time = prices.index[25]
    events = pd.DatetimeIndex([entry_time])

    labels = triple_barrier_labels(
        prices=prices,
        events=events,
        pt_sl=(2, 2),
        max_holding=5,
        vol_lookback=20,
    )

    entry_log_price = np.log(75.00)
    final_log_price = np.log(75.10)
    expected_label = int(np.sign(final_log_price - entry_log_price))

    assert labels[entry_time] == expected_label
    assert expected_label == 1


def test_truncated_window_returns_nan():
    np.random.seed(28)
    prices = _make_price_series(
        list(np.random.normal(0, 1, 25) + 100) + [100.00, 100.10, 100.05]
    )
    entry_time = prices.index[25]
    events = pd.DatetimeIndex([entry_time])

    labels = triple_barrier_labels(
        prices=prices,
        events=events,
        pt_sl=(2, 2),
        max_holding=5,
        vol_lookback=20,
    )

    assert pd.isna(labels[entry_time])


def test_upper_barrier_touched_mid_window_breaks_early():
    """Unlike test_upper_barrier_returns_positive_one, which actually resolves
    via the vertical time-expiry branch (the window never reaches the upper
    threshold there), this constructs a price path that genuinely crosses the
    upper barrier inside the scan loop. The bar right after the crossing
    (90.0) would flip the label to -1 if the loop didn't break on first touch,
    so a label of +1 here only happens if the early-exit branch actually ran."""
    prices = _make_price_series([100.0, 100.05, 99.95, 100.00, 101.0, 90.0])
    entry_time = prices.index[3]
    events = pd.DatetimeIndex([entry_time])

    labels = triple_barrier_labels(
        prices=prices, events=events, pt_sl=(1, 1), max_holding=2, vol_lookback=3,
    )

    assert labels[entry_time] == 1


def test_event_not_in_price_index_returns_nan():
    prices = _make_price_series([100.0, 100.1, 100.2, 100.3, 100.4, 100.5])
    bad_event = prices.index[2] + pd.Timedelta(seconds=30)

    labels = triple_barrier_labels(
        prices=prices, events=pd.DatetimeIndex([bad_event]),
        pt_sl=(2, 2), max_holding=2, vol_lookback=2,
    )

    assert pd.isna(labels.iloc[0])


def test_insufficient_vol_history_returns_nan():
    prices = _make_price_series([100.0, 100.1, 100.2, 100.3, 100.4, 100.5])
    events = pd.DatetimeIndex([prices.index[1]])

    labels = triple_barrier_labels(
        prices=prices, events=events, pt_sl=(2, 2), max_holding=2, vol_lookback=5,
    )

    assert pd.isna(labels.iloc[0])


def test_event_at_final_bar_returns_nan():
    """end_loc <= loc: the event is the last bar in the series, so there's no
    forward window to evaluate at all. Distinct from a truncated-but-nonempty
    window, which test_truncated_window_returns_nan covers."""
    prices = _make_price_series([100.0, 100.1, 99.9, 100.2, 99.8, 100.0])
    events = pd.DatetimeIndex([prices.index[-1]])

    labels = triple_barrier_labels(
        prices=prices, events=events, pt_sl=(2, 2), max_holding=3, vol_lookback=3,
    )

    assert pd.isna(labels.iloc[0])


def test_horizontal_wins_tie_with_vertical():
    np.random.seed(28)
    prices = _make_price_series(
        list(np.random.normal(0, 1, 25) + 100) + [100.00, 100.80, 101.5, 101.30, 99.10, 98.40, 97]
    )
    entry_time = prices.index[25]
    events = pd.DatetimeIndex([entry_time])

    labels = triple_barrier_labels(
        prices=prices,
        events=events,
        pt_sl=(2, 2),
        max_holding=6,
        vol_lookback=20,
    )

    assert labels[entry_time] == -1