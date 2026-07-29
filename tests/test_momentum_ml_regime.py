import numpy as np
import pandas as pd

from src.signals.momentum_ml_regime import (
    momentum_signal_outcome,
    non_overlapping_subsample,
)


def _prices(values, start="2020-01-01"):
    index = pd.date_range(start=start, periods=len(values), freq="1D")
    return pd.Series(values, index=index)


def test_row_count_is_n_minus_one_minus_lookback_minus_holding():
    prices = _prices(np.linspace(100.0, 200.0, 60))
    out = momentum_signal_outcome(prices, lookback=26, holding=5)
    assert len(out) == 60 - 1 - 26 - 5


def test_trailing_and_forward_windows_never_overlap():
    prices = _prices(np.linspace(100.0, 140.0, 80))
    out = momentum_signal_outcome(prices, lookback=10, holding=3)

    assert (out["forward_start"] > out["trailing_end"]).all()
    assert (out["trailing_start"] < out["trailing_end"]).all()
    assert (out["forward_start"] <= out["forward_end"]).all()


def test_forward_window_starts_the_day_after_the_trailing_window_ends():
    prices = _prices(np.linspace(100.0, 140.0, 40))
    out = momentum_signal_outcome(prices, lookback=5, holding=2)
    gap = out["forward_start"] - out["trailing_end"]
    assert (gap == pd.Timedelta(days=1)).all()


def test_returns_are_log_differences_over_the_stated_windows():
    prices = _prices(np.exp(np.arange(30) * 0.01) * 100.0)
    out = momentum_signal_outcome(prices, lookback=4, holding=2)

    # constant 1% log drift per step
    assert np.allclose(out["trailing_return"].to_numpy(), 0.04)
    assert np.allclose(out["forward_return"].to_numpy(), 0.02)


def test_flat_prices_give_zero_on_both_legs():
    out = momentum_signal_outcome(_prices(np.full(30, 100.0)), lookback=4, holding=2)
    assert np.allclose(out["trailing_return"].to_numpy(), 0.0)
    assert np.allclose(out["forward_return"].to_numpy(), 0.0)


def test_subsample_leaves_no_shared_forward_windows():
    prices = _prices(np.linspace(100.0, 200.0, 200))
    holding = 5
    out = momentum_signal_outcome(prices, lookback=10, holding=holding)
    sub = non_overlapping_subsample(out, holding=holding)

    starts = sub["forward_start"].to_numpy()
    ends = sub["forward_end"].to_numpy()
    assert (starts[1:] > ends[:-1]).all()


def test_subsample_keeps_every_holding_th_row():
    prices = _prices(np.linspace(100.0, 200.0, 100))
    out = momentum_signal_outcome(prices, lookback=5, holding=4)
    sub = non_overlapping_subsample(out, holding=4)
    assert len(sub) == int(np.ceil(len(out) / 4))
    pd.testing.assert_series_equal(sub["date"].reset_index(drop=True),
                                   out["date"].iloc[::4].reset_index(drop=True))


def test_consecutive_rows_overlap_by_lookback_minus_one():
    prices = _prices(np.linspace(100.0, 200.0, 60))
    lookback = 10
    out = momentum_signal_outcome(prices, lookback=lookback, holding=3)

    first = pd.date_range(out["trailing_start"].iloc[0], out["trailing_end"].iloc[0], freq="1D")
    second = pd.date_range(out["trailing_start"].iloc[1], out["trailing_end"].iloc[1], freq="1D")
    shared = first.intersection(second)
    assert len(shared) == lookback
