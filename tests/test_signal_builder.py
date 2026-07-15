import numpy as np
import pandas as pd
import pytest

from src.signals.signal_builder import SignalBuilder


def _make_data(n: int = 200, start: str = "2024-01-01", freq: str = "1D", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=n, freq=freq)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"price": prices}, index=index)


def _momentum_signal_fn(data: pd.DataFrame, lookback: int) -> pd.Series:
    price = data["price"]
    return np.sign(price / price.shift(lookback) - 1)


def _lookahead_signal_fn(data: pd.DataFrame, lookback: int) -> pd.Series:
    price = data["price"]
    full_sample_mean = price.mean()
    return price.shift(-1) - full_sample_mean


def _random_signal_fn(data: pd.DataFrame, lookback: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, len(data)), index=data.index)


class TestCompute:

    def test_compute_returns_series(self):
        data = _make_data()
        builder = SignalBuilder(
            signal_fn=_momentum_signal_fn,
            data=data,
            price_col="price",
            lookback=10,
            holding_period=5,
        )
        signal = builder.compute(data)

        assert isinstance(signal, pd.Series)
        assert signal.index.isin(data.index).all()
        assert len(signal) == len(data)

    def test_compute_rejects_non_series_return(self):
        data = _make_data()

        def bad_signal_fn(d, lookback):
            return d["price"].to_numpy()

        builder = SignalBuilder(signal_fn=bad_signal_fn, data=data, lookback=10, holding_period=5)
        with pytest.raises(TypeError, match="pd.Series"):
            builder.compute(data)

    def test_invalid_price_col_raises(self):
        data = _make_data()
        with pytest.raises(ValueError, match="price_col"):
            SignalBuilder(
                signal_fn=_momentum_signal_fn,
                data=data,
                price_col="close",
                lookback=10,
                holding_period=5,
            )

    def test_non_datetime_index_raises(self):
        data = _make_data().reset_index(drop=True)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            SignalBuilder(signal_fn=_momentum_signal_fn, data=data, lookback=10, holding_period=5)


class TestComputeIC:

    def test_ic_random_signal_near_zero(self):
        data = _make_data(n=500, seed=1)

        def signal_fn(d, lookback):
            return _random_signal_fn(d, lookback, seed=42)

        builder = SignalBuilder(
            signal_fn=signal_fn,
            data=data,
            price_col="price",
            lookback=10,
            holding_period=5,
        )
        forward_returns = builder.compute_forward_returns()
        ic = builder.compute_ic(forward_returns)

        assert not np.isnan(ic)
        assert abs(ic) < 0.15

    def test_ic_bounded(self):
        data = _make_data(n=300, seed=3)
        builder = SignalBuilder(
            signal_fn=_momentum_signal_fn,
            data=data,
            price_col="price",
            lookback=10,
            holding_period=5,
        )
        forward_returns = builder.compute_forward_returns()
        ic = builder.compute_ic(forward_returns)

        assert -1.0 <= ic <= 1.0


class TestComputeRollingIC:

    def test_rolling_ic_length(self):
        data = _make_data(n=200, seed=2)
        builder = SignalBuilder(
            signal_fn=_momentum_signal_fn,
            data=data,
            price_col="price",
            lookback=10,
            holding_period=5,
        )
        forward_returns = builder.compute_forward_returns()
        window = 20

        signal = builder.compute(data)
        aligned = pd.concat(
            [signal.rename("signal"), forward_returns.rename("fwd")], axis=1, join="inner"
        ).dropna()
        max_possible_windows = len(aligned) // window

        rolling_ic = builder.compute_rolling_ic(forward_returns, window=window)

        # compute_rolling_ic skips windows where signal or forward_returns is
        # constant (undefined correlation) -- see Day 44 pipeline test finding
        # (34% of 60-bar windows were constant-signal for a 78-day-lookback
        # momentum signal). So len(result) is an upper bound, not exact, on a
        # random-walk fixture -- but should still be > 0 for a reasonable window.
        assert 0 < len(rolling_ic) <= max_possible_windows

    def test_rolling_ic_window_too_small_raises(self):
        data = _make_data()
        builder = SignalBuilder(signal_fn=_momentum_signal_fn, data=data, lookback=10, holding_period=5)
        forward_returns = builder.compute_forward_returns()
        with pytest.raises(ValueError, match="window"):
            builder.compute_rolling_ic(forward_returns, window=1)

    def test_rolling_ic_skips_constant_signal_windows(self):
        # First window's signal is forced constant (all +1, since price rises
        # monotonically over the whole lookback+window span); the second
        # window uses genuinely varying prices so its signal isn't constant.
        n = 60
        lookback = 5
        window = 20
        index = pd.date_range("2024-01-01", periods=lookback + n, freq="1D")

        rng = np.random.default_rng(7)
        rising = 100 + np.arange(lookback + window)  # strictly increasing -> signal always +1
        varying = 100 + np.cumsum(rng.normal(0, 1, n - window))
        prices = np.concatenate([rising, varying + (rising[-1] - varying[0])])
        data = pd.DataFrame({"price": prices}, index=index)

        builder = SignalBuilder(
            signal_fn=_momentum_signal_fn,
            data=data,
            price_col="price",
            lookback=lookback,
            holding_period=5,
        )
        forward_returns = builder.compute_forward_returns()
        rolling_ic = builder.compute_rolling_ic(forward_returns, window=window)

        first_window_start = data.index[lookback]
        assert first_window_start not in rolling_ic.index
        assert not rolling_ic.isna().any()


class TestValidateNoLookahead:

    def test_causal_signal_passes(self):
        data = _make_data(n=200, seed=4)
        builder = SignalBuilder(
            signal_fn=_momentum_signal_fn,
            data=data,
            price_col="price",
            lookback=10,
            holding_period=5,
        )
        cutoff = data.index[100]

        assert builder.validate_no_lookahead(cutoff) is True

    def test_lookahead_signal_fails(self):
        data = _make_data(n=200, seed=5)
        builder = SignalBuilder(
            signal_fn=_lookahead_signal_fn,
            data=data,
            price_col="price",
            lookback=10,
            holding_period=5,
        )
        cutoff = data.index[100]

        assert builder.validate_no_lookahead(cutoff) is False
