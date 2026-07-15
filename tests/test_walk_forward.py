import numpy as np
import pandas as pd
import pytest

from src.framework.walk_forward import WalkForwardValidator


def _make_data(n_years: int = 10, seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range("2016-01-01", periods=252 * n_years, freq="B")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"price": 100 + rng.normal(0, 1, size=len(idx)).cumsum()}, index=idx)


def _dummy_signal_fn(data: pd.DataFrame, lookback: int) -> pd.Series:
    # Placeholder matching the SignalBuilder-style contract; unused by this
    # skeleton's run(), just needs to exist for the constructor.
    price = data["price"]
    return np.sign(price / price.shift(lookback) - 1)


def test_window_count_matches_n_windows():
    data = _make_data(n_years=10)
    n_windows = 5
    validator = WalkForwardValidator(
        signal_fn=_dummy_signal_fn,
        data=data,
        n_windows=n_windows,
        train_years=3,
        test_months=12,
        embargo_days=5,
    )

    windows = validator.generate_windows()
    assert len(windows) == n_windows

    results = validator.run()
    assert len(results["window_results"]) == n_windows
    assert results["aggregate_stats"]["n_windows"] == n_windows


def test_no_overlap_between_train_and_test():
    data = _make_data(n_years=10)
    validator = WalkForwardValidator(
        signal_fn=_dummy_signal_fn,
        data=data,
        n_windows=5,
        train_years=3,
        test_months=12,
        embargo_days=5,
    )

    for w in validator.generate_windows():
        train_df = validator._slice(w["train_start"], w["train_end"])
        test_df = validator._slice(w["test_start"], w["test_end"])

        assert len(train_df.index.intersection(test_df.index)) == 0
        assert w["test_start"] > w["train_end"]
        assert train_df.index.max() < test_df.index.min()


def test_embargo_gap_correct():
    data = _make_data(n_years=10)
    embargo_days = 7
    validator = WalkForwardValidator(
        signal_fn=_dummy_signal_fn,
        data=data,
        n_windows=5,
        train_years=3,
        test_months=12,
        embargo_days=embargo_days,
    )

    for w in validator.generate_windows():
        assert w["embargo_end"] == w["train_end"] + pd.Timedelta(days=embargo_days)
        assert w["test_start"] == w["embargo_end"]

        embargo_mask = (data.index >= w["train_end"]) & (data.index < w["embargo_end"])
        embargoed_dates = data.index[embargo_mask]

        train_df = validator._slice(w["train_start"], w["train_end"])
        test_df = validator._slice(w["test_start"], w["test_end"])

        assert len(train_df.index.intersection(embargoed_dates)) == 0
        assert len(test_df.index.intersection(embargoed_dates)) == 0


def test_rolling_not_expanding():
    data = _make_data(n_years=10)
    validator = WalkForwardValidator(
        signal_fn=_dummy_signal_fn,
        data=data,
        n_windows=3,
        train_years=3,
        test_months=12,
        embargo_days=5,
    )

    windows = validator.generate_windows()
    assert windows[1]["train_start"] > windows[0]["train_start"]

    len0 = windows[0]["train_end"] - windows[0]["train_start"]
    len1 = windows[1]["train_end"] - windows[1]["train_start"]
    # DateOffset(years=3) can differ by 1 day across a leap day depending on
    # which calendar years the window spans -- assert near-equality, not
    # exact equality.
    assert abs((len0 - len1).days) <= 1


def test_insufficient_data_raises():
    data = _make_data(n_years=4)
    validator = WalkForwardValidator(
        signal_fn=_dummy_signal_fn,
        data=data,
        n_windows=5,
        train_years=3,
        test_months=12,
        embargo_days=5,
    )

    with pytest.raises(ValueError):
        validator.generate_windows()


def test_invalid_constructor_args_raise():
    data = _make_data(n_years=10)

    with pytest.raises(ValueError):
        WalkForwardValidator(_dummy_signal_fn, data, n_windows=0, train_years=3, test_months=12)

    with pytest.raises(ValueError):
        WalkForwardValidator(_dummy_signal_fn, data, n_windows=5, train_years=0, test_months=12)

    with pytest.raises(ValueError):
        WalkForwardValidator(_dummy_signal_fn, data, n_windows=5, train_years=3, test_months=0)

    with pytest.raises(ValueError):
        WalkForwardValidator(
            _dummy_signal_fn, data, n_windows=5, train_years=3, test_months=12, embargo_days=-1
        )
