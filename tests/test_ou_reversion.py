import numpy as np
import pandas as pd

from src.signals.ou_reversion import (
    extract_excursions,
    half_life_from_theta,
    split_pools,
    zscore_deviation,
)


def _series(values, start="2020-01-01"):
    index = pd.date_range(start=start, periods=len(values), freq="1D")
    return pd.Series(np.asarray(values, dtype=float), index=index)


def test_zscore_drops_the_warmup_of_both_windows():
    prices = _series(np.linspace(100.0, 130.0, 300))
    z = zscore_deviation(prices, ma_window=20, vol_window=10)
    # each rolling window costs its own length minus one
    assert len(z) == 300 - (20 - 1) - (10 - 1)


def test_zscore_uses_only_trailing_information():
    base = np.concatenate([np.full(60, 100.0), np.linspace(100.0, 150.0, 60)])
    early = zscore_deviation(_series(base), ma_window=10, vol_window=5)

    extended = np.concatenate([base, np.full(40, 999.0)])
    late = zscore_deviation(_series(extended), ma_window=10, vol_window=5)

    shared = early.index.intersection(late.index)
    assert np.allclose(early.loc[shared].to_numpy(), late.loc[shared].to_numpy())


def test_half_life_inverts_theta():
    assert np.isclose(half_life_from_theta(np.log(2)), 1.0)
    assert np.isclose(half_life_from_theta(0.02), np.log(2) / 0.02)


def test_excursion_records_peak_and_reversion_time():
    # rises to a peak of 3.0 at index 3, then retraces past 3.0 - 1.0 at index 5
    z = _series([0.0, 1.2, 2.0, 3.0, 2.5, 1.5, 0.4])
    ex = extract_excursions(z, censoring_cap=10.0, entry_threshold=1.0, reversion_x=1.0)

    assert len(ex) == 1
    assert np.isclose(ex["peak"].iloc[0], 3.0)
    assert ex["reversion_time"].iloc[0] == 2
    assert not bool(ex["censored"].iloc[0])


def test_sign_change_counts_as_reversion():
    z = _series([0.0, 1.5, 2.0, -0.1])
    ex = extract_excursions(z, censoring_cap=10.0, entry_threshold=1.0, reversion_x=5.0)
    assert len(ex) == 1
    assert ex["reversion_time"].iloc[0] == 1


def test_unreverted_excursion_is_censored_at_the_cap():
    z = _series([0.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    ex = extract_excursions(z, censoring_cap=2.0, entry_threshold=1.0, reversion_x=1.0)
    assert len(ex) == 1
    assert bool(ex["censored"].iloc[0])
    assert np.isclose(ex["reversion_time"].iloc[0], 2.0)


def test_negative_excursions_are_detected_with_positive_peak_magnitude():
    z = _series([0.0, -1.5, -2.5, -1.2, -0.2])
    ex = extract_excursions(z, censoring_cap=10.0, entry_threshold=1.0, reversion_x=1.0)
    assert len(ex) == 1
    assert ex["peak"].iloc[0] > 0
    assert np.isclose(ex["peak"].iloc[0], 2.5)


def test_below_threshold_path_produces_no_excursions():
    z = _series([0.1, -0.3, 0.5, -0.9, 0.2])
    ex = extract_excursions(z, censoring_cap=10.0, entry_threshold=1.0)
    assert ex.empty


def test_separate_sign_runs_are_separate_excursions():
    z = _series([0.0, 2.0, 0.5, -0.2, -2.0, -0.5, 0.1])
    ex = extract_excursions(z, censoring_cap=10.0, entry_threshold=1.0, reversion_x=1.0)
    assert len(ex) == 2


def test_split_pools_partitions_on_peak_magnitude():
    ex = pd.DataFrame({
        "peak": [1.0, 1.4, 1.5, 2.2],
        "reversion_time": [3.0, 4.0, 5.0, 6.0],
        "censored": [False] * 4,
    })
    small, large = split_pools(ex, pool_split=1.5)
    assert sorted(small) == [3.0, 4.0]
    assert sorted(large) == [5.0, 6.0]
    assert len(small) + len(large) == len(ex)
