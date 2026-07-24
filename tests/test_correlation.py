import pandas as pd
import numpy as np
import pytest
from src.stats.correlation import (
    compute_correlation_matrix,
    rolling_correlation,
    detect_correlation_regime_shifts,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:

    return pd.DataFrame({
        "EURUSD": [ 0.42, -0.17,  0.89, -0.53,  0.11,
                   -0.78,  0.34, -0.29,  0.62, -0.21],
        "GBPUSD": [ 0.38, -0.22,  0.75, -0.41,  0.19,
                   -0.65,  0.28, -0.18,  0.51, -0.15],
        "USDJPY": [-0.31,  0.18, -0.64,  0.29, -0.08,
                    0.57, -0.22,  0.14, -0.48,  0.11],
    })


def _correlated_series(n, rho, seed):
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x1 = z1
    x2 = rho * z1 + np.sqrt(1 - rho**2) * z2
    idx = pd.RangeIndex(n)
    return pd.Series(x1, index=idx), pd.Series(x2, index=idx)


def test_diagonal_is_one(sample_returns):
    corr = compute_correlation_matrix(sample_returns)

    for col in corr.columns:
        assert corr.loc[col, col] == pytest.approx(1.0), (
            f"Diagonal entry for {col} is not 1.0"
        )


def test_symmetric(sample_returns):
    corr = compute_correlation_matrix(sample_returns)

    pd.testing.assert_frame_equal(corr, corr.T)


def test_rolling_corr_length(sample_returns):
    window = 5
    result = rolling_correlation(
        sample_returns["EURUSD"],
        sample_returns["GBPUSD"],
        window=window,
    )

    assert len(result) == len(sample_returns)
    assert result.iloc[: window - 1].isna().all()


def test_bounds(sample_returns):
    corr = compute_correlation_matrix(sample_returns)

    assert (corr.values >= -1).all() and (corr.values <= 1).all()


def test_column_labels_preserved(sample_returns):
    corr = compute_correlation_matrix(sample_returns)

    assert list(corr.columns) == list(sample_returns.columns)
    assert list(corr.index) == list(sample_returns.columns)


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_correlation_matrix(pd.DataFrame())


def test_single_series_raises():
    with pytest.raises(ValueError, match="at least 2"):
        compute_correlation_matrix(pd.DataFrame({"EURUSD": [0.1, 0.2]}))


def test_window_larger_than_series_raises(sample_returns):
    with pytest.raises(ValueError, match="larger than"):
        rolling_correlation(
            sample_returns["EURUSD"],
            sample_returns["GBPUSD"],
            window=999,
        )


def test_window_less_than_two_raises(sample_returns):
    with pytest.raises(ValueError, match="at least 2"):
        rolling_correlation(
            sample_returns["EURUSD"],
            sample_returns["GBPUSD"],
            window=1,
        )


def test_rolling_correlation_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        rolling_correlation(
            pd.Series([0.1, 0.2, 0.3]),
            pd.Series([0.1, 0.2]),
            window=2,
        )


def test_perfect_correlation_with_self(sample_returns):

    result = rolling_correlation(
        sample_returns["EURUSD"],
        sample_returns["EURUSD"],
        window=5,
    ).dropna()
    
    assert result.min() == pytest.approx(1.0)


def test_stable_correlation_no_flags():
    s1, s2 = _correlated_series(n=500, rho=0.7, seed=42)
    flags = detect_correlation_regime_shifts(s1, s2, window=60, threshold=3.0)

    assert flags.sum() == 0


def test_sudden_shift_flagged():
    n_each = 400
    changepoint = n_each

    s1_a, s2_a = _correlated_series(n=n_each, rho=0.8, seed=1)
    s1_b, s2_b = _correlated_series(n=n_each, rho=-0.3, seed=2)

    s1 = pd.concat([s1_a, s1_b], ignore_index=True)
    s2 = pd.concat([s2_a, s2_b], ignore_index=True)

    flags = detect_correlation_regime_shifts(s1, s2, window=60, threshold=2.0)

    assert flags.any()
    first_flag_idx = flags[flags].index[0]
    assert first_flag_idx >= changepoint


def test_output_length_matches_input():
    s1, s2 = _correlated_series(n=250, rho=0.5, seed=7)
    flags = detect_correlation_regime_shifts(s1, s2, window=60, threshold=2.0)

    assert len(flags) == len(s1)
    assert flags.index.equals(s1.index)


def test_regime_shifts_mismatched_lengths_raises():
    s1, _ = _correlated_series(n=100, rho=0.5, seed=3)
    s2, _ = _correlated_series(n=90, rho=0.5, seed=4)
    with pytest.raises(ValueError, match="same length"):
        detect_correlation_regime_shifts(s1, s2)


def test_small_window_raises():
    s1, s2 = _correlated_series(n=100, rho=0.5, seed=5)
    with pytest.raises(ValueError, match="must be > 3"):
        detect_correlation_regime_shifts(s1, s2, window=3)
