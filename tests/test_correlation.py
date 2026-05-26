import pandas as pd
import numpy as np
import pytest
from src.stats.correlation import compute_correlation_matrix, rolling_correlation


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """10-day return sample — same data as your hand-computation problem."""

    return pd.DataFrame({
        "EURUSD": [ 0.42, -0.17,  0.89, -0.53,  0.11,
                   -0.78,  0.34, -0.29,  0.62, -0.21],
        "GBPUSD": [ 0.38, -0.22,  0.75, -0.41,  0.19,
                   -0.65,  0.28, -0.18,  0.51, -0.15],
        "USDJPY": [-0.31,  0.18, -0.64,  0.29, -0.08,
                    0.57, -0.22,  0.14, -0.48,  0.11],
    })

class TestComputeCorrelationMatrix:

    def test_diagonal_is_one(self, sample_returns):
        corr = compute_correlation_matrix(sample_returns)
        for col in corr.columns:
            assert corr.loc[col, col] == pytest.approx(1.0), (
                f"Diagonal entry for {col} is not 1.0"
            )

    def test_symmetric(self, sample_returns):
        corr = compute_correlation_matrix(sample_returns)
        pd.testing.assert_frame_equal(corr, corr.T)

    def test_rolling_corr_length(self, sample_returns):
        window = 5
        result = rolling_correlation(
            sample_returns["EURUSD"],
            sample_returns["GBPUSD"],
            window=window,
        )
        assert len(result) == len(sample_returns)

        assert result.iloc[: window - 1].isna().all()

    def test_bounds(self, sample_returns):
        corr = compute_correlation_matrix(sample_returns)
        assert (corr.values >= -1).all() and (corr.values <= 1).all()

    def test_column_labels_preserved(self, sample_returns):
        corr = compute_correlation_matrix(sample_returns)
        assert list(corr.columns) == list(sample_returns.columns)
        assert list(corr.index) == list(sample_returns.columns)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_correlation_matrix(pd.DataFrame())

    def test_single_series_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_correlation_matrix(pd.DataFrame({"EURUSD": [0.1, 0.2]}))


class TestRollingCorrelation:

    def test_window_larger_than_series_raises(self, sample_returns):
        with pytest.raises(ValueError, match="larger than"):
            rolling_correlation(
                sample_returns["EURUSD"],
                sample_returns["GBPUSD"],
                window=999,
            )

    def test_window_less_than_two_raises(self, sample_returns):
        with pytest.raises(ValueError, match="at least 2"):
            rolling_correlation(
                sample_returns["EURUSD"],
                sample_returns["GBPUSD"],
                window=1,
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            rolling_correlation(
                pd.Series([0.1, 0.2, 0.3]),
                pd.Series([0.1, 0.2]),
                window=2,
            )

    def test_perfect_correlation_with_self(self, sample_returns):
        result = rolling_correlation(
            sample_returns["EURUSD"],
            sample_returns["EURUSD"],
            window=5,
        ).dropna()
        assert result.min() == pytest.approx(1.0)