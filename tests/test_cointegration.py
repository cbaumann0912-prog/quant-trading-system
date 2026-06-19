import numpy as np
import pandas as pd
import pytest

from src.signals.cointegration import engle_granger_test, cointegration_spread, johansen_test
from src.stats.regression import fit_ols
from src.data.stationarity import adf_test


@pytest.fixture
def cointegrated_pair():
    rng = np.random.default_rng(28)
    n = 1000
    shared_trend = np.cumsum(rng.standard_normal(n))
    y = shared_trend + rng.standard_normal(n) * 0.05
    x = 0.7 * shared_trend + rng.standard_normal(n) * 0.05
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    return pd.Series(y, index=dates), pd.Series(x, index=dates)


@pytest.fixture
def independent_pair():
    rng = np.random.default_rng(28)
    n = 1000
    y = pd.Series(
        np.cumsum(rng.standard_normal(n)),
        index=pd.date_range("2010-01-01", periods=n, freq="D"),
    )
    x = pd.Series(
        np.cumsum(rng.standard_normal(n)),
        index=pd.date_range("2010-01-01", periods=n, freq="D"),
    )
    return y, x


def test_result_keys_present(cointegrated_pair):
    y, x = cointegrated_pair
    result = engle_granger_test(y, x)
    expected_keys = {'alpha', 'hedge_ratio', 'residuals', 'adf_stat', 'adf_p', 'is_cointegrated'}

    assert expected_keys.issubset(result.keys())


def test_residuals_length_matches_input(cointegrated_pair):
    y, x = cointegrated_pair
    result = engle_granger_test(y, x)

    assert len(result['residuals']) == len(y)
    assert result['residuals'].index.equals(y.index)


def test_hedge_ratio_from_ols(cointegrated_pair):
    y, x = cointegrated_pair
    A = x.values.reshape(-1, 1)
    b = y.values
    direct = fit_ols(A, b)
    direct_beta = direct['coefficients'][1]
    result = engle_granger_test(y, x)

    assert abs(result['hedge_ratio'] - direct_beta) < 1e-8


def test_cointegrated_series_detected(cointegrated_pair):
    y, x = cointegrated_pair
    result = engle_granger_test(y, x)

    assert result['is_cointegrated'] is True
    assert result['adf_p'] < 0.05


def test_non_cointegrated_rejected(independent_pair):
    y, x = independent_pair
    result = engle_granger_test(y, x)
    
    assert result['is_cointegrated'] == False
    assert result['adf_p'] > 0.05


def test_spread_matches_residuals(cointegrated_pair):
    y, x = cointegrated_pair
    result = engle_granger_test(y, x)
    spread = cointegration_spread(y, x, result['alpha'], result['hedge_ratio'])

    assert (spread - result['residuals']).abs().max() < 1e-10


def test_spread_is_stationary(cointegrated_pair):
    y, x = cointegrated_pair
    result = engle_granger_test(y, x)
    spread = cointegration_spread(y, x, result['alpha'], result['hedge_ratio'])

    adf_result = adf_test(spread)
    assert adf_result['adf_p'] < 0.05


def test_spread_mean_near_zero(cointegrated_pair):
    y, x = cointegrated_pair
    result = engle_granger_test(y, x)
    spread = cointegration_spread(y, x, result['alpha'], result['hedge_ratio'])

    assert abs(spread.mean()) < 0.1


def test_johansen_known_cointegrated_pair():
    n = 1000
    y1 = np.cumsum(np.random.normal(0, 1, n))
    noise = np.random.normal(0, 0.5, n)
    y2 = 0.5 * y1 + noise  # stationary combination -> rank should be 1
    df = pd.DataFrame({"y1": y1, "y2": y2})
    result = johansen_test(df)

    assert result["rank_trace"] >= 1
    assert result["eigenvectors"].shape == (2, 2)

def test_johansen_independent_series():
    n = 1000
    y1 = np.cumsum(np.random.normal(0, 1, n))
    y2 = np.cumsum(np.random.normal(0, 1, n))  # independent random walk
    df = pd.DataFrame({"y1": y1, "y2": y2})
    result = johansen_test(df)

    assert result["rank_trace"] == 0


def test_johansen_trace_stat_monotonic():
    n = 500
    y1 = np.cumsum(np.random.normal(0, 1, n))
    y2 = np.cumsum(np.random.normal(0, 1, n))
    y3 = np.cumsum(np.random.normal(0, 1, n))
    df = pd.DataFrame({"y1": y1, "y2": y2, "y3": y3})
    result = johansen_test(df)
    trace = result["trace_stat"]

    assert all(trace[i] >= trace[i + 1] for i in range(len(trace) - 1))


def test_johansen_output_shape():
    n = 500
    df = pd.DataFrame({
        "a": np.cumsum(np.random.normal(0, 1, n)),
        "b": np.cumsum(np.random.normal(0, 1, n)),
        "c": np.cumsum(np.random.normal(0, 1, n)),
    })
    result = johansen_test(df)

    assert result["eigenvalues"].shape[0] == 3
    assert result["eigenvectors"].shape == (3, 3)
    assert result["trace_stat"].shape[0] == 3
    assert result["max_eig_stat"].shape[0] == 3


def test_johansen_rejects_nan_input():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        johansen_test(df)


def test_johansen_real_data_three_pairs():
    """
    Smoke test on real daily EUR/USD, GBP/USD, USD/JPY closes.
    No leakage: deterministic seed, no parameter tuning against this output.
    """
    data_dir = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"

    pairs = {
        "EURUSD": f"{data_dir}\\EURUSD.csv",
        "GBPUSD": f"{data_dir}\\GBPUSD.csv",
        "USDJPY": f"{data_dir}\\USDJPY.csv",
    }

    series = {}
    for name, path in pairs.items():
        df = pd.read_csv(path)
        df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
        df = df.set_index("Datetime")
        daily_close = df["Close"].resample("D").last().dropna()
        series[name] = daily_close

    combined = pd.DataFrame(series).dropna()

    assert combined.shape[0] > 100
    assert combined.shape[1] == 3

    result = johansen_test(combined)

    assert result["eigenvalues"].shape[0] == 3
    assert result["eigenvectors"].shape == (3, 3)
    assert result["rank_trace"] in (0, 1, 2, 3)
    assert result["rank_max_eig"] in (0, 1, 2, 3)
