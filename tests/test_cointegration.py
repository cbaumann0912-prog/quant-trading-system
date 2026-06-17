import numpy as np
import pandas as pd
import pytest

from src.signals.cointegration import engle_granger_test, cointegration_spread
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