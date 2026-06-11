import numpy as np
import pandas as pd
import pytest

from src.data.stationarity import adf_test, check_stationarity, kpss_test

np.random.seed(28)

STATIONARY_SERIES = pd.Series(np.random.normal(0, 1, 1000))
RANDOM_WALK = pd.Series(np.cumsum(np.random.normal(0, 1, 1000)))


def test_adf_stationary_series_passes():
    result = adf_test(STATIONARY_SERIES)

    assert result["reject_null"] is True


def test_adf_random_walk_fails():
    result = adf_test(RANDOM_WALK)

    assert result["reject_null"] is False


def test_adf_returns_expected_keys():
    result = adf_test(STATIONARY_SERIES)
    expected_keys = {
        "adf_stat",
        "adf_p",
        "n_lags",
        "n_obs",
        "critical_values",
        "reject_null",
    }

    assert expected_keys.issubset(result.keys())


def test_adf_stat_is_negative():
    result = adf_test(STATIONARY_SERIES)

    assert result["adf_stat"] < 0


def test_adf_critical_values_has_three_levels():
    result = adf_test(STATIONARY_SERIES)

    assert {"1%", "5%", "10%"}.issubset(result["critical_values"].keys())


def test_kpss_stationary_series_passes():
    result = kpss_test(STATIONARY_SERIES)

    assert result["reject_null"] is False


def test_kpss_random_walk_fails():
    result = kpss_test(RANDOM_WALK)

    assert result["reject_null"] is True


def test_kpss_returns_expected_keys():
    result = kpss_test(STATIONARY_SERIES)
    expected_keys = {
        "kpss_stat",
        "kpss_p",
        "n_lags",
        "critical_values",
        "reject_null",
    }

    assert expected_keys.issubset(result.keys())


def test_check_stationarity_stationary_series_passes():
    result = check_stationarity(STATIONARY_SERIES)

    assert result["is_stationary"] is True


def test_check_stationarity_random_walk_fails():
    result = check_stationarity(RANDOM_WALK)

    assert result["is_stationary"] is False


def test_recommendation_string_not_empty():
    for series in [STATIONARY_SERIES, RANDOM_WALK]:
        result = check_stationarity(series)

        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0


def test_check_stationarity_returns_expected_keys():
    result = check_stationarity(STATIONARY_SERIES)
    expected_keys = {
        "adf_stat",
        "adf_p",
        "kpss_stat",
        "kpss_p",
        "is_stationary",
        "recommendation",
    }
    
    assert expected_keys.issubset(result.keys())