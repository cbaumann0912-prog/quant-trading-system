import numpy as np
import pandas as pd
import pytest

from src.features.garch import fit_garch

np.random.seed(55)

def _simulate_garch(n=1500, omega=0.00001, alpha=0.08, beta=0.90):
    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    eps[0] = np.sqrt(sigma2[0]) * np.random.normal()

    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * np.random.normal()

    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(eps, index=idx)


SIM_RETURNS = _simulate_garch()


def test_persistence_between_0_and_1():
    result = fit_garch(SIM_RETURNS)

    assert 0 < result["persistence"] < 1


def test_conditional_vol_length_matches_returns():
    result = fit_garch(SIM_RETURNS)

    assert len(result["conditional_vol"]) == len(SIM_RETURNS)


def test_long_run_vol_positive():
    result = fit_garch(SIM_RETURNS)

    assert result["long_run_vol"] > 0


def test_params_dict_keys_present():
    result = fit_garch(SIM_RETURNS)
    expected_keys = {
        "omega",
        "alpha",
        "beta",
        "persistence",
        "long_run_vol",
        "conditional_vol",
    }

    assert expected_keys.issubset(result.keys())
