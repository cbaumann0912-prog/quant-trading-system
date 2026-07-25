import numpy as np
import pandas as pd
import pytest

from src.features.garch import classify_vol_regime, fit_garch

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


def _bimodal_vol(n=400, low=0.005, high=0.03, seed=7):
    rng = np.random.default_rng(seed)
    half = n // 2
    low_cluster = rng.normal(low, low * 0.05, half)
    high_cluster = rng.normal(high, high * 0.05, n - half)
    values = np.concatenate([low_cluster, high_cluster])
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(values, index=idx)


def test_classifier_returns_two_labels():
    vol = _bimodal_vol()
    regime = classify_vol_regime(vol, n_regimes=2)

    assert set(regime.unique()) == {"low", "high"}


def test_high_regime_has_higher_mean_vol():
    vol = _bimodal_vol()
    regime = classify_vol_regime(vol, n_regimes=2)

    high_mean = vol[regime == "high"].mean()
    low_mean = vol[regime == "low"].mean()
    assert high_mean > low_mean


def test_output_index_matches_input():
    vol = _bimodal_vol()
    regime = classify_vol_regime(vol, n_regimes=2)

    assert list(regime.index) == list(vol.index)


def test_nan_dropped():
    vol = _bimodal_vol().copy()
    vol.iloc[5] = np.nan
    regime = classify_vol_regime(vol, n_regimes=2)

    assert len(regime) == len(vol) - 1


def test_n_regimes_below_2_raises():
    vol = _bimodal_vol()
    with pytest.raises(ValueError, match="n_regimes"):
        classify_vol_regime(vol, n_regimes=1)


def test_three_regimes_labeled_by_rank():
    rng = np.random.default_rng(11)
    low = rng.normal(0.005, 0.0002, 100)
    mid = rng.normal(0.015, 0.0002, 100)
    high = rng.normal(0.03, 0.0002, 100)
    vol = pd.Series(
        np.concatenate([low, mid, high]),
        index=pd.date_range("2020-01-01", periods=300, freq="D"),
    )

    regime = classify_vol_regime(vol, n_regimes=3)

    assert set(regime.unique()) == {"regime_0", "regime_1", "regime_2"}
    means = {r: vol[regime == r].mean() for r in regime.unique()}
    assert means["regime_0"] < means["regime_1"] < means["regime_2"]
