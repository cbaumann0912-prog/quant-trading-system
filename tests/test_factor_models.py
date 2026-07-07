import numpy as np
import pandas as pd
import pytest

from src.analysis.factor_models import capm_expected_return, pca_factor_decomposition


def test_beta_one_for_market_itself():
    rng = np.random.default_rng(41)
    market_returns = pd.Series(rng.normal(0.0005, 0.01, 500))

    result = capm_expected_return(market_returns, market_returns, rf_rate=0.0)

    assert result["beta"] == pytest.approx(1.0, abs=1e-8)
    assert result["alpha"] == pytest.approx(0.0, abs=1e-8)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-8)


def test_alpha_zero_for_passive_benchmark():
    rng = np.random.default_rng(42)
    market_returns = pd.Series(rng.normal(0.0005, 0.01, 1000))
    true_beta = 0.65
    asset_returns = true_beta * market_returns

    result = capm_expected_return(asset_returns, market_returns, rf_rate=0.0)

    assert result["alpha"] == pytest.approx(0.0, abs=1e-8)
    assert result["beta"] == pytest.approx(true_beta, abs=1e-8)


def test_output_keys_present():
    rng = np.random.default_rng(7)
    asset_returns = pd.Series(rng.normal(0.0, 0.01, 300))
    market_returns = pd.Series(rng.normal(0.0, 0.01, 300))

    result = capm_expected_return(asset_returns, market_returns, rf_rate=0.0)

    expected_keys = {"alpha", "beta", "r_squared", "alpha_t_stat", "alpha_p_value"}
    assert set(result.keys()) == expected_keys


def test_zero_variance_market_does_not_raise():
    asset_returns = pd.Series(np.random.default_rng(3).normal(0.0, 0.01, 100))
    market_returns = pd.Series(np.zeros(100))

    result = capm_expected_return(asset_returns, market_returns, rf_rate=0.0)

    assert np.isnan(result["beta"])
    assert np.isnan(result["alpha_t_stat"])
    assert np.isnan(result["alpha_p_value"])


def test_alpha_t_stat_sign_matches_alpha_sign():
    rng = np.random.default_rng(11)
    market_returns = pd.Series(rng.normal(0.0, 0.01, 400))
    noise = rng.normal(0.0, 0.001, 400)
    asset_returns = 0.5 * market_returns + 0.002 + noise

    result = capm_expected_return(asset_returns, market_returns, rf_rate=0.0)

    assert result["alpha"] > 0
    assert result["alpha_t_stat"] > 0


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    n_obs = 500
    factor1 = rng.normal(0, 0.01, n_obs)
    factor2 = rng.normal(0, 0.005, n_obs)
    eurusd = 0.8 * factor1 + 0.3 * factor2 + rng.normal(0, 0.001, n_obs)
    gbpusd = 0.7 * factor1 - 0.2 * factor2 + rng.normal(0, 0.001, n_obs)
    usdjpy = -0.6 * factor1 + 0.5 * factor2 + rng.normal(0, 0.001, n_obs)
    
    return pd.DataFrame(
        {"EURUSD": eurusd, "GBPUSD": gbpusd, "USDJPY": usdjpy},
        index=pd.date_range("2020-01-01", periods=n_obs, freq="D")
    )


def test_factor_returns_shape(sample_returns):
    n_factors = 2
    result = pca_factor_decomposition(sample_returns, n_factors)

    assert result["factor_returns"].shape == (len(sample_returns), n_factors)
    assert result["loadings"].shape == (sample_returns.shape[1], n_factors)
    assert len(result["explained_variance"]) == n_factors


def test_residuals_lower_variance_than_raw(sample_returns):
    result = pca_factor_decomposition(sample_returns, n_factors=1)
    raw_variance = sample_returns.var().sum()
    residual_variance = result["residual_returns"].var().sum()

    assert residual_variance < raw_variance


def test_loadings_orthogonal(sample_returns):
    result = pca_factor_decomposition(sample_returns, n_factors=3)
    V = result["loadings"].values
    gram = V.T @ V
    off_diagonal = gram - np.diag(np.diag(gram))

    assert np.all(np.abs(off_diagonal) < 1e-10)