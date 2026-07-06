import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from src.stats.regression import fit_ols, r_squared, adj_r_squared, residual_diagnostics, ridge_fit, lasso_objective, lasso_fit, interaction_regression


@pytest.fixture
def simple_data():
    X = np.array([1, 2, 3, 4, 5], dtype=float).reshape(-1, 1)
    y = np.array([2.1, 3.9, 6.2, 7.8, 9.5], dtype=float)
    return X, y


def test_coefficients_match_sklearn(simple_data):
    from sklearn.linear_model import LinearRegression
    X, y = simple_data

    result = fit_ols(X, y, add_intercept=True)

    model = LinearRegression(fit_intercept=True).fit(X, y)
    sklearn_coeffs = np.array([model.intercept_, model.coef_[0]])

    np.testing.assert_allclose(
        result['coefficients'], sklearn_coeffs, atol=1e-10
    )


def test_r_squared_between_0_and_1(simple_data):
    A, b = simple_data
    result = fit_ols(A, b, add_intercept=True)

    assert 0.0 <= result['r_squared'] <= 1.0


def test_residuals_sum_to_zero(simple_data):
    A, b = simple_data
    result = fit_ols(A, b, add_intercept=True)

    assert abs(result['residuals'].sum()) < 1e-10


def test_r_squared_perfect_fit():
    y = np.array([1.0, 2.0, 3.0])
    y_hat = np.array([1.0, 2.0, 3.0])
    result = r_squared(y, y_hat)

    assert np.isclose(result, 1.0)


def test_r_squared_null_model():
    y = np.array([1.0, 2.0, 3.0])
    y_hat = np.array([2.0, 2.0, 2.0])
    result = r_squared(y, y_hat)

    assert np.isclose(result, 0)


def test_r_squared_known_value():
    y = np.array([3.0, 8.0, 4.0])
    y_hat = np.array([4.0, 5.0, 13.0])
    result = r_squared(y, y_hat)

    assert np.isclose(result, -5.5)


def test_adj_r_squared_leq_r_squared():
    np.random.seed(28)
    y = np.random.normal(0, 1, 20)
    y_hat = np.random.normal(0, 1, 20)
    adj_r2 = adj_r_squared(y, y_hat, 2)
    r2 = r_squared(y, y_hat)

    assert adj_r2 <= r2


def test_adj_r_squared_guard_fires():
    y = np.array([3.0, 8.0, 4.0])
    y_hat = np.array([4.0, 5.0, 13.0])

    with pytest.raises(ValueError): adj_r_squared(y, y_hat, 5)


def test_adj_r_squared_penalty_direction():
    np.random.seed(28)
    y = np.random.normal(0, 1, 20)
    y_hat = np.random.normal(0, 1, 20)
    adj_r2_1 = adj_r_squared(y, y_hat, 2)
    adj_r2_2 = adj_r_squared(y, y_hat, 3)

    assert adj_r2_2 < adj_r2_1


def test_residual_diagnostics_keys_present():
    np.random.seed(28)
    y = np.random.normal(0, 1, 500)
    y_hat = np.random.normal(0, 1, 500)
    result = residual_diagnostics(y, y_hat, 20)

    assert "mean" in result
    assert "variance" in result
    assert "excess_kurtosis" in result
    assert "lb_stat" in result
    assert "lb_pvalue" in result
    assert "lag1_autocorr" in result


def test_residual_diagnostics_mean_near_zero():
    np.random.seed(28)
    y = np.random.normal(0, 1, 500)
    y_hat = np.random.normal(0, 1, 500)
    result = residual_diagnostics(y, y_hat, 20)

    assert np.isclose(result["mean"], 0, atol = 0.1)


def test_residual_diagnostics_white_noise_passes_ljung_box():
    np.random.seed(28)
    y = np.random.normal(0, 1, 500)
    y_hat = np.random.normal(0, 1, 500)
    result = residual_diagnostics(y, y_hat, 20)

    assert result["lb_pvalue"] > 0.05


def test_residual_diagnostics_autocorrelated_fails_ljung_box():
    np.random.seed(28)
    n = 500
    residuals = np.zeros(n)
    for i in range(1, n):
        residuals[i] = 0.9 * residuals[i-1] + np.random.normal(0, 1)
    y = residuals
    y_hat = np.zeros(n)
    result = residual_diagnostics(y, y_hat, 20)

    assert result["lb_pvalue"] < 0.05


def test_ridge_shrinks_vs_ols():
    np.random.seed(28)
    X = np.random.randn(100, 3)
    y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(100)

    result_ols = fit_ols(X, y, False)
    result_rr = ridge_fit(X, y, 10.0)

    ols_norm = np.linalg.norm(result_ols["coefficients"])
    ridge_norm = np.linalg.norm(result_rr["coefficients"])

    assert ridge_norm < ols_norm

def test_lasso_produces_sparse_solution():
    np.random.seed(28)
    X = np.random.randn(100, 10)
    true_beta = np.array([3.0, -2.0, 0, 0, 0, 0, 0, 0, 0, 0])
    y = X @ true_beta + np.random.randn(100)

    result = lasso_fit(X, y, 10.0)

    assert result["n_nonzero"] < X.shape[1]

def test_ridge_lambda_zero_matches_ols():
    np.random.seed(28)
    X = np.random.randn(100, 3)
    y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(100)

    result_ridge = ridge_fit(X, y, lambda_=0.0)
    result_ols = fit_ols(X, y, add_intercept=True)

    np.testing.assert_allclose(
        result_ridge["coefficients"],
        result_ols["coefficients"][1:],
        rtol=1e-5
    )


def test_recovers_true_coefficients_with_low_noise():
    rng = np.random.default_rng(1)
    n = 2000
    x1 = pd.Series(rng.normal(0, 1, n))
    x2 = pd.Series(rng.normal(0, 1, n))
    noise = rng.normal(0, 0.05, n)
    y = pd.Series(0.3 + 1.0 * x1 + -0.5 * x2 + 1.5 * (x1 * x2) + noise)

    result = interaction_regression(y, x1, x2)

    assert result["coefficients"]["intercept"] == pytest.approx(0.3, abs=0.05)
    assert result["coefficients"]["x1"] == pytest.approx(1.0, abs=0.05)
    assert result["coefficients"]["x2"] == pytest.approx(-0.5, abs=0.05)
    assert result["coefficients"]["interaction"] == pytest.approx(1.5, abs=0.05)
    assert result["r_squared"] > 0.99


def test_null_interaction_gives_small_insignificant_coefficient():
    rng = np.random.default_rng(2)
    n = 1000
    x1 = pd.Series(rng.normal(0, 1, n))
    x2 = pd.Series(rng.normal(0, 1, n))
    noise = rng.normal(0, 1, n)
    y = pd.Series(0.1 + 0.4 * x1 + 0.2 * x2 + noise)

    result = interaction_regression(y, x1, x2)

    assert abs(result["t_stats"]["interaction"]) < 2.5
    assert result["p_values"]["interaction"] > 0.05


def test_output_keys_present():
    rng = np.random.default_rng(3)
    n = 200
    x1 = pd.Series(rng.normal(0, 1, n))
    x2 = pd.Series(rng.normal(0, 1, n))
    y = pd.Series(rng.normal(0, 1, n))

    result = interaction_regression(y, x1, x2)

    expected_top_keys = {
        "coefficients", "std_errors", "t_stats", "p_values",
        "r_squared", "adj_r_squared", "n_obs", "condition_number",
    }
    assert set(result.keys()) == expected_top_keys

    expected_term_keys = {"intercept", "x1", "x2", "interaction"}

    assert set(result["coefficients"].keys()) == expected_term_keys
    assert set(result["std_errors"].keys()) == expected_term_keys
    assert set(result["t_stats"].keys()) == expected_term_keys
    assert set(result["p_values"].keys()) == expected_term_keys


def test_misaligned_indices_are_aligned_by_inner_join():
    rng = np.random.default_rng(4)
    y = pd.Series(rng.normal(0, 1, 100), index=range(0, 100))
    x1 = pd.Series(rng.normal(0, 1, 100), index=range(10, 110))
    x2 = pd.Series(rng.normal(0, 1, 100), index=range(10, 110))

    result = interaction_regression(y, x1, x2)

    assert result["n_obs"] == 90


def test_nan_rows_are_dropped():
    rng = np.random.default_rng(5)
    y = pd.Series(rng.normal(0, 1, 100))
    x1 = pd.Series(rng.normal(0, 1, 100))
    x2 = pd.Series(rng.normal(0, 1, 100))
    x1.iloc[5] = np.nan
    y.iloc[10] = np.nan

    result = interaction_regression(y, x1, x2)

    assert result["n_obs"] == 98


def test_high_condition_number_flags_collinearity():
    rng = np.random.default_rng(6)
    n = 300
    x1 = pd.Series(rng.normal(1000, 0.001, n))
    x2 = pd.Series(rng.normal(1000, 0.001, n))
    y = pd.Series(rng.normal(0, 1, n))

    result = interaction_regression(y, x1, x2)

    assert result["condition_number"] > 1e6


def test_insufficient_observations_raises():
    y = pd.Series([1.0, 2.0, 3.0])
    x1 = pd.Series([1.0, 2.0, 3.0])
    x2 = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        interaction_regression(y, x1, x2)