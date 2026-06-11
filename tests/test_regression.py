import numpy as np
import pytest
from numpy.typing import NDArray
from src.stats.regression import fit_ols, r_squared, adj_r_squared, residual_diagnostics, ridge_fit, lasso_objective, lasso_fit


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