import numpy as np
import pytest
from src.stats.regression import fit_ols


@pytest.fixture
def simple_data():
    A = np.array([1, 2, 3, 4, 5], dtype=float).reshape(-1, 1)
    b = np.array([2.1, 3.9, 6.2, 7.8, 9.5], dtype=float)
    return A, b


def test_coefficients_match_sklearn(simple_data):
    from sklearn.linear_model import LinearRegression
    A, b = simple_data

    result = fit_ols(A, b, add_intercept=True)

    model = LinearRegression(fit_intercept=True).fit(A, b)
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