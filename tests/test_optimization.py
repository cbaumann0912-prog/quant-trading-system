import numpy as np
import pytest
from src.stats.optimization import gradient_descent, constrained_optimize


def test_converges_to_zero_for_quadratic():
    f = lambda x: x**2
    grad_f = lambda x: 2 * x
    x0 = np.array([5.0])

    result = gradient_descent(f, grad_f, x0, lr=0.1, n_iter=1000)

    assert np.allclose(result, np.array([0.0]), atol=1e-6)


def test_returns_ndarray():
    f = lambda x: x[0] ** 2 + x[1] ** 2
    grad_f = lambda x: np.array([2 * x[0], 2 * x[1]])
    x0 = np.array([3.0, 4.0])

    result = gradient_descent(f, grad_f, x0, lr=0.1, n_iter=1000)

    assert isinstance(result, np.ndarray)


def test_stops_at_tolerance():
    f = lambda x: x**2
    grad_f = lambda x: 2 * x
    x0 = np.array([1e-9])

    result = gradient_descent(
        f,
        grad_f,
        x0,
        lr=0.1,
        n_iter=1000,
        tol=1e-8,
    )

    assert np.allclose(result, x0, atol=1e-12)


def test_warns_and_returns_x_when_not_converged():
    f = lambda x: x**2
    grad_f = lambda x: 2 * x
    x0 = np.array([10.0])

    with pytest.warns(UserWarning, match="did not converge"):
        result = gradient_descent(f, grad_f, x0, lr=0.1, n_iter=1, tol=1e-12)

    expected = x0 - 0.1 * grad_f(x0)
    assert np.allclose(result, expected)


def test_constrained_optimize_returns_scipy_result():
    result = constrained_optimize(
        objective=lambda w: w @ w,
        x0=np.array([0.5, 0.5]),
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
    )

    assert result.success
    np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-6)


def test_constrained_optimize_raises_on_infeasible_constraints():
    with pytest.raises(RuntimeError, match="Optimization failed"):
        constrained_optimize(
            objective=lambda w: w @ w,
            x0=np.array([0.5, 0.5]),
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w: np.sum(w) - 5},
            ],
            options={"maxiter": 50},
        )