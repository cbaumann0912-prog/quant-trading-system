import numpy as np
from src.stats.optimization import gradient_descent


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