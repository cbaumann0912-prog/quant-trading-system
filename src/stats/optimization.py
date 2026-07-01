import numpy as np
from typing import Callable
from scipy.optimize import minimize

import warnings

def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float,
    n_iter: int,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Minimize f via gradient descent.

    Parameters
    ----------
    f : Callable
        Objective function, f(x) -> scalar.
    grad_f : Callable
        Gradient of f, grad_f(x) -> ndarray, same shape as x.
    x0 : np.ndarray
        Initial point.
    lr : float
        Step size (learning rate), eta.
    n_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance — stop early if gradient norm is less than tol

    Returns
    -------
    np.ndarray
        The approximate minimizer.
    """
    x = x0.copy() 
    count = 0
    while count < n_iter:
        count += 1
        if np.linalg.norm(grad_f(x)) >= tol:
            x = x - lr * grad_f(x)
        else:
            return x

    warnings.warn(f"gradient_descent did not converge within {n_iter} iterations")
    return x

def constrained_optimize(
        objective,
        x0,
        constraints,
        bounds=None,
        jac=None,
        options=None):
        """
        Thin SLSQP wrapper — no problem-specific logic.

        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Objective function
        x0 : np.ndarray
            Initial guess.
        constraints : dict or list of dict
            scipy-style constraint dict(s)
        bounds : list of tuple, optional
            Per-coordinate (min, max) bounds
        jac : Callable[[np.ndarray], np.ndarray], optional
            Exact gradient of `objective`
        options : dict, optional
            Passed through to scipy.optimize.minimize

        Returns
        -------
        scipy.optimize.OptimizeResult
            Full scipy result object

        Raises
        ------
        RuntimeError
            If SLSQP does not converge (`result.success is False`).
        """

        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints, jac=jac, options=options)
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        return result