import numpy as np
from typing import Callable
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