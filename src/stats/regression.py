import numpy as np
from numpy.typing import NDArray


def fit_ols(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    add_intercept: bool = True,
) -> dict:
    """
    Fit OLS regression via the normal equations.

    Parameters
    ----------
    A : NDArray[np.float64]
    b : NDArray[np.float64]
    add_intercept : bool

    Returns
    -------
    dict with keys:
        'coefficients' : NDArray  
        'residuals'    : NDArray 
        'r_squared'    : float    
        'std_errors'   : NDArray 
    """
    if add_intercept:
        A = np.column_stack([np.ones(len(b)), A])
    
    AtA = A.T @ A
    Atb = A.T @ b
    
    beta = np.linalg.solve(AtA, Atb)   
    
    y_hat = A @ beta        
    residuals = b - y_hat   
    
    RSS = np.sum(residuals**2)
    TSS = np.sum((b - np.mean(b))**2)
    R_squared = 1 - (RSS / TSS)
    
    n, p = A.shape
    sigma_squared = RSS / (n - p)
    var_beta = sigma_squared * np.linalg.inv(AtA)
    std_errors = np.sqrt(np.diag(var_beta))
    
    return {
    'coefficients': beta,
    'residuals':    residuals,
    'r_squared':    R_squared,
    'std_errors':   std_errors,
} 
