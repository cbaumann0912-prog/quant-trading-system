import numpy as np
from numpy.typing import NDArray
from typing import Dict
import scipy.stats
from statsmodels.stats.stattools import acorr_ljungbox

def fit_ols(A: NDArray[np.float64], b: NDArray[np.float64], add_intercept: bool = True,) -> dict:
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
    
    R_squared = r_squared(b, y_hat)
    
    n, p = A.shape
    RSS = np.sum(residuals**2)
    sigma_squared = RSS / (n - p)
    
    var_beta = sigma_squared * np.linalg.inv(AtA)
    std_errors = np.sqrt(np.diag(var_beta))
    
    return {
    'coefficients': beta,
    'residuals':    residuals,
    'r_squared':    R_squared,
    'std_errors':   std_errors,
} 

def r_squared(y: NDArray[np.float64],y_hat: NDArray[np.float64]) -> float:
    """
    Compute the coefficient of determination R².

    Parameters
    ----------
    y : array of shape (n,)
        Observed response values.
    y_hat : array of shape (n,)
        Fitted values from the model.

    Returns
    -------
    float
        R² in [0, 1] for well-specified OLS models.
    """
    RSS = np.sum((y-y_hat)**2)
    TSS = np.sum((y-np.mean(y))**2)
    r2 = 1 - (RSS / TSS)
    return r2
    


def adj_r_squared(y: NDArray[np.float64], y_hat: NDArray[np.float64], p: int) -> float:
    """
    Compute adjusted R² with degrees-of-freedom penalty.

    Parameters
    ----------
    y : array of shape (n,)
        Observed response values.
    y_hat : array of shape (n,)
        Fitted values from the model.
    p : int
        Number of predictors, excluding the intercept.

    Returns
    -------
    float
        Adjusted R².
    """
    
    r2 = r_squared(y, y_hat)
    n = y.shape[0]

    if n-p-1 <= 0:
        raise ValueError("n-p-1 must be > 0")
    else:
        adj_r_squared = 1 - ((1 - r2) * ((n-1) / (n-p-1)))

    return adj_r_squared


def residual_diagnostics(y: NDArray[np.float64], y_hat: NDArray[np.float64], lags: int = 20) -> Dict[str, float]:
    """
    Run a standard residual diagnostic suite on OLS residuals.

    Computes:
        - residual mean (should be near zero)
        - residual variance (estimated σ²)
        - excess kurtosis (fat tail indicator)
        - Ljung-Box Q-statistic and p-value at `lags` lags
        - lag-1 autocorrelation

    Parameters
    ----------
    y : array of shape (n,)
        Observed response values.
    y_hat : array of shape (n,)
        Fitted values from the model.
    lags : int
        Number of lags for the Ljung-Box test. Default 20.

    Returns
    -------
    dict with keys:
        'mean', 'variance', 'excess_kurtosis',
        'lb_stat', 'lb_pvalue', 'lag1_autocorr'
    """
    residuals = y - y_hat

    mean = np.mean(residuals)
    variance = np.var(residuals,ddof=1)
    excess_kurtosis = scipy.stats.kurtosis(residuals)
    lb_result = acorr_ljungbox(residuals, lags=[lags], return_df=False)
    lb_stat = lb_result[0][0]
    lb_pvalue = lb_result[1][0]
    lag1_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0,1]

    return {
    'mean': mean,
    'variance': variance,
    'excess_kurtosis': excess_kurtosis,
    'lb_stat': lb_stat,
    'lb_pvalue': lb_pvalue,
    'lag1_autocorr': lag1_autocorr,
}
