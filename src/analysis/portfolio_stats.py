import numpy as np
import pandas as pd

def compute_covariance_matrix(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute the sample covariance matrix from a DataFrame of returns.

    Parameters
    ----------
    returns
        Each column is a return series for one instrument. Rows are observations.

    Returns
    -------
    np.ndarray
        Covariance matrix of shape (n_assets, n_assets).
    """
    means = returns.mean()
    deviations = returns - means
    return np.cov(deviations.T)

def compute_portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Compute portfolio variance given weights and a covariance matrix.

    Parameters
    ----------
    weights
        Portfolio weight vector of shape (n_assets,). Should sum to 1.
    cov_matrix
        Covariance matrix of asset returns, shape (n_assets, n_assets).

    Returns
    -------
    float
        Portfolio variance w^T Σ w.
    """
    return np.dot(weights,np.dot(cov_matrix,weights))

def compute_portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """
    Compute expected portfolio return as the weighted sum of asset returns.

    Parameters
    ----------
    weights
        Portfolio weight vector of shape (n_assets,). Should sum to 1.
    mean_returns
        Expected return vector of shape (n_assets,).

    Returns
    -------
    float
        Weighted average return w^T μ.
    """
    return np.dot(weights, mean_returns)