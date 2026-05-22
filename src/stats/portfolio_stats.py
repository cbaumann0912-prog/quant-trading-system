import numpy as np
import pandas as pd

def compute_covariance_matrix(returns: pd.DataFrame)->np.ndarray:
    means = returns.mean()
    deviations = returns - means
    return np.cov(deviations.T)

def compute_portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray)->float:
    return np.dot(weights,np.dot(cov_matrix,weights))

def compute_portfolio_return(weights: np.ndarray, mean_returns: np.ndarray)->float:
    return np.dot(weights, mean_returns)