import numpy as np
import pandas as pd
from src.stats.portfolio_stats import (compute_covariance_matrix, compute_portfolio_variance, compute_portfolio_return)

returns = pd.DataFrame({
    'EURUSD': [0.0050, -0.0020, 0.0080, -0.0010, 0.0030],
    'GBPUSD': [0.0040, -0.0030, 0.0060,  0.0010, 0.0020],
    'USDJPY': [-0.0030, 0.0050, -0.0020, 0.0040, -0.0010]
})

weights = np.array([1/3, 1/3, 1/3])

def test_cov_matrix_symetric():
    result = compute_covariance_matrix(returns)
    assert np.allclose(result, result.T)

def test_cov_matrix_shape():
    result = compute_covariance_matrix(returns)
    assert result.shape == (len(returns.columns),len(returns.columns))

def test_portfolio_variance_pos():
    result = compute_portfolio_variance(weights,compute_covariance_matrix(returns))
    assert result > 0

def test_equal_weight_return():
    mean_returns = returns.mean().values
    result = compute_portfolio_return(weights, mean_returns)
    assert np.isclose(result, 0.001733, atol=1e-6)