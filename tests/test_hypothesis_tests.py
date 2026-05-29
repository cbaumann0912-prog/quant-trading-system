import pytest
import numpy as np
import pandas as pd

from src.stats.hypothesis_tests import t_test_mean

@pytest.fixture
def eurusd_returns():
    """
    Deterministic EUR/USD-like return series drawn from a known distribution
    so expected values can be computed analytically.
    """
    np.random.seed(8)
    returns = np.random.normal(0.0003, 0.008, 50)
    return pd.Series(returns)

@pytest.fixture
def significant_returns():
    """
    Return series whose mean is clearly non-zero so H0: mu=0 should be rejected.
    """
    np.random.seed(0)
    returns = np.random.normal(0.01, 0.008, 50)
    return pd.Series(returns)

def test_known_t_stat(eurusd_returns):
    """
    Given a series with known statistics, the computed t-stat should match
    the hand-calculated value within floating-point tolerance.

    Hand calc: W = (x_bar - 0) / (s / sqrt(n))
    """
    result = t_test_mean(eurusd_returns, null_mean=0.0, confidence=0.95)

    x_bar = eurusd_returns.mean()
    s = eurusd_returns.std()
    n = len(eurusd_returns)
    expected = (x_bar / (s / np.sqrt(n)))

    assert result["t_stat"] == pytest.approx(expected, 1e-6)

def test_reject_null_when_significant(significant_returns):
    """
    When the sample mean is far from the null, reject_null should be True
    and the p-value should be below alpha = 0.05.
    """
    result = t_test_mean(significant_returns, null_mean=0.0, confidence=0.95)

    assert result["reject_null"] == True
    assert result["p_value"] < 0.05
    

def test_confidence_interval_coverage():
    """
    For a large Monte Carlo sample, ~95% of constructed CIs should contain
    the true mean, validating the interval formula.

    Strategy: simulate many samples from a known distribution, run
    t_test_mean on each, check what fraction of CIs contain the true mean.
    """
    np.random.seed(8)
    true_mean = 0.0003
    true_std  = 0.008
    n         = 50
    n_trials  = 1000
    confidence = 0.95
    
    count = 0
    hits = 0
    while count < n_trials:
        count += 1
        returns = pd.Series(np.random.normal(true_mean, true_std, n))
        result = t_test_mean(returns, 0, confidence)
        if true_mean > result["confidence_interval"][0] and true_mean < result["confidence_interval"][1]:
            hits += 1
    
    coverage = hits / n_trials
    
    assert abs(coverage - confidence) < 0.02

def test_raises_on_insufficient_data():
    """t_test_mean must raise ValueError when n < 2."""
    with pytest.raises(ValueError):
        t_test_mean(pd.Series([0.001]), null_mean=0.0, confidence=0.95)