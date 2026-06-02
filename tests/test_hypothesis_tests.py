import pytest
import numpy as np
import pandas as pd
import scipy

from src.stats.hypothesis_tests import t_test_mean, p_value_interpretation, compute_effect_size_cohens_d, compute_required_sample_size, compute_achieved_power
from src.evaluation.bootstrap_ci import bootstrap_confidence_interval 

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


def test_interpretation_returns_string():
    """test_interpretation_returns_string must be a string"""
    result = p_value_interpretation(0.012, 0.05)
    assert isinstance(result,str)

def test_cohens_d_known_value():
    np.random.seed(8)
    result = compute_effect_size_cohens_d(pd.Series(np.random.normal(0,1,56)),(pd.Series(np.random.normal(1,1,28))))
    assert abs(result) == pytest.approx(1, 0.1)


def compute_sharpe(returns):
    return (returns.mean() - 0.0) / returns.std()


def test_ci_contains_true_mean():
    true_mean = 0  
    n_trials = 1000
    count = 0

    for d in range(n_trials):
        data = scipy.stats.t.rvs(df=5, scale=0.01, size=252)
        lower, upper = bootstrap_confidence_interval(data,np.mean , 1000, 0.95)
        if lower < true_mean < upper:
            count += 1

    coverage = count / n_trials
    assert coverage > 0.90
        

def test_output_is_tuple_of_two_floats():
    statistic_fn = compute_sharpe
    result = bootstrap_confidence_interval(scipy.stats.t.rvs(5,0.01,size=252), statistic_fn, 1000, 0.95)
    assert isinstance(result, tuple)
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


def test_lower_less_than_upper():
    statistic_fn = compute_sharpe
    result = bootstrap_confidence_interval(scipy.stats.t.rvs(5,0.01,size=252), statistic_fn, 1000, 0.95)
    assert result[0] < result[1]


def test_larger_effect_needs_fewer_samples():
    """Smaller effect size should require more samples to achieve the same power."""
   
    result1 = compute_required_sample_size(0.08,0.05,0.8)
    result2 = compute_required_sample_size(0.28,0.05,0.8)

    assert result1 > result2


def test_power_increases_with_n():
    """Larger sample size should yield higher achieved power for the same effect and alpha."""
    
    result1 = compute_achieved_power(8,0.28,0.05)
    result2 = compute_achieved_power(28,0.28,0.05)

    assert result1 < result2


def test_n80_power_reasonable():
    """compute_required_sample_size(0.2, 0.05, 0.80) should return 196."""
    
    result = compute_required_sample_size(0.2, 0.05, 0.80)

    assert result == 197
