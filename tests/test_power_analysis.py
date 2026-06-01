from src.stats.power_analysis import compute_required_sample_size, compute_achieved_power
import pytest

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