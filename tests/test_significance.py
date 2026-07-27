import numpy as np
import pandas as pd
import pytest

from src.evaluation.significance import (
    bonferroni_correction,
    benjamini_hochberg_correction,
    permutation_test,
    permutation_test_interaction_coefficient,
)

sample = [0.00448,0.39341,0.53882,
          0.00671,0.01220,0.98617,
          0.58125,0.00017,0.00907,
          0.33626]

tiny_sample = [0.00001,0.00007,0.00004,0.00009,0.00002]

def test_bonferroni_more_conservative_than_bh():
    result_bf = bonferroni_correction(sample,0.05)
    result_bh = benjamini_hochberg_correction(sample,0.05)

    assert sum(result_bf) < sum(result_bh)


def test_bh_known_example():
    result = benjamini_hochberg_correction(sample,0.05)
    
    assert result == [True,False,False,True,True,False,False,True,True,False]


def test_all_rejected_when_all_tiny():
    result_bf = bonferroni_correction(tiny_sample,0.05)
    result_bh = benjamini_hochberg_correction(tiny_sample,0.05)

    assert sum(result_bf) == len(result_bf)
    assert sum(result_bh) == len(result_bh)


def test_random_signal_high_p_value():
    rng = np.random.default_rng(21)
    n = 300
    idx = pd.RangeIndex(n)

    signal = pd.Series(rng.normal(size=n), index=idx)
    forward_returns = pd.Series(rng.normal(size=n), index=idx)

    result = permutation_test(
        signal,
        forward_returns,
        n_permutations=1000,
        seed=42,
        alternative="two-sided",
    )

    assert result["p_value"] > 0.10, (
        f"Expected non-significant p-value for unrelated random signal, "
        f"got {result['p_value']:.4f}"
    )


def test_correlated_signal_low_p_value():
    rng = np.random.default_rng(23)
    n = 300
    idx = pd.RangeIndex(n)

    forward_returns = pd.Series(rng.normal(size=n), index=idx)
    noise = rng.normal(scale=0.1, size=n)
    signal = forward_returns + noise
    signal = pd.Series(signal.to_numpy(), index=idx)

    result = permutation_test(
        signal,
        forward_returns,
        n_permutations=1000,
        seed=42,
        alternative="greater",
    )

    assert result["p_value"] < 0.05, (
        f"Expected significant p-value for deliberately correlated signal, "
        f"got {result['p_value']:.4f}"
    )


def test_null_distribution_length():
    rng = np.random.default_rng(5)
    n = 100
    idx = pd.RangeIndex(n)
    signal = pd.Series(rng.normal(size=n), index=idx)
    forward_returns = pd.Series(rng.normal(size=n), index=idx)

    result_default = permutation_test(signal, forward_returns, seed=1)

    assert len(result_default["null_distribution"]) == 1000

    n_perm_custom = 347
    result_custom = permutation_test(
        signal, forward_returns, n_permutations=n_perm_custom, seed=1
    )

    assert len(result_custom["null_distribution"]) == n_perm_custom


def test_raises_on_mismatched_index():
    rng = np.random.default_rng(32)
    n = 100
    signal = pd.Series(rng.normal(size=n), index=pd.RangeIndex(n))
    forward_returns = pd.Series(
        rng.normal(size=n), index=pd.RangeIndex(1, n + 1)
    )
 
    with pytest.raises(ValueError):
        permutation_test(signal, forward_returns)
 
 
def test_raises_on_invalid_alternative():
    rng = np.random.default_rng(7)
    n = 100
    idx = pd.RangeIndex(n)
    signal = pd.Series(rng.normal(size=n), index=idx)
    forward_returns = pd.Series(rng.normal(size=n), index=idx)

    with pytest.raises(ValueError):
        permutation_test(signal, forward_returns, alternative="bogus")


def test_dummy_permutation_null_signal_high_p_value():
    rng = np.random.default_rng(101)
    n = 800
    idx = pd.RangeIndex(n)

    x1 = pd.Series(rng.normal(0, 1, n), index=idx)
    dummy = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    y = pd.Series(0.4 * x1 + rng.normal(0, 1, n), index=idx)

    result = permutation_test_interaction_coefficient(
        y, x1, dummy, n_permutations=500, seed=1
    )

    assert result["p_value"] > 0.10, (
        f"Expected non-significant p-value with no true interaction, "
        f"got {result['p_value']:.4f}"
    )


def test_dummy_permutation_real_signal_low_p_value():
    rng = np.random.default_rng(102)
    n = 800
    idx = pd.RangeIndex(n)

    x1 = pd.Series(rng.normal(0, 1, n), index=idx)
    dummy = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    y = pd.Series(
        0.2 * x1 + 2.5 * (x1 * dummy) + rng.normal(0, 0.3, n), index=idx
    )

    result = permutation_test_interaction_coefficient(
        y, x1, dummy, n_permutations=500, seed=2, alternative="greater"
    )

    assert result["p_value"] < 0.05, (
        f"Expected significant p-value for a strong true interaction, "
        f"got {result['p_value']:.4f}"
    )


def test_dummy_permutation_preserves_base_rate():
    rng = np.random.default_rng(103)
    n = 400
    idx = pd.RangeIndex(n)
    x1 = pd.Series(rng.normal(0, 1, n), index=idx)
    dummy_vals = np.zeros(n)
    dummy_vals[:120] = 1.0
    dummy = pd.Series(dummy_vals, index=idx)
    y = pd.Series(rng.normal(0, 1, n), index=idx)

    result_a = permutation_test_interaction_coefficient(
        y, x1, dummy, n_permutations=200, seed=5
    )
    result_b = permutation_test_interaction_coefficient(
        y, x1, dummy, n_permutations=200, seed=5
    )
    
    np.testing.assert_allclose(
        result_a["null_distribution"], result_b["null_distribution"]
    )
    assert len(result_a["null_distribution"]) == 200


def test_dummy_permutation_output_keys_present():
    rng = np.random.default_rng(104)
    n = 200
    idx = pd.RangeIndex(n)
    x1 = pd.Series(rng.normal(0, 1, n), index=idx)
    dummy = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    y = pd.Series(rng.normal(0, 1, n), index=idx)
    result = permutation_test_interaction_coefficient(y, x1, dummy, n_permutations=50, seed=1)

    assert set(result.keys()) == {"observed_b3", "p_value", "null_distribution", "n_obs"}
    assert result["n_obs"] == n


def test_dummy_permutation_raises_on_invalid_alternative():
    rng = np.random.default_rng(105)
    n = 100
    idx = pd.RangeIndex(n)
    x1 = pd.Series(rng.normal(0, 1, n), index=idx)
    dummy = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    y = pd.Series(rng.normal(0, 1, n), index=idx)

    with pytest.raises(ValueError):
        permutation_test_interaction_coefficient(y, x1, dummy, alternative="bogus")