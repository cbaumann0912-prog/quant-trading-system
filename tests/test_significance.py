import numpy as np
import pandas as pd
import pytest

from src.evaluation.significance import (
    bonferroni_correction,
    benjamini_hochberg_correction,
    permutation_test,
    permutation_test_interaction_coefficient,
    paired_sign_permutation_test,
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


def test_bonferroni_raises_on_empty_p_values():
    with pytest.raises(ValueError):
        bonferroni_correction([], 0.05)


def test_bh_raises_on_empty_p_values():
    with pytest.raises(ValueError):
        benjamini_hochberg_correction([], 0.05)


def test_permutation_test_raises_on_mismatched_length():
    rng = np.random.default_rng(11)
    signal = pd.Series(rng.normal(size=100), index=pd.RangeIndex(100))
    forward_returns = pd.Series(rng.normal(size=90), index=pd.RangeIndex(90))

    with pytest.raises(ValueError):
        permutation_test(signal, forward_returns)


def test_permutation_test_less_alternative_matches_manual_count():
    rng = np.random.default_rng(17)
    n = 200
    idx = pd.RangeIndex(n)
    signal = pd.Series(rng.normal(size=n), index=idx)
    forward_returns = pd.Series(rng.normal(size=n), index=idx)

    result = permutation_test(
        signal, forward_returns, n_permutations=300, seed=9, alternative="less"
    )
    manual_count = int(np.sum(result["null_distribution"] <= result["observed_ic"]))
    expected_p = (1 + manual_count) / (300 + 1)

    assert result["p_value"] == pytest.approx(expected_p)


def test_dummy_permutation_less_alternative_matches_manual_count():
    rng = np.random.default_rng(19)
    n = 300
    idx = pd.RangeIndex(n)
    x1 = pd.Series(rng.normal(0, 1, n), index=idx)
    dummy = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    y = pd.Series(rng.normal(0, 1, n), index=idx)

    result = permutation_test_interaction_coefficient(
        y, x1, dummy, n_permutations=200, seed=3, alternative="less"
    )
    manual_count = int(np.sum(result["null_distribution"] <= result["observed_b3"]))
    expected_p = (1 + manual_count) / (200 + 1)

    assert result["p_value"] == pytest.approx(expected_p)


def test_paired_sign_permutation_raises_on_empty_diffs():
    with pytest.raises(ValueError):
        paired_sign_permutation_test(np.array([]))


def test_paired_sign_permutation_raises_on_invalid_alternative():
    with pytest.raises(ValueError):
        paired_sign_permutation_test(np.array([0.1, -0.2, 0.3]), alternative="bogus")


def test_paired_sign_permutation_null_distribution_length():
    diffs = np.array([0.01, -0.02, 0.03, 0.015, -0.005, 0.02, -0.01])
    result = paired_sign_permutation_test(diffs, n_permutations=500, seed=1)

    assert len(result["null_distribution"]) == 500
    assert result["observed_mean_diff"] == pytest.approx(diffs.mean())


def test_paired_sign_permutation_reproducible_with_same_seed():
    diffs = np.array([0.02, -0.01, 0.015, 0.03, -0.02, 0.01, -0.005, 0.025])
    result_a = paired_sign_permutation_test(diffs, n_permutations=400, seed=42)
    result_b = paired_sign_permutation_test(diffs, n_permutations=400, seed=42)

    np.testing.assert_allclose(
        result_a["null_distribution"], result_b["null_distribution"]
    )


def test_paired_sign_permutation_large_positive_mean_is_significant():
    rng = np.random.default_rng(23)
    diffs = rng.normal(loc=1.0, scale=0.05, size=50)

    result = paired_sign_permutation_test(
        diffs, n_permutations=2000, seed=23, alternative="greater"
    )

    assert result["p_value"] < 0.01, (
        f"Expected a small p-value for a mean far from zero, got {result['p_value']:.4f}"
    )


def test_paired_sign_permutation_zero_centered_is_not_significant():
    rng = np.random.default_rng(29)
    diffs = rng.normal(loc=0.0, scale=1.0, size=50)

    result = paired_sign_permutation_test(
        diffs, n_permutations=2000, seed=29, alternative="two-sided"
    )

    assert result["p_value"] > 0.10, (
        f"Expected a non-significant p-value for noise centered at zero, "
        f"got {result['p_value']:.4f}"
    )


def test_paired_sign_permutation_less_alternative_matches_manual_count():
    rng = np.random.default_rng(31)
    diffs = rng.normal(loc=-0.5, scale=0.2, size=40)

    result = paired_sign_permutation_test(
        diffs, n_permutations=800, seed=31, alternative="less"
    )
    manual_count = int(np.sum(result["null_distribution"] <= result["observed_mean_diff"]))
    expected_p = (1 + manual_count) / (800 + 1)

    assert result["p_value"] == pytest.approx(expected_p)