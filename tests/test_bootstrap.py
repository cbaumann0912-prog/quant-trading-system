import numpy as np
import pytest
from src.evaluation.bootstrap import bootstrap_confidence_interval, block_bootstrap


def test_ci_contains_true_mean():
    rng = np.random.default_rng(28)
    true_mu = 5.0
    true_sigma = 2.0
    sample_size = 200
    n_trials = 200
    confidence = 0.95

    hits = 0
    for trial in range(n_trials):
        data = rng.normal(loc=true_mu, scale=true_sigma, size=sample_size)
        lower, upper = bootstrap_confidence_interval(
            data,
            statistic_fn=np.mean,
            n_bootstrap=1000,
            confidence=confidence,
        )
        if lower <= true_mu <= upper:
            hits += 1

    empirical_coverage = hits / n_trials

    se = np.sqrt(confidence * (1 - confidence) / n_trials)
    tolerance = 3 * se
    assert abs(empirical_coverage - confidence) <= tolerance, (
        f"Empirical coverage {empirical_coverage:.3f} outside "
        f"[{confidence - tolerance:.3f}, {confidence + tolerance:.3f}]"
    )


def test_output_is_tuple_of_two_floats():
    rng = np.random.default_rng(28)
    data = rng.normal(size=100)

    result = bootstrap_confidence_interval(
        data, statistic_fn=np.mean, n_bootstrap=500, confidence=0.95
    )

    assert len(result) == 2
    lower, upper = result
    assert isinstance(lower, (float, np.floating))
    assert isinstance(upper, (float, np.floating))


def test_lower_less_than_upper():
    rng = np.random.default_rng(28)
    data = rng.normal(size=100)

    lower, upper = bootstrap_confidence_interval(data, statistic_fn=np.mean, n_bootstrap=500, confidence=0.95)

    assert lower <= upper


def test_output_shape_matches_n_samples():
    series = np.arange(50).astype(float)
    n_samples = 237

    result = block_bootstrap(series, block_size=5, n_samples=n_samples, statistic_fn=np.mean, seed=1)

    assert result.shape == (n_samples)


def test_block_bootstrap_different_from_iid_bootstrap_on_autocorrelated_series():
    rng = np.random.default_rng(28)
    phi = 0.9
    sigma_eps = 1.0
    n = 500

    eps = rng.normal(scale=sigma_eps, size=n)
    series = np.empty(n)
    series[0] = eps[0]
    for t in range(1, n):
        series[t] = phi * series[t - 1] + eps[t]

    gamma_0 = sigma_eps**2 / (1 - phi**2)
    gamma = lambda k: gamma_0 * phi**k

    theoretical_var_xbar = gamma_0 / n
    for k in range(1, n):
        theoretical_var_xbar += (2 / n) * (1 - k / n) * gamma(k)

    n_samples = 2000

    iid_means = block_bootstrap(series, block_size=1, n_samples=n_samples, statistic_fn=np.mean, seed=99)
    block_means = block_bootstrap(series, block_size=10, n_samples=n_samples, statistic_fn=np.mean, seed=99)

    iid_empirical_var = np.var(iid_means)
    block_empirical_var = np.var(block_means)

    iid_error = abs(iid_empirical_var - theoretical_var_xbar)
    block_error = abs(block_empirical_var - theoretical_var_xbar)

    assert iid_empirical_var < theoretical_var_xbar, (
        f"Expected i.i.d. bootstrap to underestimate Var(x_bar) under "
        f"positive autocorrelation: got {iid_empirical_var:.6f} vs "
        f"theoretical {theoretical_var_xbar:.6f}"
    )
    assert block_error < iid_error, (
        f"Expected block bootstrap Var(x_bar) estimate ({block_empirical_var:.6f}, "
        f"error {block_error:.6f}) to be closer to the theoretical value "
        f"({theoretical_var_xbar:.6f}) than i.i.d.'s estimate "
        f"({iid_empirical_var:.6f}, error {iid_error:.6f})"
    )


def test_seed_reproducibility():
    rng = np.random.default_rng(28)
    series = rng.normal(size=100)

    result_1 = block_bootstrap(series, block_size=5, n_samples=300, statistic_fn=np.mean, seed=7)
    result_2 = block_bootstrap(series, block_size=5, n_samples=300, statistic_fn=np.mean, seed=7)

    assert np.array_equal(result_1, result_2)

    result_3 = block_bootstrap(series, block_size=5, n_samples=300, statistic_fn=np.mean, seed=8)

    assert not np.array_equal(result_1, result_3)