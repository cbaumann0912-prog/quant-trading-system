import numpy as np
import pandas as pd
import pytest

from src.analysis.portfolio import (
    efficient_frontier,
    markowitz_weights,
    minimum_variance_portfolio,
    leverage_bounded_return_range,
    _solve_affine_weight_coefficients,
)

np.random.seed(28)

RETURNS = pd.DataFrame(
    np.random.normal(
        loc=[0.0010, 0.0015, 0.0008],
        scale=[0.0100, 0.0120, 0.0090],
        size=(1000, 3),
    ),
    columns=["A", "B", "C"],
)

TARGET_RETURN = RETURNS.mean().mean()


@pytest.fixture
def sample_returns():
    # FIX: this fixture was referenced by every new Day 33 test below but
    # never defined -- none of those tests could be collected without it.
    return RETURNS


@pytest.fixture
def result():
    return markowitz_weights(
        RETURNS,
        target_return=TARGET_RETURN,
        allow_short=True,
    )


def test_weights_sum_to_one(result):
    assert np.isclose(result["weights"].sum(), 1.0, atol=1e-10)


def test_portfolio_return_near_target(result):
    assert np.isclose(result["portfolio_return"], TARGET_RETURN, atol=1e-10)


def test_stationarity_condition(result):
    x = result["weights"]
    p_bar = RETURNS.mean().to_numpy()
    sigma = RETURNS.cov().to_numpy()
    ones = np.ones(len(x))

    target_vec = 2 * sigma @ x
    basis = np.column_stack([p_bar, -ones])  # columns: p_bar, -1

    coeffs, residuals, rank, _ = np.linalg.lstsq(basis, target_vec, rcond=None)
    reconstructed = basis @ coeffs

    assert np.allclose(reconstructed, target_vec, atol=1e-8), (
        "2*Sigma*x does not lie in span{p_bar, 1} -- stationarity violated."
    )


def test_variance_positive_constructed_psd():
    rng = np.random.default_rng(28)
    M = rng.normal(size=(3, 3))
    sigma_constructed = M @ M.T

    constructed_returns = pd.DataFrame(
        rng.multivariate_normal(
            mean=[0.001, 0.0015, 0.0008],
            cov=sigma_constructed,
            size=5000,
        ),
        columns=["A", "B", "C"],
    )

    target = constructed_returns.mean().mean()
    res = markowitz_weights(constructed_returns, target_return=target, allow_short=True)

    assert res["portfolio_variance"] > 0


def test_allow_short_false_raises():
    with pytest.raises(NotImplementedError):
        markowitz_weights(RETURNS, target_return=TARGET_RETURN, allow_short=False)


@pytest.mark.skip(reason="Known gap: no conditioning check on near-singular Sigma yet (see TODO in portfolio.py)")
def test_near_singular_sigma_conditioning():
    pass


def test_frontier_length_matches_n_points(sample_returns):
    n_points = 25

    result = efficient_frontier(sample_returns, n_points=n_points)

    assert len(result["returns"]) == n_points
    assert len(result["volatilities"]) == n_points
    assert len(result["sharpes"]) == n_points
    assert result["weights"].shape == (n_points, sample_returns.shape[1])


def test_frontier_volatilities_positive(sample_returns):
    result = efficient_frontier(sample_returns, n_points=20)

    vols = result["volatilities"]

    assert np.all(np.isfinite(vols))
    assert np.all(vols > 0)


def test_frontier_returns_monotonic_and_linspace(sample_returns):
    n_points = 30
    result = efficient_frontier(sample_returns, n_points=n_points)

    r = result["returns"]

    assert np.all(np.diff(r) > 0)  # strictly increasing

    # must match internal linspace construction
    p_bar = sample_returns.mean().to_numpy()
    r_min, r_max = p_bar.min(), p_bar.max()

    expected = np.linspace(r_min, r_max, n_points)
    np.testing.assert_allclose(r, expected)


def test_frontier_weights_sum_to_one(sample_returns):
    result = efficient_frontier(sample_returns, n_points=20)

    weights = result["weights"]

    row_sums = weights.sum(axis=1)

    np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), atol=1e-10)


def test_leverage_bounds_respect_caps(sample_returns):
    caps = np.array([10.0] * sample_returns.shape[1])

    r_min, r_max = leverage_bounded_return_range(sample_returns, caps)

    a, b = _solve_affine_weight_coefficients(sample_returns)

    for r in [r_min, r_max]:
        w = a + b * r
        assert np.all(np.abs(w) <= caps + 1e-8)


def test_leverage_bounds_infeasible_raises(sample_returns):
    caps = np.array([1e-12] * sample_returns.shape[1])  # absurdly tight

    with pytest.raises(ValueError):
        leverage_bounded_return_range(sample_returns, caps)


def test_leverage_bounds_match_manual_affine_intersection(sample_returns):
    caps = np.array([5.0, 5.0, 5.0])

    expected_r_min = -0.0017381386295733101
    expected_r_max = 0.0031198179061657123

    r_min, r_max = leverage_bounded_return_range(sample_returns, caps)

    np.testing.assert_allclose(r_min, expected_r_min, atol=1e-8)
    np.testing.assert_allclose(r_max, expected_r_max, atol=1e-8)


def test_min_variance_lowest_vol(sample_returns):
    mv = minimum_variance_portfolio(sample_returns)

    frontier = efficient_frontier(sample_returns, n_points=50)

    frontier_vars = frontier["volatilities"] ** 2

    assert mv["variance"] <= np.min(frontier_vars) + 1e-8


def test_min_variance_weights_sum_to_one(sample_returns):
    mv = minimum_variance_portfolio(sample_returns)

    np.testing.assert_allclose(mv["weights"].sum(), 1.0, atol=1e-10)


def test_affine_coefficients_reconstruct_markowitz_weights(sample_returns):
    a, b = _solve_affine_weight_coefficients(sample_returns)

    p_bar = sample_returns.mean().to_numpy()

    # use same implied feasible range as frontier
    r_min, r_max = p_bar.min(), p_bar.max()

    for r in np.linspace(r_min, r_max, 7):
        expected = markowitz_weights(sample_returns, r, allow_short=True)["weights"]
        actual = a + b * r

        np.testing.assert_allclose(actual, expected, atol=1e-10)