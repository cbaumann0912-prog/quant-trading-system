import numpy as np
import pandas as pd
import pytest

from src.analysis.portfolio import (
    efficient_frontier,
    markowitz_weights,
    minimum_variance_portfolio,
    leverage_bounded_return_range,
    _solve_affine_weight_coefficients,
    risk_parity_weights,
    _markowitz_weights_closed_form,
    kelly_fraction,
    fractional_kelly
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

@pytest.fixture
def sample_returns():
    return RETURNS

TARGET_RETURN = RETURNS.mean().mean()


DATA_DIR = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"

FILES = {
    "EURUSD": "EURUSD.csv",
    "GBPUSD": "GBPUSD.csv",
    "USDJPY": "USDJPY.csv",
}


@pytest.fixture
def fx_returns():
    pairs = {}
    for pair_name, filename in FILES.items():
        path = f"{DATA_DIR}\\{filename}"
        df = pd.read_csv(path)
        df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
        df = df.set_index("Datetime")

        daily_close = df["Close"].resample("D").last().dropna()
        log_returns = np.log(daily_close / daily_close.shift(1)).dropna()

        pairs[pair_name] = log_returns

    returns = pd.concat(pairs, axis=1, join="inner")
    returns.columns = list(FILES.keys())
    return returns


@pytest.fixture
def target_return_forces_short(fx_returns):
    from src.analysis.portfolio import _markowitz_weights_closed_form
    p_bar = fx_returns.mean().to_numpy()
    r_min, r_max = p_bar.min(), p_bar.max()

    for r in (r_max - 1e-6, r_min + 1e-6):
        w = _markowitz_weights_closed_form(fx_returns, r)["weights"]
        if np.any(w < 0):
            return r

    raise RuntimeError("Neither range endpoint produces a short position — check fx_returns data.")


@pytest.fixture
def result():
    return markowitz_weights(
        RETURNS,
        target_return=TARGET_RETURN,
        ann_factor=252.0,
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
    basis = np.column_stack([p_bar, -ones])

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
    res = markowitz_weights(constructed_returns, target_return=target, ann_factor=252.0, allow_short=True)

    assert res["portfolio_variance"] > 0


@pytest.mark.skip(reason="Known gap: no conditioning check on near-singular Sigma yet (see TODO in portfolio.py)")
def test_near_singular_sigma_conditioning():
    pass


def test_frontier_length_matches_n_points(sample_returns):
    n_points = 25
    result = efficient_frontier(sample_returns, ann_factor=252.0, n_points=n_points)

    assert len(result["returns"]) == n_points
    assert len(result["volatilities"]) == n_points
    assert len(result["sharpes"]) == n_points
    assert result["weights"].shape == (n_points, sample_returns.shape[1])


def test_frontier_volatilities_positive(sample_returns):
    result = efficient_frontier(sample_returns, ann_factor=252.0, n_points=20)
    vols = result["volatilities"]

    assert np.all(np.isfinite(vols))
    assert np.all(vols > 0)


def test_frontier_returns_monotonic_and_linspace(sample_returns):
    n_points = 30
    result = efficient_frontier(sample_returns, ann_factor=252.0, n_points=n_points)
    r = result["returns"]

    assert np.all(np.diff(r) > 0) 

    p_bar = sample_returns.mean().to_numpy()
    r_min, r_max = p_bar.min(), p_bar.max()

    expected = np.linspace(r_min, r_max, n_points)
    np.testing.assert_allclose(r, expected)


def test_frontier_weights_sum_to_one(sample_returns):
    result = efficient_frontier(sample_returns, ann_factor=252.0, n_points=20)

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
    caps = np.array([1e-12] * sample_returns.shape[1])

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

    frontier = efficient_frontier(sample_returns, ann_factor=252.0, n_points=50)

    frontier_vars = frontier["volatilities"] ** 2

    assert mv["variance"] <= np.min(frontier_vars) + 1e-8


def test_min_variance_weights_sum_to_one(sample_returns):
    mv = minimum_variance_portfolio(sample_returns)

    np.testing.assert_allclose(mv["weights"].sum(), 1.0, atol=1e-10)


def test_affine_coefficients_reconstruct_markowitz_weights(sample_returns):
    a, b = _solve_affine_weight_coefficients(sample_returns)

    p_bar = sample_returns.mean().to_numpy()

    r_min, r_max = p_bar.min(), p_bar.max()

    for r in np.linspace(r_min, r_max, 7):
        expected = markowitz_weights(sample_returns, r, ann_factor=252.0, allow_short=True)["weights"]
        actual = a + b * r

        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_risk_parity_weights_sum_to_one():
    result = risk_parity_weights(RETURNS, ann_factor=252.0)

    assert np.isclose(result["weights"].sum(), 1.0, atol=1e-8)


def test_risk_parity_contributions_equal():
    result = risk_parity_weights(RETURNS, ann_factor=252.0)
    rc = result["risk_contributions"]

    assert np.max(rc) - np.min(rc) < 1e-6


def test_risk_parity_low_vol_gets_higher_weight():
    rng = np.random.default_rng(28)
    RETURNS_VOL = pd.DataFrame({
        "low":  rng.normal(0, 0.005, 1000),
        "high": rng.normal(0, 0.030, 1000),
    })
    result = risk_parity_weights(RETURNS_VOL, ann_factor=252.0)
    assert result["weights"][0] > result["weights"][1]


def test_scipy_matches_numpy_result(fx_returns, target_return_forces_short):
    scipy_result = markowitz_weights(fx_returns, target_return_forces_short, ann_factor=252.0, allow_short=True)
    closed_form_result = _markowitz_weights_closed_form(fx_returns, target_return_forces_short)

    assert np.allclose(
        scipy_result["weights"],
        closed_form_result["weights"],
        atol=1e-4,
    )


def test_long_only_no_negative_weights(fx_returns, target_return_forces_short):
    result = markowitz_weights(fx_returns, target_return_forces_short, ann_factor=252.0, allow_short=False)

    assert np.all(result["weights"] >= -1e-6)


def test_long_only_diverges_from_unconstrained(fx_returns, target_return_forces_short):
    closed_form_result = _markowitz_weights_closed_form(fx_returns, target_return_forces_short)

    assert np.any(closed_form_result["weights"] < 0)


def test_kelly_positive_edge():
    mu = 0.10
    sigma = 0.20
    expected = mu / sigma**2

    assert kelly_fraction(mu, sigma) == pytest.approx(expected)


def test_kelly_zero_edge_returns_zero():
    result = kelly_fraction(mu=0.0, sigma=0.15)

    assert result == pytest.approx(0.0)


def test_kelly_negative_sigma_raises():
    with pytest.raises(ValueError):
        kelly_fraction(mu=0.10, sigma=0.0)
    with pytest.raises(ValueError):
        kelly_fraction(mu=0.10, sigma=-0.05)


def test_fractional_less_than_full():
    mu = 0.12
    sigma = 0.18
    full = kelly_fraction(mu, sigma)

    for f in (0.1, 0.25, 0.5, 0.75, 0.99):
        assert fractional_kelly(mu, sigma, fraction=f) < full


def test_fractional_invalid_fraction_raises():
    mu, sigma = 0.10, 0.20

    with pytest.raises(ValueError):
        fractional_kelly(mu, sigma, fraction=0.0)
    with pytest.raises(ValueError):
        fractional_kelly(mu, sigma, fraction=-0.5)
    with pytest.raises(ValueError):
        fractional_kelly(mu, sigma, fraction=1.5)
    fractional_kelly(mu, sigma, fraction=1.0)


def test_fractional_reuses_kelly_fraction():
    mu = 0.10
    sigma = 0.20
    fraction = 0.25
    expected = 0.625
    
    assert fractional_kelly(mu, sigma, fraction) == pytest.approx(expected)
    assert fractional_kelly(mu, sigma, fraction) == pytest.approx(
        fraction * kelly_fraction(mu, sigma)
    )