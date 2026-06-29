import numpy as np
import pandas as pd
import pytest

from src.analysis.portfolio import markowitz_weights

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