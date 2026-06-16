import numpy as np
import pandas as pd

from src.data.time_series import fit_arima, select_arima_order

SEED = 28


def _white_noise(n: int = 500) -> pd.Series:
    rng = np.random.default_rng(SEED)
    return pd.Series(rng.standard_normal(n))


def _ar1(phi: float, n: int = 1000) -> pd.Series:
    rng = np.random.default_rng(SEED)
    burn = 100
    y = np.zeros(n + burn)
    eps = rng.standard_normal(n + burn)
    for t in range(1, n + burn):
        y[t] = phi * y[t - 1] + eps[t]
    return pd.Series(y[burn:])


def test_fit_returns_dict_with_aic():
    result = fit_arima(_white_noise(), order=(1, 0, 0))

    assert isinstance(result, dict)
    assert "aic" in result
    assert np.isfinite(result["aic"])


def test_residuals_length_matches_series():
    series = _white_noise(n=300)
    result = fit_arima(series, order=(1, 0, 0))

    assert len(result["residuals"]) == len(series)


def test_best_order_for_ar1_is_1_0_0():
    series = _ar1(phi=0.8, n=2000)
    order = select_arima_order(series, max_p=3, max_q=2, d=0, criterion="bic")
    
    assert order == (1, 0, 0)