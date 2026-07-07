import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

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


def test_fit_arima_raises_on_nan():
    series_with_nan = _white_noise().copy()
    series_with_nan.iloc[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_arima(series_with_nan, order=(1, 0, 0))


def test_fit_arima_raises_on_too_short_series():
    short_series = pd.Series([0.1, 0.2, 0.15])
    with pytest.raises(ValueError, match="too short"):
        fit_arima(short_series, order=(5, 0, 5))


def test_fit_arima_wraps_convergence_failure():
    series = _white_noise()
    with patch("src.data.time_series.ARIMA") as mock_arima:
        mock_arima.return_value.fit.side_effect = Exception("did not converge")
        with pytest.raises(RuntimeError, match="fitting failed"):
            fit_arima(series, order=(1, 0, 0))


def test_select_arima_order_raises_on_invalid_criterion():
    series = _white_noise()
    with pytest.raises(ValueError, match="criterion"):
        select_arima_order(series, max_p=1, max_q=1, criterion="bogus")


def test_select_arima_order_skips_failed_combinations(capsys):
    series = _white_noise()
    real_fit_arima = fit_arima

    def flaky_fit(s, order):
        p, d, q = order
        if p == 1 and q == 1:
            raise RuntimeError("forced failure")
        return real_fit_arima(s, order=order)

    with patch("src.data.time_series.fit_arima", side_effect=flaky_fit):
        result = select_arima_order(series, max_p=1, max_q=1)

    captured = capsys.readouterr()
    assert "failed to converge" in captured.out
    assert result != (1, 0, 1)


def test_select_arima_order_raises_when_all_combinations_fail():
    series = _white_noise()
    with patch(
        "src.data.time_series.fit_arima",
        side_effect=RuntimeError("forced failure"),
    ):
        with pytest.raises(RuntimeError):
            select_arima_order(series, max_p=1, max_q=1)