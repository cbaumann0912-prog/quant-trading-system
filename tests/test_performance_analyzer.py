import pytest
import pandas as pd
import numpy as np
from src.analysis.performance_analyzer import (
    PerformanceAnalyzer,
    information_coefficient,
    information_ratio,
)

POSITIVE_RETURNS = pd.Series(
    [0.01, 0.02, 0.01, 0.03, 0.01],
    index=pd.date_range(start="2026-05-28", periods=5, freq="D")
)

FLAT_RETURNS = pd.Series(
    [0.0, 0.0, 0.0, 0.0, 0.0],
    index=pd.date_range(start="2026-05-28", periods=5, freq="D")
)

CONSTANT_NONZERO_RETURNS = pd.Series(
    [1 / 3] * 50,
    index=pd.date_range(start="2026-05-28", periods=50, freq="D")
)


def test_sharpe_positive_returns():
    result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).compute_sharpe()
    
    assert result > 0


def test_sharpe_zero_rf():
    analyzer = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None)
    result = analyzer.compute_sharpe()
    ann_factor = analyzer.compute_ann_factor()
    expected = (POSITIVE_RETURNS.mean() / POSITIVE_RETURNS.std()) * np.sqrt(ann_factor)
    
    assert result == expected


def test_sharpe_constant_nonzero_returns_is_nan():
    result = PerformanceAnalyzer(returns=CONSTANT_NONZERO_RETURNS, trades=None).compute_sharpe()
    
    assert np.isnan(result)


def test_max_drawdown_returns_dict_with_value_and_duration():
    result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).compute_max_drawdown()
    
    assert "value" in result
    assert "duration_days" in result
    assert "start_date" in result
    assert "end_date" in result


def test_max_drawdown_flat_series():
    result = PerformanceAnalyzer(returns=FLAT_RETURNS, trades=None).compute_max_drawdown()
    
    assert result["value"] == 0


def test_dsr_between_0_and_1():
    result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=1.5,
        n_trials=10,
        n_obs=312,
        skewness=0.0,
        kurtosis=3.0
    )
    
    assert 0 <= result <= 1


def test_more_trials_lowers_dsr():
    few_trials = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=1.5,
        n_trials=2,
        n_obs=50,
        skewness=0.0,
        kurtosis=3.0
    )
    many_trials = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=1.5,
        n_trials=50,
        n_obs=50,
        skewness=0.0,
        kurtosis=3.0
    )
    
    assert many_trials < few_trials


def test_deflated_lower_than_observed():
    observed = 1.5
    dsr_deflated = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=observed,
        n_trials=50,
        n_obs=50,
        skewness=0.0,
        kurtosis=3.0
    )
    dsr_baseline = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=observed,
        n_trials=2,
        n_obs=50,
        skewness=0.0,
        kurtosis=3.0
    )
    
    assert dsr_baseline > dsr_deflated


def test_t_stat_constant_nonzero_returns_is_nan():
    result = PerformanceAnalyzer(returns=CONSTANT_NONZERO_RETURNS, trades=None).compute_t_stat()
    
    assert np.isnan(result)


def test_ic_perfect_monotonic_signal_equals_one():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    signal = pd.Series(np.arange(10), index=idx)
    forward_returns = pd.Series(np.arange(10) * 0.01, index=idx)
    result = information_coefficient(signal, forward_returns)
    
    assert result == pytest.approx(1.0)


def test_ic_perfect_inverse_monotonic_signal_equals_negative_one():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    signal = pd.Series(np.arange(10), index=idx)
    forward_returns = pd.Series(-np.arange(10) * 0.01, index=idx)
    result = information_coefficient(signal, forward_returns)
    
    assert result == pytest.approx(-1.0)


def test_ic_spearman_vs_pearson_differ_on_nonlinear_monotonic_relationship():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    signal = pd.Series(np.arange(1, 21), index=idx)
    forward_returns = pd.Series((np.arange(1, 21).astype(float)) ** 3, index=idx)
    spearman_ic = information_coefficient(signal, forward_returns, method="spearman")
    pearson_ic = information_coefficient(signal, forward_returns, method="pearson")
    
    assert spearman_ic == pytest.approx(1.0)
    assert pearson_ic < spearman_ic


def test_ic_random_signal_near_zero():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=2000, freq="D")
    signal = pd.Series(rng.standard_normal(2000), index=idx)
    forward_returns = pd.Series(rng.standard_normal(2000), index=idx)
    result = information_coefficient(signal, forward_returns)
    
    assert abs(result) < 0.05


def test_ic_handles_misaligned_index_via_intersection():
    idx_signal = pd.date_range("2020-01-01", periods=10, freq="D")
    idx_returns = pd.date_range("2020-01-05", periods=10, freq="D")
    signal = pd.Series(np.arange(10), index=idx_signal)
    forward_returns = pd.Series(np.arange(10) * 0.01, index=idx_returns)
    result = information_coefficient(signal, forward_returns)
    
    assert not np.isnan(result)


def test_ic_invalid_method_raises():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    signal = pd.Series(np.arange(10), index=idx)
    forward_returns = pd.Series(np.arange(10) * 0.01, index=idx)
    with pytest.raises(ValueError):
        information_coefficient(signal, forward_returns, method="kendall")


def test_ir_fundamental_law_matches_hand_calc_case():
    result = information_ratio(0.03, method="fundamental_law", breadth=100)
    
    assert result == pytest.approx(0.03 * np.sqrt(100))


def test_ir_fundamental_law_missing_breadth_raises():
    with pytest.raises(ValueError):
        information_ratio(0.05, method="fundamental_law")


def test_ir_fundamental_law_nonscalar_ic_raises():
    ic_series = pd.Series([0.01, 0.02, 0.03])
    with pytest.raises(ValueError):
        information_ratio(ic_series, method="fundamental_law", breadth=50)


def test_ir_empirical_matches_hand_calc():
    ic_series = pd.Series([0.02, 0.04, -0.01, 0.03, 0.05])
    expected = ic_series.mean() / ic_series.std(ddof=1)
    result = information_ratio(ic_series, method="empirical")
    
    assert result == pytest.approx(expected)


def test_ir_empirical_single_observation_is_nan():
    result = information_ratio(pd.Series([0.05]), method="empirical")
    
    assert np.isnan(result)


def test_ir_empirical_constant_series_is_nan():
    result = information_ratio(pd.Series([0.05, 0.05, 0.05]), method="empirical")
    
    assert np.isnan(result)


def test_ir_invalid_method_raises():
    with pytest.raises(ValueError):
        information_ratio(0.05, method="not_a_real_method", breadth=50)