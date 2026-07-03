import pytest
import pandas as pd
import numpy as np
from src.analysis.performance_analyzer import PerformanceAnalyzer

POSITIVE_RETURNS = pd.Series(
    [0.01, 0.02, 0.01, 0.03, 0.01],
    index=pd.date_range(start="2026-05-28", periods=5, freq="D")
)

FLAT_RETURNS = pd.Series(
    [0.0, 0.0, 0.0, 0.0, 0.0],
    index=pd.date_range(start="2026-05-28", periods=5, freq="D")
)

class TestSharpe:
    def test_sharpe_positive_returns(self):
        result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).compute_sharpe() 
        assert result > 0

    def test_sharpe_zero_rf(self):
        analyzer = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None)
        result = analyzer.compute_sharpe()
        ann_factor = analyzer.compute_ann_factor()
        expected = (POSITIVE_RETURNS.mean() / POSITIVE_RETURNS.std()) * np.sqrt(ann_factor)
    
        assert result == expected

class TestMaxDrawdown:
    def test_max_drawdown_returns_dict_with_value_and_duration(self):
        result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).compute_max_drawdown() 
        assert "value" in result
        assert "duration_days" in result
        assert "start_date" in result
        assert "end_date" in result

    def test_max_drawdown_flat_series(self):
        result = PerformanceAnalyzer(returns=FLAT_RETURNS, trades=None).compute_max_drawdown()
        assert result["value"] == 0

class TestDeflatedSharpeRatio:
    def test_dsr_between_0_and_1(self):
        result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=1.5,
        n_trials=10,
        n_obs=312,
        skewness=0.0,
        kurtosis=3.0
        )
        assert 0 <= result <= 1

    def test_more_trials_lowers_dsr(self):
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

    def test_deflated_lower_than_observed(self):
        observed = 1.5
        dsr_deflated  = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
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

        assert dsr_baseline  > dsr_deflated
