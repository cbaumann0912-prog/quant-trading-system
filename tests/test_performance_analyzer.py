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
        result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).compute_sharpe()
        expected = (POSITIVE_RETURNS.mean() / POSITIVE_RETURNS.std()) * np.sqrt(252)
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
