# Day 27 Audit — PerformanceAnalyzer First Run

## Methodology
Ran PerformanceAnalyzer.run_report() on raw daily log returns (close-to-close, resampled from 1-min OHLCV) for EUR/USD, GBP/USD, USD/JPY, 2011-01-03 to 2026-03-31, trades=None (mechanics test, not a strategy verdict).

## Findings

| Pair    | Sharpe | Sortino | Max DD  | Calmar  | t-stat | Ann. Return | Ann. Vol |
|---------|--------|---------|---------|---------|--------|-------------|----------|
| EUR/USD | -0.112 | -0.112  | -0.381  | -0.024  | -0.439 | -0.91%      | 8.15%    |
| GBP/USD | -0.118 | -0.112  | -0.412  | -0.025  | -0.461 | -1.05%      | 8.94%    |
| USD/JPY | 0.472  | 0.458   | -0.210  | 0.214   | 1.844  | 4.50%       | 9.31%    |

win_rate and profit_factor returned NaN for all three, as expected (trades=None). No crashes, no inf values.

## Interpretation
The test passes cleanly. The PerformanceAnalyzer handles resampled FX data without crashes, NaNs outside expected fields (win_rate, profit_factor), or instability in the return-based metrics, so the pipeline is behaving correctly.

USD/JPY’s higher Sharpe and t-stat likely reflect a persistent macro-driven drift from interest rate divergence over the sample period, while EUR/USD and GBP/USD are more dominated by alternating regime cycles (risk-on/risk-off and regional shocks), which wash out returns and push their long-run drift slightly negative.