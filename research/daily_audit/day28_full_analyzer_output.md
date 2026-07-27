# Day 28 — Full Analyzer Run, All Pairs (Raw Data)

## Methodology
`PerformanceAnalyzer.run_report()` executed on raw daily log returns forEUR/USD, GBP/USD, USD/JPY (2011-01-03 to 2023-12-29, n=312.27 obs/year empirical ann_factor). No trade log and no benchmark supplied — these are raw asset return diagnostics, not strategy performance; no signal exists yet `win_rate`, `profit_factor`, and `tracking_error` are NaN by design as a result.

## Findings

| Metric | EUR/USD | GBP/USD | USD/JPY |
|---|---|---|---|
| Sharpe (annualized) | -0.1719 | -0.1660 | 0.4620 |
| Sortino | -0.1695 | -0.1570 | 0.4543 |
| Max Drawdown | -38.15% | -41.21% | -20.99% |
| Calmar | -0.0373 | -0.0370 | 0.2065 |
| t-stat (mean return = 0) | -0.6195 | -0.5981 | 1.6648 |
| Deflated Sharpe (n_trials=1) | 0.2677 | 0.2733 | 0.9520 |
| Annualized Return | -1.42% | -1.52% | 4.34% |
| Annualized Vol | 8.34% | 9.26% | 9.19% |
| Skewness | -0.0781 | -1.7896 | 0.0467 |
| Excess Kurtosis | 3.3025 | 32.4807 | 7.3796 |
| Jarque-Bera stat | 1846.87 | 180370.47 | 9202.80 |
| JB p-value | 0.0000 | 0.0000 | 0.0000 |
| Ljung-Box stat (lag=10) | 5.3683 | 14.3941 | 16.7395 |
| LB p-value | 0.8653 | 0.1558 | 0.0803 |

## Interpretation
EUR/USD and GBP/USD posted slightly negative risk-adjusted performance over the sample. USD/JPY posted a positive Sharpe ratio under 0.5, with a t-stat of 1.67 — below the conventional two-tailed significance threshold. None of the three pairs show a statistically significant mean return, which fits the expectation that raw currency exposure carries no built-in directional edge.

All three series reject normality under Jarque-Bera, driven by excess kurtosis. GBP/USD's kurtosis (32.48) is an order of magnitude above the other two pairs, traced to a cluster of known crisis dates. Ljung-Box fails to reject at lag 10 for EUR/USD and GBP/USD, so those series show no exploitable linear autocorrelation at this lag; USD/JPY at p = 0.0803 clears the 5% bar but not the 10% one and should not be called clean.