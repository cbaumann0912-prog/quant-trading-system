# Day 28 — Full Analyzer Run, All Pairs (Raw Data)

## Methodology
`PerformanceAnalyzer.run_report()` executed on raw daily log returns forEUR/USD, GBP/USD, USD/JPY (2011-01-03 to 2026-03-31, n=312.29 obs/year empirical ann_factor). No trade log and no benchmark supplied — these are raw asset return diagnostics, not strategy performance; no signal exists yet `win_rate`, `profit_factor`, and `tracking_error` are NaN by design as a result.

## Findings

| Metric | EUR/USD | GBP/USD | USD/JPY |
|---|---|---|---|
| Sharpe (annualized) | -0.1124 | -0.1181 | 0.4725 |
| Sortino | -0.1116 | -0.1116 | 0.4577 |
| Max Drawdown | -38.15% | -41.21% | -20.99% |
| Calmar | -0.0239 | -0.0255 | 0.2142 |
| t-stat (mean return = 0) | -0.4388 | -0.4609 | 1.8444 |
| Deflated Sharpe (n_trials=1) | 0.3304 | 0.3215 | 0.9672 |
| Annualized Return | -0.91% | -1.05% | 4.50% |
| Annualized Vol | 8.15% | 8.94% | 9.31% |
| Skewness | 0.0031 | -1.7118 | -0.0792 |
| Excess Kurtosis | 3.7177 | 32.0399 | 6.4539 |
| Jarque-Bera stat | 2740.63 | 205838.31 | 8264.36 |
| JB p-value | 0.0000 | 0.0000 | 0.0000 |
| Ljung-Box stat (lag=10) | 4.6200 | 15.0592 | 9.9547 |
| LB p-value | 0.9151 | 0.1299 | 0.4445 |

## Interpretation
EUR/USD and GBP/USD posted slightly negative risk-adjusted performance over the sample. USD/JPY posted a positive Sharpe ratio under 0.5, with a t-stat of 1.84 — close to but below the conventional two-tailed significance threshold. None of the three pairs show a statistically significant mean return, which fits the expectation that raw currency exposure carries no built-in directional edge.

All three series reject normality under Jarque-Bera, driven by excess kurtosis. GBP/USD's kurtosis (32.04) is an order of magnitude above the other two pairs, traced to a cluster of known crisis dates, so daily returns show no exploitable linear autocorrelation at this lag.