# Week 1 Review — Probability Foundations (Days 1–7)

## Methodology
Distributions (normal, log-normal, Student-t) implemented in `src/stats/distributions.py`; portfolio stats in `src/analysis/portfolio_stats.py`; Sharpe ratio per Sharpe (1994) in `src/analysis/performance_analyzer.py`; empirical return distribution analysis on EUR/USD, GBP/USD, USD/JPY.

## Findings
- All three forex pairs reject normality; empirical excess kurtosis confirms fat tails beyond what normal distribution predicts.
- Sharpe (1994) requires excess returns over risk-free rate, not raw returns; annualization factor is derived empirically from each series' own index (~312 obs/year on this daily data).
- CLT justifies t-tests on mean returns despite non-normal daily returns; LLN implications for small-sample backtest reliability.
- Gap identified: regression/beta mechanics from Day 3 momentum paper not yet understood — put off for a later date.

## Interpretation
The core result this week is that forex returns don't behave the way the normal distribution assumes — excess kurtosis showed up empirically across all three pairs, not just in theory, which means Student-t is the more honest baseline for anything built on top of this. Sharpe (1994) done properly means excess returns over risk-free, annualized on the right convention — details that change the number, not just the presentation. The regression/beta gap from Day 3 is still open.