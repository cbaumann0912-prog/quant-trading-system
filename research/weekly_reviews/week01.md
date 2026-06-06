# Day 7 — Week 1 Review: Probability Foundations

**Date:** 2026-05-28

## Question Investigated

Week 1 covered the probability foundations underlying return modeling: normal and log-normal distributions, the Central Limit Theorem, Student-t tails, expectation and variance, covariance structure across assets, and the Sharpe ratio as a first performance metric. The implicit question running through all of it is whether daily returns on forex pairs behave the way standard statistical models assume they do.

## Why It Matters

Every test, metric, and model built through the rest of this program rests on assumptions about the distribution of returns. If those assumptions aren't understood from first principles then the outputs of any analysis can't be properly interpreted. The Sharpe ratio is meaningless if you don't know what it assumes. Hypothesis tests are meaningless if you don't know what distribution the test statistic follows.

## Methodology

Each distribution was implemented from scratch in `src/stats/distributions.py` before being used in any applied context. Portfolio statistics were derived from the covariance matrix directly in `src/analysis/portfolio_stats.py`. The Sharpe ratio was implemented in `src/analysis/performance_analyzer.py` following Sharpe (1994) rather than the common simplified form. Return distributions for EUR/USD, GBP/USD, and USD/JPY were analyzed empirically and documented in `research/audit/day04_return_distribution_analysis.md`.

## Assumptions

The normal distribution assumption for daily returns is violated in practice. The Day 4 empirical analysis showed excess kurtosis across all three pairs, meaning the tails are fatter than a normal distribution predicts. The Student-t distribution fits better but introduces a degrees-of-freedom parameter that itself requires estimation. The Sharpe ratio as implemented assumes returns are normally distributed and serially uncorrelated. Both assumptions are likely false for intraday forex returns.

## Findings

**Distributions.** Normal and log-normal are locked in. The distinction between modeling returns as normal versus prices as log-normal is clear. Student-t accounts for fat tails by adding the degrees-of-freedom parameter, which controls how heavy the tails are. The empirical return distributions across all three pairs showed excess kurtosis — the fat-tail problem is real, not theoretical.

**CLT and LLN.** The CLT explains why sample means converge to a normal distribution regardless of the underlying return distribution, which is the justification for t-tests on mean returns even when daily returns themselves aren't normal. The LLN explains why larger samples produce more reliable estimates. Both laws have direct implications for how seriously to take backtest statistics computed on small trade samples.

**Sharpe ratio.** The Sharpe ratio measures return per unit of total volatility. The 1994 Sharpe paper clarifies that the ratio should use excess returns over the risk-free rate, not raw returns. The annualization factor of 252 assumes trading days, not calendar days. Both details matter for producing numbers that are comparable across strategies.

**Python gaps.** NumPy and pandas were new this week. The mathematical implementations came faster than the library syntax. Logging functions remain fuzzy. The gap isn't in understanding what needs to be computed — it's in translating that into idiomatic Python quickly enough to not break momentum during a build session.

## Alternative Explanations

The Student-t fit to forex returns assumes the tail behavior is stationary — that the degree of fat-tailedness stays constant across the sample period. If tail behavior is regime-dependent, a single fitted Student-t understates tail risk during the periods that matter most. The Sharpe ratio computed this week treats volatility as a symmetric measure of risk, but for a strategy with negative skew like FVG_BoS_Reversal, downside deviation is more relevant than total standard deviation.

## Open Questions

The Day 3 momentum paper introduced the concept of regressing returns on market beta to extract market-adjusted alphas. That methodology was opaque at the time of reading. It's the one topic from Week 1 that isn't locked in — the mechanics of what regression is doing, why beta is estimated by regressing returns rather than read from a formula, and what "market-adjusted" means in a forex context where there is no obvious market portfolio. Week 3 regression work addresses this directly, but the gap is worth naming now because it sits underneath a significant portion of the academic literature on systematic strategies.