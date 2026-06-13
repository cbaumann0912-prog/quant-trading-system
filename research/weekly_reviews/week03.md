# Day 21 — Week 3 Review: Regression, Linear Algebra & Stationarity

## Question Investigated
Week 3 built the regression and linear algebra tools needed to characterize the EUR/USD ~ GBP/USD relationship and the factor structure across all three pairs. Given the PCA results, should all three pairs be traded independently, or is there redundancy that has to be accounted for in portfolio construction?

## Why It Matters
OLS on currency returns is the first step of Engle-Granger cointegration testing, which determines whether a spread can mean-revert at all. PCA on the same three pairs determines whether treating them as three independent strategies is statistically defensible or whether the portfolio is one leveraged USD bet dressed as three positions.

## Methodology
Six modules were built and applied: `fit_ols` (normal equations, applied to EUR/USD ~ GBP/USD log returns and log price levels), `residual_diagnostics` (six diagnostics on OLS residuals), `fit_ridge` (next-day EUR/USD return prediction from five lagged returns at λ = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]), `eigen_decomposition` and `pca` (3-pair covariance matrix on daily log returns), and `check_stationarity` combining `adf_test` and `kpss_test` (applied to log price levels and OLS residuals for all pairwise combinations, with Benjamini-Hochberg correction across residual ADF p-values).

All data spans 2011-01-02 to 2026-03-31. PCA was run on the full sample as exploratory analysis — loadings will be re-estimated on rolling windows during signal construction.

## Assumptions
OLS assumes linearity, homoscedasticity, no serial correlation, and normally distributed errors. Serial correlation holds at the 5% level (Ljung-Box p = 0.078, marginal at 10%). Normality is violated — residual excess kurtosis of 3.769 exceeds raw EUR/USD kurtosis of 3.04. Homoscedasticity was not tested; FX residuals almost certainly exhibit GARCH effects. Engle-Granger requires I(1) inputs — confirmed for all three pairs. Ridge on lagged returns assumes autocorrelation structure exists to regularize around — Ljung-Box confirms it does not. PCA assumes covariance stationarity — PC1 variance explained drifted monotonically from ~52% to ~76% between 2011 and 2025, a direct violation.

## Findings

**OLS residual diagnostics (EUR/USD ~ GBP/USD, β = 0.5596):**

| Diagnostic | Value |
|---|---|
| Residual mean | 0.000000 |
| Residual variance | 0.000013 (~36bp/day unexplained) |
| Excess kurtosis | 3.769 (raw EUR/USD: 3.04) |
| Ljung-Box p-value (20 lags) | 0.078 — fail to reject at 5% |
| Lag-1 autocorrelation | −0.003 → half-life ≈ 0.695 days |

Ridge on five lagged EUR/USD returns produced uniformly negative coefficients (peak: lag4 at −0.012), with ~90% shrinkage between λ = 0.1 and λ = 1. Ljung-Box on the raw return series (p = 0.957 at 5 lags) confirms no autocorrelation structure — the apparent mean-reversion signal is noise.

**Stationarity — log price levels and OLS residuals:**

| Series / Spread | ADF p | KPSS stat | Verdict |
|---|---|---|---|
| EUR/USD (levels) | 0.2581 | 6.009 | I(1) |
| GBP/USD (levels) | 0.3966 | 7.750 | I(1) |
| USD/JPY (levels) | 0.7925 | 8.111 | I(1) |
| EUR/USD ~ GBP/USD residuals | 0.0576 | 0.574 | Not cointegrated |
| EUR/USD ~ USD/JPY residuals | 0.2071 | 1.242 | Not cointegrated |
| GBP/USD ~ USD/JPY residuals | 0.3729 | 2.488 | Not cointegrated |

No pair produced stationary residuals. EUR/USD ~ GBP/USD is the closest: ADF statistic −2.805 against critical value −2.86 (a 0.046-unit miss), KPSS stat 0.574 above the 0.463 critical value. Both tests agree; BH correction confirms no cointegration across any pair.

The levels-based β (0.7282) and the returns-based β (0.5596) measure different things — long-run equilibrium elasticity vs. contemporaneous daily return sensitivity. They should not be substituted for each other.

**Portfolio construction — PCA answer:**

PC1 (59.6–76% of variance, drifting post-2020) is a pure USD factor: EUR/USD and GBP/USD load negatively, USD/JPY positively. A simultaneous long EUR/USD and long GBP/USD position is two leveraged USD short positions with correlated residuals, not two independent bets. PC2 (28.9%) is a JPY safe-haven factor largely orthogonal to PC1 — USD/JPY provides real diversification from the EUR/GBP block. Three pairs produce at most two genuinely distinct risk dimensions.

## Alternative Explanations
The EUR/USD ~ GBP/USD null result at ADF p = 0.0576 warrants caution before being treated as final. ADF has known low power against near-unit-root alternatives in finite samples — a spread with a half-life of several months would look nearly identical to a unit root over this horizon. Johansen's multivariate procedure is a more sensitive test, particularly across all three series simultaneously.

Residual kurtosis rising from 3.04 to 3.769 after hedging is counterintuitive. The most plausible explanation: the SNB shock, Brexit, and COVID are episodes where EUR/USD and GBP/USD decoupled sharply — GBP/USD did not hedge EUR/USD in those moments, so the residual absorbed the full tail event. The PC1 drift post-2020 likely reflects Fed rate divergence dominating FX price action during that cycle, though the data cannot rule out a more permanent structural shift.

## Open Questions
EUR/USD ~ GBP/USD missed the ADF cointegration threshold by 0.046 units. Whether this is a genuine null or a power problem in Engle-Granger is unresolved pending Johansen testing across all three pairs.

The returns-based hedge ratio β = 0.5596 is a full-sample average. Walk-forward stability has not been tested — a parameter that drifts materially across sub-periods cannot be used as a live hedge ratio.

If PC1 USD factor loading is higher now than full-sample PCA implies — and the post-2020 drift suggests it is — a portfolio sized on full-sample covariances is underestimating current EUR/USD and GBP/USD correlation. Whether that requires time-weighted covariance estimation or rolling PCA is unresolved.