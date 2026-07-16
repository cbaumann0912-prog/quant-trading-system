# Day 16 Research Audit — Residual Diagnostics: EUR/USD ~ GBP/USD Regression

## 1. Question Investigated
Do the residuals from the EUR/USD ~ GBP/USD OLS regression satisfy the classical assumptions — no serial correlation, normality, zero mean? And what do violations imply for the reliability of the hedge ratio and its standard errors?

## 2. Why It Matters
The t-statistics, confidence intervals, and significance tests on the hedge ratio are only valid if the residuals satisfy OLS assumptions. Serial correlation inflates significance. Non-normality — specifically fat tails — makes tail probabilities unreliable. Either violation means the regression output looks more precise than it is.

## 3. Methodology
Both variables are daily log returns computed from 1-minute close prices resampled to daily frequency, spanning 2011-01-02 to 2026-03-31 — 4,758 aligned trading days after dropping missing values from log differencing. `residual_diagnostics` computes six diagnostics on the residuals from the Day 15 regression: mean, variance, excess kurtosis, Ljung-Box Q-statistic and p-value at 20 lags, and lag-1 autocorrelation.

## 4. Assumptions
- **Zero conditional mean:** E(ε|X) = 0 — supported; residual mean = 0.000000 by construction.
- **Homoscedasticity:** Var(ε|X) = σ² — not tested today
- **No serial correlation:** Cov(εᵢ, εⱼ) = 0 for i ≠ j — supported at the 5% level; marginal at 10%.
- **Normality:** ε ~ N(0, σ²) — challenged; excess kurtosis of 3.769 indicates fat tails. CLT provides asymptotic cover at n = 4,758, but tail probabilities remain understated.

## 5. Findings
| Diagnostic | Value | Interpretation |
|------------|-------|----------------|
| Mean | 0.000000 | Mathematical guarantee for OLS with an intercept — minimizing RSS with respect to β₀ forces Σεᵢ = 0 by construction. |
| Variance | 0.000013 | Estimated σ² of the residuals. The typical residual is √0.000013 ≈ 36 basis points per day — the basis risk that GBP/USD does not explain. |
| Excess Kurtosis | 3.768928 | Residual tails are materially fatter than normal. Confidence intervals and t-statistics on the hedge ratio are less reliable than they appear, particularly in the tails. |
| Ljung-Box Stat | 29.507682 | Compared to χ²(20) critical value of 31.4 at the 5% level. Since 29.507 < 31.4, we fail to reject the null of no serial correlation. |
| Ljung-Box p-value | 0.078233 | Fail to reject at the 5% level. Marginal at 10% — the result sits in a grey zone worth monitoring. |
| Lag-1 Autocorr | −0.003149 | Economically negligible. Today's residual carries virtually no information about tomorrow's. |

## 6. Alternative Explanations
Day 4 measured excess kurtosis of 3.04 on raw EUR/USD log returns. After removing the GBP/USD linear relationship, residual kurtosis rises to 3.769. If GBP/USD were absorbing tail events proportionally, residual kurtosis would fall. It does not — it increases. This means GBP/USD is not the driver of EUR/USD during extreme market days. The tail events are largely orthogonal to the hedge.

A second explanation: the 15-year window spans structurally different regimes — the 2015 SNB shock, Brexit, COVID. Extreme residuals from these episodes inflate kurtosis even if within-regime errors are approximately normal. The fat tails may be episodic rather than a permanent feature of the distribution.

## 7. Open Questions
The hedge ratio β = 0.5596 is a static full-sample estimate. A single coefficient estimated across 15 years of structurally different regimes may not be stable. Walk-forward validation, put off for a later date, will test whether this number holds out-of-sample or drifts materially across sub-periods. A rolling hedge ratio is worth evaluating against the static estimate.

Lag-1 autocorrelation of −0.003 is near zero, implying a half-life of approximately 0.7 days (t₁/₂ = ln(2) / (1 − 0.003)). Mean reversion at sub-daily frequency means any stat-arb strategy on this pair would need to operate intraday to capture the reversion timing. Note: this autocorrelation is measured on the OLS regression residuals, not the spread itself — the formal OU half-life will be computed at a later date using the cointegration spread directly.

Residual stationarity has not been tested. If the residuals are non-stationary, the spread drifts rather than mean-reverts and the hedge ratio cannot be treated as a stable parameter. 