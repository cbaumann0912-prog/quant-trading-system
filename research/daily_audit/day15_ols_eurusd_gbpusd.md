# Day 15 — OLS: EUR/USD vs GBP/USD Log Returns

## Question
Does a linear relationship exist between EUR/USD and GBP/USD daily log returns, and if so, what does the estimated slope tell us about the hedge ratio needed for a market-neutral pairs position?

## Why It Matters
Both EUR/USD and GBP/USD carry substantial USD exposure. If the relationship between their returns is stable and well-estimated, the OLS slope gives the hedge ratio — the number of GBP/USD units to short per unit of EUR/USD to neutralize the common USD factor. The residual from this regression is also the candidate signal for any mean-reversion strategy. If it is stationary, there is a tradeable spread. If not, the regression is spurious and the relationship is an artifact.

## Methodology
$$\text{EUR/USD}_t = \alpha + \beta \cdot \text{GBP/USD}_t + \epsilon_t$$

Both variables are daily log returns computed from 1-minute close prices resampled to daily frequency.

`fit_ols` estmates via normal equations $\hat{\beta} = (X^\top X)^{-1} X^\top y$, solved with `np.linalg.solve` rather than explicit matrix inversion.

Sampled from 2011-01-02 to 2023-12-29 | 4,054 aligned trading days after dropping missing values from log differencing.

## Assumptions
- The relationship between EUR/USD and GBP/USD returns is linear
- Errors have constant variance (homoskedasticity) — likely violated for forex returns
- Errors are uncorrelated across time — lag-1 autocorrelation is a first check on this
- The predictor GBP/USD return is measured without error — in practice both series have microstructure noise

## Results
| Parameter | Estimate | Std Error | t-statistic |
|-----------|----------|-----------|-------------|
| $\alpha$ (intercept) | -0.00001956 | 0.00005959 | -0.3282 |
| $\beta$ (hedge ratio) | 0.53545910 | 0.01137353 | 47.0794 |

| Metric | Value |
|--------|-------|
| $R^2$ | 0.3536 |
| Observations | 4,054 |

| Metric | Value | Threshold |
|--------|-------|-----------|
| Residual mean | 7.00e-20 | ~0 (intercept included) |
| Residual std | 0.003793 | — |
| Skewness | -0.0541 | ~0 for symmetric errors |
| Excess kurtosis | 3.1203 | ~0 for normal errors |
| Lag-1 autocorrelation | -0.001125 | <0.1 acceptable |
| max\|X'e\| | 3.26e-16 | ~machine epsilon |

## Findings
The hedge ratio is 0.5355, meaning a one-unit EUR/USD long requires shorting 0.5355 units of GBP/USD to neutralize the common USD exposure. The magnitude makes sense. GBP is a higher-volatility currency than EUR — it responds more aggressively to a given USD shock — so you need less of it on the short side to achieve balance. The t-statistic of 47.1 puts this estimate 47 standard errors from zero. The relationship is not in question.

Alpha is -0.0000196 with a t-statistic of -0.328, statistically indistinguishable from zero. This is the right answer. A nonzero, significant alpha would imply a persistent daily return premium to one side of the spread. In liquid forex markets, that kind of persistent differential gets arbitraged away fast.

GBP/USD explains 35.4% of EUR/USD daily return variance. The remaining 64.6% is pair-specific. An $R^2$ in this range is internally consistent: high enough to confirm that a common USD factor dominates both pairs, low enough to confirm the pairs carry distinct information. The 64.6% unexplained variance is  the space where any EUR/GBP-specific signal has to operate.

The residuals are well-behaved on the dimensions tested today. Skewness of -0.054 is negligible — the spread residuals are symmetric around zero. Lag-1 autocorrelation of -0.001 is effectively zero, so there is no evidence that the errors are serially correlated; the t-statistics and standard errors are not corrupted on this dimension. The orthogonality check at 3.26e-16 confirms the normal equations solved correctly — this is machine epsilon and validates the implementation.

The one number that stands out is excess kurtosis of 3.77. The residuals have substantially fatter tails than a normal distribution assumes. Large spread dislocations happen more frequently than Gaussian-based risk estimates predict. Any position sizing or stop-loss logic derived from these residuals under a normality assumption will systematically underestimate tail exposure.

## Alternative Explanations
The $R^2$ of 35.4% is a full-sample statistic computed across 13 years that include the European debt crisis, Brexit, COVID, and aggressive Fed tightening cycles. During periods of extreme global USD stress, EUR/USD and GBP/USD tend to move almost in lockstep as the USD dominates everything else, which would inflate the full-sample $R^2$. It is plausible that the relationship is stronger in crisis regimes and weaker in normal regimes, and that the aggregate $R^2$ is averaging across two quite different underlying states. This cannot be tested today but will become relevant in the regime-detection module 

The hedge ratio of 0.5355 is a single number summarizing 13 years of a relationship that has no reason to be stationary. The EUR/GBP cross-rate has moved substantially over this period — it traded near 0.70 in 2015 and above 0.92 post-Brexit. Each regime change in EUR/GBP represents a structural shift in the relative USD sensitivities of the two pairs. During those transitions, the actual hedge ratio would have been different from 0.5355, and using the full-sample estimate in those subperiods would have left systematic residual exposure.

## What Cannot Be Concluded Today
The most important missing piece is stationarity of the residual. A significant $\hat{\beta}$ and a reasonable $R^2$ on returns says nothing about whether the price-level spread mean-reverts. The residual here is a return residual, not a price-level spread. The cointegration question — whether EUR/USD and GBP/USD prices share a common stochastic trend — requires the Engle-Granger test. Until then, no claim about a tradeable mean-reversion signal in the price spread is warranted.

The hedge ratio estimated today is a full-sample quantity. Whether it is stable enough to apply in a walk-forward setting — where you estimate on a training window and apply to unseen data — is unanswered.

This analysis was conducted on log returns. The resulting β = 0.5355 is a return hedge ratio measuring the contemporaneous return relationship, not a cointegration hedge ratio. Engle-Granger cointegration testing requires OLS on log price levels.