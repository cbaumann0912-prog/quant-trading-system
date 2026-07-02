# Week 3 Review — Regression, Linear Algebra & Stationarity (Days 15–21)

## Methodology
`fit_ols`, `residual_diagnostics`, `fit_ridge`, `eigen_decomposition`/`pca`, `check_stationarity` (ADF+KPSS) applied to EUR/USD, GBP/USD, USD/JPY log returns/levels, 2011-01-02–2026-03-31, full sample.

## Findings
- OLS EUR/USD~GBP/USD: β=0.5596 (returns), β=0.7282 (levels) — not interchangeable. Residual excess kurtosis 3.769 (up from raw 3.04). Ljung-Box p=0.078 (marginal).
- Ridge on lagged returns: uniformly negative coefficients, ~90% shrinkage λ=0.1→1; Ljung-Box p=0.957 on raw returns confirms no autocorrelation to regularize around.
- Stationarity: all three pairs I(1) in levels. No pair's residuals stationary — EUR/USD~GBP/USD closest (ADF p=0.0576, missed by 0.046 units; KPSS 0.574 vs 0.463 critical). BH-corrected, no cointegration confirmed across any pair.
- PCA: PC1 (59.6–76%, drifting post-2020) = USD factor; EUR/USD & GBP/USD are correlated, not independent bets. PC2 (28.9%) = JPY factor, largely orthogonal — real diversification source.
- Open: ADF near-miss on EUR/USD~GBP/USD may be a power problem, not a true null — Johansen pending. Hedge ratio walk-forward stability untested.

## Interpretation
The PCA result is the one that actually changes how I think about the portfolio — EUR/USD and GBP/USD load onto the same USD factor, so trading both isn't two independent bets, it's one leveraged position with a second name on it. PC2 as a JPY safe-haven factor is where the real diversification lives. Cointegration came back negative across every pair combination this week, EUR/USD~GBP/USD closest but still short of the threshold — Johansen is the next check on whether that's genuine.