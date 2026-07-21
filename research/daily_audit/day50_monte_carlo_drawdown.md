# Day 50 Research Audit: Monte Carlo Drawdown Analysis, Momentum Leg

## Question
Given `simulate_gbm`, what does the momentum leg's realized drawdown look like against a GBM null process calibrated to its own empirical mean and volatility? Is the historical max drawdown consistent with pure noise, or does it stand out from what a driftless random process would produce anyway?

## Methodology
- **Pooling convention**: Day 49's `pd.concat(chunks)` stacks each pair's full 13-year history end to end — defensible for a Sharpe/DSR mean, but not a real equity curve, so `compute_max_drawdown()` fails. Day 50 instead aligns the three pairs by calendar date and takes an equal-weighted mean across whichever pairs are regime-active on a given day, giving one coherent daily timeline.
- **GBM calibration**: daily mean/std of the date-aligned momentum pnl series, annualized via `PerformanceAnalyzer(momentum_pnl).compute_ann_factor()` (`n_obs / years_spanned`). Empirical factor ≈210.27 obs/year, reflecting turbulent-regime-active observations per year, not all trading days.
- **Simulation horizon**: `n_steps = len(momentum_pnl)` (1,472), matching the full development-period series exactly rather than an arbitrary fixed window. `T = n_steps / ann_factor ≈ 7.00` years.
- **Annualization cancels out**: `T` and `dt` are both defined off the same `ann_factor` used to build `annual_mu`/`annual_sigma`, so the annualization cancels out exactly in the simulated increment (`annual_mu * dt = daily_mu`, `annual_sigma * sqrt(dt) = daily_sigma`). Simulated path statistics depend only on `daily_mu`, `daily_sigma`, and `n_steps` — `ann_factor` only changes how `annual_mu`/`annual_sigma`/`T` are labeled.
- **Simulation**: `simulate_gbm(S0=1.0, mu=annual_mu, sigma=annual_sigma, T=n_steps/ann_factor, n_steps=1472, n_paths=1000, seed=28)`, exact lognormal scheme — `seed=28` matches the project-wide convention (`simulate_gbm`'s own default, and the convention used throughout `src/stats/` and most existing tests). Max drawdown computed per simulated path via `PerformanceAnalyzer.compute_max_drawdown()` on that path's simple returns (1,000 individual calls, not vectorized — reuses the existing tested drawdown logic exactly rather than reimplementing it).
- **Historical comparison**: same `compute_max_drawdown()` applied to the actual date-aligned momentum pnl series, full sample.

## Findings

| Quantity | Value |
|---|---|
| n_observations (date-aligned momentum leg) | 1,472 |
| daily_mu | 0.000041 |
| daily_sigma | 0.004221 |
| empirical ann_factor | 210.27 |
| annual_mu | 0.86% |
| annual_sigma | 6.12% |
| n_steps (= full development-period series length) | 1,472 |
| T (years) | 7.00 |
| Simulated 5th percentile max drawdown | −27.31% |
| Simulated median max drawdown | −15.33% |
| P(max drawdown ≤ −20%) over the full 7.00-year development horizon | 23.90% (239/1000) |
| Historical realized max drawdown (full sample) | −12.16% |

The historical realized drawdown (−12.16%) is *better* than the simulated median (−15.33%) and nowhere near the simulated 5th-percentile tail (−27.31%). Over a 7-year horizon, a pure-noise GBM process calibrated to this leg's own near-zero drift and 6.12% annual vol typically produces a worse drawdown than what was  realized by the strategy. `P(max drawdown ≤ −20%) = 23.9%` means roughly 1 in 4 noise-only paths at this horizon breach −20% at some point, purely from volatility with no real signal — useful context for how much of a "large" drawdown is just what 7 years of this vol level looks like regardless of edge.

## Interpretation
*(Clay's own reasoning — not filled in here per project convention. Consider: annual_mu of 0.86% is barely above zero and annual_sigma of 6.12% is not small relative to it — what does that ratio alone already tell you about this leg's edge, independent of any drawdown comparison? The historical drawdown beating the simulated median is a mildly favorable signal on risk, but says nothing about the negative IC from Day 49 — a process can have unremarkable drawdown and still have no real predictive edge. Does "drawdown looks fine against a noise null" change your read on this leg at all, or is it a separate axis entirely from the IC/Sharpe verdict?)*

## Next Steps
- Decide whether the equal-weighted-across-active-pairs pooling convention introduced here should retroactively replace Day 49's stacked-concatenation convention for any future work, or whether the two constructions should coexist for different purposes (cross-sectional stats vs. path-dependent stats).
- `P(max drawdown ≤ −20%) = 23.9%` on `n_paths=1000` has a standard error of roughly ±1.3 percentage points, rerun with more paths if a tighter bound matters later.
- Feeds directly into the Day 51 month-2 candidate-selection assessment: this is the one remaining candidate, and its drawdown profile is statistically unremarkable (if anything, mildly favorable) against a noise null calibrated to its own weak drift and volatility over the full 7-year development span.
