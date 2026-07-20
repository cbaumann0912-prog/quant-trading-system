# SignalReport -- Volatility Regime Breakout/Mean-Reversion

## Momentum leg

| Metric | Value |
|---|---|
| IC mean (n=14 windows) | -0.1037 |
| IC std | 0.3084 |
| IC-derived IR (mean/std) | -0.3361 |
| IC frac. positive windows | 28.57% |
| OOS Sharpe mean (n=15 windows) | -0.2688 |
| OOS Sharpe std | 1.3205 |
| Primary regression p(b3) | 0.00000 |
| BH-significant (alpha=0.05, 2-leg family) | True |
| Survives project-wide pre-registered bar (p<0.0125) | True |
| Deflated Sharpe (n_trials=4, n_obs=3234) | 0.1646 |
| DSR input: observed Sharpe | 0.0289 |
| DSR input: skewness / excess kurtosis | -0.2172 / 6.1080 |

## Reversion leg

| Metric | Value |
|---|---|
| IC mean (n=14 windows) | -0.2355 |
| IC std | 0.2349 |
| IC-derived IR (mean/std) | -1.0025 |
| IC frac. positive windows | 14.29% |
| OOS Sharpe mean (n=14 windows) | -1.0262 |
| OOS Sharpe std | 1.5376 |
| Primary regression p(b3) | 0.56308 |
| BH-significant (alpha=0.05, 2-leg family) | False |
| Survives project-wide pre-registered bar (p<0.0125) | False |
| Deflated Sharpe (n_trials=4, n_obs=2433) | 0.0007 |
| DSR input: observed Sharpe | -0.7099 |
| DSR input: skewness / excess kurtosis | -7.7990 / 152.9295 |

**Multiple-testing-corrected verdict (2-leg BH, alpha=0.05): FAIL**

## Caveats

- OOS Sharpe (per-window and the DSR input) is a raw, unsized, no-transaction-cost signal-level proxy: sign(signal) x forward return on regime-active days only. Section 7 vol-targeted sizing and the Day 57 transaction-cost model are not applied here -- this is not a claim of tradable performance.
- DSR's observed Sharpe is annualized from a DatetimeIndex pooled across all 3 pairs over the same calendar span, so compute_ann_factor()'s n_obs/years_spanned is inflated roughly 3x versus a single-pair basis -- DSR here is optimistically biased. Known, documented gap; not corrected in this build.
- The BH correction above (alpha=0.05) is applied only across this strategy's 2 legs, not the full project-wide 4 strategies the spec pre-registers (Section 1, item 5: needs p<0.0125 at rank 1 of 4). The other 3 strategies' final p-values are not logged anywhere in the repo yet, so a true 4-way BH cannot be run. The project-wide bar column above is shown for comparison, not as a substitute for that correction.
- This report covers only the two-leg multiple-testing check and DSR. It does not replace the full Section 10 verdict (reliability gate + both robustness checks) in research/daily_audit/day48_two_leg_validation.md, and the Section 10 lockbox holdout (2024-2026) has still not been opened.
