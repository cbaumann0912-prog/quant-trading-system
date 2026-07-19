# SignalReport -- Volatility Regime Breakout/Mean-Reversion
## Momentum leg

| Metric | Value |
|---|---|
| IC mean (n=16 windows) | -0.1059 |
| IC std | 0.2882 |
| IC-derived IR (mean/std) | -0.3676 |
| IC frac. positive windows | 25.00% |
| OOS Sharpe mean (n=19 windows) | -0.1997 |
| OOS Sharpe std | 1.3353 |
| Primary regression p(b3) | 0.00016 |
| BH-significant (alpha=0.05, 2-leg family) | True |
| Survives project-wide pre-registered bar (p<0.0125) | True |
| Deflated Sharpe (n_trials=4, n_obs=3805) | 0.1193 |
| DSR input: observed Sharpe | -0.0410 |
| DSR input: skewness / excess kurtosis | -0.1312 / 5.4022 |

## Reversion leg

| Metric | Value |
|---|---|
| IC mean (n=22 windows) | -0.1711 |
| IC std | 0.2312 |
| IC-derived IR (mean/std) | -0.7403 |
| IC frac. positive windows | 22.73% |
| OOS Sharpe mean (n=22 windows) | -0.5104 |
| OOS Sharpe std | 1.5200 |
| Primary regression p(b3) | 0.00549 |
| BH-significant (alpha=0.05, 2-leg family) | True |
| Survives project-wide pre-registered bar (p<0.0125) | True |
| Deflated Sharpe (n_trials=4, n_obs=4155) | 0.0214 |
| DSR input: observed Sharpe | -0.2952 |
| DSR input: skewness / excess kurtosis | -6.1840 / 152.7705 |

**Multiple-testing-corrected verdict (2-leg BH, alpha=0.05): PASS**

## Caveats

- OOS Sharpe (per-window and the DSR input) is a raw, unsized, no-transaction-cost signal-level proxy: sign(signal) x forward return on regime-active days only. Section 7 vol-targeted sizing and the Day 57 transaction-cost model are not applied here -- this is not a claim of tradable performance.
- DSR's observed Sharpe is annualized from a DatetimeIndex pooled across all 3 pairs over the same calendar span, so compute_ann_factor()'s n_obs/years_spanned is inflated roughly 3x versus a single-pair basis -- DSR here is optimistically biased. Known, documented gap; not corrected in this build.
- The BH correction above (alpha=0.05) is applied only across this strategy's 2 legs, not the full project-wide 5 strategies the spec pre-registers (Section 1, item 5: needs p<0.0125 at rank 1 of 5). The other 4 strategies' final p-values are not logged anywhere in the repo yet, so a true 5-way BH cannot be run. The project-wide bar column above is shown for comparison, not as a substitute for that correction.
- This report covers only the two-leg multiple-testing check and DSR. It does not replace the full Section 10 verdict (reliability gate + both robustness checks) in research/daily_audit/day48_two_leg_validation.md, and the Section 10 lockbox holdout (2024-2026) has still not been opened.
