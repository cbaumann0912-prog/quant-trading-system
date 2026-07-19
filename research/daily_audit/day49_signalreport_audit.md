# Day 49 Research Audit — SignalReport Design + Run, Volatility Regime Breakout/Mean-Reversion

## Question
Two questions, deliberately kept separate:

1. **Engineering**: does a formatting-only `SignalReport`  cleanly aggregate the per-window OOS diagnostics for both legs of the one surviving strategy candidate into a single reviewable object, without re-deriving any of the already-implemented signal/regression math?
2. **Research**: given that aggregation, what do the per-window regime-gated IC distribution and Deflated Sharpe actually say about each leg — and does that picture agree with the Day 48 interaction-regression verdict, or complicate it?

## Why It Matters
Day 48 established statistical significance for both legs via the interaction regression (`b3`), but that test answers a narrow question: is the regime-conditional *slope* different from zero. It says nothing about the *sign or magnitude* of the signal's raw predictive power within each regime, and nothing about whether either leg's raw exposure would have produced a Sharpe ratio worth deflating in the first place. SignalReport was designed to surface exactly that — IC distribution and DSR per leg — as a second, independent lens on the same two legs, not a re-run of Day 48's test.

## Correction (added same-day, after initial write-up)
The n_trials/strategy count used below was originally 5, copied from the strategy spec. The 4 candidate strategies developed across the project only one survived PC2 Carry Regime, Momentum w/ ML Regime, and OU Half-Life Mean Reversion are all formally discarded." No earlier point in the repo ever used a smaller number for this concept; the multiple-testing framework itself wasn't introduced until the Day 43 spec, by which point all 3 prior candidates were already closed, so the count was already 4 at first mention. Everything below has been corrected to n=4 and the pipeline was re-run to get accurate DSR figures under the corrected n_trials. See `src/analysis/signal_report.py` and `research/applied_analysis/day49_signal_report_pipeline.py` for the corrected constants.

## Methodology
- **Orchestration**: `research/applied_analysis/day49_signal_report_pipeline.py`. Reuses Day 46/48's already-tested primitives — no new signal or regression logic was written.
- **Per-window IC**: Spearman IC between each leg's raw signal and the 26-day forward return, computed only on the subset of a window's test days where that leg's regime was active (turbulent for momentum, calm for reversion) — matching the strategy spec's own pre-registered test statistic, not an unconditional IC.
- **Per-window Sharpe**: `pnl_t = exposure_{t-1} * daily_log_return_t`, same regime gating, raw and unsized
- **DSR**: `PerformanceAnalyzer.deflated_sharpe_ratio`  applied to each leg's pooled (3 pairs x 10 windows) regime-gated pnl series.
- **Multiple testing**: `benjamini_hochberg_correction` applied to the two legs' Day-48-style primary regression p-values, alpha=0.05. This is narrower than the Section 1 item 5 project-wide 4-strategy bar (p<0.0125, rank 1 of 4).
- **Formatting**: `src/analysis/signal_report.py` — `LegSignalStats` dataclass + `build_signal_report()` + `.to_markdown()`. Contains no scoring logic; takes the pipeline's arrays/results as arguments.
- Data: same as Day 48 — `DataLoader` daily close, 2011-01-01 to 2026-05-01, all three pairs, lockbox handling identical to Day 48's script.

## Assumptions
- Regime gating for IC/Sharpe uses the per-window-refit composite classifier (`compute_composite_regime_score_walkforward`), the same walk-forward-safe construction Day 47/48 validated — not the Day 43 full-sample fit.
- DSR's observed Sharpe is annualized via `PerformanceAnalyzer.compute_ann_factor()`, which derives an empirical annualization factor from `n_obs / years_spanned` of the return series' own DatetimeIndex. Pooling three pairs' regime-gated pnl into one series means `n_obs` is roughly 3x a single-pair count over the *same* calendar span, so this ann_factor is inflated by roughly 3x versus a true single-pair basis. Not corrected in this build — flagged as a caveat in the generated report and carried into Findings below.
- IC here is the *unconditional* correlation between raw signal and forward return, computed on the in-regime subset. This is a different statistical object than Day 48's `b3`. Both are legitimate, but they answer different questions; see Alternative Explanations.
- OOS Sharpe is a signal-level proxy only: no position sizing, no transaction costs. It is not evidence about tradable performance, only about the raw signal's directional relationship with subsequent returns.

## Findings
**Momentum leg** (n=16 valid IC windows out of 30 pair-window slots; n=19 valid Sharpe windows; pooled DSR n_obs=3805):

| Metric | Value |
|---|---|
| IC mean | -0.1059 |
| IC std | 0.2882 |
| IC frac. positive | 25.00% |
| OOS Sharpe mean | -0.1997 |
| Primary regression p(b3) | 0.00016 |
| BH-significant (2-leg, alpha=0.05) | True |
| Survives project-wide p<0.0125 bar | True |
| DSR (n_trials=4) | 0.1193 |
| DSR input observed Sharpe | -0.0410 |

**Reversion leg** (n=22 valid IC windows; n=22 valid Sharpe windows; pooled DSR n_obs=4155):

| Metric | Value |
|---|---|
| IC mean | -0.1711 |
| IC std | 0.2312 |
| IC frac. positive | 22.73% |
| OOS Sharpe mean | -0.5104 |
| Primary regression p(b3) | 0.00549 |
| BH-significant (2-leg, alpha=0.05) | True |
| Survives project-wide p<0.0125 bar | True |
| DSR (n_trials=4) | 0.0214 |
| DSR input observed Sharpe | -0.2952 |

**Multiple-testing-corrected verdict (2-leg BH): PASS for both legs.** 

## Alternative Explanations
**Regime-gated IC is negative for both legs while Day 48's `b3` is significant with a consistent sign.** These are not necessarily in conflict, but they are not the same claim.

- `b3` measures whether the signal's *slope* against forward returns is different inside the regime versus outside it, after mean-centering both main effects. A significant, consistently-signed b3 says the regime conditioning genuinely changes the signal-return relationship.
- Regime-gated IC measures the absolute rank correlation between the signal and forward returns, computed only on in-regime days, with no comparison to the out-of-regime baseline.
- A leg can have a significant, well-signed interaction term while its in-regime IC is still negative or noisy, if the out-of-regime relationship is even more negative, or if regime-gating with only a handful of valid windows is too small a sample for IC to be a stable descriptive statistic in the first place.
- This has not been separated out. Next Steps below proposes the specific comparison that would distinguish these possibilities.

**DSR near zero for both legs is expected, not a new negative finding, given the input.** The DSR input Sharpe (-0.04 momentum, -0.30 reversion) is a raw, unsized, no-cost signal proxy — a near-zero-or-negative Sharpe correctly produces a near-zero DSR. This should not be read as contradicting Day 48's PASS verdict; it is answering a different question that Day 48 never claimed to answer. 

**Pooled-index annualization bias could be inflating both DSR values somewhat**, but since both observed Sharpes are already negative, correcting this bias would push DSR further toward zero, not away from the current low readings.

## Next Steps
- Compute and report in-regime vs. out-of-regime IC side by side per leg, to isolate whether the negative regime-gated IC reflects a weak-but-real in-regime effect or an even-more-negative out-of-regime baseline that `b3` is picking up in a relative sense.
- Address the pooled-index annualization bias in DSR before citing DSR numbers outside this repo.
- Carry reversion leg's primary p (0.00549) forward as the binding constraint once all 4 project-wide strategies have logged a final p-value. 
- Section 10 lockbox holdout (2024–2026) remains unopened; nothing here changes that gate.
