# Day 49 Research Audit — SignalReport Design + Run, Volatility Regime Breakout/Mean-Reversion

## Question
Two questions, deliberately kept separate:

1. **Engineering**: does a formatting-only `SignalReport` cleanly aggregate the per-window OOS diagnostics for both legs of the one surviving strategy candidate into a single reviewable object, without re-deriving any of the already-implemented signal/regression math?
2. **Research**: given that aggregation, what do the per-window regime-gated IC distribution and Deflated Sharpe actually say about each leg — and does that picture agree with the Day 48 interaction-regression verdict, or complicate it?

## Why It Matters
Day 48 tested statistical significance for both legs via the interaction regression (`b3`), but that test answers a narrow question: is the regime-conditional *slope* different from zero. It says nothing about the *sign or magnitude* of the signal's raw predictive power within each regime, and nothing about whether either leg's raw exposure would have produced a Sharpe ratio worth deflating in the first place. SignalReport was designed to surface exactly that.

## Correction (project-wide trial count)
The n_trials/strategy count used throughout this report is 4, not 5 as originally used in an earlier draft. The project's roster has only ever had 4 candidate strategies — PC2 Carry Regime, Momentum w/ ML Regime, and OU Half-Life Mean Reversion (all formally discarded), plus this one. `src/analysis/signal_report.py` and `research/applied_analysis/day49_signal_report_pipeline.py` both use n_trials=4.

## Methodology
- **Orchestration**: `research/applied_analysis/day49_signal_report_pipeline.py`. Reuses Day 46/48's already-tested primitives (`momentum_signal`, `price_zscore_signal`, `compute_composite_regime_score_walkforward`, `classify_regime`, `interaction_regression_centered`) — no new signal or regression logic was written.
- **Per-window IC**: Spearman IC between each leg's raw signal and the 26-day forward return, computed only on the subset of a window's test days where that leg's regime was active 
- **Per-window Sharpe**: `pnl_t = exposure_{t-1} * daily_log_return_t`, same regime gating, raw and unsized 
- **DSR**: `PerformanceAnalyzer.deflated_sharpe_ratio` applied to each leg's pooled (3 pairs x 7 windows) regime-gated pnl series. `n_trials=4`.
- **Multiple testing**: `benjamini_hochberg_correction` applied to the two legs' Day-48-style primary regression p-values, alpha=0.05. 
- **Formatting**: `src/analysis/signal_report.py` — `LegSignalStats` dataclass + `build_signal_report()` + `.to_markdown()`. Contains no scoring logic; takes the pipeline's arrays/results as arguments.
- **Data**: `DataLoader` daily close, 2011-01-01 to 2023-12-31 (development set), all three pairs. 

## Assumptions
- Regime gating for IC/Sharpe uses the per-window-refit composite classifier (`compute_composite_regime_score_walkforward`), the same walk-forward-safe construction Day 47/48 validated — not the Day 43 full-sample fit.
- DSR's observed Sharpe is annualized via `PerformanceAnalyzer.compute_ann_factor()`, which derives an empirical annualization factor from `n_obs / years_spanned` of the return series' own DatetimeIndex. Pooling three pairs' regime-gated pnl into one series means `n_obs` is roughly 3x a single-pair count over the *same* calendar span, so this ann_factor is inflated by roughly 3x versus a true single-pair basis. Not corrected in this build.
- IC here is the *unconditional* correlation between raw signal and forward return, computed on the in-regime subset. This is a different statistical object than Day 48's b3.
- OOS Sharpe is a signal-level proxy only: no position sizing, no transaction costs.

## Findings

**Momentum leg** (n=14 valid IC windows out of 21 pair-window slots; n=15 valid Sharpe windows; pooled DSR n_obs=3234):

| Metric | Value |
|---|---|
| IC mean | -0.1037 |
| IC std | 0.3084 |
| IC frac. positive | 28.57% |
| OOS Sharpe mean | -0.1646 |
| Primary regression p(b3) | 0.00000 |
| BH-significant (2-leg, alpha=0.05) | True |
| Survives project-wide p<0.0125 bar | True |
| DSR (n_trials=4) | 0.1646 |
| DSR input observed Sharpe | 0.0289 |

**Reversion leg** (n=14 valid IC windows; n=14 valid Sharpe windows; pooled DSR n_obs=2433):

| Metric | Value |
|---|---|
| IC mean | -0.2355 |
| IC std | 0.2349 |
| IC frac. positive | 14.29% |
| OOS Sharpe mean | -0.8526 |
| Primary regression p(b3) | 0.56308 |
| BH-significant (2-leg, alpha=0.05) | False |
| Survives project-wide p<0.0125 bar | False |
| DSR (n_trials=4) | 0.0007 |
| DSR input observed Sharpe | -0.7099 |

**Multiple-testing-corrected verdict (2-leg BH): FAIL** 

## Alternative Explanations

**Regime-gated IC and Day 48's `b3` now agree on the reversion leg — both say null.** Reversion's IC is negative (-0.2355) and its regression p-value is insignificant (0.563). Momentum's IC is also negative (-0.1037) despite a significant, well-signed `b3` — that gap is still open and is a different statistical-object issue (`b3` measures the conditional slope, not the same thing as raw in-regime rank correlation).

**DSR is uniformly low, most strikingly for reversion (0.0007).** The DSR input Sharpe is negative for both legs even after removing the lockbox (momentum +0.03, reversion -0.71). This is a raw, unsized, no-cost signal proxy, not a tradable-performance claim, but the direction is consistent with reversion having no real edge on the development set.

**Pooled-index annualization bias could still be inflating both DSR values somewhat**, but given both observed Sharpes are near zero or negative, this doesn't change the qualitative read.

## Next Steps
- Momentum-only edge: the test itself already passed cleanly and doesn't need redoing. What's unresolved is the surrounding spec — sizing and risk controls were built for a strategy active most of the time, not one that only trades the ~9-16% of days classified turbulent.
- GBP/USD-only reversion: dev-only threshold check suggests reversion may hold only in GBP/USD (EUR/USD, USD/JPY wrong-signed on clean data). Exploratory, not pre-registered
- Address the pooled-index annualization bias in DSR before citing DSR numbers outside this repo.
- Section 10 lockbox holdout (2024-2026) remains unopened. 