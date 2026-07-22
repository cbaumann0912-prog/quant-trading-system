# Day 52 Research Audit: Kelly Sizing Mechanics Demo, Momentum-Only Pooled Book

## Question
Given `kelly_fraction` and `fractional_kelly`, what full- and quarter-Kelly position size does the momentum-only pooled book (`research/strategies/momentum_only_pooled_book.md`) imply, computed from its own real mu/sigma?

## Illustrative-Only Scope
This is a mechanics demonstration of `kelly_fraction`/`fractional_kelly`, not a new strategy candidate or a new statistical test. It does not count toward `n_trials` (stays at 4 per `framework_map.md`) and nothing here is logged in any strategy vault, spec, or decision log.

## Methodology
`research/applied_analysis/day52_kelly_position_sizing.py` reuses the already-tested momentum signal logic unchanged (`momentum_signal`, `compute_composite_regime_score_walkforward`, `classify_regime`, `regime_gated_pnl` — Days 46-49), pooled across all three pairs (EUR/USD, GBP/USD, USD/JPY), development window only (2011-01-01 to 2023-12-31, lockbox excluded). Pooling convention follows Day 50, not Day 49: the three pairs' regime-gated daily pnl are aligned by calendar date and averaged (equal-weighted mean across whichever pairs are regime-active that day) rather than stacked end to end, giving one coherent daily series rather than a concatenated one. Mu/sigma are computed in-file from that real series — daily mean and daily standard deviation, annualized via `PerformanceAnalyzer(momentum_pnl).compute_ann_factor()` (empirical `n_obs / years_spanned`, not a fixed 252 or 312). Kelly fraction = mu / sigma^2, evaluated both full and at a 0.25 fraction, sized against a hypothetical $100,000 in capital.

## Findings

| Quantity | Value |
|---|---|
| n_observations (pooled, regime-gated, lockbox-excluded) | 1,472 |
| daily_mu | 0.000041 |
| daily_sigma | 0.004221 |
| empirical ann_factor (`compute_ann_factor`) | 210.2652 |
| annual mu | 0.86% |
| annual sigma | 6.12% |
| full kelly f* | 2.2870 |
| 0.25 fractional kelly f | 0.5717 |
| position on $100,000 (quarter-Kelly) | $57,174.68 |

## Interpretation
Full Kelly here comes out above 2x leverage on the account, which is mechanical rather than a sign of a strong edge: `f* = mu / sigma^2` is inversely proportional to the square of volatility, so a low-vol signal (6.12% annualized) produces a large fraction even off a small mean return (0.86% annualized). Quarter-Kelly still allocates $57,175 of a $100,000 account to a single leg. That is worth reading against Day 49's own DSR finding on this leg — 0.1646 at n_trials=4, off an observed raw Sharpe of only 0.0289 — which `momentum_only_pooled_book.md` Section 9 already flags as a clean p-value sitting on a weak edge. Sizing at even a quarter of full Kelly here is sizing as if that edge were considerably more certain than the DSR supports. This exercise demonstrates the sizing mechanics correctly; it does not by itself justify using either the full or quarter-Kelly number as a real position size.

## Next Steps
- Classical Kelly assumes i.i.d. per-period returns; Day 37 found volatility clustering material at daily frequency for FX (Ljung-Box on |r_t| rejected white noise decisively across all three pairs), so this f* is likely biased before any other consideration.
- No estimation error on mu is propagated through to f*; Kelly is well known to be highly sensitive to mu estimation noise, and mu here is a single point estimate over 1,472 regime-gated observations. A shrinkage or CI-propagated treatment is future work, not attempted here.
- Section 7 of `momentum_only_pooled_book.md` (position sizing rule) remains an open decision; this demo does not resolve it and should not be read as a proposed answer to it.
