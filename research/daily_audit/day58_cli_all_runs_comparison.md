# Day 58 Research — CLI Run All Pairs Both Signals

## Verdict
Infrastructure: PASS. `run_research.py` runs the chain end to end on all 6 pair/signal combinations and writes 6 well-formed JSON reports to `results/`. 11 CliRunner tests pass.

## Methodology
`run_research.py` (Click 8.4.1): `DataLoader.load`, signal construction, `WalkForwardValidator.generate_windows`, per-window OOS scoring, JSON. Six runs at `--train-start 2015-01-01 --train-end 2022-12-31 --windows 10 --train-years 5 --test-months 3`, momentum lookback 78, z-score lookback 26, forward horizon 26, embargo 5.

Scoring is a CLI-local loop, not a framework component. `WalkForwardValidator.run()` takes `signal_fn` and never calls it, so every pipeline in the repo hand-writes its own scoring off `generate_windows()`. `src/framework/walk_forward.py` untouched.

Two deliberate departures from `vol_regime_signal_report_pipeline.py`. Forward returns are masked over the final `holding_period` bars of each test window so no scored bar reads past `test_end`; Day 49 does not do this. And `build_signal_report` is not called: its contract needs a regime-interaction p(b3) that an ungated single-signal run cannot produce, and a one-leg BH correction is a no-op printing "multiple-testing-corrected verdict" over a raw p-value.

## Findings
### Pipeline output
| Pair | Signal | IC mean | IC std | IC IR | IC windows | IC frac pos | Sharpe mean | Sharpe std |
|---|---|---|---|---|---|---|---|---|
| EUR/USD | momentum | −0.3681 | 0.3954 | −0.9311 | 6/10 | 33% | +0.4212 | 2.1894 |
| EUR/USD | mean-reversion | −0.3236 | 0.3017 | −1.0726 | 10/10 | 10% | +0.0572 | 2.0982 |
| GBP/USD | momentum | −0.1728 | 0.3974 | −0.4348 | 7/10 | 43% | +0.0156 | 2.0532 |
| GBP/USD | mean-reversion | −0.4372 | 0.2566 | −1.7039 | 10/10 | 0% | −0.2576 | 1.5627 |
| USD/JPY | momentum | −0.0801 | 0.3196 | −0.2505 | 7/10 | 43% | +0.9877 | 1.9000 |
| USD/JPY | mean-reversion | −0.3752 | 0.2980 | −1.2591 | 10/10 | 10% | −0.2120 | 2.5981 |

Sign convention differs by leg. Reversion IC is on the raw z-score, so negative is the predicted direction and all three rows point the right way. Momentum IC is on the sign of the lookback return, so negative is anti-predictive and all three point wrong, matching Day 49's −0.1037.

### Why none of it is measurable
Test windows are 3 months, 77–79 bars, of which 51–53 survive forward-return masking. At a 26-bar horizon consecutive forward returns overlap by 25 bars, so those 52 bars carry roughly 2 independent observations. A Spearman IC on n≈2 is one realization, not a statistic, and GBP/USD mean-reversion runs from −0.03 to −0.78 across its ten windows.

The geometry is forced. Ten windows over 2015–2022 with a 5-year train leaves only 3-month test windows; twelve-month windows, which Day 49 used, fit twice in the same span.

Momentum fails a second way: the 78-day sign is constant across an entire test window in 4, 3 and 3 of 10 windows, leaving Spearman undefined. Logged as `ic_status: constant_signal` rather than a bare NaN.

### Data quality
| Metric | EUR/USD | GBP/USD | USD/JPY |
|---|---|---|---|
| Minute rows | 4,766,619 | 4,759,785 | 4,700,657 |
| Span | 2011-01-02 17:00 → 2023-12-29 16:58 | same | same |
| Daily bars after resample | 4,056 | 4,055 | 4,056 |
| Daily bars 2015–2022 | 2,498 | 2,497 | 2,498 |
| Duplicate minute timestamps | 300 | 300 | 300 |
| Volume non-zero rows | 0 | 0 | 0 |
| Flat bars (High = Low) | 1.49% | 1.51% | 2.12% |
| Max gap between daily bars | 3 days | 3 days | 3 days |
| Empirical annualization factor | 312.3 | 312.2 | 312.3 |

No NaN closes, non-positive prices, or OHLC violations. Volume is identically zero across all 14.2 million rows, so no volume feature is constructible; with Day 57's finding that the files carry no bid or ask, this is a price-only dataset. The duplicate timestamps fall exclusively in October 2019–2023, daylight-saving transitions against local-time-naive stamps. Daily resampling hides them, which is why `DataLoader`'s duplicate check has never fired, but they are live for intraday work.

### The annualization claim I retracted
I claimed the 312 factor was an artifact of counting Sunday part-sessions as full days, inflating every Sharpe by √(312/260) ≈ 1.09. Wrong, and the second time: `.claude/CLAUDE.md` logs it under Day 57 errors, and `README.md` line 89 already says both factors are correct for their own series.

The session structure is real. Median minute bars: Mon 1,435, Tue–Thu 1,437, Fri 1,020, Sun 416. Resampling emits 672 Sunday bars from roughly seven-hour sessions, 16.6% of the sample, mean |r| 0.00159 against 0.0032–0.0043 on weekdays.

The inference is what fails. Dropping Sunday bars merges the move into Friday-to-Monday rather than deleting it: total log return is −0.1861 with Sundays and −0.1904 without, one path partitioned two ways. Annual variance is the sum of per-period variances and does not care whether periods are equal length, so the coarser partition raises σ by almost exactly the factor it lowers √K.

| Basis | n | K | std | Annualized SR |
|---|---|---|---|---|
| All daily bars | 4,055 | 312.3 | 0.004718 | −0.1719 |
| Sunday dropped | 3,383 | 260.6 | 0.005116 | −0.1776 |

Ratio 0.97, not 1.09, and the opposite direction. The residual 3% is serial dependence, the generic Lo (2002) point, not anything Sunday-specific. `compute_ann_factor()` returns `n_obs / years_spanned`, a measurement rather than a convention. A hardcoded 252 gives −0.1544 against the measured −0.1719, which is what a wrong constant actually does.

## Interpretation
Both things worth keeping came out of the plumbing, not the research question. Wiring four components into one command was the first time they ran as a chain, and composition turned out to be a test nobody had written: it exposed that `WalkForwardValidator.run()` has never scored anything, and that the calendar's window geometry cannot support the strategy's own forward horizon.

The n≈2 result reaches backward into every walk-forward IC in this repo, not just Day 58's. It overturns no verdict, since thin data makes a FAIL easier to reach rather than harder, but it does mean the IC dispersion figures I have been quoting are close to worthless as evidence of instability. The annualization miss is worse, and it is a process failure rather than a math one: the error was already written in my own notes, and four lines of arithmetic settled it the moment someone asked. Restating is cheaper than remeasuring, which is why the rule has to run the other way.

## Next steps
- No annualization change. `compute_ann_factor()` stays.
- Report effective sample size beside every walk-forward IC, retroactively where published. At a 26-bar horizon twelve-month windows give n_eff ≈ 9; the real fix is a longer sample or shorter horizon.
- Wire up or delete `WalkForwardValidator.run()`'s dead `signal_fn`. Five scripts route around it with their own scoring loops, five places for the leakage contract to drift.
- Re-run Day 49 with forward-return masking. On a 90-bar window at a 26-bar horizon the unmasked path scores 26 bars, 29% of the window, whose target falls past `test_end`.
- Nothing here justifies opening the lockbox or changes any strategy's status.
