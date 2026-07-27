# Day 58 Research — CLI Run All Pairs Both Signals

## Verdict

**Infrastructure: PASS.** `run_research.py` runs the chain end to end on all 6 pair/signal combinations and writes 6 well-formed JSON reports to `results/`. 11 CliRunner tests pass.

**No research verdict is issued and `n_trials` is not incremented.** The window geometry the calendar specifies produces roughly **2 effective independent observations per window**, which is not enough to measure an IC. The ranked table the calendar asks for is not published because the quantity it would rank is noise.

## Methodology

Entry point `run_research.py` (Click 8.4.1). Chain: `DataLoader.load` → signal construction → `WalkForwardValidator.generate_windows` → per-window OOS scoring → JSON. Signals from `src/signals/momentum.py` and `src/signals/mean_reversion.py`. IC via `information_coefficient` in `src/analysis/performance_analyzer.py`. Six invocations, `--train-start 2015-01-01 --train-end 2022-12-31 --windows 10 --train-years 5 --test-months 3`, momentum lookback 78, z-score lookback 26, forward horizon 26, embargo 5 days.

Scoring is a CLI-local loop, not a framework component. `WalkForwardValidator.run()` accepts `signal_fn` but never calls it — the docstring reads "Reserved for future use (scoring integration)." Every existing pipeline in the repo works around this the same way, by calling `generate_windows()` and hand-writing its own scoring. `src/framework/walk_forward.py` was left untouched.

Two deviations from the shape of `vol_regime_signal_report_pipeline.py`, both deliberate. Forward returns are masked over the final `holding_period` bars of every test window, so no scored bar reads a price beyond `test_end` into the embargo or the next training region; the Day 49 script does not do this. And `build_signal_report` is not invoked — its contract requires a regime-interaction p(b3) that an ungated single-signal run does not produce, and a one-leg Benjamini-Hochberg correction is a no-op that would print "multiple-testing-corrected verdict" over a raw p-value.

Data-quality figures come from a direct scan of the three raw minute files, not from cached numbers.

## Findings

### Pipeline output — 6 runs

| Pair | Signal | IC mean | IC std | IC IR | IC windows | IC frac pos | Sharpe mean | Sharpe std |
|---|---|---|---|---|---|---|---|---|
| EUR/USD | momentum | −0.3681 | 0.3954 | −0.9311 | 6/10 | 33% | +0.4212 | 2.1894 |
| EUR/USD | mean-reversion | −0.3236 | 0.3017 | −1.0726 | 10/10 | 10% | +0.0572 | 2.0982 |
| GBP/USD | momentum | −0.1728 | 0.3974 | −0.4348 | 7/10 | 43% | +0.0156 | 2.0532 |
| GBP/USD | mean-reversion | −0.4372 | 0.2566 | −1.7039 | 10/10 | 0% | −0.2576 | 1.5627 |
| USD/JPY | momentum | −0.0801 | 0.3196 | −0.2505 | 7/10 | 43% | +0.9877 | 1.9000 |
| USD/JPY | mean-reversion | −0.3752 | 0.2980 | −1.2591 | 10/10 | 10% | −0.2120 | 2.5981 |

Sign convention differs by leg. Mean-reversion IC is measured on the raw z-score, so a **negative** IC is the predicted direction — all three reversion rows point the right way. Momentum IC is measured on the sign of the lookback return, so a negative IC is **anti**-predictive — all three momentum rows point the wrong way, consistent with the Day 49 result (momentum IC −0.1037).

### Why none of the above is measurable

| Quantity | Value |
|---|---|
| Test window length | 3 months, 77–79 daily bars |
| Bars scored after forward-return masking | 51–53 |
| Forward-return horizon | 26 bars |
| **Effective independent observations per window** | **1.96 – 2.04** |

Consecutive forward returns overlap by 25 of 26 bars. Fifty-two scored bars of 26-bar overlapping returns carry about two independent observations. A Spearman IC on n≈2 is a single realization, not a statistic, and the per-window spread bears that out — GBP/USD mean-reversion ranges from −0.03 to −0.78 across its ten windows.

The window geometry is forced, not chosen. `--windows 10` over 2015–2022 with `--train-years 5` leaves only 3-month test windows. Twelve-month test windows, which is what the Day 49 pipeline used, fit only 2 windows in the same span.

Momentum fails a second way: the 78-day momentum sign is constant across an entire 3-month test window in **4/10, 3/10 and 3/10** windows respectively, leaving Spearman undefined. These are recorded as `ic_status: constant_signal` in the JSON rather than collapsed into a bare NaN.

### Data quality — 3 pairs, raw minute files

| Metric | EUR/USD | GBP/USD | USD/JPY |
|---|---|---|---|
| Minute rows | 4,766,619 | 4,759,785 | 4,700,657 |
| Span | 2011-01-02 17:00 → 2023-12-29 16:58 | same | same |
| Years | 12.99 | 12.99 | 12.99 |
| Daily bars after resample | 4,056 | 4,055 | 4,056 |
| Daily bars 2015–2022 | 2,498 | 2,497 | 2,498 |
| NaN / non-positive close | 0 / 0 | 0 / 0 | 0 / 0 |
| OHLC violations | 0 | 0 | 0 |
| Duplicate minute timestamps | 300 | 300 | 300 |
| Volume non-zero rows | 0 | 0 | 0 |
| Flat bars (High = Low) | 1.49% | 1.51% | 2.12% |
| Max gap between daily bars | 3 days | 3 days | 3 days |
| Empirical annualization factor | 312.3 | 312.2 | 312.3 |

**Volume is identically zero across all 14.2 million rows in all three files.** No volume-based feature is constructible from this dataset. Combined with the Day 57 finding that the files carry no bid or ask, the dataset supports price-only research and nothing else.

**Duplicate minute timestamps occur exclusively in October** — 2019-10, 2020-10, 2021-10, 2022-10 and 2023-10. These are daylight-saving transitions against local-time-naive stamps. `resample("D").last()` hides them at daily frequency, which is why `DataLoader`'s duplicate check has never fired, but they are live for any intraday work.

### The annualization factor — a claim I made and then retracted

The session structure is real. Median minute bars per calendar day: Mon 1,435 · Tue 1,437 · Wed 1,437 · Thu 1,437 · Fri 1,020 · **Sun 416** · Sat 0. The FX week opens 17:00 ET Sunday, so `resample("D").last()` emits a Sunday bar built from a ~7-hour session, 672 of them, 16.6% of daily bars, with mean |r| of 0.00159 against 0.0032–0.0043 on weekdays.

From that I concluded the 312 factor was an artifact inflating every Sharpe by √(312/260) ≈ 1.09. **That conclusion is wrong.**

Dropping Sunday bars does not delete the Sunday price move. It merges that move into the Friday→Monday return. Total log return over the sample is −0.1861 with Sundays and −0.1904 without: the same asset path, partitioned two ways. Annual variance is the sum of per-period variances, and that sum does not care whether the periods are equal in length, so σ√K estimates the same annual volatility on either partition. The coarser partition raises σ (0.004718 → 0.005116) by almost exactly the amount it lowers √K.

Measured on EUR/USD across the development sample:

| Basis | n | K | std | Annualized SR |
|---|---|---|---|---|
| All daily bars | 4,055 | 312.3 | 0.004718 | −0.1719 |
| Sunday dropped | 3,383 | 260.6 | 0.005116 | −0.1776 |

Ratio 0.97, not 1.09, and in the opposite direction to the one I claimed. The residual 3% is aggregation and serial dependence, which is the generic Lo (2002) point about Sharpe annualization under autocorrelation and is not Sunday-specific.

`compute_ann_factor()` returns `n_obs / years_spanned` — the empirical bar count of whatever series it is handed. That is a measurement, not a convention, and it cannot be "the wrong constant" the way a hardcoded 252 could be. For contrast, on this same series a hardcoded 252 gives SR −0.1544 against the empirical −0.1719; that is what an actually wrong constant looks like.

**This is the second time I have made this claim.** `.claude/CLAUDE.md` records it under Day 57 errors as "claimed 312 was a wrong convention when it is the empirical bar count of the data," alongside the instruction to compute before stating. I asserted it again on Day 58 and called it the day's most consequential finding, then measured it only when challenged. `README.md` line 89 already documents the Sunday bucketing and already says both factors are correct for their own series.

## Interpretation

*Per the Day 27 amendment this section is yours to write. Draft reasoning below — replace it.*

The infrastructure works and the day's stated deliverable is met, but the research question the calendar attached to it was not answerable with the parameters it specified. The useful output of Day 58 is the window-geometry finding: a 3-month test window cannot support a 26-bar forward horizon, and every walk-forward IC in this repo should be read against its effective sample size rather than its nominal one.

The annualization episode is the other thing worth keeping. A large, confident, wrong claim survived from Day 57 to Day 58 because it was restated rather than remeasured, and it took four lines of arithmetic to settle once someone asked.

## Next steps

- No annualization change is warranted. `compute_ann_factor()` stays as is.
- Fix the window geometry before treating any walk-forward IC as evidence. Test windows should span several multiples of the forward horizon. At a 26-bar horizon, 12-month windows give n_eff ≈ 9, which is still thin; the real fix is a longer sample or a shorter horizon.
- `WalkForwardValidator.run()`'s dead `signal_fn` parameter should either be wired up or removed. Five scripts currently route around it, each with its own scoring loop, which is five places for the leakage contract to drift apart.
- The forward-return masking in `run_research.py` is not in the Day 49 script. On a 90-bar test window at a 26-bar horizon, the unmasked path scores 26 bars — **29% of the window** — whose target price falls beyond `test_end`. Whether the published vol-regime numbers move once that is corrected is untested and worth one run to find out.
- No part of this justifies opening the lockbox, and nothing here changes any strategy's status.
