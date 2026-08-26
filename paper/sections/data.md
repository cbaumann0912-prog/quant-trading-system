# Data

*Rescoped Day 72 from the Day 58 three-pair draft. Every figure below is from a direct scan of the raw files by `research/applied_analysis/day72_paper_data_section_scan.py`, not a cached number.*

## Source and coverage

The study uses 1-minute OHLCV bars for ten currency pairs — EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY and EUR/CHF. The development sample spans 2 January 2011 through 29 December 2023, a period of 12.99 years, and contains 47,287,601 minute bars in total. Timestamps are local-time-naive and formatted `%Y%m%d %H%M%S`.

| Pair | Minute bars | Daily closes | Flat bars | Duplicate stamps |
|---|---:|---:|---:|---:|
| EUR/USD | 4,766,619 | 4,056 | 1.49% | 300 |
| GBP/USD | 4,759,785 | 4,055 | 1.51% | 300 |
| USD/JPY | 4,700,657 | 4,056 | 2.12% | 300 |
| USD/CHF | 4,722,674 | 4,055 | 3.29% | 299 |
| AUD/USD | 4,749,069 | 4,055 | 1.76% | 300 |
| USD/CAD | 4,691,825 | 4,055 | 2.49% | 299 |
| NZD/USD | 4,715,513 | 4,055 | 2.49% | 300 |
| EUR/GBP | 4,725,060 | 4,055 | 1.96% | 300 |
| EUR/JPY | 4,782,753 | 4,056 | 0.55% | 300 |
| EUR/CHF | 4,673,646 | 4,055 | 3.65% | 300 |

Coverage is continuous. The largest gap between consecutive daily observations is 3 days in every pair, corresponding to ordinary weekend closures; no pair exhibits a gap exceeding 4 days anywhere in the sample. Across all 47.3 million bars there are no missing closing prices, no non-positive prices, and no bars violating the OHLC ordering constraints.

Eight three-month interbank rate series accompany the price data, one each for the United States, euro area, United Kingdom, Japan, Switzerland, Canada, Australia and New Zealand. These are monthly, not daily: 156 observations apiece over 2011-01 through 2023-12. Any carry or rate-differential factor built from them inherits that frequency, and a monthly rate aligned onto a daily price panel is a step function, not a daily observation.

## Known limitations of the source

Three properties of this dataset constrain what can be asked of it, and each is a hard limit rather than a nuisance.

**Volume is identically zero.** The volume column contains zero in all 47,287,601 rows across all ten files. No volume-based feature — turnover, volume-weighted price, order-flow proxy — is constructible here, and none is used anywhere in this study.

**No bid or ask is recorded.** The files carry a single price series per bar. Every transaction cost figure in this paper therefore rests on assumed spreads drawn from typical ECN quotes, not on observed ones. No spread in this study has been measured.

**Timestamps duplicate across daylight-saving transitions.** Each pair contains 299 or 300 duplicated minute timestamps, occurring exclusively in October of 2019, 2020, 2021, 2022 and 2023. Because the stamps carry no timezone, the repeated autumn hour appears twice. Daily-frequency resampling collapses these silently, so they do not affect the daily results reported here, but they are a live concern for the intraday work in §4.6, which reads minute bars directly.

Between 0.55% and 3.65% of minute bars are flat, with high equal to low. These concentrate in thin sessions and, for EUR/CHF, in the SNB floor period: that pair is the highest in the universe at 3.65% overall, and its flat share by year peaks at 7.00% in 2012 and 7.55% in 2014 against 1.75% to 3.62% in every year from 2015 on. Flat bars are retained rather than filtered.

## Resampling to daily frequency

Statistical work outside §4.6 is conducted at daily frequency. Minute bars are resampled by taking the last observation within each calendar day and dropping days with no observations, yielding 4,055 or 4,056 daily closes per pair. The ten series are aligned on a shared, sorted, unique DatetimeIndex, producing a panel of 4,056 rows over 2011-01-02 to 2023-12-29 with 7 missing cells; complete-case daily log returns number 4,053.

Returns are computed as log differences of consecutive daily closes. Log returns are time-additive across periods, which is required for aggregating over arbitrary lookback and holding windows by summation, and are closer to normal for small daily FX moves than simple returns. Where returns must be aggregated across pairs at a single point in time rather than across time, simple returns are used instead, since log returns are not additive across a portfolio.

### Reconciling the evaluation window

The Day 58 draft of this section named a 2015–2022 evaluation window. That window describes no result in this study and has been dropped. The program's window is **2 January 2011 to 31 December 2023**: 26 research scripts carry `DEV_END = "2023-12-31"` as a module constant, `START = "2011-01-01"` is the corresponding convention, and every strategy verdict in §4 was computed on that range. Two 2015 dates do survive in the repository and neither defines an evaluation window — `GARCH_DEV_START` in the figure and table generators, which is a volatility-model warm-up boundary, and the `--train-start` default on the command-line runner, which is a CLI convenience.

### The Sunday session and annualization

The FX week opens at 17:00 ET on Sunday. Median minute-bar counts per calendar day are 1,435 on Monday, 1,437 on Tuesday through Thursday, 1,020 on Friday, and 416 on Sunday, with no Saturday bars. Daily resampling therefore emits a Sunday observation built from a roughly seven-hour session — 671 to 672 such bars per pair, 16.5% to 16.6% of the daily sample. These partial sessions behave as their length implies: on EUR/USD, mean absolute daily log return of 0.00159 against 0.00319 to 0.00428 on weekday bars.

The daily index consequently yields an empirical annualization factor of 312.19 or 312.27 for every pair in the universe, against the 260.57 that a weekday-only index would give. This study annualizes using the empirical factor computed from each series' own index rather than any fixed convention.

The larger factor is not a source of bias, and the point is worth making explicitly because the arithmetic is counterintuitive. Excluding Sunday bars does not remove the Sunday price move; it merges that move into the Friday-to-Monday return. The two partitions describe the same asset path — total log return over the EUR/USD sample is −0.1861 including Sunday bars and −0.1904 excluding them. Because annual variance is the sum of per-period variances, and that sum is invariant to how the year is partitioned, σ√K estimates the same annual volatility on either basis: the coarser partition raises the per-period standard deviation from 0.004718 to 0.005116 by almost exactly the factor by which it lowers √K. Annualized Sharpe on EUR/USD is −0.1719 including Sunday bars and −0.1776 excluding them, a ratio of 0.97.

The residual three percent reflects aggregation and serial dependence rather than session structure, and is an instance of the general result that Sharpe annualization by √K is sensitive to autocorrelation in the underlying returns (Lo, 2002). That sensitivity applies at any sampling frequency and is treated in the limitations section.

What the empirical factor does protect against is a fixed constant. Annualizing this same series with a hardcoded 252 gives −0.1544 against the measured −0.1719, a discrepancy with no justification behind it.

### Two annualization factors, both correct

Two different factors appear in this paper and the distinction matters. The daily panel measures 312.19–312.27 observations per year because the vendor buckets the Sunday open as its own date. The intraday session book of §4.6 measures 259.44 sessions per year over the same 12.99 years, because that strategy trades one 09:00–13:00 session per weekday and never trades the Sunday stub at all. Neither number is a correction of the other. They describe two different observation streams drawn from the same underlying bars, and each is computed from its own index. Applying either to the other's return series would misstate the annualized Sharpe by roughly 10%, which is the kind of error that survives review precisely because both constants look plausible.

## Embargo and train/test separation

Walk-forward evaluation enforces a directional gap between the end of each training window and the start of the corresponding test window. The embargo is 5 rows of the resampled daily index throughout — weekends are already dropped, so an embargo "day" is a row and not a calendar day — sized to exceed the longest feature lookback so that no test-set feature reads training data. The gap is directional: trailing training rows adjacent to the boundary are not purged, since features are causal and the leakage risk runs only forward.

A second, distinct leakage channel arises from the target rather than the features. Forward returns over a horizon of h bars are, by construction, forward-looking; for the final h bars of any test window the target price falls beyond the window boundary, inside the embargo or the following training region. Those observations are masked rather than scored, so every scored bar resolves its full holding horizon within its own test window.

Overlapping forward returns remain a limitation of the design. Consecutive h-bar forward returns share h−1 bars, so the effective number of independent observations in a test window of n scored bars is approximately n/h rather than n. This ratio governs how much statistical weight any per-window estimate can carry, and it is reported alongside the results rather than left implicit.

## Reserved holdout

The interval from 1 January 2024 onward is reserved for a single unbiased evaluation of a strategy that has already survived every development-phase test. No result reported in this paper is computed on it, and it has never been opened. The command-line research entry point enforces this mechanically: `guard_lockbox` raises on any end date at or beyond `LOCKBOX_START = 2024-01-01` unless `--allow-lockbox` is passed explicitly, so the holdout cannot be spent by accident.

The guard was added partway through the project rather than at the start, and two artifacts in the repository still carry pre-guard figures computed on data running through 2026. Both are identified in §4.1 and neither bears on a verdict.

All results reported in this paper are drawn from data ending 29 December 2023.

## The universe was chosen without checking what it cost

One design decision deserves stating here rather than only in the limitations, because it conditions every result that follows. Ten pairs at 13 years was chosen for breadth. The breadth it actually delivered, measured on the one book that pooled all ten, was 2.34 effective independent signals out of 10 (§4.6).

The span that breadth cost is quantified. The 2011 start is a download choice, not a vendor limit; the binding constraint on a common start date across all ten pairs is NZD/USD at 2005-08. Dropping that single pair moves the common start to 2002-03 and buys 8.8 years, taking the Sharpe required for t = 2 from 0.555 to 0.428. The trade was breadth against span, and at a realized breadth of 2.34 the marginal pair was worth far less than the marginal year. §6 returns to this as the study's central limitation.
