# Data

*First draft — Day 58. Framework Map assigns the paper data section to Day 61; the Day 58 calendar pulls it forward as reclaimed time. Every figure below is from a direct scan of the raw files, not a cached number.*

## Source and coverage

The study uses 1-minute OHLCV bars for three major currency pairs — EUR/USD, GBP/USD and USD/JPY. The development sample spans 2 January 2011 through 29 December 2023, a period of 12.99 years, and contains 4,766,619, 4,759,785 and 4,700,657 rows respectively, 14.23 million bars in total. Timestamps are local-time-naive and formatted `%Y%m%d %H%M%S`.

Coverage is continuous. The largest gap between consecutive daily observations is 3 days in every pair, corresponding to ordinary weekend closures; no pair exhibits a gap exceeding 4 days anywhere in the sample. There are no missing closing prices, no non-positive prices, and no bars violating the OHLC ordering constraints.

## Known limitations of the source

Three properties of this dataset constrain what can be asked of it, and each is a hard limit rather than a nuisance.

**Volume is identically zero.** The volume column contains zero in all 14.23 million rows across all three files. No volume-based feature — turnover, volume-weighted price, order-flow proxy — is constructible here, and none is used anywhere in this study.

**No bid or ask is recorded.** The files carry a single price series per bar. Every transaction cost figure in this paper therefore rests on assumed spreads drawn from typical ECN quotes, not on observed ones. No spread in this study has been measured.

**Timestamps duplicate across daylight-saving transitions.** Each pair contains exactly 300 duplicated minute timestamps, occurring exclusively in October of 2019, 2020, 2021, 2022 and 2023. Because the stamps carry no timezone, the repeated autumn hour appears twice. Daily-frequency resampling collapses these silently, so they do not affect the results reported here, but they remain a live concern for any intraday extension of this work.

A further 1.49% to 2.12% of minute bars are flat, with high equal to low. These concentrate in thin sessions and are retained rather than filtered.

## Resampling to daily frequency

All analysis is conducted at daily frequency. Minute bars are resampled by taking the last observation within each calendar day and dropping days with no observations, yielding 4,056, 4,055 and 4,056 daily closes per pair. The three series are aligned on a shared, sorted, unique DatetimeIndex. The 2015–2022 evaluation window used throughout the results section contains 2,498, 2,497 and 2,498 daily bars.

Returns are computed as log differences of consecutive daily closes. Log returns are time-additive across periods, which is required for aggregating over arbitrary lookback and holding windows by summation, and are closer to normal for small daily FX moves than simple returns. Where returns must be aggregated across pairs at a single point in time rather than across time, simple returns are used instead, since log returns are not additive across a portfolio.

### The Sunday session and annualization

The FX week opens at 17:00 ET on Sunday. Median minute-bar counts per calendar day are 1,435 on Monday, 1,437 on Tuesday through Thursday, 1,020 on Friday, and 416 on Sunday, with no Saturday bars. Daily resampling therefore emits a Sunday observation built from a roughly seven-hour session — 672 such bars, 16.6% of the daily sample. These partial sessions behave as their length implies: mean absolute daily log return of 0.00159 against 0.00319 to 0.00428 on weekday bars.

The daily index consequently yields an empirical annualization factor of 312.3, 312.2 and 312.3 for the three pairs, against the 260.6 that a weekday-only index would give. This study annualizes using the empirical factor computed from each series' own index rather than any fixed convention.

The larger factor is not a source of bias, and the point is worth making explicitly because the arithmetic is counterintuitive. Excluding Sunday bars does not remove the Sunday price move; it merges that move into the Friday-to-Monday return. The two partitions describe the same asset path — total log return over the EUR/USD sample is −0.1861 including Sunday bars and −0.1904 excluding them. Because annual variance is the sum of per-period variances, and that sum is invariant to how the year is partitioned, σ√K estimates the same annual volatility on either basis: the coarser partition raises the per-period standard deviation from 0.004718 to 0.005116 by almost exactly the factor by which it lowers √K. Annualized Sharpe on EUR/USD is −0.1719 including Sunday bars and −0.1776 excluding them, a ratio of 0.97.

The residual three percent reflects aggregation and serial dependence rather than session structure, and is an instance of the general result that Sharpe annualization by √K is sensitive to autocorrelation in the underlying returns (Lo, 2002). That sensitivity applies at any sampling frequency and is treated in the limitations section.

What the empirical factor does protect against is a fixed constant. Annualizing this same series with a hardcoded 252 gives −0.1544 against the measured −0.1719, a discrepancy with no justification behind it.

## Embargo and train/test separation

Walk-forward evaluation enforces a directional gap between the end of each training window and the start of the corresponding test window. The embargo is 5 days throughout, sized to exceed the longest feature lookback so that no test-set feature reads training data. The gap is directional: trailing training rows adjacent to the boundary are not purged, since features are causal and the leakage risk runs only forward.

A second, distinct leakage channel arises from the target rather than the features. Forward returns over a horizon of h bars are, by construction, forward-looking; for the final h bars of any test window the target price falls beyond the window boundary, inside the embargo or the following training region. Those observations are masked rather than scored, so every scored bar resolves its full holding horizon within its own test window.

Overlapping forward returns remain a limitation of the design. Consecutive h-bar forward returns share h−1 bars, so the effective number of independent observations in a test window of n scored bars is approximately n/h rather than n. This ratio governs how much statistical weight any per-window estimate can carry, and it is reported alongside the results rather than left implicit.

## Reserved holdout

The interval from 1 January 2024 onward is reserved for a single unbiased evaluation of a strategy that has already survived every development-phase test. It entered aggregate descriptive statistics during early development but was never used for strategy selection or evaluation, and no result reported in this paper is computed on it. The command-line research entry point refuses any end date reaching this interval unless an explicit override flag is supplied, so that the holdout cannot be spent by accident.

All results reported in this paper are drawn from data ending 31 December 2022.
