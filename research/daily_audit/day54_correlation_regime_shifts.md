# Day 54 Research Audit: Correlation Regime Shifts, All Pairs

## Question
Do correlation regime shifts cluster in time across the full FX universe, and does that clustering line up with known macro events? Day 5 only checked three pairs. This runs all 10 pairs, 45 combinations, where a real shock should show up as many pairs breaking at once, not just one.

## Scope Note
A descriptive census, not a trading signal or backtest: the output is a boolean flag series, no sizing applied. Development window only, 2011-2023, lockbox excluded.

## Methodology
`day54_correlation_regime_shifts.py` runs `detect_correlation_regime_shifts` (Fisher z-transformed CUSUM with re-baselining after each alarm, `src/stats/correlation.py`) on daily log returns for all 10 pairs (4,053 aligned observations), all 45 combinations, window=60, threshold=3.0 rather than the function's default of 2.0, since 45 parallel tests raise false-discovery risk beyond what one pair carries.

Threshold calibration was checked against real data, not synthetic noise: a paired block bootstrap (block_size=21, matching the Day 37 convention) resampled EUR/USD-GBP/USD's longest flag-free stretch (2014-08-29 to 2018-04-26), preserving the fat tails and autocorrelation of actual FX returns. Across 100 resamples, the empirical false-alarm rate came out to 4.3% per evaluation point. A window=30 variant was checked the same way (4.6%, comparable) before being run on the full dataset as a robustness cross-check.

## Findings
273 flags, 45 combinations, about 6 per combination over 13 years. All combinations share the same evaluation dates, so clustering is a per-date count. Detection lags its trigger by roughly one window, 20-60 trading days, so each date below is matched to the nearest documented event inside that lookback:

| Date | Combos flagged (of 45) | Driver |
|---|---|---|
| 2013-07-03 | 13 | Taper Tantrum (Bernanke's May 22 testimony, June 19 FOMC) |
| 2022-12-05 | 13 | Q4 2022 USD reversal (October CPI print, BOJ/MOF yen intervention) |
| 2016-05-23 | 12 | Brexit referendum run-up (vote June 23) |
| 2020-03-29 | 11 | COVID dash-for-cash (Mar 9-23 liquidity crisis, Fed QE Mar 23) |
| 2015-03-27 | 10 | SNB floor removal (Jan 15) + ECB QE launch (Jan 22) |
| 2013-02-13 | 9 | Abenomics yen depreciation (Abe's Dec 26 election win, USD/JPY 80 to 90 by late Jan) |
| 2022-02-28 | 8 | Russia's invasion of Ukraine (Feb 24), 2-4 days' lag, the tightest of any event here |
| 2012-05-08 | 8 | Greek election (May 6), Grexit fears, EUR to a 4-month low within days |

Most flagged dates, 25 of 64, are isolated: one or two combinations. A handful, 8 dates, have 8 or more.

The window=30 cross-check reproduces the same major events but with a different lag profile: it catches the actual Brexit result (2016-06-27, days after the vote) alongside window=60's pre-vote date, and roughly doubles total flags (477 vs. 273) at a similar false-alarm rate.

## Interpretation
The cluster-size split is the finding, not any single date. At a 4.3% real false-alarm rate, a meaningful share of the 25 isolated single-combination flags are plausibly noise rather than signal. Multi-combination clusters are a different story: treating combinations as independent, the chance of 8 or more of 45 flagging the same date by coincidence is about 6×10⁻⁴, falling to roughly 4×10⁻⁸ at 13/45. Every date in the table above clears that bar comfortably, spanning most of the currency universe rather than pairs sharing a leg, consistent with genuine market-wide decorrelation rather than a lucky alignment of noisy detectors.

Day 5's 3-pair audit put GBP/USD-USD/JPY's Brexit break at May-August 2016. This run shows the decorrelation was broader, hitting most GBP and EUR crosses, and earlier, a month before the vote rather than after. A materially different picture from the single-pair result.

## Alternative Explanations
The 4.3% false-alarm rate comes from bootstrapping one pair's quiet stretch (EUR/USD-GBP/USD). Whether that rate generalizes to all 45 combinations, some of which involve thinner-traded crosses with different tail behavior, is untested. If the true rate varies meaningfully across combinations, the isolated-flag noise estimate above is a rough average, not a per-pair guarantee. The clustering argument is more robust to this uncertainty: it would take a substantially higher false-alarm rate across the board to make an 8+ combination coincidence plausible.

## Next Steps
- Identify the drivers behind the window=30 cross-check's new candidate dates not yet matched to events: 2014-11-07, 2011-11-13, 2022-10-31, 2021-06-28, 2019-11-10, 2020-05-03.
- Extend the false-alarm bootstrap beyond EUR/USD-GBP/USD to a sample of combinations spanning majors and thinner crosses, to check whether 4.3% holds broadly or varies by pair.
