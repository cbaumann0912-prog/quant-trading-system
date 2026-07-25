# Strategy Specification: Volatility-Conditioned Cross-Sectional FX Carry

**Date drafted:** Day 57 (2026-07-25)
**Status:** Pre-registered. Written before any test was run.

## Provenance
Strategy #6. Five tested previously, all null: PC2 Carry Regime, Momentum w/ ML Regime, OU Half-Life Mean Reversion, Volatility Regime Breakout/Mean-Reversion (and its momentum-only successor, closed in `momentum_book_invalidation.md`), Month-End FX Flow (closed in `day57_month_end_fx_flow_h1.md`). Honest `n_trials` for the deflated Sharpe is 6.

The carry premium itself is documented back to Fama (1984) and is not a discovery. What is being tested here is the volatility conditioning in H2. H1 exists to establish the premium is present in this sample at all, which is not guaranteed.

## 0. Feasibility, computed before the hypothesis
Every strategy this project has tested was specified first and costed later, and all six died on effect sizes smaller than their own trading costs. This section runs the arithmetic first. If the strategy cannot clear both hurdles, it does not get tested.

**Cost hurdle.** Monthly rebalance across 6 positions, historically about 2 position changes per month, so roughly 24 legs per year. At 1.0 pip round trip (about 0.9bp on the majors) that is **0.22%/year**. FX majors run about 8% annualized vol, so a Sharpe-0.3 book earns about 2.4%/year gross. Cost consumes 9% of gross.

For comparison, the same 0.9bp at daily turnover costs 2.27%/year and consumes the entire gross return. That difference is the whole reason this spec is monthly.

**Power hurdle, and the binding constraint.** The t-statistic on an annualized Sharpe is approximately SR x sqrt(years). The development sample is 13 years, so t = 2 requires **SR >= 0.55**. Carry's long-run Sharpe is usually reported in the 0.4 to 0.6 range, but 2011-2023 spans the ZIRP era, when policy rates across all eight currencies compressed toward zero and the cross-sectional dispersion carry depends on largely disappeared.

Stated plainly: if carry's true Sharpe in this sample is 0.3, this test cannot detect it, and a null result would carry almost no information. FX price data starts in 2011, so extending the sample is not available. Confirm using `compute_required_sample_size` and `compute_achieved_power` from `src/stats/hypothesis_tests.py` before running H1, and record the achieved power in the audit alongside the p-value.

This is the first spec in the project to state its own detection floor up front. Doing so earlier would have saved most of Days 42-57.

## 1. Hypothesis
**H1, base premium.** Cross-sectional FX carry earns a positive risk premium: a monthly-rebalanced book long the highest-yielding currencies and short the lowest-yielding earns a positive Sharpe over 2011-2023, net of cost.

**H2, the contribution.** Carry returns are conditionally worse in high-volatility regimes. Carry is compensation for crash risk (Brunnermeier, Nagel & Pedersen 2008), and crashes cluster in risk-off episodes, so a volatility-regime filter should improve risk-adjusted returns by standing aside when the crash risk being compensated is most likely to realize.

Falsification criteria, binding:

1. Direction requirement: every coefficient must be significant **and** carry the predicted sign. A significant coefficient of the wrong sign is a FAIL. Carried over from the momentum book and the month-end test, both of which produced exactly that.
2. Primary threshold: p < 0.05 on H1's Sharpe, block-bootstrap standard errors.
3. Reliability gate on any regression: condition number < 1e10, all VIF < 10, main effects mean-centered before interaction.
4. Multiple testing: Benjamini-Hochberg across 6 strategies. The other 5 are null, so this needs p < 0.0083 at rank 1 of 6.
5. Cost gate: net-of-cost Sharpe positive at 1.0 pip round trip, computed on realized turnover rather than assumed turnover.
6. Power disclosure: achieved power reported with every result. A null below the detection floor is recorded as inconclusive, not as evidence against.

## 2. Economic rationale
Carry earns a premium because it is short crash risk. High-yield currencies depreciate suddenly during risk-off episodes, so the carry trader is paid to hold an exposure that loses badly and correlates with everything else when it does. The counterparty is an investor buying insurance against exactly that event. This is why the premium survives publication: it is compensation for a risk, not a mispricing to be arbitraged.

The conditioning in H2 follows directly. If the premium is payment for crash risk, and crashes cluster in high-volatility regimes, then carry's return distribution should be materially worse in those regimes. That is a mechanism-derived prediction, not a fitted parameter.

What would make this uninteresting: if H2's improvement comes entirely from reduced exposure rather than better timing, the filter is doing nothing a smaller position size would not do. Section 10 tests that explicitly.

## 3. Data
Eight currencies: USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD.

Rates: FRED OECD 3-month interbank, `data/{region}_3m_interbank.csv`, monthly, all eight covering 2011-2023. Shifted forward 2 months before forward-filling, per the publication-lag convention already used in this repo.

Prices: daily closes via `DataLoader`. Each non-USD currency is expressed against USD, inverting where the pair is USD-quoted (USD/JPY, USD/CHF, USD/CAD).

Development 2011-01-01 to 2023-12-31. Lockbox 2024-01-01 to 2026-05-01, sealed.

## 4. Signal logic
All parameters fixed as of this document.

1. On the last trading day of each month, rank all 8 currencies by lagged 3-month interbank rate.
2. Long the top 3, short the bottom 3, equal weight, dollar-neutral. USD nets out by construction when long and short notionals match.
3. Hold one calendar month. Rebalance only at month end.
4. Volatility regime (H2 only): fit `fit_garch` per currency-vs-USD return series, average conditional vol across the book, classify with `classify_vol_regime(n_regimes=2)`. GARCH refit walk-forward per Day 47, never full-sample.

| Parameter | Value |
|---|---|
| Universe | 8 currencies |
| Ranking variable | 3m interbank rate, 2-month publication lag |
| Long / short | top 3 / bottom 3, equal weight |
| Rebalance | monthly, last trading day |
| Regime classifier | GARCH conditional vol, 2-means, walk-forward refit |
| Development sample | 2011-01-01 to 2023-12-31 |

## 5. Entry rule
Enter at the close of the last trading day of each month, equal weight across the 3 long and 3 short currencies. H2 variant: hold the same book in low-vol regimes, flat in high-vol regimes.

## 6. Exit rule
Positions roll monthly. A currency leaving the top or bottom 3 is closed at the next month-end rebalance. No stops, no intra-month discretion.

## 7. Position sizing
Deferred until H1 passes, per the pattern that left the momentum book's Section 7 open through invalidation. Validation uses equal-weight unit exposure, a measurement convention rather than a sizing proposal.

## 8. Risk controls
Deferred with Section 7. Named now: carry books have strongly negative skew and this one has no stop-loss. The 25% drawdown halt carried through earlier specs was judgmental and has never been derived. Compute realized skew and CVaR (`src/analysis/portfolio.py`) alongside Sharpe rather than after.

## 9. Failure conditions
- Net-of-cost Sharpe non-positive at 1.0 pip round trip on realized turnover.
- H1 null with achieved power below 0.5, recorded as inconclusive rather than as a rejection.
- H2 improvement explained entirely by reduced time in market, per the Section 10 exposure-matched control.
- Rate dispersion across the 8 currencies falling low enough that the top-3 and bottom-3 baskets are not meaningfully different, which would make the signal degenerate regardless of what the returns do.

## 10. Statistical validation plan
Gatekeeping, as in the month-end spec. H2 runs only if H1 passes.

**H1.** Monthly book returns, 2011-2023. Report Sharpe with block-bootstrap confidence interval (`block_bootstrap`, 12-month blocks to preserve annual seasonality), t-statistic, deflated Sharpe at `n_trials=6`, skew, excess kurtosis, max drawdown, realized turnover, and net-of-cost Sharpe. Achieved power reported alongside.

**H1 robustness.** Long-2/short-2 and long-4/short-4 baskets. The premium should not depend on the basket size chosen.

**H2.** Split monthly book returns by volatility regime and compare Sharpe via `regime_conditional_performance`. Prediction: low-vol Sharpe materially exceeds high-vol Sharpe. Then run the filtered book (flat in high vol) against the unfiltered book.

**H2 exposure-matched control.** Compare the filtered book against an unfiltered book scaled to the same average exposure. If the filtered book's advantage disappears, the regime filter is a position-size reduction wearing a costume, and H2 fails regardless of the raw Sharpe comparison.

**Standard errors.** Block bootstrap throughout. Monthly carry returns are serially correlated through persistent rate differentials, so analytic errors will be optimistic.

**Lockbox.** Opened once, only on a PASS, only after the development verdict is written down.

## 11. Open questions and known gaps
The ZIRP power problem in Section 0 is the largest risk and cannot be engineered around with the data available. If H1 comes back null with low achieved power, the correct conclusion is that this sample cannot answer the question, not that carry does not exist.

Eight currencies at top-3/bottom-3 leaves little cross-sectional room. Two currencies moving one rank can change a third of the book.

3-month interbank rates are a proxy for the actual forward points a carry trade earns. Forward points include a basis that has been persistently non-zero since 2008 and would change the ranking in some months. No forward data is available here, and this gap should be stated in any result.

Monthly rebalancing at month-end close puts this book's execution in the same window the month-end flow audit found to be unusual. Whether that matters at monthly holding periods is untested.
