# Strategy Specification: Month-End FX Rebalancing Flow

**Date drafted:** Day 57 (2026-07-25)
**Status:** Pre-registered. Written before any test was run.

## Provenance
Nothing in `research/` has examined month-end effects, fix windows, or calendar-timed flows. Every threshold and window in Section 4 is fixed here, before the first test.

This is strategy #5. Four were tested previously (PC2 Carry Regime, Momentum w/ ML Regime, OU Half-Life Mean Reversion, Volatility Regime Breakout/Mean-Reversion, the last surviving as `momentum_only_pooled_book.md` until `day57_momentum_book_invalidation.md` closed it). Day 56-57 session work explored roughly 26 configurations of that dead hypothesis, all null and all logged there. Honest `n_trials` for the deflated Sharpe here is 5.

**What is borrowed and what is not.** The directional month-end hypothesis (H1) is a replication of Melvin & Prins. 

Three things are not in that literature as far as this repo's reading goes, and they are the contribution:

1. The post-fix reversal test (H2) as a *falsification* of the mechanism rather than a confirmation of the effect. The published work establishes the flow exists; H2 asks whether the price impact is temporary, which is what "non-informational" actually commits you to.
2. Volatility-conditioned flow magnitude (H3), derived from the mechanism rather than fitted, using the GARCH regime classifier built on Days 55-56.
3. A cross-sectional dollar-neutral construction across 10 pairs, where the published treatment is largely USD-centric and per-pair.

If H1 replicates and H2 and H3 are null, the honest result is "known effect reproduced, proposed mechanism refinement unsupported." That is still a result.

## 1. Hypothesis
FX spot drifts predictably in the final trading days of each calendar month, concentrated around the 16:00 London fix, driven by mechanical portfolio-hedge rebalancing rather than information.

Direction is conditioned on the preceding month's hedging need, proxied by the month's cumulative FX return (Section 4). The effect must be stronger at month end than on other days, and stronger inside the fix window than outside it. Either alone is consistent with something other than the stated mechanism.

Falsification criteria, binding:

1. Test statistic: the month-end × fix-window interaction coefficient from the Section 10 primary regression.
2. Direction requirement: the coefficient must be significant and carry the predicted sign. A significant coefficient of the wrong sign is a FAIL. Added in response to Day 48/57, where a significance-only rule passed a leg whose effect ran backwards.
3. Primary threshold: p < 0.05, uncorrected.
4. Reliability gate: condition number < 1e10, all VIF < 10, main effects mean-centered before the interaction. Precedent: Day 41 PC2 discarded at condition number 2.27e10.
5. Robustness: primary plus both robustness checks must agree unanimously, same sign. Any single null kills it.
6. Multiple testing: must survive Benjamini-Hochberg across 5 strategies. The other 4 are null, so this needs p < 0.01 at rank 1 of 5.
7. Cost gate: net-of-cost annualized return must stay positive at 1.0 pip round trip. Added because the Day 57 session variant produced a positive gross Sharpe entirely consumed by turnover.

## 2. Economic rationale
The mechanism is a structural, non-informational flow. Institutions holding foreign equity and bond portfolios hedge the currency exposure with forwards. When foreign markets rise over a month, the hedge is now too small relative to the asset position, so the hedger sells foreign currency to restore the ratio. These adjustments cluster at month end and execute disproportionately at the 16:00 London fix, because the fix is the benchmark most hedging mandates are judged against.

Documented in Melvin & Prins (Journal of Empirical Finance, 2015) and related BIS work on fix-window liquidity. Unlike the momentum hypothesis this project spent Days 42-57 on, there is a named counterparty and a reason the flow persists: the hedgers are price-insensitive and mandate-bound, executing at a benchmark regardless of level.

Why it may still fail. Post-2013 manipulation scandals changed fix execution, the window widened from 1 minute to 5 in 2015, and banks changed how they internalize flow. Any effect may have decayed or moved. The sample spans that break, which is why the structural-break check in Section 10 is required rather than optional.

What would make it uninteresting even if significant: an effect smaller than the round-trip spread is a fact about microstructure, not a strategy. Criterion 7 binds.

## 3. Data
All 10 pairs from the outset, so universe selection cannot become a post-hoc degree of freedom the way it did in `momentum_book_regime_conditional_robustness.md`.

1-minute bars, `data/{pair}.csv`, 2011-01-01 to 2023-12-31 development. Lockbox 2024-01-01 to 2026-05-01 sealed.

Timestamps: files are fixed-offset UTC-5, so convert per `src/features/sessions.py` then DST-aware convert to Europe/London. The fix is a London-local 16:00 event and moves in UTC across the year; a naive UTC cut would smear it.

No rate or equity data needed. The hedging-need proxy is built from FX returns alone, which avoids the publication-lag machinery that complicated the previous strategy.

## 4. Signal logic
All parameters fixed as of this document. None may be tuned after seeing results.

1. Fix window: 15:30-16:15 Europe/London. Wider than the official 5-minute window because pre-positioning and unwind fall outside it, and 5 minutes of 1-minute bars is too thin to estimate on.
2. Control window: 10:00-10:45 Europe/London. Same duration, liquid, no scheduled benchmark event.
3. Month end: the last 2 trading days of the calendar month. Binary indicator.
4. Hedging-need proxy: `sign(cumulative log return from month start through t-1)`. Uses only information available before the window opens. A month where the base currency appreciated implies hedgers must sell it, so the predicted direction is opposite the month's accumulated move.
5. Signal: `-hedge_need_t` on month-end days, 0 otherwise.
6. Return measured: log return of the fix window, and separately the control window, per pair per day.

| Parameter | Value |
|---|---|
| Fix window | 15:30-16:15 Europe/London |
| Control window | 10:00-10:45 Europe/London |
| Month end | last 2 trading days |
| Hedge-need lookback | month-to-date through t-1 |
| Universe | all 10 pairs |
| Development sample | 2011-01-01 to 2023-12-31 |

## 5. Entry rule
On each of the last 2 trading days of a month, enter at the open of the 15:30 London bar in the direction `-sign(month-to-date return through t-1)`. No entry otherwise. The control window is measured, never traded.

## 6. Exit rule
Flat at the close of the 16:15 London bar. No overnight holding, no stop, no discretionary exit. Holding period is 45 minutes by construction.

## 7. Position sizing
Not decided here, deliberately. Sizing only matters if Section 10 passes, and pre-committing a sizing scheme to an unvalidated signal was a documented weakness of the previous book, whose Section 7 stayed open right through invalidation. If this validates, sizing gets specified in an amendment before any sized backtest runs.

For validation only, the test uses unit exposure per pair, equal-weighted. That is a measurement convention, not a sizing proposal.

## 8. Risk controls
Deferred with Section 7. The structural risk worth naming now: the strategy is concentrated in about 24 trading days per year with a 45-minute holding period, so sample size accumulates slowly. Expect roughly 312 pair-days per pair in development, 3,120 pooled.

## 9. Failure conditions
- Net-of-cost return non-positive at 1.0 pip round trip.
- Effect present in the control window at similar magnitude, which would mean general month-end drift rather than fix-specific flow.
- Effect present on non-month-end days at similar magnitude, same reasoning.
- Sign opposite to prediction, regardless of significance.
- Post-2015 subsample null while pre-2015 carries the result, dating the effect to the old fix regime and making it untradeable now.

## 10. Statistical validation plan

Structured as a primary endpoint plus a fixed sequence of secondary tests, borrowing the gatekeeping design used in clinical trials. Only H1 decides PASS/FAIL. H2 and H3 are tested only if the test before them passed, so family-wise error stays controlled without raising H1's bar. Testing them out of sequence, or reporting one as a standalone finding after H1 fails, voids the correction.

**H1, primary.** Pooled OLS across all 10 pairs and all development days:

`window_return = b0 + b1*signal + b2*month_end + b3*fix_window + b4*(signal × month_end × fix_window) + eps`

via `interaction_regression_centered` extended to a 3-way interaction, reliability gate per criterion 4. Predicts b4 > 0. Significance and sign both required.

Three robustness checks on H1, all of which must agree:

- *Window definition.* Narrow the fix window to 15:45-16:05. A real fix-driven flow should survive tightening around the event.
- *Structural break.* Split at 2015-02-15, fit both subsamples. Same sign in both, and the post-reform subsample significant on its own. This decides whether the effect is still live.
- *Permutation.* 1,000-permutation shuffle of the month-end label, two-sided empirical p on b4, via `permutation_test_interaction_coefficient`.

**H2, post-fix reversal.** Tested only if H1 passes. Regress the return of the 16:15-17:00 London window on the realized fix-window return, restricted to month-end days. Predicts a negative coefficient: mechanical pressure from a price-insensitive counterparty moves price temporarily and must partly unwind.

Estimate the reversion speed with `ou_half_life` on the month-end fix-to-post-fix series. A half-life materially longer than one session is inconsistent with a 45-minute liquidity effect and would point at something informational instead.

This is the sharpest test in the spec because it discriminates rather than confirms. A generic month-end drift, an unmodelled risk premium, or a calendar artifact all predict H1 while predicting no reversal. Only the flow mechanism predicts both. **A null or positive H2 does not fail the strategy, but it does falsify the stated mechanism**, and the write-up must then say the effect exists for reasons the spec does not explain.

**H3, volatility conditioning.** Tested only if H2 passes. Fit `fit_garch` per pair on daily log returns, label each month by `classify_vol_regime` applied to the month's mean conditional volatility, then re-fit H1 separately within high-vol and low-vol months.

Predicts a larger b4 in high-vol months. Hedge-ratio drift away from target is proportional to realized volatility over the month, so a high-vol month leaves a bigger gap to close and demands a bigger rebalancing trade. Direct implication of the mechanism, not a free parameter.

GARCH must be fit walk-forward, not full-sample, per the Day 47 finding that full-sample fitting changed regime labels by 39-68%. Both the 4-way interaction and the split-sample version will be reported; the 4-way carries real collinearity risk and the reliability gate may reject it, in which case the split-sample version stands.

**Portfolio construction variant.** The signal above is traded per pair, time-series. A cross-sectional version will also be reported: rank all 10 pairs by hedging need each month-end day, long the top 3, short the bottom 3, dollar-neutral. This strips common USD moves and raises breadth per the Fundamental Law (`information_ratio`, Day 40).

This is not a fifth hypothesis. It is a second construction of the same signal, so it does not enter the gatekeeping sequence and does not increment `n_trials`. The time-series version is designated primary in advance; if the cross-sectional version looks better, that is an observation for a future spec, not a result to promote after the fact.

**Standard errors.** All pooled p-values use `block_bootstrap` with date-blocks rather than OLS analytic errors. Month-end flows are common across pairs on the same date, so ten pairs on one date are nowhere near ten independent observations and analytic errors would be optimistic. Block length selected by the same rule as `research/notes/bootstap_block_length_selection.md`, fixed before testing.

**Verdict rule.** PASS requires: reliability gate on H1 and its window-definition check; H1 p(b4) < 0.05 with b4 > 0 on block-bootstrap errors; all three robustness checks agreeing in sign, with the window check p < 0.05 and the post-2015 subsample p < 0.05; permutation p < 0.05; and the criterion 7 cost gate. Any single failure kills the strategy. H2 and H3 do not affect PASS/FAIL; they determine whether the mechanism claim survives alongside it.

**Lockbox.** 2024-2026 stays sealed until the development verdict is written down. Opened once, for a PASS, and the result stands either way.

## 11. Open questions and known gaps
The hedge-need proxy is a crude stand-in for actual equity-portfolio hedging demand, which would properly need foreign equity index returns and holdings data. A pure-FX proxy may be too noisy to detect the flow even if it is real. Most likely route to a false negative.

45-minute holding on 1-minute bars raises execution-realism questions, particularly slippage inside a high-volume window, that unit-exposure backtesting does not capture. H2's fade trade is worse on this count, since it trades immediately after the most crowded window of the day.

The 2-day month-end definition comes from the literature rather than being derived here. Fixed, not tuned, but also not optimized, and the true flow window may differ.

H3 splits an already-thin sample. Roughly 3,120 pooled pair-days become two halves of about 1,560, and the power calculation from `day57_momentum_book_invalidation.md` applies with full force. A null H3 is close to uninformative and should not be read as evidence against the volatility mechanism.

The cross-sectional variant needs at least 6 pairs with valid signals on a given date to form 3-and-3 legs. Days failing that are dropped, which is a mild selection effect on which dates enter that book.
