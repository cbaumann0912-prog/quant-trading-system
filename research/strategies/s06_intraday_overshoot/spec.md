# Strategy Specification: Intraday Overshoot Reversal, London-NY Overlap

**Date drafted:** Day 57 (2026-07-25)
**Status:** CLOSED, Day 57. Section 10 verdict FAIL. Figures regenerated Day 72: the strategy fails on robustness 3 per-trade permutation (p = 0.1349) and H2 (b3 = -0.0465, p = 0.679). Robustness 1 threshold monotonicity, recorded as a third failure on the Day 57 run, passes on regeneration and is uninformative in either direction. Result in `research/strategies/s06_intraday_overshoot/intraday_overshoot_section10_validation.md`. Lockbox never opened.

Written and pre-registered before any test was run; the sections below are unedited from that pre-registration.

## Provenance
Strategy #6. Five tested previously, all null: PC2 Carry Regime, Momentum w/ ML Regime, OU Half-Life Mean Reversion, Volatility Regime Breakout/Mean-Reversion (momentum-only successor closed in `momentum_book_invalidation.md`), Month-End FX Flow (closed in `month_end_fx_flow_h1_result.md`). Honest `n_trials` is 6.

## 0. Feasibility, computed before the hypothesis
**Cost hurdle.** EUR/USD 09:00-12:00 ET session return std is 27.6bp, annualizing to 4.39% vol. Against a 0.9bp round trip:

| Trades/yr | Cost %/yr | Break-even Sharpe |
|---|---|---|
| 252 (unconditional) | 2.27 | 0.517 |
| 60 | 0.54 | 0.123 |
| 30 | 0.27 | 0.062 |
| 15 | 0.14 | 0.031 |

Selectivity is the entire cost argument. Trading the window unconditionally requires Sharpe 0.52 before earning anything, which is why the Day 57 session variants failed. A 2-sigma entry threshold fires on 9.6% of days, about 24 trades/year/pair, dropping the hurdle to 0.062.

Because this strategy fades a fast move, entry slippage will exceed the quoted spread. The cost gate is therefore evaluated at **1.0, 2.0 and 3.0 pips**, and the strategy must clear at 2.0 pips to pass. Clearing only at 1.0 pip is recorded as a fail.

**Power hurdle.** t on an annualized Sharpe is SR x sqrt(years), so 12.9 years still requires **SR >= 0.55** for t = 2. Sampling intraday does not change this: the constraint is calendar span, not observation count, as verified by simulation during the carry work.

The lever available here that carry lacked is breadth. Ten pairs with imperfectly correlated intraday shocks give an effective breadth well above 1, so a per-pair Sharpe near 0.30 can plausibly produce a book Sharpe above 0.55 (Fundamental Law, `information_ratio`, Day 40). Compute realized effective breadth from the cross-pair correlation of trade returns and report it. If effective breadth turns out near 1, the power argument collapses and the result should be read as inconclusive.

Achieved power is reported with every result via `compute_achieved_power`.

## 1. Hypothesis
**H1.** Within the 09:00-12:00 ET window, price displacement away from the session open partially reverses before the window closes. Fading a large displacement earns a positive net-of-cost return.

**H2, the contribution.** The reversal is stronger following *fast* displacement than slow displacement of the same magnitude. Fast moves are impatient liquidity demand and overshoot; slow moves of equal size are more likely informed and should not revert.

Falsification criteria, binding:

1. Direction requirement: significant **and** predicted sign. A significant coefficient of the wrong sign is a FAIL. Two prior strategies produced exactly that.
2. Primary threshold: p < 0.05 on H1, block-bootstrap standard errors clustered by date.
3. Reliability gate on any regression: condition number < 1e10, all VIF < 10, main effects mean-centered before interaction.
4. Multiple testing: Benjamini-Hochberg across 6 strategies, so p < 0.0083 at rank 1 of 6.
5. Cost gate: net-of-cost Sharpe positive at **2.0 pips** round trip on realized trade count.
6. Power disclosure: achieved power and realized effective breadth reported alongside every result.

## 2. Economic rationale
The counterparty is a market maker with unwanted inventory. When a large directional order arrives in a three-hour window, dealers absorb it, price moves further than the information warrants, and dealers unwind into the reversion. The trader on the other side is buying immediacy and is price-insensitive about it.

This mechanism does not get arbitraged away because the liquidity demander is not trying to predict anything, and the compensation is payment for bearing inventory risk. That is a structurally different situation from the momentum hypothesis this project spent Days 42-57 on, which never identified a counterparty at all.

Choosing 09:00-12:00 ET is not arbitrary within that story: it is the London-New York overlap, the highest-flow window of the FX day, so it should carry both the most liquidity demand and the most dealer inventory turnover.

**What would falsify the mechanism rather than the effect.** If reversal is equally strong for slow and fast displacement (H2 null), the story about impatient flow is wrong even if H1 passes, and the write-up must say the effect exists for reasons this spec does not explain.

## 3. Data
All 10 pairs from the outset, so universe selection cannot become a post-hoc degree of freedom as it did in `momentum_book_regime_conditional_robustness.md`.

1-minute bars, `data/{pair}.csv`. Timestamps carry a fixed UTC-5 file offset, converted per `src/features/sessions.py` then DST-aware converted to America/New_York, since 09:00 ET is a local-clock event.

Development 2011-01-01 to 2023-12-31. Lockbox 2024-01-01 to 2026-05-01, sealed.

## 4. Signal logic
All parameters fixed as of this document. None may be tuned after seeing results.

1. Entry scan runs 09:00 to 12:00 ET, exit at 13:00 ET. Reference price is the 09:00 bar open.
2. Fit `fit_garch` on daily log returns per pair, walk-forward refit per the Day 47 finding that full-sample fitting shifts regime labels by 39-68%. Scale to a session-equivalent sigma.
3. Entry threshold is `k x sigma_t` where sigma_t is the GARCH conditional vol as of t-1, with **k = 2.0**.
4. Entry fires the first time within the window that `|log(P_t / P_open)| > k x sigma_t`. Direction is the fade: short if price is above the open, long if below.
5. One entry per pair per day, maximum. No re-entry after exit.

GARCH is used to **scale the threshold, not to gate trading**. A fixed threshold fires constantly in high vol and never in low vol, so trade count and risk swing by regime. Scaling holds the trigger rate roughly stable across regimes, which is what a liquidity-provision book requires.

| Parameter | Value |
|---|---|
| Entry scan window | 09:00-12:00 America/New_York |
| Exit | 13:00 America/New_York |
| Reference | 09:00 bar open |
| Threshold k | 2.0 conditional sigma |
| Max entries | 1 per pair per day |
| Universe | all 10 pairs |
| Development sample | 2011-01-01 to 2023-12-31 |

## 5. Entry rule
Enter at the close of the first 1-minute bar whose displacement from the 09:00 open exceeds `2.0 x sigma_t`, in the direction opposite the displacement. Unit exposure per pair.

## 6. Exit rule
Flat at the 13:00 ET bar close. No target, no stop, no discretion. Holding period is whatever remains after entry.

This is deliberately parameter-free. A reversion target or stop would improve the return distribution but each adds a tunable degree of freedom to a pre-registered test, and the project's history is that free parameters are where results go to die.

## 7. Position sizing
Deferred until H1 passes, per the pattern that left the momentum book's Section 7 open through invalidation. Validation uses unit exposure per pair, equal weighted, as a measurement convention.

## 8. Risk controls
Deferred with Section 7. Named now, because this strategy's risk profile is its central weakness: liquidity provision is structurally negative skew. Many small wins, occasional large losses when the move keeps running, and a hard 13:00 exit with no stop guarantees the left tail is uncapped within the holding period. Report realized skew, excess kurtosis, and CVaR (`src/analysis/portfolio.py`) alongside Sharpe, not after.

## 9. Failure conditions
- Net-of-cost Sharpe non-positive at 2.0 pips on realized trade count.
- Effect present only in the smallest displacement bucket, which would suggest bid-ask bounce rather than genuine overshoot.
- Realized effective breadth near 1, collapsing the power argument.
- H2 null, falsifying the impatient-flow mechanism even if H1 survives.
- Trade count materially below 15/year/pair, which would leave too few observations to say anything.

## 10. Statistical validation plan
Gatekeeping. H2 runs only if H1 passes.

**H1 primary.** Pooled mean trade return across all pairs and trades, tested against zero. Block-bootstrap standard errors clustered by **date**, since displacement events cluster across pairs on common shocks and 10 pairs on one day are nowhere near 10 independent observations. Report pooled book Sharpe, achieved power, effective breadth, deflated Sharpe at `n_trials=6`, skew, excess kurtosis, and max drawdown.

**H1 robustness 1, threshold.** Repeat at k = 1.5 and k = 2.5. The effect should not depend on the specific threshold, and the monotonicity matters: if reversal is stronger at larger k, that supports overshoot; if stronger at smaller k, it suggests bid-ask bounce and fails Section 9.

**H1 robustness 2, structural break.** Split at 2017-06-30 and fit both halves. Same sign required in both, with the later half significant on its own, since an effect that died mid-sample is untradeable now.

**H1 robustness 3, permutation.** 1,000-permutation shuffle of the fade direction, holding entry timing fixed, two-sided empirical p.

**H2.** Split trades by time-to-threshold (fast = threshold crossed within 30 minutes of the open, slow = after). Interaction regression of trade return on displacement size, speed dummy, and their interaction, via `interaction_regression_centered`. Prediction: fast trades revert more.

**Verdict.** PASS requires reliability gate on H1 and robustness 1; H1 p < 0.05 on block-bootstrap errors with the predicted sign; all three robustness checks agreeing in sign with robustness 1 and the post-break half both p < 0.05; permutation p < 0.05; and the 2.0-pip cost gate. Any single failure kills it.

**Lockbox.** Opened once, only on a PASS, only after the development verdict is written down.

## 11. Open questions and known gaps
Entry slippage is the biggest threat. Fading a fast move means crossing the spread into adverse flow, and realized slippage on a 2-sigma displacement bar will exceed the quoted spread by an unknown amount. The 2.0-pip gate is a guess at this, not a measurement, and only tick data would settle it.

1-minute OHLCV closes are not tradeable prices. At a 2-sigma threshold the displacement is around 55bp against a 0.9bp spread, so bid-ask bounce should not dominate, but robustness check 1's monotonicity is the actual test of that and it should be read carefully.

GARCH conditional vol is fit on daily returns and rescaled to a session horizon, which assumes the daily-to-intraday vol ratio is stable. It is not, since intraday vol has strong time-of-day structure. A session-native vol estimate would be better and is not built here.

The hard 13:00 exit is honest but almost certainly suboptimal. If H1 passes, the first extension worth specifying is an exit study, as its own hypothesis with its own trial count.