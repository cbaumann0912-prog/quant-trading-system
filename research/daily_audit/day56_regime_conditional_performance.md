# Day 56 Research Audit: Regime-Conditional Performance, Momentum-Only Pooled Book

## Question
Does the momentum-only pooled book's edge hold up under a different volatility-regime definition? Split its realized PnL into high/low-vol days using GARCH(1,1) conditional vol clustered with `classify_vol_regime`, instead of the deployed classifier's realized-vol input, and compare Sharpe.

## Scope Note
Only the momentum leg is validated, as `momentum_only_pooled_book.md`, on EUR/USD, GBP/USD, and USD/JPY (see Day 53's audit for the same caveat on the other three discarded candidates). The remaining 7 pairs run through the identical pipeline here for diagnostic breadth only, following Day 55's precedent of testing GARCH across the full 10-pair universe. Results on those 7 are not a validation; that requires a Section 10-style test of their own.

## Methodology
The deployed gate calls a day "turbulent" from a composite of 78-day realized vol and a rate differential, thresholded at |z|>1.5. Day 43 found the rate differential barely matters, so vol was already carrying the composite. If momentum's edge is a real vol-regime effect, an independently-fit vol measure should find it too. That's the test here: a stress check on an already-validated result, not a new trial, no `n_trials` increment.

Rebuilt the deployed book unchanged (`compute_composite_regime_score_walkforward`, `classify_regime`, `momentum_signal`, `regime_gated_pnl`) for all 10 pairs, 2011-2023, lockbox excluded. Fit `fit_garch` per pair for a conditional vol path and split with `classify_vol_regime`. Day 49's rate-differential loader only covers 3 pairs, so it's extended locally to all 10 with the same base-minus-quote convention (e.g. EUR/GBP -> ea - uk). Compared three ways: per pair, pooled across the validated 3, pooled across all 10. GARCH is fit full-sample, not walk-forward refit (see Alternative Explanations).

Script: `research/applied_analysis/day56_regime_conditional_performance.py`.

## Findings
Rebuild check: the validated-3 pooled book comes out to n=1,472, Sharpe 0.140, matching Day 53's 0.141 within rounding.

**Per pair** (GARCH regime inside each pair's already-turbulent days):

| Pair | Status | High-vol Sharpe | Low-vol Sharpe | High-vol % | n |
|---|---|---|---|---|---|
| EUR/USD | validated | -0.695 | -0.088 | 14% | 917 |
| GBP/USD | validated | -0.471 | +0.223 | 8% | 990 |
| USD/JPY | validated | +0.341 | +0.161 | 28% | 1,327 |
| USD/CHF | diagnostic | n/a | -0.303 | 0% | 1,036 |
| AUD/USD | diagnostic | -1.148 | +0.230 | 15% | 832 |
| USD/CAD | diagnostic | -0.584 | +0.391 | 59% | 564 |
| NZD/USD | diagnostic | -0.050 | +0.349 | 24% | 1,027 |
| EUR/GBP | diagnostic | +0.007 | +0.173 | 45% | 337 |
| EUR/JPY | diagnostic | +0.044 | -1.040 | 20% | 496 |
| EUR/CHF | diagnostic | n/a | +0.558 | 0% | 530 |

**Pooled, validated 3 pairs:** high-vol Sharpe +0.396, low-vol -0.232, high-vol 29% of days, n=1,472. Unconditional Sharpe 0.140.

**Pooled, all 10 pairs (diagnostic):** high-vol Sharpe -0.377, low-vol +0.007, high-vol 22% of days, n=1,988. Unconditional Sharpe -0.260.

## Interpretation
Per pair, direction is mixed: 6 of 10 pairs show GARCH-high-vol underperforming GARCH-low-vol, 4 go the other way. Not strong enough to call an effect without a significance test, which this audit doesn't run.

The pooled comparison is the sharper result. Pooling just the validated 3, high-vol beats low-vol and beats the unconditional Sharpe, matching the strategy's hypothesis. Pooling all 10 flips the unconditional Sharpe negative (0.140 to -0.260) and the regime split flips with it. The validated edge doesn't survive mechanical extension to a wider universe, a caution against widening this book's pairs without separate validation.

## Alternative Explanations
USD/CHF and EUR/CHF show 0% high-vol days inside their own turbulent subset. Their GARCH-identified high-vol clusters simply don't overlap with the days the realized-vol composite flagged turbulent. USD/CHF's GARCH fit also hit the persistence upper bound, likely because one model can't span both the near-zero-vol peg years and the 2015 de-peg shock. The conditional vol path itself still looks sane, so the clustering isn't broken, but the GARCH parameters for this pair shouldn't be trusted at face value.

The all-10 result may also just reflect parameters tuned for the wrong pairs: momentum lookback, entry rule, and regime thresholds were all chosen against the 3-pair set. A negative Sharpe on 7 untested pairs under those fixed parameters looks more like a mismatched tool than a real test of the underlying hypothesis.

Full-sample GARCH isn't a like-for-like comparison against the deployed classifier either, which is walk-forward refit (Day 47 found only 34-56% label agreement between full-sample and refit versions). Every regime label here is descriptive, not something to trade on as-is.

## Next Steps
Run a bootstrap CI or interaction regression on the validated-3 Sharpe gap before trusting it. If it holds, that argues for a book-level vol input to the regime gate, but that's a new entry rule needing its own spec and an `n_trials` increment, not a rerun of this audit.

The 7 diagnostic pairs need their own validation pass (interaction regression, reliability gate, robustness checks) before today's numbers on them mean more than "worth a look." The negative all-10 Sharpe says the current parameter set doesn't transfer, not that momentum fails on those pairs.

If GARCH is pursued further, it needs the same walk-forward refit treatment the deployed classifier gets before anything built on it can be trusted out of sample. USD/CHF and EUR/CHF specifically need a look at whether one GARCH model should span the pre- and post-2015 franc regimes, or whether the break should be modeled explicitly.
