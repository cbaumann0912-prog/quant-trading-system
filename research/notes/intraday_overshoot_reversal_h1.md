# Research Note: H1 Result, Intraday Overshoot Reversal

## Question
Does price displacement away from the 09:00 ET session open partially reverse before 13:00? Pre-registered as H1 in `research/strategies/intraday_overshoot_reversal.md`.

## Verdict
**H1 PASS** on all five pre-registered criteria at the execution-realistic entry. **H2 FAIL**: the proposed mechanism is not supported. The effect exists; the reason given for it does not hold. Both belong in any summary.

## Methodology
All 10 pairs, 2011-2023, lockbox sealed. Signal definition and look-ahead audit: `research/signals/intraday_overshoot_fade.md`. Book is equal-weight across pairs active that day, flat when none trigger.

Reporting runs through the existing repo modules rather than reimplemented statistics: `PerformanceAnalyzer.run_report` for Sharpe, Sortino, Calmar, drawdown, deflated Sharpe, Jarque-Bera and Ljung-Box; `block_bootstrap` for confidence intervals; `t_test_mean` for per-trade significance; `var_historical` and `cvar` for tail risk; `information_ratio` for the cross-pair IR analogue; `interaction_regression_centered` for H2; `compute_achieved_power` for power; `build_signal_report` for the multiple-testing summary.

Script: `research/applied_analysis/intraday_overshoot_h1.py`.

## Headline result

Primary is the +5 min delayed entry. The pre-registered crossing-bar entry is reported alongside because the gap between them measures how much of the effect is microstructure.

| Metric | Crossing bar | +5 min (primary) |
|---|---|---|
| Ann return | +3.846% | +2.185% |
| Ann volatility | 2.265% | 2.190% |
| Sharpe | +1.667 (t +6.01) | +0.987 (t +3.56) |
| Sortino | +1.232 | +0.703 |
| Calmar | +1.273 | +0.554 |
| Max drawdown | -3.02% | -3.95% |
| VaR 95% daily | 0.174% | 0.179% |
| CVaR 95% daily | 0.309% | 0.311% |
| Net Sharpe @1 pip | +1.505 | +0.848 |
| Net Sharpe @2 pips (gate) | +1.391 | +0.737 |
| Net Sharpe @3 pips | +1.278 | +0.625 |
| Bootstrap 95% CI | [+1.101, +2.235] | [+0.444, +1.525] |
| Bootstrap p | <0.00001 | <0.00001 |
| Achieved power | 0.982 | 0.651 |

Max drawdown in both cases runs 294 days, 2022-10-14 to 2023-08-04.

## Entry delay

| Entry | Trades | Mean trade | t | p | Hit rate | Gross SR | Net SR @2 pips |
|---|---|---|---|---|---|---|---|
| crossing bar | 3,828 | +1.423bp | +2.92 | 0.0035 | 53.1% | +1.667 | +1.391 |
| +1 min | 3,780 | +1.126bp | +2.31 | 0.0208 | 52.6% | +1.416 | +1.146 |
| +5 min | 3,616 | +0.832bp | +1.67 | 0.0942 | 51.1% | +0.987 | +0.737 |
| +15 min | 3,304 | +1.124bp | +2.30 | 0.0214 | 51.5% | +0.805 | +0.562 |

About 40% of the crossing-bar edge disappears within five minutes. That component is microstructure captured by entering at the exact bar that crosses the threshold, and it is not tradeable. The +15 min row breaks the otherwise monotone decay in mean trade return while its Sharpe keeps falling; that is a sample-composition effect, since 312 fewer trades survive the delay inside the scan window, not the edge recovering.

## Robustness: entry threshold

| k | Trades | Mean trade | Hit rate | Gross SR | Net SR @2 pips |
|---|---|---|---|---|---|
| 1.5 | 7,545 | +0.267bp | 50.8% | +1.124 | +0.660 |
| 2.0 | 3,616 | +0.832bp | 51.1% | +0.987 | +0.737 |
| 2.5 | 1,759 | +1.181bp | 52.0% | +0.795 | +0.646 |

Reversion strengthens as the threshold rises. This is the spec's test for bid-ask artifacts: microstructure would be strongest at small thresholds, and it is the opposite. Mean trade return quadruples from k=1.5 to k=2.5. This is the single strongest piece of evidence that the effect is real.

## Trade profile and stability

Hit rate 51.1%, mean win +20.18bp against mean loss -19.45bp, win/loss ratio 1.038, profit factor 1.088, 270 trades/year, active on 42.3% of days.

Structural break at 2017-07-01 gives 1.015 pre and 0.977 post, essentially identical. Every prior candidate in this project either died in one half of the sample or reversed sign.

Annual returns are positive in **11 of 13 years**. Negative only in 2011 (12 trades, before the vol-ratio warmup completes) and 2023 (-1.28%, Sharpe -0.524). Best years 2022 (+5.22%), 2016 (+4.85%), 2020 (+4.74%). No single year drew down more than 3.29%.

Distribution: skew +1.102, excess kurtosis +19.259, Jarque-Bera rejects normality at any level, Ljung-Box p=0.136 so no detectable autocorrelation in book returns. The positive skew is unusual for a fade strategy and most likely comes from the hard 13:00 exit truncating losers. It is favourable but fragile, entirely dependent on exit discipline. The kurtosis means any Gaussian risk figure on this book understates the extremes.

## Trigger timing

Median 117 minutes into the 180-minute scan, q25 75, q75 148. Only 1.7% of triggers fire in the first half hour, and there is a pronounced spike of 101 triggers at exactly minute 30, the NYSE equity open.

| Window | Triggers | Share |
|---|---|---|
| 09:00-09:30 | 64 | 1.7% |
| 09:30 exactly | 101 | 2.6% |
| 09:31-09:45 | 257 | 6.7% |
| 09:45-10:00 | 238 | 6.2% |
| 10:00-10:30 | 558 | 14.6% |
| 10:30-11:00 | 770 | 20.1% |
| 11:00-11:30 | 916 | 23.9% |
| 11:30-12:00 | 924 | 24.1% |

## Per pair

Seven of ten positive, none individually significant, largest |t| is 1.40.

| Pair | Trades | Mean trade | t | Hit rate | Sharpe |
|---|---|---|---|---|---|
| NZD/USD | 360 | +2.345bp | +1.20 | 52.2% | +0.347 |
| AUD/USD | 438 | +2.173bp | +1.40 | 51.8% | +0.405 |
| EUR/JPY | 315 | +1.894bp | +1.08 | 56.2% | +0.312 |
| EUR/CHF | 284 | +1.544bp | +1.10 | 56.3% | +0.322 |
| EUR/GBP | 361 | +1.368bp | +1.09 | 52.4% | +0.315 |
| USD/CAD | 379 | +0.712bp | +0.51 | 49.9% | +0.149 |
| EUR/USD | 340 | +0.117bp | +0.07 | 50.0% | +0.020 |
| USD/JPY | 381 | -0.357bp | -0.23 | 50.1% | -0.067 |
| GBP/USD | 397 | -0.389bp | -0.26 | 47.4% | -0.074 |
| USD/CHF | 361 | -0.928bp | -0.62 | 46.5% | -0.178 |

Signal report: cross-pair IC mean +0.0287, IC std 0.0388, IC-derived IR +0.7387, 70% of pairs with positive IC, per-pair Sharpe mean +0.1551 (std 0.2129). Deflated Sharpe 0.9905 at n_trials=6.

## H2: does reversal depend on displacement speed?

**FAIL.**

| Bucket | Trades | Mean trade | t | Hit rate |
|---|---|---|---|---|
| fast (trigger within 30 min) | 165 | -4.918bp | -1.12 | 40.0% |
| slow | 3,451 | +1.107bp | +2.32 | 51.6% |

Interaction regression of trade return on displacement size, fast dummy, and their interaction gives b3 = +0.031 with p=0.809. Reliability gate passes (condition number 6.5e6, max VIF 1.235). Prediction was b3 > 0 and significant. It is neither.

### Follow-up hypothesis, proposed and falsified in the same session

101 of the 165 fast trades fire at exactly minute 30, the NYSE equity open. That raised the possibility the fast penalty is not about speed but about the open, where displacement would be scheduled information-driven repricing rather than temporary order-flow noise. The prediction was stated before looking: if the equity open is the cause, the 64 genuinely pre-open triggers should behave like the slow bucket.

| Bucket | Trades | Mean trade | t | Hit rate |
|---|---|---|---|---|
| t < 30, before equity open | 64 | -5.881bp | -0.76 | 45.3% |
| t = 30, NYSE open exactly | 101 | -4.307bp | -0.83 | 36.6% |
| 30 < t <= 45 | 275 | +9.163bp | +3.24 | 53.1% |
| t > 45 | 3,176 | +0.409bp | +0.90 | 51.5% |

The prediction failed. Pre-open triggers are marginally the worst bucket, not the best. The equity-open explanation is dead and speed remains the operative variable. Recorded because a mechanism proposed and killed is part of the evidence.

The 30-45 minute bucket at +9.163bp with t=+3.24 is the loudest cell in the entire analysis and should be treated as noise. Trigger time has now been sliced four ways; one cell reaching p=0.0014 across that many splits is unremarkable, and building a rule around it would be fitting the largest piece of randomness in the sample.

## Interpretation

Three things distinguish this from the five strategies that failed before it: the structural break is stable, the threshold monotonicity runs the right way, and 11 of 13 years are positive.

Against that, the mechanism is unexplained. The spec argued reversal comes from dealers absorbing impatient liquidity demand, which predicts fast displacement reverts more. The data says the opposite. The equity-open alternative was the obvious second candidate and failed too. Per Section 2 of the spec, the honest statement is that the effect exists for reasons the spec does not explain, and no substitute explanation has survived a test. An effect this stable with no working mechanism is a standing red flag for an unmodelled data artifact.

## Two caveats that belong in any summary

**The portfolio construction does more work than the signal.** Per-trade hit rate is 51.1%, per-trade t is 1.674, `t_test_mean` returns p=0.094 on the per-trade mean, and mean win and mean loss are nearly symmetric. If trades were independent within a day the implied book Sharpe would be 0.458; the actual is 0.987. The gap is cross-pair variance reduction, since fading correlated pairs on the same day partially hedges the common USD factor and leaves idiosyncratic reversion. That is a legitimate source of return and exactly the breadth argument Section 0 was built on, but it means this is a diversification result resting on a marginal directional edge.

Realized cross-pair trade-return correlation is +0.390, giving effective breadth of **2.22 of 10**, below the 3.13 that Section 0 estimated from trigger-day overlap. The breadth lever is real but weaker than the pre-registration assumed.

The deflated Sharpe of 0.99 should not be read at face value. The moment correction is derived under assumptions not tested at an excess kurtosis of 19, and `signal_report`'s standing caveats flag the pooled annualization as optimistically biased. Treat it as directional.

**The two permutation tests disagree, and the disagreement is informative.** Flipping each trade's sign independently gives p=0.087. Flipping the sign per day, preserving which pairs traded together, gives p<0.0001. Both are correct answers to different questions. The per-trade version destroys the cross-pair structure and tests the directional signal in isolation, where the evidence is marginal. The per-day version tests the book as constructed. Anyone reading only the second number would overstate the result.

## On not patching the result

Dropping the fast bucket and writing reasoning afterwards was considered and rejected. The fast trades are 4.6% of the sample at t=-1.12, so the filter would be fitted to an insignificant subgroup; the gain is small, moving the mean from +0.832bp to +1.107bp; and it would cost the pre-registration, an `n_trials` increment, and the credibility that reporting a mechanism failure buys. An effect with an honest unexplained mechanism is worth more than a patched Sharpe with a story attached.

## Alternative explanations

The residual edge after a 5-minute delay could still be slower microstructure rather than a genuine reversion premium. The threshold monotonicity argues against this, but 1-minute OHLCV closes are not tradeable prices and only tick data would settle it.

Entry slippage remains the largest unquantified risk. Fading a displacement means crossing the spread into adverse flow, and the 2-pip gate is an assumption rather than a measurement. The strategy still nets 0.625 at 3 pips, which provides margin, but nothing here rules out worse fills.

The three pairs that come back negative (USD/CHF, GBP/USD, USD/JPY) may reflect the daily-to-session vol scaling being least accurate where intraday vol structure differs most from the daily pattern. Untested.

The lockbox (2024-2026) remains sealed.

## Next steps

Resolve whether the directional signal stands on its own before treating this as validated. The obvious test is a single-pair-per-day book, which removes the diversification benefit entirely, plus an equal-weight book with randomised pair selection as a control.

Find a mechanism that survives a test, or state plainly that there is none. Two candidates have now failed.

Specify and test an exit rule. The 13:00 exit was chosen for being parameter-free, not for being good, and the positive skew suggests exit timing carries real weight. Its own hypothesis, its own trial count.

Measure realized slippage properly if tick data becomes available. Everything about the cost gate is currently an assumption.

Only after the above should the lockbox be opened, and it is opened once.
