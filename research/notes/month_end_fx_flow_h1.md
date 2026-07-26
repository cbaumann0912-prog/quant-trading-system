## Question
Does the hedging-need signal predict returns specifically in month-end fix windows? Pre-registered as H1 in `research/strategies/month_end_fx_flow.md`, prediction b4 > 0, significance and sign both required.

## Verdict
FAIL on direction. Every other criterion passed. This is the first use of the direction requirement added to the falsification criteria after `day57_momentum_book_invalidation.md` found the momentum book had passed a significance-only rule with its effect running backwards.

## Methodology
Fully specified 3-way interaction across all 10 pairs, 2011-2023, lockbox sealed:

`window_return = b0 + b1*signal + b2*month_end + b3*fix + b4*(signal x month_end x fix) + all 2-way terms + eps`

All main effects mean-centered before products. `signal = -sign(month-to-date log return through t-1)`, defined on every day; `month_end` = last 2 trading days; `fix` = 15:30-16:15 London vs. a 10:00-10:45 control window. No walk-forward refit: nothing in H1 estimates a parameter, so there is no leakage channel to close.

Standard errors from `block_bootstrap` over 21-day date blocks rather than OLS analytic errors, since month-end flows hit every pair on the same date and 10 pairs on one date are nowhere near 10 independent observations.

Script: `research/applied_analysis/month_end_fx_flow_validation.py`.

## Findings
n = 67,057 pooled. Condition number 48.2, max VIF 1.000, reliability gate clean on every fit.

| Test | b4 | p |
|---|---|---|
| Primary, OLS | -1.200e-4 | 0.00093 |
| Primary, block bootstrap | 95% CI [-2.52e-4, -2.03e-5] | 0.024 |
| Robustness 1, narrow window 15:45-16:05 | -6.885e-5 | 0.024 |
| Robustness 2a, pre-reform | -8.378e-5 | 0.240 |
| Robustness 2b, post-reform | -1.362e-4 | 0.0010 |
| Robustness 3, 1000-permutation | -1.200e-4 | 0.044 |

Bootstrap SE (5.88e-5) is 1.6x the OLS SE (3.62e-5), confirming the cross-pair date clustering the analytic errors ignore.

Conditional decomposition, mean signal-directed return in basis points:

| month_end | window | n | signal x y (bp) | t |
|---|---|---|---|---|
| 0 | fix | 30,419 | +0.075 | 0.91 |
| 0 | control | 30,420 | -0.030 | -0.41 |
| 1 | fix | 3,108 | -0.950 | -3.44 |
| 1 | control | 3,110 | +0.142 | 0.73 |

## Interpretation
The conditioning structure is exactly what the hypothesis predicted. The effect sits in the month-end fix cell and nowhere else: non-month-end fix windows are flat, month-end control windows are flat, and only the month-end fix cell carries a t-statistic worth anything. Three of the four cells behaving as the null would predict is strong evidence the test was correctly targeted.

The direction is wrong. Since `signal = -sign(month-to-date return)`, a negative b4 means month-end fix windows **continue** the month's move rather than reversing it. A price-insensitive hedger unwinding an appreciated position produces reversal. This is the opposite.

The effect also strengthened after the 2015 fix reform (post p=0.0010 vs. pre p=0.240), which rules out the most obvious dismissal. If this were a residue of pre-reform fix manipulation it would live in the early sample. It does not.

Magnitude kills it independently of sign. The month-end fix cell earns 0.95bp per trade against a 1.0 pip round trip costing roughly 0.9-1.0bp across the majors. Criterion 7 fails on the reversed version too, so even a legitimate route to trading the other direction would not produce a tradeable strategy.

## Alternative Explanations
The most likely reading is that the proxy is misspecified rather than the mechanism falsified, and Section 11 of the spec flagged this in advance as the most likely route to a false negative. Hedge-ratio drift is driven by the foreign **equity** return, not the FX rate: a rising European equity book leaves a US investor's EUR hedge too small regardless of where EUR/USD went. Substituting month-to-date FX return for equity return does not measure hedging need. What the regression actually detected is some month-end fix-window continuation effect, which the spec never hypothesised and this audit cannot explain.

A second reading is that the hedging flow is real but swamped in the fix window by a larger opposing flow, for instance index rebalancing pushing the same direction as the month's move. Untestable with the data here.

## What Was Not Done
H2 (post-fix reversal) and H3 (volatility conditioning) were not run. The Section 10 gatekeeping sequence only controls family-wise error if it binds when the primary fails, and it failed.

The sign was not flipped and re-tested. Discovering a coefficient runs backwards and then trading the inverse uses the same data to generate and confirm a hypothesis, which is the failure mode the pre-registration exists to prevent. The lockbox stays sealed.

## Next Steps

Testing the actual mechanism requires foreign equity index returns as the conditioning variable. That is a new hypothesis with a new spec and `n_trials` at 6, not an amendment to this one, and the 0.95bp effect size suggests it would fail the cost gate even if the sign came out right.

Before any further strategy is specified, compute the cost hurdle first. Six strategies have now been tested, all of them high-turnover, all of them producing per-trade edges comparable to or smaller than the round-trip spread. FX majors run roughly 8% annualized vol, so a Sharpe-0.3 strategy earns about 2.4%/year gross; a daily round trip at 0.9bp costs 2.27%/year. The same predictive ability is roughly 20x more tradeable at monthly turnover. That arithmetic takes five minutes and would have killed both this hypothesis and the Day 56-57 session variants before either was coded.
