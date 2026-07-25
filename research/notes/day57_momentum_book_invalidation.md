# Day 57 Research Audit: Invalidation of the Momentum-Only Pooled Book

## Question
Does the momentum-only pooled book survive extension to all 10 pairs under its own Section 10 rules, and does the evidence behind the original 3-pair pass hold up on re-examination?

## Verdict
FAIL on both counts. The 10-pair extension fails Section 10, and re-examination turned up a sign problem in the original 3-pair result. This audit closes `momentum_only_pooled_book.md` as invalidated.

## Methodology
Replicated the book's existing validation sequence (Day 47 classifier stability, Day 48 Section 10 battery, Day 49 signal report) across all 10 pairs, reproducing the 3-pair numbers first as a rebuild check each time. All three rebuilds matched exactly, including Day 48's b3 = -0.00356, p < 0.0001, PASS.

Scripts: `research/applied_analysis/day57_*`.

## Findings
Section 10, all 10 pairs, n = 21,873 pooled OOS:

| Test | b3 | p | Cond. number | Gate |
|---|---|---|---|---|
| Primary (78d regime window) | -0.00137 | 0.000032 | 4.36 | Pass |
| Robustness 1 (156d alt window) | -0.00030 | 0.3992 | 4.54 | Pass |
| Robustness 2 (1000-perm shuffle) | -0.00137 | 0.0010 | — | — |

Robustness 1 is null. Section 10 says any single null kills the leg, the same rule that killed the reversion leg on Day 48.

Signal report, all 10 pairs: IC mean -0.094, IC-derived IR -0.333, 31.6% of windows positive (n = 38). Deflated Sharpe 0.0299 on an observed Sharpe of -0.260.

Classifier stability, all 10 pairs: every pair's rho flips sign at least once across 9 window transitions. Walk-forward vs. full-sample label agreement runs 32.2% to 60.8%. USD/CHF (42.9%) joins USD/JPY (57.4%) above the spec's 40% turbulent-share failure condition, with NZD/USD at 40.4%.

## Interpretation
The 10-pair failure is narrow. Primary and permutation tests both still clear the bar; what breaks is sensitivity to the regime window. Moving 78d to 156d shrinks b3 roughly 6x and takes p from 0.00003 to 0.399. At 3 pairs the same swap shrank it about 2x and held significance. More pairs made the result fragile to one design choice rather than uniformly noisier.

The sign problem is the bigger finding, and it sits in the original 3-pair result, not the extension. Section 1 hypothesises momentum works better in turbulent regimes. The regression says the opposite. For the validated 3, b1 = +0.0014 and b3 = -0.00356, so momentum's total effect inside the turbulent regime is b1 + b3 = -0.0022. The signal is ±1, so a +1 momentum reading predicts about -0.22% over the next 26 days in exactly the regime the strategy trades. The 10-pair fit agrees at -0.0021, and Day 49's independently computed IC agrees again at -0.104 and -0.094, with 28.6% of windows positive.

Section 10 tested whether b3 differs from zero. It never tested the sign. Day 48's audit records "sign is consistent (negative) across the primary and alternate-window fits" as support for the pass, without noting that negative is backwards for the hypothesis. The original pass is better described as: the interaction is reliably non-zero and reliably points the wrong way.

Underneath both problems is a power ceiling. The t-statistic on an annualized Sharpe is roughly SR × √years, so a 13-year sample needs SR = 0.55 to reach t = 2. The book's observed Sharpe is 0.140, t = 0.37. Any true effect below about 0.55 Sharpe is invisible at daily frequency on this sample no matter what method is applied.

## Session variant (also negative)
A provisional NY-session variant was tested alongside: same 78-day momentum sign, held only during a fixed intraday window. Volatility fell as predicted, annualized 0.0612 down to 0.0294-0.0456, and the best window (09:00-12:00 ET) lifted pooled Sharpe to +0.217 on the validated 3.

It is cost-dominated. Going flat each day raises turnover from 5.9 legs/year/pair to 166. At half a pip per leg that costs roughly 0.75%/year against 0.68%/year gross. Unprofitable before significance even enters the picture. About 26 window/lookback/regime/universe configurations were explored in that pass, best of them t = +0.89. Logged here so the exploration is on the record.

## Alternative explanations
Parameters tuned on 3 pairs may simply be wrong for the other 7, in which case the 10-pair failure says nothing about momentum on those currencies. That defence does not cover the sign problem or the power ceiling, both of which show up in the original 3-pair sample.

The sign finding has one benign reading worth ruling out. IC and the interaction regression both use 26-day forward returns while the reported Sharpe uses daily exposure × daily return, so a horizon mismatch could in principle produce a positive Sharpe alongside a negative IC. Two independent measures point negative, so the burden sits with that explanation.

## Next steps
- Close the book as invalidated. The 2024-2026 lockbox stays sealed; there is nothing here worth spending it on.
- Resolve the sign/horizon question as a standalone check. If the interaction is genuinely negative, the effect ran opposite to the hypothesis throughout and Day 48 needs an amendment saying the test never checked direction.
- Add a direction requirement to any future Section 10-style verdict rule. Testing that a coefficient differs from zero is not the same as testing the hypothesis behind it.
- Move future work to intraday frequency. The power calculation is the binding constraint, and 15 years of 1-minute bars per pair sit unused in this repo.
