# Section 10 Validation: Intraday Overshoot Reversal

## Verdict
**FAIL.** Section 10 kills the strategy on any single failed criterion. Three fail.

| Criterion | Result |
|---|---|
| H1 primary p < 0.05, block-bootstrap | PASS, p < 0.00001 |
| H1 predicted sign | PASS, +1.043 Sharpe |
| R1 threshold monotone | **FAIL** |
| R1 same sign across k | PASS |
| R2 same sign both halves | PASS |
| R2 post-break half p < 0.05 | PASS, p = 0.0093 |
| R3 permutation, per-trade | **FAIL**, p = 0.114 |
| R3 permutation, per-day | PASS, p = 0.0010 |
| Reliability gate | PASS, cond 4.26e6, max VIF 1.151 |
| Cost gate at 2.0 pips | PASS, net Sharpe +0.814 |
| BH rank 1 of 6 | PASS, bar 0.00833 |
| H2 speed interaction | **FAIL**, b3 = −0.048, p = 0.669 |

The 2024–2026 lockbox stays sealed. Section 10 permits opening only on a PASS.

## Method
`intraday_overshoot_section10_validation.py`, rebuilt from raw 1-minute bars every run via `src/signals/intraday_overshoot.py`. No cached intermediates. Ten pairs, 2011–2023, k = 2.0, +5 minute entry, 259.44 obs/yr empirical annualization over 12.99 years.

Statistics come from the repo's tested modules: `block_bootstrap`, `paired_sign_permutation_test`, `benjamini_hochberg_correction`, `interaction_regression_centered`, `PerformanceAnalyzer.run_report`, `compute_achieved_power`, `compute_required_sample_size`.

## R1 cannot resolve what it claims to test
| k | trades | mean bp | SE bp | t | 95% CI bp |
|---|---|---|---|---|---|
| 1.5 | 6,708 | +0.149 | 0.346 | 0.38 | [−0.548, +0.809] |
| 2.0 | 3,206 | +0.852 | 0.548 | 1.52 | [−0.242, +1.905] |
| 2.5 | 1,584 | +0.795 | 0.853 | 1.03 | [−0.790, +2.553] |

The ladder is supposed to rise with k. It doesn't. The check also could not have told us much either way.

Every CI spans zero. The deciding gap, k=2.0 to k=2.5, is +0.050 bp against SE 1.014, so t = 0.05. Roughly four trades out of 1,588 flip the ordering, and at excess kurtosis +22.5 a handful of extremes sets it. Year by year the ladder is monotone in 2 of 11 tradeable years.

The superseded note called this ladder "the single strongest piece of evidence that the effect is real." That reading wasn't supportable then either, since the standard errors were the same. Treat R1 as uninformative in both directions. What survives from Section 9 is narrower: the effect isn't concentrated at low k, so bid-ask bounce stays ruled out.

## R3 fails on the test the spec chose
Section 10 asks for "a 1,000-permutation shuffle of the fade direction, holding entry timing fixed." That's the per-trade test, and it gives p = 0.114.

The per-day version gives p = 0.0010, but it preserves which pairs traded together, so it asks whether the book beats a coin flip rather than whether the direction does. Both answer real questions. The spec pre-committed to the first.

The plain t-test agrees with the strict reading: pooled mean trade return has t = +1.56, p = 0.119.

## Where the return comes from
| Pair | trades | mean bp | t | Sharpe |
|---|---|---|---|---|
| EUR/JPY | 259 | +3.498 | +1.69 | +0.468 |
| EUR/GBP | 318 | +1.738 | +1.26 | +0.350 |
| AUD/USD | 402 | +2.029 | +1.21 | +0.337 |
| NZD/USD | 333 | +2.288 | +1.07 | +0.296 |
| USD/CAD | 309 | +1.066 | +0.65 | +0.181 |
| EUR/CHF | 296 | +0.171 | +0.13 | +0.037 |
| GBP/USD | 372 | +0.143 | +0.09 | +0.025 |
| USD/CHF | 278 | +0.022 | +0.01 | +0.003 |
| EUR/USD | 296 | −0.486 | −0.27 | −0.076 |
| USD/JPY | 343 | −1.748 | −1.02 | −0.284 |

The book looks strong: Sharpe +1.043, bootstrap p < 0.00001, CI [+0.493, +1.584], ten of thirteen years positive, both structural-break halves significant at +1.199 and +0.949.

No individual pair reaches significance. Largest |t| is 1.69, and the bootstrap CI on cross-pair mean IR is [−0.0002, +0.0543], which includes zero. If trades were independent within a day the book Sharpe would be +0.432 rather than +1.043. Cross-pair correlation is +0.364, effective breadth 2.34 of 10.

Almost all of the book Sharpe comes from that gap between +0.432 and +1.043. Diversification is a real source of return and Section 0 said so upfront, but it leaves the strategy resting on a directional edge that nothing here separates from zero.

## Power
Achieved power 0.639. Reaching 80% needs 1,871 active days; there are 1,278. The study cannot resolve its own effect size. Same calendar ceiling that killed the momentum book: 13 years needs Sharpe ≈ 0.55 for t = 2, and sampling intraday doesn't change it.

## H2 and the mechanism
| Bucket | trades | mean bp | t | hit rate |
|---|---|---|---|---|
| fast, ≤30 min | 159 | −6.988 | −1.43 | 38.4% |
| slow | 3,047 | +1.261 | +2.45 | 52.1% |

b3 = −0.048, p = 0.669, Cohen's d −0.267. The prediction was b3 > 0 and significant. It's neither, and the point estimate runs backwards.

The NYSE-open alternative fails too. Splitting trigger time four ways: t<30 gives −6.175 bp (n=63), t=30 exactly −7.521 (n=96), 30<t≤45 gives +8.116 (n=264), t>45 gives +0.611 (n=2,783). Pre-open triggers are as bad as the open itself. The 30–45 cell at t = +2.93 is the loudest number in the analysis and should be read as noise after this many splits.

Section 2 pre-committed to the consequence: a null H2 means the mechanism story is wrong even if H1 passes. Two mechanisms have now failed and there is no working explanation for the effect.

## What changed since the superseded note
| Metric | previous | rebuilt |
|---|---|---|
| trades, +5 min | 3,616 | 3,206 |
| gross Sharpe | +0.987 | +1.043 |
| k ladder, mean bp | +0.267 / +0.832 / +1.181 | +0.149 / +0.852 / +0.795 |
| 2011 trades | 12 | 0 |
| structural break, pre / post | 1.015 / 0.977 | 1.199 / 0.949 |
| years positive | 11 / 13 | 10 / 13 |
| per-trade permutation p | 0.087 | 0.114 |
| Ljung-Box p | 0.136 | 0.0004 |
| H2 b3 | +0.031, p 0.809 | −0.048, p 0.669 |

Headline figures barely moved. The ladder changed shape, and that flips the verdict, though as shown above the shape was never resolvable in either version. Ljung-Box moving from 0.136 to 0.0004 also matters: book returns are autocorrelated, which breaks the independence assumption behind the naive Sharpe t-stat and is another reason to quote the bootstrap number.

**The 2011 row dates the old note.** It reports 12 trades in 2011; the current pipeline produces zero and cannot produce any. `walk_forward_conditional_vol` fits GARCH per year on data strictly before that year and needs 500 prior observations. At ~312 obs/yr and a data start of 2011-01-02 across all ten files, 2011 has no prior history and 2012 has ~312. First fitted sigma is 2013, and sigma is verified null for every pair on every day of 2011 and 2012.

A run with 2011 trades therefore had sigma a walk-forward fit cannot supply. That matches commit a84039f's own message flagging the note as pending a rerun, and it means the old note's early trades were leakage-contaminated. The guard costs 2 of 13 years, leaving 1,278 active days.

The rebuild is the trustworthy side of this. `open` and `exit_px` reproduce the old cache bit for bit, and a 0.03% sigma perturbation moves about one trigger day in 311, so an 11.3% trade-count difference is a methodology change rather than numerical drift.

## Next
- Strategy closed. Six of six candidates have now failed.
- Lockbox unopened, held for strategy #7.
- Entry slippage remains unmeasured, as it was before this run. 1-minute closes are not tradeable prices.
- The exit rule was never tested. The 13:00 exit was chosen for being parameter-free, not for being good, and skew +1.254 suggests exit timing carries weight. That's strategy #7 with its own pre-registration, not a patch to this one.
- The per-day versus per-trade split is worth keeping for the paper: ten marginal signals at rho 0.36 produced a book significant at p = 0.001 while no component cleared p = 0.05.
