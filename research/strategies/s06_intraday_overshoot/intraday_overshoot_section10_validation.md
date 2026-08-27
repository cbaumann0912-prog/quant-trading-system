# Section 10 Validation: Intraday Overshoot Reversal

## Verdict
**FAIL.** Section 10 kills the strategy on any single failed criterion. Two fail.

*Figures regenerated Day 72 at `N_BOOTSTRAP = 10,000`. The Day 57 figures this table previously carried are preserved in the addendum below; they cannot be regenerated from any committed state of this repository and are superseded rather than corrected.*

| Criterion | Result |
|---|---|
| H1 primary p < 0.05, block-bootstrap | PASS, p < 0.00001 |
| H1 predicted sign | PASS, +1.0464 Sharpe |
| R1 threshold monotone | PASS, and uninformative — see below |
| R1 same sign across k | PASS |
| R2 same sign both halves | PASS |
| R2 post-break half p < 0.05 | PASS, p = 0.0072 |
| R3 permutation, per-trade | **FAIL**, p = 0.1349 |
| R3 permutation, per-day | PASS, p = 0.0010 |
| Reliability gate | PASS, cond 4.26e6, max VIF 1.153 |
| Cost gate at 2.0 pips | PASS, net Sharpe +0.816 |
| BH rank 1 of 6 | PASS, bar 0.00833 |
| H2 speed interaction | **FAIL**, b3 = −0.0465, p = 0.679 |

The 2024–2026 lockbox stays sealed. Section 10 permits opening only on a PASS.

## Method
`intraday_overshoot_section10_validation.py`, rebuilt from raw 1-minute bars every run via `src/signals/intraday_overshoot.py`. No cached intermediates. Ten pairs, 2011–2023, k = 2.0, +5 minute entry, 259.44 obs/yr empirical annualization over 12.99 years. `N_BOOTSTRAP = 10,000`, `SEED = 42` threaded through every randomness consumer including the per-pair IR interval.

Statistics come from the repo's tested modules: `block_bootstrap`, `paired_sign_permutation_test`, `benjamini_hochberg_correction`, `interaction_regression_centered`, `PerformanceAnalyzer.run_report`, `compute_achieved_power`, `compute_required_sample_size`.

## R1 cannot resolve what it claims to test
| k | trades | mean bp | SE bp | t | 95% CI bp | hit rate | gross SR |
|---|---|---|---|---|---|---|---|
| 1.5 | 6,706 | +0.135 | 0.346 | +0.39 | [−0.543, +0.813] | 50.2% | +0.993 |
| 2.0 | 3,211 | +0.832 | 0.547 | +1.52 | [−0.240, +1.904] | 51.4% | +1.046 |
| 2.5 | 1,588 | +0.874 | 0.853 | +1.03 | [−0.799, +2.547] | 52.0% | +0.757 |

The ladder rises with k, so R1 passes. **It should not be read as evidence, and the reason is the point of this section.**

Every one of the three intervals spans zero. The deciding gap, k=2.0 to k=2.5, is **+0.042 bp against a standard error of 1.013, a t-statistic of +0.04**. The ladder's ordering is therefore decided at four one-hundredths of a standard error. Four trades out of 1,588 flip it, and at excess kurtosis +22.5 a handful of extremes sets it. Year by year the ladder is monotone in 2 of 11 tradeable years.

These standard errors are now emitted by the validation script itself rather than derived elsewhere and copied in, which is the provenance gap that let the Day 57 figures drift undetected. They are also almost exactly the Day 57 values — 0.346, 0.548 and 0.853 then against 0.346, 0.547 and 0.853 now — so the dispersion of this ladder never moved. Only its ordering did.

The criterion has now returned FAIL on the Day 57 run, PASS on a Linux rebuild, and PASS on the declared Windows environment, on the same data under the same pre-registered rule.

The superseded note called this ladder "the single strongest piece of evidence that the effect is real." That reading was not supportable then and is not supportable now that it passes. **Treat R1 as uninformative in both directions.** What survives from Section 9 is narrower: the effect isn't concentrated at low k, so bid-ask bounce stays ruled out.

## R3 fails on the test the spec chose
Section 10 asks for "a 1,000-permutation shuffle of the fade direction, holding entry timing fixed." That's the per-trade test, and it gives p = 0.1349.

The per-day version gives p = 0.0010, but it preserves which pairs traded together, so it asks whether the book beats a coin flip rather than whether the direction does. Both answer real questions. The spec pre-committed to the first.

The plain t-test agrees with the strict reading: pooled mean trade return has t = +1.52, p = 0.1281.

## Where the return comes from
| Pair | trades | mean bp | t | Sharpe |
|---|---|---|---|---|
| EUR/JPY | 259 | +3.498 | +1.69 | +0.468 |
| AUD/USD | 402 | +2.029 | +1.21 | +0.337 |
| EUR/GBP | 322 | +1.810 | +1.32 | +0.367 |
| NZD/USD | 330 | +2.171 | +1.00 | +0.278 |
| USD/CAD | 309 | +1.032 | +0.63 | +0.175 |
| EUR/CHF | 296 | +0.171 | +0.13 | +0.037 |
| GBP/USD | 373 | +0.049 | +0.03 | +0.009 |
| USD/CHF | 282 | +0.046 | +0.02 | +0.007 |
| EUR/USD | 292 | −0.462 | −0.26 | −0.071 |
| USD/JPY | 346 | −1.776 | −1.05 | −0.291 |

The book looks strong: Sharpe +1.0464, bootstrap p < 0.00001, CI [+0.5019, +1.5785], ten of thirteen years positive, both structural-break halves significant at +1.1980 and +0.9555.

No individual pair reaches significance. Largest |t| is 1.69, and the bootstrap CI on cross-pair mean IR is [−0.0011, +0.0551] at 10,000 resamples, which includes zero. If trades were independent within a day the book Sharpe would be +0.4221 rather than +1.0464. Cross-pair correlation is +0.3652, effective breadth 2.33 of 10.

Almost all of the book Sharpe comes from that gap between +0.4221 and +1.0464. Diversification is a real source of return and Section 0 said so upfront, but it leaves the strategy resting on a directional edge that nothing here separates from zero.

## Power
Achieved power 0.6419. Reaching 80% needs 1,860 active days; there are 1,279. The study cannot resolve its own effect size. Same calendar ceiling that killed the momentum book: 13 years needs Sharpe ≈ 0.55 for t = 2, and sampling intraday doesn't change it.

## H2 and the mechanism
| Bucket | trades | mean bp | t | hit rate |
|---|---|---|---|---|
| fast, ≤30 min | 159 | −7.064 | −1.44 | 37.7% |
| slow | 3,052 | +1.244 | +2.42 | 52.1% |

b3 = −0.0465, p = 0.679, Cohen's d −0.2686. The prediction was b3 > 0 and significant. It's neither, and the point estimate runs backwards.

The NYSE-open alternative fails too. Splitting trigger time four ways: t<30 gives −6.368 bp (n=63), t=30 exactly −7.521 (n=96), 30<t≤45 gives +8.186 (n=263), t>45 gives +0.589 (n=2,789). Pre-open triggers are as bad as the open itself. The 30–45 cell at t = +2.95 is the loudest number in the analysis and should be read as noise after this many splits.

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

## Addendum, Day 72 — the published figures were not reproducible, and R1 flips

Day 71 left the full section 10 re-run outstanding and flagged `N_BOOTSTRAP = 3,000` as too noisy. Both are now closed. `N_BOOTSTRAP` is 10,000, the `bootstrap_confidence_interval` call at line 277 receives `seed=SEED` like every other randomness consumer in the script, and the validation was re-executed against the raw minute bars for all ten pairs on the project's declared environment — Windows, Python 3.12.10, the pinned library set. Those are the figures now published above. The lockbox was not touched.

**The Day 57 figures do not reproduce.** Two independent re-runs were performed: one on Linux / Python 3.10.12 with the same pinned library versions, and one on the declared Windows / Python 3.12 environment. Neither matches what this document previously published, and they do not differ from it in the same places.

| Pair | Day 57 published | Linux rebuild | Windows rebuild |
|---|---:|---:|---:|
| EUR/USD | 296 | 292 | 292 |
| GBP/USD | 372 | 373 | 373 |
| USD/JPY | 343 | 346 | 346 |
| NZD/USD | 333 | 330 | 330 |
| USD/CHF | 278 | 277 | 282 |
| EUR/GBP | 318 | 318 | 322 |
| AUD/USD | 402 | 402 | 402 |
| USD/CAD | 309 | 309 | 309 |
| EUR/JPY | 259 | 259 | 259 |
| EUR/CHF | 296 | 296 | 296 |
| **total** | **3,206** | **3,202** | **3,211** |

The R1 ladder on the Linux rebuild, for the record, since the published ladder above is the Windows one:

| k | trades | mean bp | hit rate | gross SR | net SR |
|---:|---:|---:|---:|---:|---:|
| 1.5 | 6,697 | +0.134 | 50.2% | +0.991 | +0.541 |
| 2.0 | 3,202 | +0.825 | 51.4% | +1.038 | +0.808 |
| 2.5 | 1,588 | +0.874 | 52.0% | +0.761 | +0.631 |

Monotone increasing, same sign across k, so R1 passes on this build as it does on the Windows one. The two ladders differ by nine trades at k = 1.5, nine at k = 2.0 and none at k = 2.5, and by 0.001, 0.007 and 0.000 basis points respectively.

The two rebuilds agree on eight of ten pairs. Four — EUR/USD, GBP/USD, USD/JPY, NZD/USD — moved off the Day 57 figures **identically on both platforms**, which rules out the platform as their cause. Only USD/CHF and EUR/GBP genuinely differ between the two builds, and that residual is the GARCH maximum-likelihood optimum landing differently against a different linear-algebra backend, at the scale this document already bounds: a 0.03% sigma perturbation moves about one trigger day in 311.

**What was ruled out for the other four.** The validation script is byte-identical to commit `d356b08` apart from the two edits named above; `ef2e8fa` moved it between directories with zero content change. `src/signals/intraday_overshoot.py`, `src/features/garch.py` and `src/features/sessions.py` have changed only under `c89739f`, and only in module docstrings, logging calls and I/O exception guards — the sole non-docstring lines in `garch.py` are two whitespace deletions. `requirements.txt` has changed only its line endings; the pinned versions are unchanged. The raw CSVs predate this document. And the overshoot chain consumes no randomness whatsoever: no seed, no generator, no `random`, no k-means anywhere between the minute bars and the trade list.

No committed change accounts for the four-pair difference. The conclusion this supports is that **the Day 57 figures were produced by a state that is not in the history** — an uncommitted local edit, an earlier data vintage, or a different interpreter. They are superseded rather than corrected, because there is nothing to correct them against.

This is the same defect Day 71 found in `bootstrap_confidence_interval`, one level up the stack. There the interval could not be regenerated because the function drew from an unseeded global generator. Here a headline result cannot be regenerated at all from the code that is supposed to produce it. The methodological note is the same in both cases: reproducibility is a property that has to be tested for, not assumed from the fact that a script exists.

### Superseded Day 57 figures, for the record

| Statistic | Day 57 published | Day 72 published |
|---|---|---|
| trades, k = 2.0, +5 min | 3,206 | 3,211 |
| k ladder trades | 6,708 / 3,206 / 1,584 | 6,706 / 3,211 / 1,588 |
| k ladder mean bp | +0.149 / +0.852 / +0.795 | +0.135 / +0.832 / +0.874 |
| **R1 threshold monotone** | **FAIL** | **PASS** |
| book Sharpe | +1.043 | +1.0464 |
| Sharpe bootstrap CI | [+0.493, +1.584] | [+0.5019, +1.5785] |
| cross-pair correlation | +0.364 | +0.3652 |
| effective breadth | 2.34 | 2.33 |
| book Sharpe if iid | +0.432 | +0.4221 |
| per-pair mean IR CI | [−0.0002, +0.0543] at N=3,000 | [−0.0011, +0.0551] at N=10,000 |
| per-trade permutation p | 0.114 | 0.1349 |
| R2 post-break p | 0.0093 | 0.0072 |
| achieved power | 0.639 | 0.6419 |
| active days | 1,278 | 1,279 |
| n for 80% power | 1,871 | 1,860 |
| Ljung-Box p | 0.0004 | 0.0006 |
| H2 b3 | −0.048, p 0.669 | −0.0465, p 0.679 |
| **Section 10 verdict** | **FAIL** | **FAIL** |

The "What changed since the superseded note" table above compares an older cached run against the Day 57 rebuild. Its "rebuilt" column is the Day 57 column here, and is itself now superseded.

**The bootstrap changes are not what moved the interval.** Holding one trade set fixed, the per-pair mean IR interval is [−0.0014, +0.0555] at N = 3,000 with seed 42, [−0.0011, +0.0546] at N = 10,000 with seed 42, and [−0.0010, +0.0543] at N = 10,000 unseeded — a ±0.001 spread matching the sensitivity Day 71 measured. The gap between the Day 57 interval and the current one is the underlying trade set, not the resampling.

**What this does to R1.** The criterion has now returned FAIL, PASS and PASS on the same data under the same pre-registered rule, differing only in which machine and which vintage of the pipeline produced the trade list. Its verdict turns on four trades in 1,588. A pre-registered criterion whose outcome is decided at that margin is not measuring the market, and the strategy verdict does not depend on it: Section 10 fails on any single criterion, and R3 per-trade and H2 fail in every run. R1 is uninformative in both directions, and the fact that it now passes is a cleaner demonstration of that than its original failure was.

## Next
- Strategy closed. Six of six candidates have now failed.
- Lockbox unopened, held for strategy #7.
- Entry slippage remains unmeasured, as it was before this run. 1-minute closes are not tradeable prices.
- The exit rule was never tested. The 13:00 exit was chosen for being parameter-free, not for being good, and skew +1.254 suggests exit timing carries weight. That's strategy #7 with its own pre-registration, not a patch to this one.
- The per-day versus per-trade split is worth keeping for the paper: ten marginal signals at rho 0.36 produced a book significant at p = 0.001 while no component cleared p = 0.05.
