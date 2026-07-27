# Day 40 — IC/IR on All Signals

## 1. What Question Was Investigated?
For PC2 Carry Strategy, compute the Information Coefficient (IC), rolling IC, and Information Ratio (IR). Estimate how many independent bets per year the strategy makes, and what the Fundamental Law of Active Management (`IR ≈ IC × sqrt(BR)`) predicts for its IR given that breadth.

## 2. Why Does the Question Matter?
IC/IR is the standard framework for checking whether a forecast's skill is economically meaningful once bet frequency and independence are accounted for. A strategy with modest IC but many genuinely independent bets can beat one with high IC but few — but only if breadth is estimated honestly rather than assumed from raw observation count. Getting breadth wrong overstates how much independent evidence a signal has actually accumulated, which undermines any IR reported downstream.

## 3. Methodology
**Scope constraint:** no strategy has coded entry/exit logic yet.  What's computable now is IC/IR on **existing statistical research artifacts**, following the Day 38 precedent.
- **PC2 Carry Regime Signal**: computable factor score (built Day 38, reconstructed here) — the test-period PC2 projection, sign-normalized to positive USD/JPY loading, correlated against one-day-forward PC2 factor-mimicking-portfolio returns.
- **PC2 computation**: `information_coefficient` (Spearman) applied to the aligned signal/forward-return pair gives a pooled IC. Rolling IC is computed across a window grid (20, 40, 60, 90, 120) rather than one arbitrary choice — a conclusion that only holds at one window length isn't trustworthy yet, same principle as the Day 5 correlation-regime work. `information_ratio` runs two ways per window: `method="empirical"` (mean/std of the rolling IC series) and `method="fundamental_law"` (IC × sqrt(breadth)), the latter using both raw day-count (`BR_raw`) and autocorrelation-adjusted effective breadth (`BR_eff`).
- **BR_eff estimation**: the signal's own lag-`k` autocorrelation is computed directly rather than assuming independent daily bets. Lags are checked from 1 upward until autocorrelation drops below 0.1; the mean across those lags becomes `rho` in `BR_eff = N / (1 + (N-1)*rho)`.
- **Regime split**: positive/negative subsets follow the exact Day 38 definition (`signal > 0` / `signal < 0`), extended across every computation above rather than reported pooled-only.
- **Block bootstrap IC confidence interval**: `block_bootstrap` operates on a single series, but IC is a paired statistic, so it's applied to the position-index array with a statistic function that looks up both signal and forward-return arrays at the resampled positions. Each resample gets reset to a clean positional index — bootstrap resampling with replacement produces duplicate dates, and duplicate-indexed series passed into `information_coefficient`'s internal `.align()` step could silently corrupt the correlation. Block size is set to each regime's own decorrelation lag (minimum 2), not a fixed constant. 2000 resamples per regime; reported interval is the 2.5th–97.5th percentile.
- **Run-level correlation-adjusted breadth**: an alternative to lag-based `BR_eff` that doesn't assume every contiguous run is independent of its neighbors. Contiguous same-sign spans are identified from the positional, calendar-adjacent signal sequence — same adjacency convention as the Day 5 regime-break detection — giving `n_runs` and average run length as descriptive statistics. Each run's mean forward return becomes one observation in a run-indexed series; that series' own lag-1 autocorrelation (`rho_runs`) is computed and fed into the same `BR_eff` formula one level up (`BR_eff_runs = n_runs / (1 + (n_runs-1)*rho_runs)`), floored at `rho_runs ≥ 0` since negative autocorrelation between runs isn't the claim being tested here and can otherwise push the formula to a negative or undefined breadth.

## 4. Assumptions
- Spearman IC is used throughout, consistent with the framework's distribution-agnostic approach given confirmed excess kurtosis in the underlying returns.
- The window grid (20–120) spans roughly one to six trading months; it isn't derived from an economic argument about carry-regime duration.
- `rho` for `BR_eff` is lag-1-dominated, so `BR_eff` reflects short-horizon persistence only — it misses any longer-horizon structure in the signal's sign or magnitude.
- Fundamental-law IR assumes constant IC across all bets. Pooled IC is statistically indistinguishable from zero (Day 38: p = 0.256), so this assumption is being applied to a signal that may not have a stable IC to begin with.
- The bootstrap's block size comes from each regime's own decorrelation lag — a stated rule, not a validated one. Same open block-size question as prior bootstrap work, just applied to a signal instead of a returns series.
- Positive/negative subsets are non-contiguous in calendar time. Lag-`k` autocorrelation on these filtered subsets measures correlation between the `k`-th most recent *same-sign* observation, not `k` calendar days prior — weaker and different from the contiguous-run method's genuine calendar adjacency.
- The run-level correlation-adjusted breadth tests one specific form of dependence between neighboring runs (correlation of mean forward returns) and floors negative autocorrelation at zero. It doesn't rule out other forms of shared structure between runs (correlated volatility, clustering around common macro events without correlated returns) — finding no evidence of this one form of dependence isn't the same as proving true independence.

## 5. Findings
**PC2 Carry Regime Signal:**
| Metric | Pooled | Positive | Negative |
|---|---|---|---|
| n_obs | 1638 | 890 | 748 |
| IC (Spearman) | 0.0378 | 0.0405 | 0.0910 |
| Lag-1 autocorrelation | 0.0423 | 0.0766 | 0.2339 |
| Decorrelation lag | 1 | 2 | 3 |
| rho estimate (BR_eff input) | 0.0423 | 0.0766 | 0.1660 |
| BR_raw (day count) | 1638 | 890 | 748 |
| BR_eff (autocorrelation-adjusted) | 74.68 | 10.30 | 7.70 |
| IR, fundamental law, BR_raw | 1.1549 | 0.9099 | 1.8847 |
| IR, fundamental law, BR_eff | 0.1816 | 0.1445 | 0.2220 |
| Block bootstrap IC 95% CI | [-0.0290, 0.1008] | [-0.0459, 0.1255] | [-0.0045, 0.1914] |
| CI excludes zero | No | No | No |

No regime's CI excludes zero — including negative, which had the most encouraging point estimate (IC = 0.0910) and a permutation p-value that clears the raw 5% bar but not the multiple-testing correction (p = 0.040, Day 38). Its CI lower bound of -0.0045 sits just below zero, which is exactly the disagreement you would expect between a raw p-value and a block bootstrap that respects serial dependence. The bootstrap is the more conservative and more defensible of the two under these conditions, so it is correcting an overstated significance claim here, not confirming one.

**Run-level correlation-adjusted breadth:**
| Metric | Positive | Negative |
|---|---|---|
| n_runs (contiguous same-sign spans, descriptive) | 425 | 426 |
| avg_run_length (days, descriptive) | 2.09 | 1.76 |
| rho_runs (lag-1 autocorr of per-run mean forward return) | 0.0535 | -0.0813 |
| BR_eff_runs | 17.93 | 426.00 |
| IR, fundamental law, breadth=BR_eff_runs | 0.0737 | 1.1797 |

The correction lands very differently on the two regimes. Positive regime: `rho_runs` is positive, so the method finds real dependence between neighboring runs' outcomes and shrinks breadth hard — 425 down to 17.93, IR falling from an earlier raw-count estimate to 0.0737. Negative regime: `rho_runs` is negative, floors to zero, and `BR_eff_runs` comes back unchanged from the raw run count (426), leaving IR at 1.1797 — numerically identical to what a naive independent-runs assumption would have given. This should not be read as validating that 1.18 figure. It means this specific test — correlation between consecutive runs' mean forward returns — found nothing to correct for the negative regime, not that the regime's runs are proven independent, and not that the IR is more credible. The block bootstrap CI on this regime's IC already includes zero (above); a breadth correction, found or not found, doesn't change that the underlying IC isn't distinguishable from zero in the first place. Average run length under two days for both regimes still contradicts the working assumption that PC2 regimes persist for weeks.

**Rolling IC / empirical IR, pooled:**
| Window | n_windows | mean IC | std IC | IR (empirical) |
|---|---|---|---|---|
| 20 | 81 | -0.0631 | 0.2241 | -0.2814 |
| 40 | 40 | -0.0274 | 0.1507 | -0.1818 |
| 60 | 27 | -0.0195 | 0.1096 | -0.1777 |
| 90 | 18 | -0.0098 | 0.1078 | -0.0907 |
| 120 | 13 | -0.0090 | 0.0964 | -0.0930 |

Mean IC and IR magnitude decay toward zero as window length grows, with estimation noise falling in step. That's the signature of small-sample noise, not a real effect — a genuine relationship wouldn't shrink as more observations feed each estimate.

**Rolling IC / empirical IR, negative regime:**
| Window | n_windows | mean IC | std IC | IR (empirical) |
|---|---|---|---|---|
| 20 | 37 | 0.0685 | 0.1809 | 0.3785 |
| 40 | 18 | 0.0574 | 0.0991 | 0.5795 |
| 60 | 12 | 0.0441 | 0.0718 | 0.6137 |
| 90 | 8 | 0.0541 | 0.0793 | 0.6819 |
| 120 | 6 | 0.0651 | 0.0531 | 1.2261 |

Unlike pooled, this doesn't decay toward zero — it holds in a 0.04–0.07 band, a genuinely different pattern from the pooled noise signature. But the largest windows (90, 120) rest on 6–8 data points per mean/std calculation, so those IR values (0.68, 1.23) owe as much to a thin denominator as to any real effect.

## 6. Alternative Explanations
- The negative IC at small windows could reflect genuine short-horizon mean-reversion rather than noise — but the monotonic decay toward zero as window length grows argues against it; a real effect should persist or intensify at short windows, not simply track the width of the confidence interval.
- Near-zero pooled lag-1 autocorrelation could be sign-cancellation: if the signal autocorrelates strongly within regimes but the sample mixes opposite-sign regimes roughly evenly, naive full-sample lag-1 autocorrelation understates true within-regime persistence. Untested — would need autocorrelation conditional on a regime label that doesn't currently exist.
- `BR_eff` from lag-1 alone could understate breadth reduction (if correlation at lags 2–10 matters and the stopping rule misses it) or overstate it (if lag-1 itself is noisy at this sample size).
- The permutation test's borderline p (0.056) and the bootstrap's failure to exclude zero aren't necessarily in conflict — p ≈ 0.05 corresponds to a CI that just barely excludes zero, and small differences in how each method handles serial dependence could tip either one across that line. More likely the true effect sits right on the detection boundary than that one test is simply wrong.
- The negative regime's `rho_runs` coming back negative rather than zero could itself be informative — mild anti-persistence between neighboring runs (a strong negative run tending to follow a strong positive one, or vice versa) rather than pure noise. Untested; would need more runs than 426 to distinguish genuine mean-reversion between runs from sampling noise around zero.

## 7. Next Steps
- Treat the block bootstrap CI as the primary significance check from here, not the permutation test. It excluded zero for none of the three regimes.
- The negative regime's non-decaying rolling IC deserves a closer look — but not the largest-window IR numbers themselves (90, 120), which lean on too few observations to quote standalone.
- Don't cite the small-window pooled IR figures (20, 40) as evidence of a real negative edge anywhere downstream. They're noise, and this is that flag.
- Wherever these breadth numbers get reused, say plainly that the run-level correction worked for the positive regime (real dependence found, breadth cut ~24x) but found nothing for the negative regime — and that the negative regime's unchanged breadth doesn't make its IR more credible, given the IC's own confidence interval already includes zero.
- The window grid (20–120) still needs an actual rationale instead of being a first-pass guess.