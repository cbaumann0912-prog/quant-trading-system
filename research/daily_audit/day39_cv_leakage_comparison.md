# Day 39 Research Audit — CV Leakage Comparison: Standard vs Purged K-Fold

## 1. Question
Does standard (unpurged) k-fold cross-validation inflate cross-validated IC relative to purged cross-validation, when evaluating a EUR/USD momentum signal on a 5-day holding period, and if so, is that inflation statistically distinguishable from zero?

## 2. Why It Matters
This is the leakage-magnitude check that either justifies or fails to justify keeping purged CV as the default validation protocol going into signal construction. If unpurged and purged IC are indistinguishable at this horizon, purging is an insurance policy rather than a correction for a measured problem. It also matters given the decision to add ML components to future strategies: if leakage isn't detectable even with a fitted model at k=5, that's a data point on how aggressive the embargo needs to be once real hyperparameter tuning enters the pipeline.

## 3. Methodology
- Signal: 20-day log-return momentum on EUR/USD daily closes (lookback chosen as ~1 trading month).
- Target: 5-day forward log return, `log(close[t+5] / close[t])` — a genuine cumulative holding-period return, not a 1-day return sampled 5 days out. Through-dates for purging are looked up exactly from the underlying trading-day index (position + 5), not approximated.
- Model: KNeighborsRegressor(k=5), fit on each fold's training set, predicted on that fold's test set. A fitted, capacity-bearing model was required rather than a deterministic signal transform, since IC on a deterministic signal doesn't depend on which observations are in the training set and can't reveal leakage.
- IC: Spearman rank correlation between model predictions and realized forward return, per fold.
- Three CV variants compared: standard KFold (shuffle=False), standard KFold (shuffle=True, random_state=42), and purged_cross_validation (embargo_pct=0.01).
- Run at n_splits=5 and n_splits=15 as a fold-count robustness check.
- Sensitivity check: mean inflation recomputed excluding the single largest-|gap| fold, to test whether any detected effect is broad or concentrated in one fold.
- Significance check: paired_sign_permutation_test (own tool, sign-flip permutation, 10,000 draws) applied to the per-fold (unshuffled − purged) differences. Applied only to that comparison, since unshuffled and purged share identical test partitions per fold — a valid pairing. Not applied to shuffled vs. purged, since shuffled folds evaluate on entirely different test data and can't be paired fold-by-fold against purged.

## 4. Assumptions
- embargo_pct=0.01 is a heuristic, not derived from the label construction. At h=5 on daily data it removes roughly 1% of the sample around each fold boundary — a small buffer relative to the 5-day window each label actually spans, so it's plausible the embargo undershoots the true overlap window at the boundaries.
- KNN with k=5 is a low-capacity, high-variance-to-noise choice. It was picked because it's genuinely path-dependent on which points are in the training set, but it's also the kind of model least likely to exploit subtle boundary leakage compared to something with more memorization capacity.
- Daily-close resampling assumes no meaningful information in intraday structure for this signal, consistent with prior days' work.
- 15 years of EUR/USD is treated as stationary enough for a single pooled test. That's almost certainly false in a strict sense.
- The sign-permutation test was chosen over a parametric paired t-test because the per-fold differences are small in number (5 or 15), and there's no reason to assume they're normally distributed.

## 5. Findings
**n_splits = 5**

| fold | unsh | shuf | purg | diff (unsh − purg) |
|---|---|---|---|---|
| 1 | 0.0760 | 0.0764 | 0.0724 | 0.0036 |
| 2 | 0.0088 | 0.0452 | 0.0092 | -0.0004 |
| 3 | 0.0265 | -0.0112 | 0.0210 | 0.0055 |
| 4 | 0.0435 | 0.0774 | 0.0484 | -0.0049 |
| 5 | -0.0310 | 0.0267 | -0.0295 | -0.0015 |

mean_ic: unshuffled 0.0248, shuffled 0.0429, purged 0.0243
Inflation (unshuffled − purged), all folds: 0.0005
Inflation (shuffled − purged), all folds: 0.0186
Sign count (unshuffled − purged): 2 positive / 3 negative / 0 zero
Largest-gap fold: fold 3 (|diff|=0.0055)
Inflation excluding fold 3: -0.0008
Paired sign-permutation p-value (unsh − purg ≠ 0): 0.8719

**n_splits = 15**

| fold | unsh | shuf | purg | diff (unsh − purg) |
|---|---|---|---|---|
| 1 | 0.0679 | 0.0914 | 0.0600 | 0.0079 |
| 2 | 0.1629 | 0.1004 | 0.1444 | 0.0185 |
| 3 | 0.0440 | 0.0982 | 0.0242 | 0.0197 |
| 4 | -0.0190 | 0.0631 | -0.0223 | 0.0032 |
| 5 | 0.0259 | 0.0930 | 0.0310 | -0.0051 |
| 6 | 0.0794 | 0.1056 | 0.0797 | -0.0003 |
| 7 | 0.0745 | -0.0051 | 0.0762 | -0.0017 |
| 8 | -0.0028 | -0.0091 | -0.0098 | 0.0070 |
| 9 | 0.1174 | -0.0638 | 0.1151 | 0.0023 |
| 10 | 0.0108 | 0.0227 | 0.0220 | -0.0111 |
| 11 | 0.1081 | 0.0534 | 0.1136 | -0.0055 |
| 12 | 0.1068 | 0.1181 | 0.1034 | 0.0034 |
| 13 | -0.0077 | 0.0569 | 0.0102 | -0.0179 |
| 14 | -0.0007 | 0.0723 | 0.0069 | -0.0076 |
| 15 | -0.0971 | -0.0161 | -0.0981 | 0.0009 |

mean_ic: unshuffled 0.0447, shuffled 0.0521, purged 0.0438
Inflation (unshuffled − purged), all folds: 0.0009
Inflation (shuffled − purged), all folds: 0.0083
Sign count (unshuffled − purged): 8 positive / 7 negative / 0 zero
Largest-gap fold: fold 3 (|diff|=0.0197)
Inflation excluding fold 3: -0.0004
Paired sign-permutation p-value (unsh − purg ≠ 0): 0.7302

The sign-permutation test doesn't reject the null that unpurged and purged IC have the same mean, at either fold count. The sign counts back this up directly — roughly an even split between folds where unpurged IC is higher and folds where it's lower. The exclusion check confirms the mean inflation isn't being carried by one bad fold: pulling out the single largest-gap fold barely moves the estimate either direction.

At h=5, k=5 KNN, on this EUR/USD momentum signal, standard unpurged CV does not detectably inflate cross-validated IC relative to purged CV. If the effect exists at all, it's smaller than what 5 or 15 folds of this data can resolve from noise.

## 6. Alternative Explanations
Shuffled KFold breaks temporal order completely, so a shuffled fold's training set can contain observations from both before and after its test window, anywhere in the sample — full look-ahead contamination. That's the more plausible reason it's a larger effect (0.0186, 0.0083). But "larger" here is a magnitude comparison, not a tested one — no significance test was run on that comparison specifically.

## 7. Next Steps
This closes the leakage-magnitude question for this configuration. Chasing a longer horizon or a higher-capacity model purely to force a detectable effect out of KNN isn't worth the time right now. The more useful next step is to treat purged CV as standing protocol rather than something whose necessity needs re-proving every signal. Given the standing decision to bring in ML components with hyperparameter tuning, a fitted model here showing no measurable leakage is reassuring, but it's not a reason to drop the discipline. Tuning introduces its own leakage surface (selecting hyperparameters using test-fold information) that this audit didn't test at all. That's the more relevant leakage risk to audit once those models exist, not overlap-purging on a static k=5 KNN.