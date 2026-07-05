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
| 1 | 0.0713 | 0.0767 | 0.0764 | -0.0052 |
| 2 | 0.0630 | 0.0970 | 0.0652 | -0.0021 |
| 3 | -0.0147 | 0.0711 | -0.0181 | 0.0034 |
| 4 | 0.0330 | 0.0532 | 0.0343 | -0.0013 |
| 5 | -0.0128 | 0.0179 | -0.0131 | 0.0004 |

mean_ic: unshuffled 0.0280, shuffled 0.0632, purged 0.0289
Inflation (unshuffled − purged), all folds: -0.0010
Inflation (shuffled − purged), all folds: 0.0342
Sign count (unshuffled − purged): 2 positive / 3 negative / 0 zero
Largest-gap fold: fold 1 (|diff|=0.0052)
Inflation excluding fold 1: 0.0001
Paired sign-permutation p-value (unsh − purg ≠ 0): 0.5627

**n_splits = 15**

| fold | unsh | shuf | purg | diff (unsh − purg) |
|---|---|---|---|---|
| 1 | 0.0945 | 0.1371 | 0.1000 | -0.0055 |
| 2 | 0.0279 | 0.0428 | 0.0387 | -0.0108 |
| 3 | 0.0247 | 0.0686 | 0.0179 | 0.0068 |
| 4 | 0.0486 | -0.0034 | 0.0261 | 0.0225 |
| 5 | 0.0751 | 0.1459 | 0.0823 | -0.0072 |
| 6 | 0.1018 | 0.0761 | 0.1041 | -0.0024 |
| 7 | 0.0066 | 0.1934 | 0.0026 | 0.0040 |
| 8 | 0.0969 | 0.0411 | 0.0925 | 0.0043 |
| 9 | -0.0214 | 0.0091 | -0.0258 | 0.0044 |
| 10 | 0.1001 | -0.0197 | 0.1110 | -0.0110 |
| 11 | 0.0007 | 0.0708 | -0.0020 | 0.0028 |
| 12 | 0.0263 | 0.0190 | 0.0303 | -0.0041 |
| 13 | -0.0842 | 0.0225 | -0.0848 | 0.0006 |
| 14 | 0.0898 | 0.0652 | 0.0784 | 0.0114 |
| 15 | 0.0845 | -0.0471 | 0.0855 | -0.0010 |

mean_ic: unshuffled 0.0448, shuffled 0.0548, purged 0.0438
Inflation (unshuffled − purged), all folds: 0.0010
Inflation (shuffled − purged), all folds: 0.0110
Sign count (unshuffled − purged): 8 positive / 7 negative / 0 zero
Largest-gap fold: fold 4 (|diff|=0.0225)
Inflation excluding fold 4: -0.0005
Paired sign-permutation p-value (unsh − purg ≠ 0): 0.6796

The sign-permutation test doesn't reject the null that unpurged and purged IC have the same mean, at either fold count. The sign counts back this up directly — roughly an even split between folds where unpurged IC is higher and folds where it's lower. The exclusion check confirms the mean inflation isn't being carried by one bad fold: pulling out the single largest-gap fold barely moves the estimate either direction.

At h=5, k=5 KNN, on this EUR/USD momentum signal, standard unpurged CV does not detectably inflate cross-validated IC relative to purged CV. If the effect exists at all, it's smaller than what 5 or 15 folds of this data can resolve from noise.

## 6. Alternative Explanations
Shuffled KFold breaks temporal order completely, so a shuffled fold's training set can contain observations from both before and after its test window, anywhere in the sample — full look-ahead contamination. That's the more plausible reason it's a larger effect (0.0342, 0.0110). But "larger" here is a magnitude comparison, not a tested one — no significance test was run on that comparison specifically.

## 7. Next Steps
This closes the leakage-magnitude question for this configuration. Chasing a longer horizon or a higher-capacity model purely to force a detectable effect out of KNN isn't worth the time right now. The more useful next step is to treat purged CV as standing protocol rather than something whose necessity needs re-proving every signal. Given the standing decision to bring in ML components with hyperparameter tuning, a fitted model here showing no measurable leakage is reassuring, but it's not a reason to drop the discipline. Tuning introduces its own leakage surface (selecting hyperparameters using test-fold information) that this audit didn't test at all. That's the more relevant leakage risk to audit once those models exist, not overlap-purging on a static k=5 KNN.