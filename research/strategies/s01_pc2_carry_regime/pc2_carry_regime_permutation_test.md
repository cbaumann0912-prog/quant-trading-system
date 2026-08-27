# Day 38 Research Audit — Permutation Test vs. t-Test on PC2 Carry Regime Signal

## 1. Question Investigated
Does the raw PC2 factor score (Day 19) have statistically detectable predictive content for one-day-forward returns on its own factor-mimicking portfolio, and does the empirical permutation test agree with a conventional parametric (Spearman) correlation t-test on the same data?

## 2. Why It Matters
The PC2 Carry Regime Signal is currently the strongest confirmed candidate in the strategy vault (Day 19: +0.865 USD/JPY loading, 29.2% variance explained). Before any signal construction work is invested in, the raw factor needs to clear a baseline significance bar. Separately, this is the first head-to-head test in this project comparing the empirical permutation method against the parametric t-test approach that Day 8's original (now-excluded) strategy relied on, which matters directly for methodology decisions in the paper.

## 3. Methodology
- Data: 1-minute OHLCV for EUR/USD, GBP/USD, USD/JPY, resampled to daily close, log returns computed as ln(Pₜ/Pₜ₋₁)
- Train/test split: train ≤ 2020-12-31, test ≥ 2021-01-01, chosen to align with the structural break in PC1/PC2 variance share identified in Day 19 (Section 6.5)
- PCA (`pca()`, `src/features/pca.py`) fit on train-period returns only; PC2 loadings extracted and sign-normalized so USD/JPY loading is positive, matching Day 19's convention
- Test-period returns centered using the **train-period mean** (not test-period mean) before projection, to avoid leaking test-period statistics into the out-of-sample PC2 scores
- Factor-mimicking portfolio return series built on the test period using raw (unnormalized) train-period PC2 loadings as weights: r_PC2,t = w_EUR·r_EUR,t + w_GBP·r_GBP,t + w_JPY·r_JPY,t
- Signal forward-shifted by one day: PC2 score at t tested against factor-mimicking portfolio return at t+1
- Three empirical permutation tests (`permutation_test`, `src/evaluation/significance.py`, empirical/rank-based p-values, n_permutations=1000, seed=42):
  - Pooled sample, alternative="two-sided"
  - Positive-signal subset (PC2ₜ > 0), alternative="greater"
  - Negative-signal subset (PC2ₜ < 0), alternative="greater" — corrected from an initial run using alternative="less", after identifying that the stated hypothesis (same-direction co-movement within each regime) implies a positive correlation in both subsets, not a sign flip between them
- Bonferroni and Benjamini-Hochberg correction (`src/evaluation/significance.py`) applied across all three permutation p-values, α=0.05
- Spearman correlation t-test computed on the pooled sample as the parametric comparison point, in place of the original Day 8 t-test (Day 8's original target, FVG_BoS_Reversal, has since been excluded from the strategy vault for lacking economic backing, making it an invalid comparison object; the same underlying data — PC2 vs. forward returns — is instead tested via both methods to isolate the effect of methodology alone)

## 4. Assumptions
- A single train/test split (rather than rolling re-estimation) is treated as sufficient to remove the most severe form of lookahead for this validation question; Day 19 already established that full rolling re-estimation is required before any live signal is deployed, and that requirement is unchanged by this result
- The factor-mimicking portfolio (raw PC2 loadings as weights) is treated as a legitimate proxy for "the return the factor itself represents," not as a capital-allocated, investable portfolio
- Split-sample subsets (positive/negative PC2) are exploratory given the ~45–46% reduction in sample size relative to the pooled test; they are not treated as independently powered confirmatory tests

## 5. Hypothesis
- Pooled: PC2ₜ has a nonzero (unspecified direction) relationship with r_PC2,t+1
- Positive/negative subsets: within each regime, PC2 and forward returns co-move in the same direction, motivated by PC2's economic role as a cross-pair carry/risk-off factor and its documented negative skew and crash asymmetry

## 6. Findings
**Sample size:** n = 934 (pooled, test period, post-alignment). Positive-signal subset: n = 505. Negative-signal subset: n = 429.

**PC2 loadings (train period, sign-normalized to USD/JPY positive):**

| Pair | Loading |
|------|---------|
| EUR/USD | 0.1266 |
| GBP/USD | 0.4557 |
| USD/JPY | 0.8811 |

**Permutation test results (empirical, rank-based p-values):**

| Test | n | observed IC | p-value | alternative |
|------|---|---|---|---|
| Pooled | 934 | 0.0378 | 0.2557 | two-sided |
| Positive-signal subset | 505 | 0.0405 | 0.1928 | greater |
| Negative-signal subset | 429 | 0.0910 | 0.0400 | greater |

**Multiple testing correction (α = 0.05, m = 3):**

| Test | p-value | Bonferroni reject H0 | BH reject H0 |
|------|---------|----------------------|--------------|
| Pooled | 0.2557 | False | False |
| Positive subset | 0.1928 | False | False |
| Negative subset | 0.0400 | False | False |

No test survives multiple testing correction.

**Permutation vs. t-test comparison (pooled sample):**

| Method | p-value |
|--------|---------|
| Empirical permutation (two-sided) | 0.2557 |
| Spearman correlation t-test | 0.2486 |

The two methods agree closely (Δp ≈ 0.007). This is consistent with the fact that the observed relationship is far from any significance boundary and the sample is reasonably large (n=934) — the conditions under which a normal-approximation-based test and an empirical permutation test are expected to converge. This does not establish that the two methods agree in general; Day 37's finding that block bootstrap and i.i.d. bootstrap CIs diverge by 30–38% on std, and diverge in direction on Sharpe, is a reminder that parametric/empirical agreement is data- and statistic-dependent, not universal.

## 7. Alternative Explanations
- **PC2 may still carry information the pooled/split design doesn't capture.** The tests here are limited to a linear/monotonic relationship (Spearman) between the raw PC2 score and one-day-forward factor-mimicking returns. A regime-based approach or nonlinear relationships would not be detected by this design and remain untested.
- **Single train/test split may understate the true relationship if PC2's predictive content is itself regime-dependent and concentrated in a sub-period of the 2021–2026 test window** (e.g., 2022 hiking cycle vs. calmer periods). This audit does not test for that.
- **The negative-subset result now clears the raw 5% bar (p=0.0400) but not the correction**, and neither Bonferroni nor BH rejects it as one of three tests run today. A single sub-threshold p-value among three exploratory splits is the textbook shape of a selection artifact, and the pooled test it came from is nowhere near significance at p=0.2557. It could still reflect a real but weak effect this sample is underpowered to confirm — the split design's reduced n (429 vs. 934 pooled) is a genuine power limitation — but nothing here warrants treating it as a finding.

## 8. Next Steps
- Do not proceed with raw PC2 score as a standalone tradable signal on the basis of this result — it does not clear significance after correction
- If pursuing the PC2 Carry Regime Signal further, consider regime/threshold-based constructions rather than the raw linear score, consistent with Day 19's own findings on threshold crossing frequency and the same-day execution requirement (near-zero lag-1 autocorrelation)
- Revisit the negative-subset near-significant result once rolling PCA re-estimation (raised earlier today as parallel exploratory work) is available, to check whether the effect strengthens, weakens, or is an artifact of the single static train/test split
- Third strategy shortlist slot remains open; this result does not by itself resolve that gap

## 9. Interpretation
Although PC2 remains the most economically interpretable latent factor identified so far, this audit provides no statistical evidence that its raw score alone predicts next-day returns on its factor-mimicking portfolio. The close agreement between the empirical permutation test and the Spearman correlation t-test suggests that, for this large-sample, near-null case, the choice of significance test does not affect the conclusion. The weak signal observed within the negative-PC2 regime is worth monitoring in future rolling-PCA analyses, but it is not strong enough to justify treating the raw PC2 score as a standalone predictive trading signal.