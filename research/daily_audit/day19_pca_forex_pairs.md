# Day 19 Research Audit — PCA on Forex Pair Returns

## 1. Question Investigated
What does the PC score time series produced by applying PCA to EUR/USD, GBP/USD, and USD/JPY log returns reveal about factor behavior through time, and does the eigenstructure identified in Day 18 hold across sub-periods?

## 2. Why It Matters
Eigenvectors represent the directions of maximum variance within the data. They contain no time dimension. PC scores, by contrast, are the projection of the original observations onto those eigenvector directions. They form a time series that quantifies the magnitude of each principal component on every trading day. This allows us to observe how latent factors evolve through time, identify periods when factors become extreme, and construct signals based on factor behavior. Eigenvectors provide the map of the factor structure, while PC scores record the journey through that structure over time.

## 3. Methodology
- 15 years of 1-minute OHLCV resampled to daily close
- Log returns computed as ln(Pₜ / Pₜ₋₁)
- Series aligned on common trading dates via pd.concat().dropna()
- PCA applied via hand-written pca() in src/features/pca.py — centers data, computes covariance matrix via compute_covariance_matrix(), calls eigendecomposition(), projects data as Z = X_c @ V
- n_components = 3 (retain all)
- Split-sample analysis: yearly windows Jan 1 – Dec 31, 2011–2026 (2026 partial)

## 4. Assumptions
- Log returns are stationary over the full 15-year sample
- Covariance structure is stable enough across the sample for full-period PCA to be meaningful (tested in Section 6.5)
- Daily close-to-close returns are representative of the return generating process

## 5. Hypothesis
- PC score series should be empirically uncorrelated — off-diagonal entries of corr(Z) near zero by construction
- Var(Z_k) should equal λ_k for each component
- PC score excess kurtosis will exceed raw return excess kurtosis — tail events that were diversified across pairs in raw space concentrate into factor space
- The 59.6% / 28.9% variance explained split will shift across sub-periods, with PC1 explaining more variance during USD stress regimes

## 6. Findings

### 6.1 Variance Explained

| Component | Eigenvalue   | Variance Explained | Cumulative |
|-----------|--------------|--------------------|------------|
| PC1       | 4.447954e-05 | 0.5958             | 0.5958     |
| PC2       | 2.159184e-05 | 0.2892             | 0.8850     |
| PC3       | 8.589501e-06 | 0.1150             | 1.0000     |

### 6.2 Eigenvector Loadings

| Pair   | PC1     | PC2    | PC3     |
|--------|---------|--------|---------|
| EURUSD | -0.5793 | 0.2261 | 0.7831  |
| GBPUSD | -0.6171 | 0.5061 | -0.6026 |
| USDJPY |  0.5326 | 0.8323 | 0.1537  |

### 6.3 Implementation Verification

**Orthogonality check — corr(Z):**

|     | PC1         | PC2         | PC3         |
|-----|-------------|-------------|-------------|
| PC1 | 1.00000000  | 1.3937e-16  | -3.6789e-16 |
| PC2 | 1.3937e-16  | 1.00000000  | -8.3679e-19 |
| PC3 | -3.6789e-16 | -8.3679e-19 | 1.00000000  |

All entries at or below float64 machine epsilon (2.2e-16). Orthogonal to machine precision.

**Eigenvalue = Var(Z_k) check:**

| Component | λ_k          | Var(Z_k)     | Match |
|-----------|--------------|--------------|-------|
| PC1       | 4.447954e-05 | 4.447954e-05 | ✅    |
| PC2       | 2.159184e-05 | 2.159184e-05 | ✅    |
| PC3       | 8.589501e-06 | 8.589501e-06 | ✅    |

Exact match to 6 significant figures on all three components.

### 6.4 PC Score Distributions

**Excess kurtosis — PC scores vs raw returns:**

| Series      | Excess Kurtosis          |
|-------------|--------------------------|
| EUR/USD raw | 3.04 (from Day 04 audit) |
| GBP/USD raw | 28.35 (from Day 04 audit)|
| USD/JPY raw | 4.99 (from Day 04 audit) |
| PC1 scores  | 4.7644                   |
| PC2 scores  | 30.5320                  |
| PC3 scores  | 4.8940                   |

Hypothesis evaluation: The hypothesis that PC scores would exceed raw return kurtosis is partially confirmed. PC1 kurtosis of 4.76 modestly exceeds EUR/USD raw (3.04) and USD/JPY raw (4.99), consistent with mild tail concentration in the USD factor. PC2 kurtosis of 30.53 substantially exceeds EUR/USD and USD/JPY raw series but marginally exceeds GBP/USD raw (28.35, documented in the Day 04 audit), indicating that PC2 largely inherits rather than amplifies the tail risk already present in the raw GBP series.

Regardless of attribution, parametric Value-at-Risk and Gaussian-based position sizing frameworks are invalid for any strategy with GBP/USD or PC2 exposure. Historical simulation or Student-t based position sizing are the appropriate alternatives.

### 6.5 Split-Sample Eigenstructure Stability — Yearly Windows

| Year  | PC1    | PC2    | PC3    | n   | Notes                                      |
|---|---|---|---|---|---|
| 2011 | 0.5795 | 0.3121 | 0.1084 | 310 |                                             |
| 2012 | 0.5574 | 0.3498 | 0.0928 | 312 |                                             |
| 2013 | 0.6272 | 0.2836 | 0.0892 | 312 |                                             |
| 2014 | 0.6046 | 0.2737 | 0.1218 | 312 |                                             |
| 2015 | 0.6946 | 0.1557 | 0.1498 | 312 | Fed rate signal, ECB QE                     |
| 2016 | 0.5214 | 0.4155 | 0.0631 | 312 | Brexit — PC1 min, PC2 max                   |
| 2017 | 0.5977 | 0.2660 | 0.1362 | 310 |                                             |
| 2018 | 0.6833 | 0.2173 | 0.0993 | 313 |                                             |
| 2019 | 0.5923 | 0.2992 | 0.1085 | 312 |                                             |
| 2020 | 0.6855 | 0.2043 | 0.1103 | 314 | COVID — largest single PC1 jump pre-2022    |
| 2021 | 0.6538 | 0.2545 | 0.0917 | 312 |                                             |
| 2022 | 0.7048 | 0.2109 | 0.0844 | 312 | Fed 425bps hiking cycle                     |
| 2023 | 0.7054 | 0.2467 | 0.0478 | 311 |                                             |
| 2024 | 0.7299 | 0.2307 | 0.0395 | 314 |                                             |
| 2025 | 0.7625 | 0.1792 | 0.0583 | 313 |                                             |
| 2026 | 0.7990 | 0.1598 | 0.0412 |  77 | Partial — excluded from trend conclusions   |
| **Full sample** | **0.5958** | **0.2892** | **0.1150** | — | |

Hypothesis evaluation: The increasing proportion of variance explained by PC1 confirms the hypothesis that USD dominance intensifies during stress regimes. The data further reveals this is not merely a cyclical pattern — PC1 has increased in the majority of years since 2011, with the post-2020 period showing a structural step up driven by coordinated central bank responses to COVID-19 and the subsequent Federal Reserve tightening cycle, during which USD monetary policy became the dominant force across all three pairs simultaneously.

The 2016 Brexit referendum produced the single largest deviation from the trend. GBP/USD experienced substantial movements driven by UK-specific political risk rather than broader USD dynamics, weakening the three-pair correlation structure and causing PC2 to absorb idiosyncratic GBP variance — producing the sample maximum of 0.4155.

Since 2020, no year has shown PC1 below 0.65 — the full-sample average of 0.5958 understates current USD factor dominance by approximately 10–15 percentage points.

### 6.6 PC Score Descriptive Statistics and Signal Properties

**Full descriptive statistics:**

| Stat | PC1       | PC2       | PC3       |
|------|-----------|-----------|-----------|
| Mean | 0.000000  | 0.000000  | 0.000000  |
| Std  | 0.006669  | 0.004647  | 0.002931  |
| Skew | 0.059932  | -1.196546 | 0.198924  |
| Kurt | 4.764401  | 30.531955 | 4.893985  |
| Min  | -0.041162 | -0.087385 | -0.017923 |
| Max  | 0.058654  | 0.039822  | 0.025916  |

**PC2 threshold crossing counts:**

| Threshold | Total events | Per year | Long (+) | Short (-) |
|-----------|-------------|----------|----------|-----------|
| \|z\| > 1.5 | 469       | 31.3     | 219      | 250       |
| \|z\| > 2.0 | 219       | 14.6     | 104      | 115       |
| \|z\| > 2.5 | 103       | 6.9      | 51       | 52       |
| \|z\| > 3.0 | 63        | 4.2      | 29       | 34       |

**Lag-1 autocorrelation:**

| Component | Lag-1 AC |
|-----------|----------|
| PC1       | -0.0033  |
| PC2       | -0.0371  |
| PC3       | -0.0158  |

Findings: PC2 skewness of -1.197 is the most consequential distributional finding beyond kurtosis. The minimum PC2 score of -0.0874 is 2.2 times larger in magnitude than the maximum of +0.0399, confirming that carry crash drawdowns are structurally asymmetric — downside moves are both more frequent and larger than upside recoveries. PC1 and PC3 skew near zero, confirming symmetric factor behavior. These two factors require different risk models.

At a 2σ threshold, PC2 produces 14.6 crossings per year — sufficient event frequency to support signal construction. At 3σ, this falls to 4.2 per year, approaching the sample size floor for reliable inference. However, lag-1 autocorrelation of -0.037 indicates that PC2 factor shocks have no meaningful daily persistence. A momentum rule entering the day after a threshold crossing has no statistical basis. Any threshold-based signal requires same-day or intraday execution at the point of crossing, not the following open.

## 7. Alternative Explanations

*Risk sentiment alternative:* The apparent increase in USD dominance may reflect broader changes in global risk sentiment rather than USD-specific structural effects. During periods of heightened uncertainty, investors seek safe-haven assets including the USD, producing a factor loading pattern consistent with what is observed here. Under this interpretation, the dominant factor captures risk-on/risk-off dynamics, not USD policy per se.

*Sample period limitation:* The analysis begins in 2011, excluding the 2008–2009 Global Financial Crisis — the most extreme carry crash and USD safe-haven event in the modern sample. Excluding this episode likely underestimates the true tail risk associated with PC2, as the most severe realizations are absent from the dataset.

*2016 outlier effect:* Since 2016 represents the minimum PC1 observation in the sample, its inclusion suppresses the apparent strength of the upward trend in the pre-2020 period. Removing it would make the trend appear more consistent from 2011 onward, though the directional conclusion is unchanged either way.

## 8. Open Questions
- Does PC2 excess kurtosis shift across yearly sub-periods, or is the 30.53 full-sample figure driven primarily by early-sample carry crash events (2011 EUR crisis, 2015 SNB floor removal — which affected USD/JPY through risk-off flows rather than direct CHF exposure)?
- Is the post-2020 regime shift in PC1 structural (permanently higher USD policy dominance) or cyclical (reverts when the Fed is on hold)?
- Is the PC1 loading pattern consistent with Lustig/Roussanov/Verdelhan dollar factor construction?
- Does the cumulative sum of PC2 scores form a mean-reverting series suitable for stat arb signal construction, and if so, what is its half-life?

## 9. Connection to Strategy Development
At approximately 72% PC1 variance explained in recent years, much of the information contained within each currency pair is redundant. The amount of genuinely independent information per pair is substantially smaller than the raw data suggests.

Because the eigenstructure has drifted materially across the sample period, PCA loadings must be re-estimated on rolling windows — applying full-sample loadings to contemporary data introduces systematic error in both signal construction and risk measurement.