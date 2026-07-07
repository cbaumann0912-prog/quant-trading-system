# Week 6 Review — Research Engine Continued (Days 38–42)

## Methodology
`permutation_test`, `purged_cross_validation` — `src/evaluation/`. `information_coefficient`, `information_ratio` — `src/analysis/performance_analyzer.py`. `capm_expected_return`, `pca_factor_decomposition` — `src/analysis/factor_models.py`.

## Findings
- PC2 Carry Regime Signal closed as a strategy candidate: null unconditional IC (Day 38, permutation test), no leakage-driven inflation under purged CV (Day 39), and conditional predictability killed by a near-singular condition number of 2.27×10¹⁰ (Day 41) — above the ~1e10 reliability threshold, discarded despite a borderline p=0.070.
- PCA factor decomposition (Day 42) extracts orthogonal, data-driven factors from the 3-pair covariance matrix, distinct from CAPM's pre-specified market factor — reconstruction and residual decomposition implemented and understood, not just called.

## Interpretation
PC2's death is a clean result, not a disappointing one — three independent tests failing it for three different reasons (no signal, no leakage artifact, ill-conditioned regime split) is exactly the kind of negative result that's citable rather than swept aside. The near-singular condition number is the more interesting finding methodologically: it's a reminder that a borderline p-value can still be meaningless if the matrix behind it is numerically unreliable, and that threshold now stands as a hard discard rule rather than a judgment call. PCA factor extraction closes out the "data-driven vs. theory-driven" factor comparison started with CAPM on Day 41 — worth carrying that contrast (pre-specified factor vs. extracted factor) directly into the paper's methodology section later.