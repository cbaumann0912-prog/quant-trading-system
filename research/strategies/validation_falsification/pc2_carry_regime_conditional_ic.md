# Day 41 Research Audit — PC2 Carry Regime Conditional Predictability

## Question
Does the PC2 Carry Regime signal, which showed null unconditional predictive power (Day 38: pooled IC = -0.0017, p = 0.951; Day 39: no detectable leakage-inflation artifact from overlap-purging), carry conditional predictive power once the sample is split by a volatility regime? I.e., is PC2's average null IC masking a regime in which the signal genuinely predicts forward returns, diluted in the pooled estimate by a regime in which it does not?

## Why It Matters
This question determines whether the PC2 Carry Regime strategy — currently the leading candidate for the third strategy slot, framed as an ML-gated signal (classifier on rolling volatility, rate differentials, vol index proxy) — has any genuine statistical basis to build on, or whether the entire candidate should be retired before any classifier-gating work begins. Since the classifier's proposed role was to identify exactly this kind of conditional regime, this test is a direct evaluation of whether that premise is worth pursuing at all.

## Methodology
Pre-registered before any results were viewed:

- **Regime variable**: rolling realized volatility of the PC2-implied combined return series (`r_pc2_test`), 26-day window (~1 trading month at the empirical ~312 day/year FX annualization factor).
- **Primary test**: interaction regression, R_{t+1} = b0 + b1·PC2_t + b2·Vol_t + b3·(PC2_t × Vol_t) + eps_t, fit via the newly built `interaction_regression` function (`src/stats/regression.py`), cross-validated against `statsmodels.OLS` to floating-point precision before use. Decision hinges on the significance and magnitude of b3, the interaction term.
- **Regime threshold (primary test's regime-split companion)**: 156-day trailing rolling window (~6 months), true rolling, not expanding, chosen to track the PC distributional drift already observed empirically without leaking future information into historical regime classifications.
- **Robustness check**: full-sample median split on the same rolling volatility series, with conditional IC (Spearman) computed and permutation-tested (`permutation_test`, 1000 permutations) separately in each half.
- Both the primary test and the robustness check were run and reported regardless of outcome, with no threshold definition chosen after seeing results.

PC2 signal construction (train/test split at 2021-01-01, sign normalization to USD/JPY positive, three-pair loadings) is identical to the Day 40 IC/IR audit's signal definition, so this test evaluates the same signal already characterized there.

## Assumptions
- The 26-day vol-estimation window and 156-day regime-threshold window are fixed a priori, not tuned against results.
- The interaction regression's standard errors assume the usual OLS conditions (homoskedastic, uncorrelated errors); this assumption is itself in tension with the finding below (see Alternative Explanations).
- Regime definitions (rolling-window and median-split) are treated as two independent operationalizations of the same underlying hypothesis, not as two separate hypotheses each deserving its own multiple-testing correction — consistent with the pre-registration's intent to prevent post-hoc threshold selection, not to inflate the number of tested hypotheses.

## Findings
**Primary test (interaction regression, n = 1613):**

| Term | Coefficient | Std. Error | t-stat | p-value |
|---|---|---|---|---|
| Intercept | -4.52e-06 | 3.24e-04 | -0.014 | 0.989 |
| PC2 (x1) | 0.1601 | 0.0849 | 1.885 | 0.060 |
| Vol (x2) | 0.0514 | 0.0750 | 0.685 | 0.493 |
| PC2 × Vol (interaction) | -29.379 | 16.179 | -1.816 | 0.070 |

R² = 0.0024. **Condition number of X'X = 2.27 × 10¹⁰** — above the ~1e10 threshold at which the design matrix is considered near-singular, meaning the standard errors above (and the p-values derived from them) are themselves numerically unreliable, not just individually non-significant.

**Robustness check 1 — 156-day rolling threshold, split-sample conditional IC:**

| Regime | n | IC | p-value |
|---|---|---|---|
| High-vol | 712 | -0.0003 | 0.996 |
| Low-vol | 926 | -0.0014 | 0.973 |

**Robustness check 2 — full-sample median split, conditional IC:**

| Regime | n | IC | p-value |
|---|---|---|---|
| High-vol | 807 | 0.0047 | 0.902 |
| Low-vol | 806 | -0.0078 | 0.830 |

All four robustness-check cells are indistinguishable from zero across both regime definitions and both regime sides.

**Verdict: hypothesis (a) is rejected. PC2 does not carry conditionally predictive power under either tested regime definition.**

The reasoning, combining all three pieces of evidence:

1. The interaction term's p = 0.070 is not a near-miss worth taking at face value — it comes from a regression whose design matrix is already flagged as near-singular. Severe collinearity between PC2, volatility, and their product inflates and destabilizes the coefficient's standard error, so the p-value itself is an unreliable estimate of significance rather than a borderline-but-trustworthy one.
2. Even taken at face value, R² = 0.0024 means the full three-term model explains roughly a quarter of one percent of the variance in forward returns — a magnitude with no practical relevance regardless of whether any coefficient cleared a significance threshold.
3. Most decisively: the pre-registered robustness checks — built specifically to prevent a single noisy primary-test result from being accepted uncritically — returned four independent, unanimous nulls across two different regime definitions and both regime sides each. A genuine conditional effect should surface under more than one reasonable way of asking the same question; this one did not surface under any.

## Alternative Explanations
- **Regime variable misspecification**: it remains possible that volatility of the PC2-implied combined return series is the wrong conditioning variable, and that rate differentials or a vol index proxy (the other two features originally proposed for the ML-gated version) would show a different result. This has not been tested and is not covered by today's audit.
- **Sample-period dependence**: the test window is the full post-2021 test split; a genuine but transient regime effect present only in a sub-period could be masked by pooling across the whole test period. This was not separately tested and would itself require new pre-registration to avoid post-hoc period selection.
- **Ill-conditioning as a symptom of a deeper problem**: the near-singular design matrix may partly reflect that PC2 and rolling volatility are themselves correlated (both derived from the same underlying return series), which is a structural property of this specific regime-variable choice rather than a data artifact. A regime variable constructed independently of the PC2 series itself (e.g., an external vol index) would not share this problem.

## Next Steps
- Treat hypothesis (a) as closed for the volatility-regime definition tested here. Do not revisit this specific regime/threshold combination without new evidence motivating it.
- PC2 has now failed three independent tests: null unconditional IC (permutation test), no leakage-inflation artifact under purged cross-validation, and no conditional predictability under either regime definition tested today. Continuing to pursue PC2 as a signal is not supported by any evidence gathered so far.
- If PC2 is revisited in the future, the only remaining untested avenue is a genuinely independent regime variable (rate differential or vol index proxy, not derived from the PC2 series itself), which would need its own pre-registration before testing to avoid repeating today's threshold-shopping risk in a different guise.
- Otherwise, redirect strategy-search effort toward other candidates: statistical arbitrage revival via a nonlinear or rolling ML hedge ratio, or a genuinely new candidate independent of both PC2 and the cointegration-based pairs approach already ruled out.