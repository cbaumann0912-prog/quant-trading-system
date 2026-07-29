# Day 48 Research Audit — Section 10 Validation, Volatility Regime Breakout/Mean-Reversion

## Question
Does the volatility-regime breakout/mean-reversion strategy have a genuine, regime-dependent edge, or does the regime classifier add nothing beyond what an unconditional signal already captures? 
Concretely: is b3 reliably different from zero for the momentum leg and the reversion leg, pooled out-of-sample across walk-forward folds and all three pairs?

## Why It Matters
This is the strategy's pre-registered falsification test (Section 1, item 1 and Section 10). Everything upstream, the entry/exit/sizing rules — is provisional until this test clears. A momentum-only or reversion-only pass would also mean the strategy as specified fails: Section 10 requires both legs to pass, since a single-leg edge is a different, smaller strategy than the one hypothesized

## Methodology
Pre-registered in Section 10 before this script was run; no threshold was chosen after seeing results.

- **Primary test**: two interaction regressions, pooled across all 3 pairs (EUR/USD, GBP/USD, USD/JPY) and all out-of-sample walk-forward test folds (`WalkForwardValidator`, 5-year train / 12-month test / 5-day embargo, 10 folds per pair, same fold geometry validated for leakage in the Day 47 regime-refit stability audit):
  - Momentum: R_{t+26} = b0 + b1*momentum_signal + b2*turbulent_dummy + b3*(momentum_signal x turbulent_dummy) + eps
  - Reversion: R_{t+26} = b0 + b1*price_z + b2*calm_dummy + b3*(price_z x calm_dummy) + eps
  Fit via `interaction_regression_centered`, which mean-centers both main effects before forming the interaction termand reports both a condition-number and a VIF diagnostic per fit.
- **Reliability gate**: condition number of X'X < 1e10 and every VIF < 10, both per leg. Precedent: the Day 41 PC2 interaction regression returned condition number 2.27e10 and was discarded as unreliable regardless of its p-value.
- **Regime construction**: 2-feature PCA composite refit inside every walk-forward training fold, hard-switch classified at \|z\| > 1.5 (turbulent) / < 1.0 (calm) / between (deadzone).
- **Signal construction**: `momentum_signal` (78-day sign-of-return), `price_zscore_signal` (26-day rolling z-score).
- **Robustness check 1** (alternate regime window): identical pipeline with the regime composite's volatility feature recomputed on a 156-day window instead of 78-day.
- **Robustness check 2** (permutation test): `permutation_test_interaction_coefficient` — shuffles the regime-dummy labels, 1000 permutations, rebuilds the null distribution for b3, two-sided empirical p-value.
- **Verdict rule** (Section 10, exactly as written): a leg passes only if the reliability gate passes for both the primary and robustness-check-1 fits, primary p(b3) < 0.05, robustness check 1 also has p(b3) < 0.05 with the same sign as the primary estimate, and robustness check 2 has p < 0.05. Any single null kills the leg. Strategy-level PASS requires both legs to pass.
- Data: `DataLoader` daily close, 2011-01-01 to 2023-12-31 all three pairs.

## Assumptions
- REGIME_WINDOW (78), ALT_REGIME_WINDOW (156), N_PERMUTATIONS (1000), and both gate thresholds (1e10, 10) are fixed by Section 10 and were not adjusted after viewing output.
- Pooling is across pairs and folds, matching Section 10's "pooled across walk-forward out-of-sample folds" wording
- Centering in `interaction_regression_centered` uses the pooled out-of-sample sample's own mean, computed once after alignment. This is a conditioning step applied to already-realized OOS data for inference purposes; it does not feed into any forward trading decision, so it is not subject to the same look-ahead constraint as the regime classifier's own per-fold refit.
- Forward return horizon is shared across both legs 
- Standard OLS assumptions (homoskedastic, uncorrelated errors) underlie the reported p-values; this audit does not separately run residual diagnostics (e.g. Ljung-Box) on the pooled fits, an unaddressed gap noted below.

## Findings

**Momentum leg** (n = 6563 pooled OOS observations, 3 pairs x 7 folds):

| Test | b3 (interaction) | p-value | Condition number | Reliability gate |
|---|---|---|---|---|
| Primary (78-day regime window) | -0.00356 | <0.0001 | 4.06 | Pass |
| Robustness 1 (156-day alt window) | -0.00170 | 0.00430 | 4.17 | Pass |
| Robustness 2 (1000-permutation dummy shuffle) | -0.00360 (observed) | 0.00100 | — | — |

All VIFs ≈ 1.00-1.01 for both fits — no meaningful collinearity between `momentum_signal`, `turbulent_dummy`, and their interaction once centered. Sign is consistent (negative) across the primary and alternate-window fits.

**Reversion leg** (n = 6563 pooled OOS observations):

| Test | b3 (interaction) | p-value | Condition number | Reliability gate |
|---|---|---|---|---|
| Primary (78-day regime window) | +0.00028 | 0.56308 | 7.35 | Pass |
| Robustness 1 (156-day alt window) | +0.00160 | 0.00060 | 7.01 | Pass |
| Robustness 2 (1000-permutation dummy shuffle) | +0.00030 (observed) | 0.54850 | — | — |

VIFs ≈ 1.00 for both fits. The reliability gate passes, but the primary interaction term is no longer significant (p=0.563), and robustness check 2 agrees (p=0.549) — both main effects remain individually insignificant on their own (`price_z` p=0.249, `calm_dummy` p=0.787), and now so is the interaction itself. Robustness check 1 (156-day window) alone shows significance (p=0.0006), but primary significance is required regardless of what the alternate window shows.

**Verdict: FAIL** Momentum leg passes independently — reliability gate, primary p<0.0001, robustness 1 p=0.0043 with matching sign, robustness 2 p=0.0010 all clear. Reversion leg fails: primary p=0.563 does not clear the 0.05 threshold, and robustness check 2 (permutation, p=0.549) agrees the effect is null.

## Alternative Explanations
- **Reversion leg is cleanly null on the two tests least correlated with each other**. The 156-day alternate window, which is not fully independent of the 78-day primary shows significance. That pattern is more consistent with robustness check 1's isolated result being noise from a non-independent replication than with a genuine effect the primary test simply missed.
- **Shared forward-return horizon across pairs induces cross-sectional correlation not accounted for in standard errors.** The pooled regression treats all 6563 observations as independent, but overlapping 26-day forward windows within a pair mean effective degrees of freedom are lower than n suggests. This would inflate the apparent precision of the p-values reported above (for both legs, not just reversion).
- **156-day alternate window is not fully independent of the 78-day primary window** so robustness check 1 is a weaker independent replication than an unrelated regime definition would be — relevant here since it's the one test where reversion still shows significance.

## Next Steps
- Momentum-only edge: the test itself already passed cleanly and doesn't need redoing. What's unresolved is the surrounding spec — sizing and risk controls were built for a strategy active most of the time, not one that only trades the ~9-16% of days classified turbulent.
- Address the cross-sectional/overlapping-window standard error concern before relying on any of today's p-values at face value — likely via a cluster-robust or block-bootstrap standard error estimator added to `interaction_regression_centered` or a new variant of it.
- Residual diagnostics (Ljung-Box, kurtosis) on both pooled fits' residuals were not run in this audit; worth checking before treating standard errors as trustworthy, independent of the clustering concern above.
- The Section 10 lockbox holdout (2024-2026) stays reserved and unopened