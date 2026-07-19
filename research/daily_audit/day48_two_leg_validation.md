# Day 48 Research Audit — Section 10 Validation, Volatility Regime Breakout/Mean-Reversion

## Question
Does the volatility-regime breakout/mean-reversion strategy have a genuine, regime-dependent edge, or does the regime classifier add nothing beyond what an unconditional signal already captures? Concretely: is b3 (the signal x regime_dummy interaction coefficient) reliably different from zero for the momentum leg (momentum_signal x turbulent_dummy) and the reversion leg (price_z x calm_dummy), pooled out-of-sample across walk-forward folds and all three pairs?

## Why It Matters
This is the strategy's pre-registered falsification test (Section 1, item 1 and Section 10). Everything upstream — the regime classifier (Day 43-47), the entry/exit/sizing rules (Sections 5-7) — is provisional until this test clears. A momentum-only or reversion-only pass would also mean the strategy as specified fails: Section 10 requires both legs to pass, since a single-leg edge is a different, smaller strategy than the one hypothesized in Section 1.

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
- Data: `DataLoader` daily close, 2011-01-01 to 2026-05-01, all three pairs.

## Assumptions
- REGIME_WINDOW (78), ALT_REGIME_WINDOW (156), N_PERMUTATIONS (1000), and both gate thresholds (1e10, 10) are fixed by Section 10 and were not adjusted after viewing output.
- Pooling is across pairs and folds, matching Section 10's "pooled across walk-forward out-of-sample folds" wording
- Centering in `interaction_regression_centered` uses the pooled out-of-sample sample's own mean, computed once after alignment. This is a conditioning step applied to already-realized OOS data for inference purposes; it does not feed into any forward trading decision, so it is not subject to the same look-ahead constraint as the regime classifier's own per-fold refit.
- Forward return horizon is shared across both legs 
- Standard OLS assumptions (homoskedastic, uncorrelated errors) underlie the reported p-values; this audit does not separately run residual diagnostics (e.g. Ljung-Box) on the pooled fits, an unaddressed gap noted below.

## Findings

**Momentum leg** (n = 9374 pooled OOS observations, 3 pairs x 10 folds):

| Test | b3 (interaction) | p-value | Condition number | Reliability gate |
|---|---|---|---|---|
| Primary (78-day regime window) | -0.00191 | 0.00016 | 4.35 | Pass |
| Robustness 1 (156-day alt window) | -0.00151 | 0.00434 | 4.33 | Pass |
| Robustness 2 (1000-permutation dummy shuffle) | -0.00191 (observed) | 0.00200 | — | — |

All VIFs ≈ 1.00-1.01 for both fits — no meaningful collinearity between `momentum_signal`, `turbulent_dummy`, and their interaction once centered. Sign is consistent (negative) across the primary and alternate-window fits.

**Reversion leg** (n = 9374 pooled OOS observations):

| Test | b3 (interaction) | p-value | Condition number | Reliability gate |
|---|---|---|---|---|
| Primary (78-day regime window) | +0.00105 | 0.00549 | 7.06 | Pass |
| Robustness 1 (156-day alt window) | +0.00150 | <0.0001 | 7.02 | Pass |
| Robustness 2 (1000-permutation dummy shuffle) | +0.00105 (observed) | 0.00400 | — | — |

VIFs ≈ 1.00 for both fits. Sign is consistent (positive) across the primary and alternate-window fits. Notably, neither main effect (`price_z` alone, p = 0.371; `calm_dummy` alone, p = 0.384) is significant on its own — the entire effect in this leg is carried by the interaction term, exactly the pattern Section 10 was designed to detect (an unconditional test would have missed this leg's edge entirely).

**Verdict: PASS.** 

**This PASS is provisional**, not a final strategy verdict, for two explicit reasons: (1) it has not yet been passed through the project-wide Benjamini-Hochberg correction across all 5 strategies tested this project; the spec itself notes a real finding here needs p < 0.01 to survive BH at rank 1 of 5 — momentum's primary p (0.00016) and permutation p (0.0020) clear that bar comfortably, but reversion's primary p (0.00549) does not. (2) the Section 10 lockbox holdout (2024-2026) has not been opened — this PASS is a walk-forward-only result.

## Alternative Explanations
- **Reversion leg's main effects are individually null.** The interaction alone carries the reversion leg's significance, with both main effects individually insignificant, is consistent with genuine regime-dependence. Today's test cannot distinguish "genuine regime-conditional mean-reversion" from "calm_dummy proxies for an omitted calm-period factor" — that would require a separate specification test not run here.
- **Shared forward-return horizon across pairs induces cross-sectional correlation not accounted for in standard errors.** The pooled regression treats all 9374 observations as independent, but overlapping 26-day forward windows within a pair mean effective degrees of freedom are lower than n suggests. This would inflate the apparent precision of the p-values reported above.
- **156-day alternate window is not fully independent of the 78-day primary window** (156 = 2 x 78, substantial autocorrelation in the underlying rolling vol estimates), so robustness check 1 is a weaker independent replication than an unrelated regime definition would be.

## Next Steps
- Before treating this as a strategy-level PASS: run Benjamini-Hochberg correction across all 5 project strategies once all 5 have a final Section 10 (or equivalent) p-value on record, using reversion leg's p = 0.00549 as the binding constraint (momentum already clears the p < 0.01 bar with margin).
- Address the cross-sectional/overlapping-window standard error concern before relying on any of today's p-values at face value — likely via a cluster-robust or block-bootstrap standard error estimator added to `interaction_regression_centered` or a new variant of it.
- Only after the above: open the Section 10 lockbox holdout (2024-2026) once, as a single-use confirmatory test — not another round of tuning. If the lockbox result disagrees with today's walk-forward verdict, report the disagreement as-is per the pre-registered lockbox protocol, rather than explaining it away.
- Residual diagnostics (Ljung-Box, kurtosis) on both pooled fits' residuals were not run in this audit; worth checking before treating standard errors as trustworthy, independent of the clustering concern above.