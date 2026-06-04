# Day 12 Research Audit — Multiple Testing Correction: Zero-Drift Test Per Pair

## Objective

Apply Bonferroni and Benjamini-Hochberg corrections across three hypothesis tests. Each test targets the null of zero mean log return (no drift) using `t_test_mean` from `src/stats/hypothesis_tests.py`.

This is a methodology demonstration. Per-pair strategy-level p-values require trade-level return series from signal execution, which is blocked until Phase 2. The correction framework will be reapplied to strategy returns post-signal-build.

## Hypothesis Setup

H0: mean log return = 0 (no drift)
H1: mean log return ≠ 0 (drift exists)

alpha = 0.05, m = 3
Bonferroni threshold: alpha/m = 0.01667

## Results

| Pair | p-value | Bonferroni | BH |
|------|---------|------------|----|
| EUR/USD | 0.67620 | False | False |
| GBP/USD | 0.66042 | False | False |
| USD/JPY | 0.07412 | False | False |

## Notes

- Zero-drift t-test on raw log returns is not a test of strategy edge. It is a test of whether each pair  has statistically detectable mean drift over the full 15-year sample. Failure to reject here is expected.
- Bonferroni and BH agree here. Disagreement only arises when
some p-values are close to but not below the threshold. With all three p-values well above alpha, both methods reach the same conclusion.
- USD/JPY has the lowest p-value at 0.074, approaching but not clearing the uncorrected threshold. Worth noting for Phase 2 — if per-pair strategy returns show a similar pattern, USD/JPY warrants closer examination.
- EUR/USD and GBP/USD share USD exposure and are positively correlated. BH assumes independence or positive correlation — the c_m = 1 assumption is an approximation here, noted for the paper's limitations section.