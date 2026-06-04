# Day 12 Research Audit — Multiple Testing Correction: Strategy Return Test Per Pair

## Objective

Apply Bonferroni and Benjamini-Hochberg corrections across three hypothesis tests. Each test targets the null of zero mean daily return using `t_test_mean` from `src/stats/hypothesis_tests.py`, applied to per-pair daily returns from the archived FVG_BoS_Reversal strategy.

This is a methodology demonstration using sandbox data from a pre-curriculum, largely AI-assisted strategy. Any null rejections observed here should not be interpreted as evidence of real edge. The correction framework will be reapplied to properly built and validated strategy returns post-Phase 2.

## Hypothesis Setup

H0: mean daily return = 0 (no edge)
H1: mean daily return ≠ 0 (edge exists)

alpha = 0.05, m = 3
Bonferroni threshold: alpha/m = 0.01667

## Results

| Pair | p-value | Bonferroni | BH |
|------|---------|------------|----|
| EUR/USD | 0.00001 | True | True |
| GBP/USD | 0.00020 | True | True |
| USD/JPY | 0.00012 | True | True |

## Notes

- All three pairs reject the null under both Bonferroni and BH. This result comes from sandbox strategy data and carries no inferential weight about real edge. It confirms the correction machinery is working correctly.
- Bonferroni and BH agree across all three pairs. Disagreement only arises when some p-values are near but not below the threshold — not the case here with all three well below alpha/m.
- EUR/USD and GBP/USD share USD exposure and are positively correlated. BH assumes independence or positive correlation — the c_m = 1 assumption is an approximation here, noted for the paper's limitations section.
- This analysis will be rerun against properly backtested, walk-forward validated strategy returns in Phase 2. Results at that stage will carry actual inferential weight.