# Day 14 — Week 2 Review: Hypothesis Testing & Significance

## Question Investigated

Week 2 covered the statistical machinery required to evaluate whether observed strategy returns represent genuine signal or noise. Specifically: what do p-values actually measure, how do effect size and statistical power bound what a test can conclude, and how does testing multiple strategies on the same data corrupt those conclusions.

## Why It Matters

A backtest number is not a finding. Before any metric from a strategy can be treated as evidence of an edge, the statistical properties of that metric need to be understood. The work this week builds the foundation for evaluating the FVG_BoS_Reversal results and every subsequent candidate without confusing statistical machinery for economic reality.

## Methodology

Each concept was implemented from scratch in Python, validated against hand-calculated examples, and applied to the existing strategy results. The sequence moved from the t-test and p-value interpretation through effect size, power analysis, bootstrap confidence intervals, the deflated Sharpe ratio, and finally multiple testing correction via Bonferroni and Benjamini-Hochberg.

## Assumptions

The t-test assumes returns are approximately normally distributed and independent across observations. For an intraday strategy with correlated trade returns, both assumptions are questionable. Cohen's d assumes the same. Power calculations assume the effect size used as input is the true population effect size, which it isn't — the estimated d from a backtest carries substantial uncertainty at low trade counts. The deflated Sharpe calculation assumes the number of trials is known and recorded honestly.

## Findings

**P-value interpretation.** The p-value is the probability of observing a result as extreme as the one measured, given the null is true. It does not confirm the alternative hypothesis. Rejecting the null at p = 0.03 means the observed return is unlikely under the assumption of no edge — it says nothing about whether the edge is large enough to trade.

**Statistical vs. economic significance.** Cohen's d fills the gap the p-value leaves. A strategy can produce a p-value well below alpha while its d sits below 0.2, meaning the effect is real but too small to survive transaction costs. Pre-cost d measures whether the signal exists. Post-cost d measures whether it survives implementation. The gap between the two measures implementation cost sensitivity. A large gap is grounds to retire the strategy regardless of p-value.

**Power and sample size.** At the trade frequencies observed in FVG_BoS_Reversal (~13 trades per year), statistical power per fold is insufficient to detect even moderate effects reliably. The strategy is undersampled. This does not mean the edge doesn't exist — it means the current sample cannot confirm or deny it with confidence.

**Deflated Sharpe Ratio.** Manually searching parameter combinations until the Sharpe peaks is a form of selection bias. Each additional optimization trial increases the probability that the reported Sharpe is a false positive. DSR corrects for this by adjusting the benchmark Sharpe upward as the number of trials grows. The implication for prior work on the FVG strategy is direct: any Sharpe reported after extended manual optimization should be treated as inflated until an out-of-sample run confirms it.

**Multiple testing correction.** Bonferroni controls FWER — the probability of any false rejection across all tests — by dividing alpha by the number of tests. As the number of strategy variants grows, the threshold collapses toward zero and genuine signals get rejected alongside false ones. Benjamini-Hochberg controls FDR instead — the expected proportion of false discoveries among all rejections — which is the appropriate constraint when scanning a strategy space where some real signals are expected to exist. Both are now implemented in `src/evaluation/significance.py`.

## Alternative Explanations

The power analysis conclusion that FVG_BoS_Reversal is undersampled assumes the effect size used in the power calculation is accurate. If the true per-trade edge is larger than estimated, the power at low trade counts may be sufficient. This cannot be resolved without more data or a longer backtest period, neither of which is currently available. The DSR findings depend on an honest count of trials — if the number of optimization runs conducted on the original strategy was not logged, the DSR calculation rests on a guess.

## Open Questions

Bonferroni and BH operate on a list of p-values. That list needs to include every strategy and variant tested, not only the ones that looked promising. The trial budget going into Phase 2 has already been partially spent. Whether the full trial count has been accurately tracked against the correction methods built this week remains to be confirmed before the Day 30 strategy shortlist is finalized.