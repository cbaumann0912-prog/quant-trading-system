# Week 2 Review — Hypothesis Testing & Significance (Days 8–14)

## Methodology
t-test, Cohen's d, power analysis, bootstrap CI, deflated Sharpe ratio, and multiple testing correction (Bonferroni, Benjamini-Hochberg) implemented from scratch and applied to existing strategy results.

## Findings
- P-value ≠ confirmation of edge — only that the result is unlikely under the null.
- Cohen's d separates statistical from economic significance; pre-cost vs. post-cost d gap measures cost sensitivity.
- FVG_BoS_Reversal (~13 trades/year) is underpowered — insufficient sample to confirm or deny moderate effects.
- DSR: any Sharpe from manual parameter search should be treated as inflated absent OOS confirmation.
- Bonferroni (FWER) vs. BH (FDR) — BH more appropriate when scanning a strategy space with expected real signals. Both live in `src/evaluation/significance.py`.
- Open: full trial count not confirmed as accurately tracked against correction methods ahead of Day 30 shortlist.

## Interpretation
The real output this week is a way to separate "statistically real" from "tradable" — Cohen's d, pre-cost vs. post-cost, does that job directly, and DSR does the same for optimization bias: a Sharpe found by manual parameter search isn't the same claim as a Sharpe that survives out-of-sample. Bonferroni vs. BH is the right frame for the strategy shortlist ahead — FDR control makes sense when I actually expect some real signals in the space I'm scanning, not just noise.