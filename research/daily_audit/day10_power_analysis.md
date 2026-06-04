# Day 10 — Power Analysis: All Strategies

**Date:** 2026-06-01  
**Alpha:** 0.05  
**Target Power:** 0.80  

## Methodology
For each strategy, effect size d = mean / std of trade returns. Required n computed using n = ((z_half_alpha + z_beta) / d)^2. Achieved power computed using power = Phi(d * sqrt(n) - z_half_alpha). Verdict: sufficient if achieved power >= 0.80.

## Results

| Strategy | Effect Size (d) | Required n | Actual n | Achieved Power | Verdict |
|----------|----------------|------------|----------|----------------|---------|
|BOS_FVG_Reversal|0.33|73|216|0.998|sufficient|

## Conclusions
BOS_FVG_Reversal is well-powered at 216 trades. This is nearly 3x the required sample size of 73. Achieved power of 0.998 means there is a 99.8% probability of detecting the observed effect if it is real. Target power is set at 0.80 (Cohen's convention). Raising it to 0.90 would increase required n to ~97 — still comfortably met — but is not justified at this stage. Backtest length is not a limiting factor for this strategy, the binding constraints are multiple testing correction and overfitting risk, not statistical power. 

## Notes
- Effect size of 0.33 is below medium by Cohen's conventions (small=0.2, medium=0.5)
- Power analysis assumes effect size is stable across the sample — regime shifts could invalidate this assumption
- As new strategies are added, append rows to the results table and rerun the script