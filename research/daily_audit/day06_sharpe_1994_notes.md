# Day 06 — Sharpe (1994) "The Sharpe Ratio" — Reading Notes

**Date:** 2026-05-27  
**Source:** Sharpe, W.F. (1994). "The Sharpe Ratio." *Journal of Portfolio Management* 

## 1. Exact Definition

> *The ratio of expected added return per unit of added risk*
 
**In my own words:** Sharpe ratio normalizes excess return by its volatility so that strategies with different risk levels can be compared on equal footing 

## 2. Assumptions
**Assumption 1:** Ex-ante Sharpe ratios can be estimated from ex-post data, implying some return predictability.

**Assumption 2:** Mean and SD are sufficient statistics for evaluating a strategy

**Assumption 3:** The differential return over T periods is
measured by summing the one-period differential returns and that the latter have zero correlation

**Assumption 4:** All the strategies in a set have similar correlations with
the other holdings.

## 3. Stated Limitations
**Limitation 1:** It doesn't account for serial correlation in returns 

**Limitation 2:** It's designed for zero-investment strategies only. 

**Limitation 3:** It penalizes upside volatility equally with downside volatility

## 4. Your Take
It assumes zero serial correlation in returns, which breaks immediately for any momentum strategy. If wins and losses cluster, you're lying to yourself with annualized Sharpe. It's also designed only for zero-investment strategies, so unless you're explicitly benchmarking against a cash rate, the number is measuring something slightly different than intended. BY penalizing upside volatility the same as downside, meaning a strategy that occasionally rips higher looks riskier than one that bleeds steadily — that's backwards in my opinion. None of these are dealbreakers, but they're reasons to never look at Sharpe alone.

## 5. Other Notes
- Sharpe updated the original "reward-to-variability ratio" to use a differential return rather than excess return over cash. 
- Sharpe discusses scaling by T^.5
- A strategy can have a lower Sharpe but still be preferable if it has low correlation with existing holdings