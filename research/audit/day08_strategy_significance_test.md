# Day 08 — Strategy Significance Test

## 1. Strategy Overview

**Strategy name:** FVG_BoS_Reversal
**Pair(s) tested:** EUR/USD, USD/JPY, GBP/USD 
**Timeframe:** Daily 
**Backtest period:** 2011-01-01 to 2026-03-31
**Number of observations (n):** 206

## 2. Return Series Summary Statistics

| Statistic | Value |
|---|---|
| Mean return ($\bar{X}$) |0.004023|
| Std deviation ($S$) |0.009529|
| Standard error ($\hat{se}$) |0.000664|
| Min |-0.018149|
| Max |0.028424|

## 3. Hypothesis Test

**Null hypothesis:** $H_0: \mu = 0$  
**Alternative hypothesis:** $H_1: \mu \neq 0$  
**Confidence level:** 95% ($\alpha = 0.05$)

### Test Statistic

$$W = \frac{\bar{X} - 0}{\hat{se}} = \frac{}{} = 6.0589$$

### Critical Value

$$z_{\alpha/2} = z_{0.025} = 1.96$$

### p-value

$$p = 0.0000$$

### Confidence Interval

$$CI = (\bar{X} - 1.96 \cdot \hat{se},\ \bar{X} + 1.96 \cdot \hat{se}) = (0.002714,0.005332)$$

## 4. Results
The FVG_BoS_Reversal strategy produces a mean per-trade log return of 0.4023% across 206 trades over the 2011–2026 backtest period. The one-sample t-test yields a test statistic of  W = 6.06 with a p-value of approximately 0.0000, leading us to reject the null hypothesis that the mean return is zero at the 95% confidence level. The 95% confidence interval of (0.2714%, 0.5332%) lies entirely above zero, confirming both statistical and scientific significance. Transaction costs on EUR/USD, GBP/USD, and USD/JPY are well within the lower bound of the interval, supporting deployment to live demo trading.

## 5. Notes & Next Steps
- **Strategy provenance** — the current implementation was largely AI-assisted and predates a full understanding of the underlying signal logic, backtesting engine, and statistical methodology. While the t-test results are encouraging, the strategy will be rebuilt from scratch as the 90-day plan progresses to ensure every layer of the stack is fully understood and defensible. The significance testing methodology developed today and all the methods developed over the past days will be reapplied to the rebuilt strategy on a proper walk-forward basis.