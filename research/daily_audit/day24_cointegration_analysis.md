# Day 24 Audit — Engle-Granger Cointegration: All Pair Combinations

## 1. Question
Do any of the three pair combinations share a cointegrating relationship at the 5% level? If so, is the estimated hedge ratio stable across rolling windows?

## 2. Why It Matters
Day 20 confirmed all three pairs are I(1) on price levels. The natural follow-up is whether any linear combination of them is I(0). That is the prerequisite for a stat arb strategy — not correlation, not shared macro drivers, but a spread that actually mean-reverts. The strongest of the three p-values on EUR/USD ~ GBP/USD appeared in the levels-based OLS from Day 15/16, though it was not close to rejection. Today tests whether that holds formally across all combinations, and whether the hedge ratio is stable enough to trade.

## 3. Methodology
- Daily close prices from 1-minute OHLCV, resampled to last observation per day
- Price levels, not log returns — cointegration is a levels concept; differencing destroys the signal
- First-named pair as dependent variable; ordering is arbitrary and a known Engle-Granger limitation
-  5% on ADF p-values applied to first-stage OLS residuals
- 252-day window, one-day step; OLS only at each step, no ADF, to keep runtime under a minute

## 4. Predictions
| Pair | Expected Result | Reasoning |
|------|-----------------|-----------|
| EUR/USD ~ GBP/USD | Borderline, p around 0.10 | Day 15/16 levels OLS gave ADF p = 0.1003; expect similar here on the longer sample |
| EUR/USD ~ USD/JPY | No cointegration | EUR and JPY driven by different regimes; no shared stochastic trend |
| GBP/USD ~ USD/JPY | No cointegration | Same reasoning |

## 5. Results
| Pair | Hedge Ratio (β) | ADF Stat | ADF p-value | Cointegrated? |
|---|---|---|---|---|
| EUR/USD ~ GBP/USD | 0.6106 | -2.5403 | 0.1060 | No |
| EUR/USD ~ USD/JPY | -0.0051 | -2.3625 | 0.1526 | No |
| GBP/USD ~ USD/JPY | -0.0058 | -1.7542 | 0.4034 | No |

Rolling hedge ratio summary (312-day window, 3,743 observations):

| | EUR/USD ~ GBP/USD | EUR/USD ~ USD/JPY | GBP/USD ~ USD/JPY |
|---|---|---|---|
| mean | 0.6257 | -0.0042 | -0.0003 |
| std | 0.4194 | 0.0062 | 0.0083 |
| min | -0.0715 | -0.0234 | -0.0250 |
| max | 1.5440 | 0.0126 | 0.0169 |

## 6. Findings
EUR/USD ~ GBP/USD lands at p = 0.1060, consistent with the Day 15/16 result. The difference in β (0.6106 vs 0.7125 from Day 15/16) is expected — those were different regressions on different inputs. Day 15/16 regressed log returns on log returns (a return hedge ratio). Today regresses price levels on price levels (a cointegration hedge ratio).

The rolling hedge ratio kills the pairs trade argument more decisively than the p-value does. A standard deviation of 0.42 around a mean of 0.63 means the hedge ratio is not a stable quantity. You can't run a stat arb strategy on a hedge ratio that has swung from -0.07 to +1.54 across the sample. The full-sample estimate of 0.61 is an average of regimes that don't look anything like each other.

EUR/USD ~ USD/JPY and GBP/USD ~ USD/JPY are not close. ADF p-values of 0.15 and 0.40, hedge ratios near zero (-0.0051 and -0.0058), and spread plots that trend persistently in one direction for years. No case to make here.

## 7. Alternative Explanations
Two concerns about the EUR/USD ~ GBP/USD result. First, Engle-Granger critical values are stricter than standard ADF tables — OLS picks the hedge ratio that minimizes residual variance, which gives the spread an artificial head start toward looking stationary. The true p-value is probably higher than 0.1060. Second, the spread plot shows a multi-year negative excursion from 2014 to 2017, followed by a partial recovery. A full-sample ADF test can't distinguish a slow structural break that happened to partially reverse from genuine mean reversion. The visual evidence points to the former.

## 8. Implications for Strategy Development
The stat arb strategy candidate requires a stable hedge ratio to trade. EUR/USD ~ GBP/USD doesn't clear that bar. If anything, the rolling analysis here is the more useful diagnostic: even if a cointegrating vector exists formally, a strategy built on it would have spent years trading the wrong hedge ratio.

EUR/USD ~ USD/JPY and GBP/USD ~ USD/JPY don't warrant further analysis. They're not cointegrated and the economics don't support expecting them to be.

## 9. Next Steps
- Johansen test on the full 3-variable system — tests for cointegration symmetrically and can detect multiple cointegrating vectors
- OU half-life on EUR/USD ~ GBP/USD if Johansen finds a cointegrating vector worth characterizing