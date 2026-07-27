# Day 34 Research Audit — Markowitz (Min Variance) vs Risk Parity

## 1. Question Investigated
Does minimum-variance or ERC portfolio weighting produce better out-of-sample performance on EUR/USD, GBP/USD, USD/JPY across a 14-year daily return series (2011–2026)?

## 2. Why It Matters
Position sizing is a first-order decision. Markowitz minimizes variance directly; ERC distributes risk evenly across assets. The choice affects realized vol, Sharpe, and how the weighting scheme interacts with a signal layer on top.

## 3. Methodology
- Data: daily log returns, EUR/USD / GBP/USD / USD/JPY, 2011–2026
- Estimation: annual window (~312 obs), weights held fixed for OOS year
- OOS reporting: quarterly intervals within each OOS year
- Markowitz variant: global minimum variance, closed-form, unconstrained
- ERC variant: equal risk contribution via SLSQP, bounds [-1.0, 1.0]
- Metrics: annualized realized vol, annualized Sharpe (rf = 0)
- Total OOS quarters: 57 across 14 estimation years

## 4. Assumptions
- Log returns approximate daily P&L adequately
- Weights estimated on year Y held fixed through year Y+1, no rebalancing
- Risk-free rate = 0.0 (framework standard)
- Annualization factor computed empirically per OOS quarter from that quarter's own index (~313-316 obs/year)
- No transaction costs applied

## 5. Findings

### OOS Scoreboard (57 quarters)
| Metric        | MV Wins    | ERC Wins   |
|---------------|------------|------------|
| Lower Vol     | 24 (42.1%) | 33 (57.9%) |
| Higher Sharpe | 31 (54.4%) | 26 (45.6%) |

### Average OOS Metrics
| Method | Avg Vol | Avg Sharpe | Avg Ann Return |
|--------|---------|------------|----------------|
| MV     | 0.0462  | 0.5197     | 0.0185         |
| ERC    | 0.0453  | 0.4936     | 0.0159         |

### Statistical Significance (57 matched quarters)
| Test                    | Sharpe p-val | Vol p-val |
|-------------------------|--------------|-----------|
| Binomial (vs 50/50)     | 0.597        | 0.111     |
| Paired t-test           | 0.581        | 0.056     |
| Wilcoxon signed-rank    | 0.796        | 0.085     |

Mean Sharpe diff (MV − ERC): +0.026 (std: 0.355)  
Mean Vol diff (MV − ERC): +0.0009 (std: 0.003)

Neither result clears p = 0.05 on any test. The vol result comes 
closest on the paired t-test (0.056) but does not cross the threshold.

## 6. Alternative Explanations
MV's higher average Sharpe may partly reflect its tendency to concentrate in whichever asset happened to perform well in the OOS quarter rather than any structural advantage — concentration occasionally gets lucky. The mean vol difference of 0.0009 with a standard deviation of 0.003 gives a signal-to-noise ratio well below one, meaning estimation error in the sample covariance matrix could account for most of the observed gap on its own.

## 7. Interpretation
Neither method statistically outperforms the other across 57 quarters and 14 years. The scoreboard percentages look directional but three independent tests say they're not distinguishable from a coin flip. Given that, the choice between MV and ERC comes down to something other than raw performance — ERC's more even weight distribution makes it less sensitive to covariance estimation error and easier to reason about when layering a directional signal on top