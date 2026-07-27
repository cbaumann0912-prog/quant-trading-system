# Day 34 Research Audit — Markowitz (Min Variance) vs Risk Parity

## 1. Question Investigated
Does minimum-variance or ERC portfolio weighting produce better out-of-sample performance on EUR/USD, GBP/USD, USD/JPY across a 12-year daily return series (2011–2023)?

## 2. Why It Matters
Position sizing is a first-order decision. Markowitz minimizes variance directly; ERC distributes risk evenly across assets. The choice affects realized vol, Sharpe, and how the weighting scheme interacts with a signal layer on top.

## 3. Methodology
- Data: daily log returns, EUR/USD / GBP/USD / USD/JPY, 2011–2023
- Estimation: annual window (~312 obs), weights held fixed for OOS year
- OOS reporting: quarterly intervals within each OOS year
- Markowitz variant: global minimum variance, closed-form, unconstrained
- ERC variant: equal risk contribution via SLSQP, bounds [-1.0, 1.0]
- Metrics: annualized realized vol, annualized Sharpe (rf = 0)
- Total OOS quarters: 48 across 12 estimation years

## 4. Assumptions
- Log returns approximate daily P&L adequately
- Weights estimated on year Y held fixed through year Y+1, no rebalancing
- Risk-free rate = 0.0 (framework standard)
- Annualization factor computed empirically per OOS quarter from that quarter's own index (~313-316 obs/year)
- No transaction costs applied

## 5. Findings

### OOS Scoreboard (48 quarters)
| Metric        | MV Wins    | ERC Wins   |
|---------------|------------|------------|
| Lower Vol     | 20 (41.7%) | 28 (58.3%) |
| Higher Sharpe | 24 (50.0%) | 24 (50.0%) |

### Average OOS Metrics
| Method | Avg Vol | Avg Sharpe | Avg Ann Return |
|--------|---------|------------|----------------|
| MV     | 0.0475  | 0.4080     | 0.0144         |
| ERC    | 0.0467  | 0.3854     | 0.0121         |

### Statistical Significance (48 matched quarters)
| Test                    | Sharpe p-val | Vol p-val |
|-------------------------|--------------|-----------|
| Binomial (vs 50/50)     | 1.000        | 0.193     |
| Paired t-test           | 0.684        | 0.110     |
| Wilcoxon signed-rank    | 0.923        | 0.186     |

Mean Sharpe diff (MV − ERC): +0.023 (std: 0.383)  
Mean Vol diff (MV − ERC): +0.0008 (std: 0.004)

Neither result clears p = 0.05 on any test, and neither comes close. 
The vol result is the strongest of the two on the paired t-test (0.110) 
and still sits at roughly twice the threshold.

## 6. Alternative Explanations
MV's higher average Sharpe may partly reflect its tendency to concentrate in whichever asset happened to perform well in the OOS quarter rather than any structural advantage — concentration occasionally gets lucky. The mean vol difference of 0.0009 with a standard deviation of 0.003 gives a signal-to-noise ratio well below one, meaning estimation error in the sample covariance matrix could account for most of the observed gap on its own.

## 7. Interpretation
Neither method statistically outperforms the other across 48 quarters and 12 years. The Sharpe scoreboard is an exact 24-24 split and the vol scoreboard leans ERC, but three independent tests say neither're not distinguishable from a coin flip. Given that, the choice between MV and ERC comes down to something other than raw performance — ERC's more even weight distribution makes it less sensitive to covariance estimation error and easier to reason about when layering a directional signal on top