# Sharpe Ratio Audit — Day 2

## Files Inspected

| # | File | Lines Inspected | Primary Concern |
|---|------|----------------|-----------------|
| 1 | portfolio.py | 581-592 | Annualization Factor is overstated on two accounts |

## The Formula
Returns is the normalized percentage that a sigular trade increases the equity of the account based off of the equity of the account when the trade was entered. Ann_factor is a scaling factor equal to the square root of the trades per year. The final calculation of the sharpe ratio is tring to find the avg return per unt of risk over a single year. 

## What's Valid
The denomenator of the equation which calculates returns is valid because sharpe ratio eventually need to compute to returns per unit of risk and since this strategy uses a set percentage of the current equity of the portfloio as the risk per trade this fits well. Since the pandas default std code uses N-1 and this is a sample this makes since to use. 

## What's Broken
The annualization factor is overstated in two ways. The first being that there actually is some covariance between the pairs as the all relate to the US dollar some positlivley proportional and others inversly proportional. Also it is not fair to assume that all trades are independent of eachother because of this covariance so the square root of n over estimates too

## Directional Impact
The reported Sharpe is over estimated than the true Sharpe because because the annualiztion factor has been overestimated

## Open Problems
- I need to find covariance between each combination of the pairs using the compute_covariance_matrix function that I wrote
- I would also need to try and figure out how to find the effective sample size 

## Findings
The unit of risk in the Sharpe ratio is the standard deviation of per-trade returns, which measures the risk per trade because as the std of the returns increases the riskiness of the trade does too. 

## Questions to Resolve
- How would this calculation change if the risk per trade was determined by another factor such as when using a statistical arbitrage or moving average strategy
- Would it be smart to adjust strategy in a way which decreases its ability to deviate far from the mean return just so the sharpe would increase