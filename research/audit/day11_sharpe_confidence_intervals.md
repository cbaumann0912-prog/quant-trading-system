# Day 11 Research — Sharpe Ratio Bootstrap Confidence Interval

## Objective
Todays goal was to run and interpret the data from the percentile cofidence interval funtion I wrote today on the strategy that I have backtest return data from. The key things that i was looking for were that the CI didnt include 0 and take note of the width of the interval.  

## Methodology
By taking daily portfolio equity balance csv file from my previous backtesting results and converting it into log daily returns without 0 days I was able to run the bootstrap_confidence_interval function along with exact annual factor calculations that do not assume normality over the 252 trading days in a year. 

### What is Bootstrap CI?
It builds a confidence interval empirically rather than assuming a distribution. It works by drawing random samples with replacement from the observed data, computing the statistic of interest on each resample. Then it uses the resulting distribution of that statistic to define the interval bounds.

### Why Bootstrap Over Parametric CI?
Parametric confidence intervals assume the underlying data follows a normal distribution. Forex returns exhibit fat tails and excess kurtosis, making normality assumptions inappropriate.

### Parameters
- Iterations: 10000
- Confidence Level: 0.95
- Annualization Factor: 13.63
- Risk-Free Rate: 0 - used today for simlicity's sake but easily changed to fixed or time-varying version when the backtest is to be run seriously in the future

## Results

| Metric | Value |
|--------|-------|
| Point Estimate Sharpe |1.5584|
| CI Lower Bound (2.5%) |0.9849|
| CI Upper Bound (97.5%) |2.2267|
| CI Width |1.2434|


## Interpretation
If we repeated this bootstrap procedure many times, 95% of the constructed intervals would contain the true Sharpe ratio. We are 95% confident the true Sharpe lies between 0.9849 and 2.2267

### What the Width Tells Us
This width is quite large which tells me there is significant uncertainty around the estimated value of the sharpe ratio. This is most likely due to the fact that the strategy is such low frequency. 

### What the Lower B Tells Us
Since the lower bound is > 0 the strategy is unlikely to just be a fluke. Even the conservative estimate of the sharpe ration of 0.9849 is a respectable value. 

## Limitations 
- Overfitting: Parameters were optimized on the full dataset before splitting 
  into evaluation windows. The return series is likely in-sample, meaning the Sharpe and CI are measuring fit to historical data the strategy was tuned to. Therefore the 1.5584 point estimate is likely inflated.
- Normality assumptions exist elsewhere in the backtesting framework that feed 
  into this return series, which are inconsistent with real return distributions.
- These two issues are known invalidators of today's results. This analysis 
  should be considered preliminary pending a properly structured walk-forward 
  backtest and assumption-corrected framework.