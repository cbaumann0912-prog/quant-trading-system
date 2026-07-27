# Day 04 Return Distribution Analysis

## Objective
The goal of this analysis was to determine whether modeling daily forex returns as approximately normal is justified, or whether the Student-t distribution is a better fit.

## Data
1-minute OHLCV data spanning 01-01-2011 to 12-31-2023 was used for all three pairs. Data was resampled to daily frequency using the last close price of each day. Log returns were computed as ln(P_t / P_{t-1}). n = 3,376 daily returns for EUR/USD and USD/JPY, 3,375 for GBP/USD.

## Results

| Pair   | Excess Kurtosis | Student-t df |
|--------|----------------|--------------|
| EURUSD | 2.65           | 5.97         |
| USDJPY | 5.67           | 1.99         |
| GBPUSD | 28.69          | 5.33         |
| Normal | 0              | ∞            |

## Interpretation
Positive excess kurtosis across all pairs indicates that extreme moves occur more often than a normal distribution would predict. GBP/USD's extreme kurtosis of 28.69 is likely driven by historic outlier events such as the 2016 Brexit flash crash and the 2022 Liz Truss mini-budget crash. The t-fit degrees of freedom of 5.33 is more robust to these outliers and better represents the bulk of the distribution.

USD/JPY's fitted df of 1.99 sits at the edge of where the Student-t has finite variance, and should not be read as a precise tail index — maximum-likelihood df estimates are unstable at low df, and a value near 2 mostly says the tails are heavy enough that the fit is pushing against the boundary of the parameter space rather than settling on an interior optimum.

## Conclusion
With degrees of freedom ranging from 1.99 to 5.97 — all far below 30 — there is no statistical justification for modeling daily forex returns as approximately normal. The Student-t distribution is a significantly better fit across all three pairs.