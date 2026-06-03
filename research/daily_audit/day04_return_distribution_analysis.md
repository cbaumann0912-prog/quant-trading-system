# Day 04 Return Distribution Analysis

## Objective
The goal of this analysis was to determine whether modeling daily forex returns as approximately normal is justified, or whether the Student-t distribution is a better fit.

## Data
1-minute OHLCV data spanning 01-01-2011 to 03-31-2026 was used for all three pairs. Data was resampled to daily frequency using the last close price of each day. Log returns were computed as ln(P_t / P_{t-1}).

## Results

| Pair   | Excess Kurtosis | Student-t df |
|--------|----------------|--------------|
| EURUSD | 3.04           | 5.59         |
| USDJPY | 4.99           | 3.55         |
| GBPUSD | 28.35          | 5.45         |
| Normal | 0              | ∞            |

## Interpretation
Positive excess kurtosis across all pairs indicates that extreme moves occur more often than a normal distribution would predict. GBP/USD's extreme kurtosis of 28.35 is likely driven by historic outlier events such as the 2016 Brexit flash crash and the 2022 Liz Truss mini-budget crash. The t-fit degrees of freedom of 5.45 is more robust to these outliers and better represents the bulk of the distribution.

## Conclusion
With degrees of freedom ranging from 3.55 to 5.59 — all far below 30 — there is no statistical justification for modeling daily forex returns as approximately normal. The Student-t distribution is a significantly better fit across all three pairs.