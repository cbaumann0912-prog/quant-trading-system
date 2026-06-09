# Day 17 Research — Ridge on EURUSD Prediction

## Setup
- **Target:** Next-day EURUSD return
- **Features:** 5 lagged daily returns (lag1 through lag5)
- **Model:** Ridge regression at λ = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
- **Baseline:** OLS (λ = 0)

## Coefficient Comparison Table
| lambda | lag1 | lag2 | lag3 | lag4 | lag5 |
|--------|------|------|------|------|------|
| OLS    | -0.003099 | -0.004659 | -0.006866 | -0.012214 | -0.002786 |
| 0.001  | -0.003077 | -0.004623 | -0.006810 | -0.012106 | -0.002770 |
| 0.01   | -0.002817 | -0.004243 | -0.006257 | -0.011124 | -0.002538 |
| 0.1    | -0.001525 | -0.002329 | -0.003452 | -0.006140 | -0.001382 |
| 1      | -0.000273 | -0.000423 | -0.000630 | -0.001121 | -0.000249 |
| 10     | -0.000030 | -0.000046 | -0.000069 | -0.000122 | -0.000027 |
| 100    | -0.000003 | -0.000005 | -0.000007 | -0.000012 | -0.000003 |
| 1000   | -0.0000003 | -0.0000005 | -0.0000007 | -0.000001 | -0.0000003 |

## Shrinkage Analysis
**Shrinkage becomes significant at λ = 1**

**Which lag shrinks first and why:**
All lags shrink proportionally — no single lag disappears before the others. This is expected behavior for Ridge, which shrinks all coefficients toward zero simultaneously rather than eliminating any. If one lag had shrunk dramatically faster than the others it would suggest that predictor was carrying very little signal relative to the rest. Lag4 retains the largest absolute magnitude at every λ level.

## OLS vs Ridge
All five OLS coefficients are negative and small in magnitude. No lag has a meaningfully large coefficient. The largest is lag4 at -0.012, which represents a predicted next-day return of -1.2% for every 100% lagged return. This is an extremely weak signal. As λ increases from 0.001 to 1000 the coefficients shrink toward zero. Roughly 90% shrinkage occurring between λ = 0.1 and λ = 1. By λ = 10 all coefficients are negligible.

## Interpretation
The negative signs are consistent and suggest a weak mean-reversion tendency in EURUSD at the daily frequency. Past positive returns weakly predict future negative returns. However the magnitudes are too small to survive transaction costs. A strategy built purely on these lag signals would be unprofitable in practice. The signal exists in the data but is not exploitable.

Ljung-Box test on the raw return series finds no statistically significant autocorrelationat. The negative coefficients produced by OLS and Ridge are not genuine autocorrelation structure they are noise. The slight negative signs seen in the coefficient table are a statistical artifact of fitting on a finite sample, not a real signal in the data. This directly invalidates any mean-reversion interpretation of the Ridge output. EURUSD daily returns are consistent with being serially uncorrelated, which aligns with weak-form market efficiency at the daily frequency.

| Lags | LB Statistic | p-value |
|------|-------------|---------|
| 5    | 1.071       | 0.957   |
| 10   | 4.616       | 0.915   |
| 20   | 21.127      | 0.390   |

If the underlying return series has no autocorrelation structure, then Ridge and Lasso applied to lagged returns are fitting noise by construction. Regularization prevents the model from overfitting to spurious patterns, which is exactly what happened here. The OLS coefficients looked meaningful in isolation but Ljung-Box confirmed they were artifacts. The correct response is not to adjust λ but to test richer feature sets where genuine predictive structure might exist.

## Conclusions
- Ridge regression on 5-day lagged EURUSD returns produces uniformly negative coefficients, indicating apparent mean-reversion that Ljung-Box confirms is noise.
- Shrinkage becomes significant at λ = 1, reflecting how little variance the lag features genuinely explain.
- Lag4 is the most persistent predictor across all regularization levels but is not economically meaningful.
- Ridge does not perform variable selection — all five lags remain in the model at every λ. Lasso is the natural next step.
- Richer features — volume, volatility regime, cross-pair signals — would be required to build a viable predictive model.