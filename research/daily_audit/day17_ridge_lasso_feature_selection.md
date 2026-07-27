# Day 17 Research — Ridge on EURUSD Prediction

## Setup
- **Target:** Next-day EURUSD return
- **Features:** 5 lagged daily returns (lag1 through lag5)
- **Model:** Ridge regression at λ = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
- **Baseline:** OLS (λ = 0)

## Coefficient Comparison Table
| lambda | lag1 | lag2 | lag3 | lag4 | lag5 |
|--------|------|------|------|------|------|
| OLS    | -0.004970 | -0.001067 | -0.005834 | -0.009998 | 0.003890 |
| 0.001  | -0.004951 | -0.001093 | -0.005813 | -0.009933 | 0.003803 |
| 0.01   | -0.004504 | -0.000994 | -0.005288 | -0.009038 | 0.003468 |
| 0.1    | -0.002365 | -0.000521 | -0.002779 | -0.004754 | 0.001844 |
| 1      | -0.000411 | -0.000091 | -0.000484 | -0.000828 | 0.000324 |
| 10     | -0.000044 | -0.000010 | -0.000052 | -0.000089 | 0.000035 |
| 100    | -0.000004 | -0.000001 | -0.000005 | -0.000009 | 0.000004 |
| 1000   | -0.0000004 | -0.0000001 | -0.0000005 | -0.0000009 | 0.0000004 |

## Shrinkage Analysis
**Shrinkage becomes significant at λ = 1**

**Which lag shrinks first and why:**
All lags shrink proportionally — no single lag disappears before the others. This is expected behavior for Ridge, which shrinks all coefficients toward zero simultaneously rather than eliminating any. If one lag had shrunk dramatically faster than the others it would suggest that predictor was carrying very little signal relative to the rest. Lag4 retains the largest absolute magnitude at every λ level.

## OLS vs Ridge
Four of the five OLS coefficients are negative and all five are small in magnitude; lag5 is weakly positive. No lag has a meaningfully large coefficient. The largest is lag4 at -0.010, which represents a predicted next-day return of -1.0% for every 100% lagged return. This is an extremely weak signal. As λ increases from 0.001 to 1000 the coefficients shrink toward zero. Roughly 90% shrinkage occurring between λ = 0.1 and λ = 1. By λ = 10 all coefficients are negligible.

## Interpretation
The predominantly negative signs suggest a weak mean-reversion tendency in EURUSD at the daily frequency, though lag5 reverses sign. Past positive returns weakly predict future negative returns. However the magnitudes are too small to survive transaction costs. A strategy built purely on these lag signals would be unprofitable in practice. The signal exists in the data but is not exploitable.

Ljung-Box test on the raw return series finds no statistically significant autocorrelationat. The negative coefficients produced by OLS and Ridge are not genuine autocorrelation structure they are noise. The signs seen in the coefficient table are a statistical artifact of fitting on a finite sample, not a real signal in the data. This directly invalidates any mean-reversion interpretation of the Ridge output. EURUSD daily returns are consistent with being serially uncorrelated, which aligns with weak-form market efficiency at the daily frequency.

| Lags | LB Statistic | p-value |
|------|-------------|---------|
| 5    | 0.665       | 0.985   |
| 10   | 5.397       | 0.863   |
| 20   | 21.057      | 0.394   |

If the underlying return series has no autocorrelation structure, then Ridge and Lasso applied to lagged returns are fitting noise by construction. Regularization prevents the model from overfitting to spurious patterns, which is exactly what happened here. The OLS coefficients looked meaningful in isolation but Ljung-Box confirmed they were artifacts. The correct response is not to adjust λ but to test richer feature sets where genuine predictive structure might exist.

## Conclusions
- Ridge regression on 5-day lagged EURUSD returns produces uniformly negative coefficients, indicating apparent mean-reversion that Ljung-Box confirms is noise.
- Shrinkage becomes significant at λ = 1, reflecting how little variance the lag features genuinely explain.
- Lag4 is the most persistent predictor across all regularization levels but is not economically meaningful.
- Ridge does not perform variable selection — all five lags remain in the model at every λ. Lasso is the natural next step.
- Richer features — volume, volatility regime, cross-pair signals — would be required to build a viable predictive model.