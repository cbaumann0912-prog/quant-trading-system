# Day 22 Research Audit — ACF, PACF, and Ljung-Box: All Forex Pairs

## 1. Question
Do daily log return series for EUR/USD, GBP/USD, and USD/JPY exhibit statistically significant autocorrelation structure, and if so, what process class does each series belong to?

## 2. Why It Matters
Autocorrelation in returns means past values carry predictive information about future values. That's a necessary condition for any momentum or mean-reversion strategy at that frequency. Without it, returns are a martingale difference sequence and the independence assumption in the Week 2 hypothesis tests holds — no linear time-series model produces edge. Whether autocorrelation survivesaggregation from 1-minute to daily bars determines the right modeling frequency for strategy specs.

## 3. Methodology
- Daily close prices computed by resampling 1-minute OHLCV to daily last close
- Log returns computed as ln(Close_t / Close_{t-1}) on daily bars
- ACF and PACF plotted to lag 40, significance bands at ±1.96/√n
- Ljung-Box Q tested to lag 20; null hypothesis: no autocorrelation up to lag m
- Analysis run on full 15-year sample (2011–2026)

## 4. Results

### EUR/USD

| Metric | Value |
|--------|-------|
| Obs | 4,759 |
| Ljung-Box Q (lag 5) | 1.0684 |
| p-value (lag 5) | 0.9569 |
| Ljung-Box Q (lag 20) | 21.1693 |
| p-value (lag 20) | 0.3872 |
| Lags with p < 0.05 | 0 / 20 |

**ACF pattern:** All spikes fall within ±1.96/√n at every lag. No decay pattern. Consistent with immediate cutoff at lag 0.

**PACF pattern:** Same

**Implied process class:** White noise

### GBP/USD

| Metric | Value |
|--------|-------|
| Obs | 4,758 |
| Ljung-Box Q (lag 5) | 12.2538 |
| p-value (lag 5) | 0.0315 |
| Ljung-Box Q (lag 20) | 35.1609 |
| p-value (lag 20) | 0.0193 |
| Lags with p < 0.05 | 3 / 20 |

**ACF pattern:** Spikes at low lags with gradual rather than abrupt decay. No clean cutoff, ruling out a pure MA process.

**PACF pattern:** Partial autocorrelations at early lags without a sharp cutoff, ruling out pure AR.

**Implied process class:** Weak ARMA or AR(1)-adjacent. Neither ACF nor PACF cuts off cleanly, suggesting both AR and MA components. The structure is weak — 3 of 20 lags reject — but the series is distinguishable from white noise at lag 5 and lag 20.

### USD/JPY

| Metric | Value |
|--------|-------|
| Obs | 4,759 |
| Ljung-Box Q (lag 5) | 4.0484 |
| p-value (lag 5) | 0.5425 |
| Ljung-Box Q (lag 20) | 16.0335 |
| p-value (lag 20) | 0.7145 |
| Lags with p < 0.05 | 0 / 20 |

**ACF pattern:** All spikes within confidence bands. No detectable structure.

**PACF pattern:** Same

**Implied process class:** White noise

## 5. Cross-Pair Comparison
EUR/USD and USD/JPY are structurally identical at daily frequency: white noise, zero lags rejecting, p-values well above 0.05 across the board. GBP/USD diverges. Lag 5 and lag 20 both reject, and the Q statistic at lag 20 (35.16 vs. 21.17 and 16.03) is roughly double the other two pairs. A strategy thatassumes homogeneous return structure across all three pairs is misspecified.GBP/USD carries autocorrelation the others don't, so any ARIMA-based signal applies to GBP/USD only.

## 6. Alignment with Day 20 Stationarity Results
Day 20 ADF tests rejected the unit root null on all three series; KPSS failed  to reject stationarity. Both results align with the ACF findings here. A stationary series has an ACF that decays to zero — white noise is just the extreme case where that happens immediately. EUR/USD and USD/JPY fit that strongly. GBP/USD is stationary too, but its ACF decays more slowly, which is exactly what the Ljung-Box rejections pick up. No contradictions between Day 20 and Day 22.

## 7. Implications for Strategy Research
The Week 2 hypothesis tests assumed independent observations. EUR/USD and USD/JPY satisfy that at daily frequency — no Ljung-Box rejection across 20 lags. GBP/USD technically violates independence at the margin (3 lags reject), introducing mild upward bias in test statistics. Worth flagging when interpreting any GBP/USD significance results from Week 2. More importantly, autocorrelation at 1-minute frequency doesn't survive aggregation to daily bars for EUR/USD and USD/JPY. Edge on those two pairs, if it exists, comes from cross-pair relationships — cointegration, PCA factors — not serial return predictability.

## 8. Next Steps
The GBP/USD finding defers to tomorrow fitting ARIMA on daily returns and check whether AIC selects a non-zero order. EUR/USD and USD/JPY won't benefit from ARIMA at this frequency. For the possible strategy list, any strategy built on serialreturn predictability should be GBP/USD-scoped or dropped in favor of cross-pairstructure. Cointegration and PCA from Days 15–21 remain the stronger candidates.