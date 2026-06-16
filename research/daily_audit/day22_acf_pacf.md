# Day 22 Research Audit — ACF, PACF, and Ljung-Box: All Forex Pairs

## 1. Question
Do the 1-minute log return series for EUR/USD, GBP/USD, and USD/JPY exhibit
statistically significant autocorrelation structure, and if so, what process
class does each series belong to?

## 2. Why It Matters
<!-- 2–3 sentences. Connect to strategy research — what does autocorrelation
     (or its absence) imply about predictability and the independence assumption
     underlying your hypothesis tests from Week 2? -->

## 3. Methodology
- Log returns computed as ln(Close_t / Close_{t-1}) on 1-minute bars
- ACF and PACF plotted to lag 40, significance bands at ±1.96/√n
- Ljung-Box Q tested to lag 20; null hypothesis: no autocorrelation up to lag m
- Analysis run on full 15-year sample

## 4. Results

### EUR/USD

| Metric | Value |
|--------|-------|
| Obs | |
| Ljung-Box Q (lag 5) | |
| p-value (lag 5) | |
| Ljung-Box Q (lag 20) | |
| p-value (lag 20) | |
| Lags with p < 0.05 | / 20 |

**ACF pattern:**
<!-- Describe — cuts off, tails off, where are the significant spikes? -->

**PACF pattern:**
<!-- Describe -->

**Implied process class:**
<!-- AR / MA / ARMA / white noise, with reasoning -->

### GBP/USD

| Metric | Value |
|--------|-------|
| Obs | |
| Ljung-Box Q (lag 5) | |
| p-value (lag 5) | |
| Ljung-Box Q (lag 20) | |
| p-value (lag 20) | |
| Lags with p < 0.05 | / 20 |

**ACF pattern:**

**PACF pattern:**

**Implied process class:**

### USD/JPY

| Metric | Value |
|--------|-------|
| Obs | |
| Ljung-Box Q (lag 5) | |
| p-value (lag 5) | |
| Ljung-Box Q (lag 20) | |
| p-value (lag 20) | |
| Lags with p < 0.05 | / 20 |

**ACF pattern:**

**PACF pattern:**

**Implied process class:**

## 5. Cross-Pair Comparison
<!-- Is the autocorrelation structure consistent across all three pairs or does
     one diverge? If one pair behaves differently, what does that imply for a
     multi-pair strategy that assumes homogeneous structure? -->

## 6. Alignment with Day 20 Stationarity Results
<!-- ADF and KPSS on Day 20 tested whether the series have unit roots. A
     stationary series should have ACF that decays to zero. Does what you see
     in the ACF plots align with those conclusions? Any contradictions? -->

## 7. Implications for Strategy Research
<!-- The hypothesis tests in Week 2 assumed independent observations. What do
     the Ljung-Box results say about that assumption? Does autocorrelation at
     1-minute frequency survive aggregation to daily bars — and why does that
     distinction matter for your Day 30 strategy specs? -->

## 8. Next Steps
<!-- What does this analysis change or confirm about your Day 30 strategy
     shortlist? Any open questions deferred to Day 23 (ARIMA)? -->