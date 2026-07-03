# Day 37 — Block Bootstrap CI on Sharpe Ratio, Variance, and Volatility Clustering

## Methodology
Four analyses run via `research/applied_analysis/day37_applied_analysis_full.py` against 1-minute OHLCV data resampled to daily close, log returns computed as `log(close_t / close_t-1)`, full available history per pair (2011–2026, n≈4759 per pair). N_BOOTSTRAP=1000, CONFIDENCE=0.95, RISK_FREE_RATE=0.0, ann_factor computed empirically per pair via `PerformanceAnalyzer.compute_ann_factor()` (≈312 for all three).

1. **Sharpe CI, i.i.d. vs block** — `bootstrap_confidence_interval` vs `block_bootstrap`, `statistic_fn` = annualized Sharpe, `block_size=21` (one trading month, picked as a starting point rather than derived — see the block size sweep below for why that's still unresolved).
2. **Std CI, i.i.d. vs block (isolation test)** — same setup, `statistic_fn=np.std`, isolating the variance component of Sharpe from the mean component.
3. **Ljung-Box on |r_t|** — lags [5, 10, 20, 40, 60], testing for autocorrelation in return magnitude, separate from autocorrelation in return level (already tested on Day 22).
4. **Block size sweep** — `block_size` ∈ [5, 10, 21, 40, 60, 100, 150, 250], Sharpe CI width at each, single seed (42) per block size, not averaged across multiple resampling draws. Limitations below.

## Findings
### 1. Sharpe CI comparison (block_size=21)

| Pair | i.i.d. CI (95%) | i.i.d. width | Block CI (95%) | Block width | Block vs iid |
|------|------------------|--------------|-----------------|--------------|--------------|
| EURUSD | (−0.6180, 0.3710) | 0.9890 | (−0.5758, 0.3589) | 0.9348 | −5.5% |
| GBPUSD | (−0.6016, 0.3529) | 0.9545 | (−0.5771, 0.3553) | 0.9324 | −2.3% |
| USDJPY | (−0.0316, 0.9723) | 1.0040 | (−0.0048, 0.9218) | 0.9266 | −7.7% |

### 2. Std CI comparison — isolation test (block_size=21)

| Pair | i.i.d. CI (95%) | i.i.d. width | Block CI (95%) | Block width | Block vs iid |
|------|------------------|--------------|-----------------|--------------|--------------|
| EURUSD | (0.004460, 0.004784) | 0.000324 | (0.004388, 0.004810) | 0.000422 | +30.0% |
| GBPUSD | (0.004706, 0.005496) | 0.000791 | (0.004598, 0.005640) | 0.001042 | +31.8% |
| USDJPY | (0.005035, 0.005502) | 0.000467 | (0.004940, 0.005584) | 0.000644 | +38.1% |

### 3. Ljung-Box on |r_t| (all pairs, all lags)
All three pairs reject white noise decisively at every lag, statistics running from ~184 up to ~2182. Day 22 tested raw returns instead of absolute returns and got mostly the opposite: EUR/USD and USD/JPY were white noise, GBP/USD had weak autocorrelation at 3 of 20 lags. Makes sense once you separate the two questions — whether direction is predictable vs. whether magnitude is.

### 4. Block size sweep (Sharpe CI width, single seed=42, not averaged)

| block_size | EURUSD | GBPUSD | USDJPY |
|------------|--------|--------|--------|
| 5 | 0.9936 | 0.9890 | 0.9332 |
| 10 | 0.9444 | 0.9084 | 0.9855 |
| 21 | 0.9348 | 0.9324 | 0.9266 |
| 40 | 0.9244 | 0.8812 | 0.9624 |
| 60 | 0.9139 | 0.8599 | 1.0544 |
| 100 | 0.9568 | 0.8753 | 1.0294 |
| 150 | 1.0171 | 0.8731 | 1.0756 |
| 250 | 0.9913 | 0.8153 | 1.0330 |

No clean interior optimum here, unlike the synthetic AR(1) test earlier in the day. EURUSD bounces around, GBPUSD mostly drifts down, USDJPY mostly drifts up. Only one seed per block size, so some of this is probably just noise from a single draw rather than a real block-length effect.

## Interpretation
The standard deviation isolation test confirms that volatility clustering materially increases uncertainty in volatility estimates. However, that larger uncertainty does not translate directly into wider Sharpe confidence intervals. This is because the Sharpe ratio depends jointly on the estimates of the mean and standard deviation. Their covariance can offset or even reverse the effect seen in either component individually. These results suggest that dependence affects the ratio in a more complex way than it affects volatility alone. Finally, the inconsistent block-size sweep indicates that a single choice such as 21 trading days cannot yet be justified empirically, so any future use of that block length should be treated as provisional until a formal block-length selection procedure.

## Next Steps
- Why does the ratio transformation reverse the direction of the effect rather than merely dampening it?
- Block size sweep needs rerunning with multiple seeds per block_size and averaged widths before it can support a specific block_size decision.
- Formal optimal block-length selection (Politis & White 2004) remains deferred — see separate note, `research/daily_audit/day37_block_length_note.md`.