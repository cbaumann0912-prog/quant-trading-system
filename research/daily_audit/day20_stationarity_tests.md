# Day 20 Research Audit — Stationarity Analysis: EUR/USD, GBP/USD, USD/JPY

## 1. Question Investigated
Are any bivariate combinations of EUR/USD, GBP/USD, and USD/JPY log price levels cointegrated? Specifically, whether OLS residuals from regressing one log price series on another are stationary (I(0)), indicating a stable long-run relationship between the two series.

## 2. Why It Matters
Mean-reversion pair trading only works if the spread has a mean to revert to. That requires cointegration — a stable long-run equilibrium between two non-stationary price series. Without it, any apparent spread convergence is noise in a drifting process, and fading it is directional speculation dressed up as arbitrage.

## 3. Methodology
- 13 years of 1-minute OHLCV data resampled to daily closing prices.
- Log price levels computed as ln(Pt).
- Series aligned on common trading dates via index intersection across all three pairs.
- Individual pair stationarity tested using `check_stationarity()` from `src/data/stationarity.py`, which combines ADF and KPSS.
- OLS regression run for each bivariate combination to estimate the hedge ratio (beta).
- Residuals extracted and tested for stationarity — Engle-Granger step 2.
- Benjamini-Hochberg correction applied to residual ADF p-values to control false discovery rate across the three tests.

## 4. Assumptions
- Log price levels are I(1). Engle-Granger requires both inputs to be integrated of order one. If any series is I(2) or already I(0), the framework breaks down. Layer 1 checks this.
- The cointegrating relationship is linear. OLS estimates a linear hedge ratio and will miss non-linear structure.
- The hedge ratio is stable over time. One regression over 13 years assumes beta is constant. Structural breaks violate this. Walk-forward stability is put off for a later date.
- Residuals are approximately homoskedastic. ADF and KPSS are sensitive to volatility clustering. FX residuals likely exhibit GARCH effects — a known violation, addressed formally at a later date.
- No lookahead bias. Only daily closing prices are used.

## 5. Findings

### 5a. Raw Log Price Levels — Individual Pairs

| Series  | ADF Stat | ADF p  | KPSS Stat | KPSS p | Verdict        |
| ------- | -------- | ------ | --------- | ------ | -------------- |
| EUR/USD | -1.7979  | 0.3815 | 5.9382    | 0.0100 | I(1) unit root |
| GBP/USD | -1.4998  | 0.5336 | 7.7829    | 0.0100 | I(1) unit root |
| USD/JPY | -1.0594  | 0.7310 | 6.1231    | 0.0100 | I(1) unit root |

All three series came back I(1) as expected. ADF failed to reject the unit root null in all cases; KPSS rejected stationarity decisively. The KPSS statistics are far outside the lookup table range — the actual p-values are smaller than 0.01, as flagged by statsmodels InterpolationWarning. The I(1) assumption for Engle-Granger holds.

### 5b. OLS Residuals — All Pair Combinations

| Spread            | Beta    | ADF Stat | ADF p  | KPSS Stat | KPSS p | Raw Verdict            | BH-Corrected |
| ----------------- | ------- | -------- | ------ | --------- | ------ | ---------------------- | ------------ |
| EUR/USD ~ GBP/USD | 0.7125  | -2.5658  | 0.1003 | 0.7230    | 0.0115 | I(1), no cointegration | False        |
| EUR/USD ~ USD/JPY | -0.4532 | -2.4280  | 0.1340 | 1.0291    | 0.0100 | I(1), no cointegration | False        |
| GBP/USD ~ USD/JPY | -0.4416 | -1.8308  | 0.3653 | 3.0776    | 0.0100 | I(1), no cointegration | False        |

No pair passed, and none came close. EUR/USD ~ GBP/USD is the strongest of the three at ADF p = 0.1003, with a test statistic of -2.5658 against a critical value of approximately -2.86 — a 0.29-unit miss, not a near-rejection. KPSS rejected stationarity at the 5% level (stat = 0.7230 vs critical value 0.463). Both tests land on non-stationary, so the verdict stands.

The other two pairs are not close. EUR/USD ~ USD/JPY and GBP/USD ~ USD/JPY show no evidence of mean-reverting residuals under either test.

The levels-based beta for EUR/USD ~ GBP/USD is 0.7125, not 0.5596. The Day 15 beta was estimated on log returns and measures contemporaneous daily return sensitivity. Today's beta is estimated on log price levels and measures long-run equilibrium elasticity. These are different quantities answering different questions.

## 6. Alternative Explanations
The null result does not rule out a relationship between these pairs — it rules out a static, full-sample, linear one.

- Over 15 years, the EUR/GBP relationship absorbed the European sovereign debt crisis, Brexit, COVID-19, and multiple divergent rate cycles. A single regression treats all of that as one regime. It probably is not.
- The borderline EUR/USD ~ GBP/USD result is consistent with a cointegrated spread with very slow mean reversion — a half-life of several months. At that speed, ADF has low power to distinguish the spread from a unit root in finite samples. Johansen testing will provide a more sensitive check.
- The Engle-Granger procedure is a two-step estimator with known efficiency losses. Johansen's maximum likelihood approach handles this better, particularly with three series.
- Daily sampling may be too coarse. Intraday cointegration could exist and disappear under daily aggregation.
- FX residual volatility clusters sharply around macro events. GARCH-corrected stationarity tests may produce different inference.

## 7. Open Questions
- Do rolling-window Engle-Granger tests reveal periods where EUR/USD ~ GBP/USD cointegration holds within specific regimes?
- Does Johansen identify a multivariate cointegrating vector across all three pairs that pairwise testing misses?
- How much does the EUR/USD ~ GBP/USD hedge ratio move across walk-forward windows?
- Does GARCH-adjusted stationarity testing on the EUR/USD ~ GBP/USD residuals push the ADF stat past -2.86?

## Conclusion
EUR/USD, GBP/USD, and USD/JPY log price levels are all I(1). No pairwise combination produced stationary residuals under the combined ADF-KPSS framework. After BH correction, none of the three spreads meets the statistical threshold for cointegration. These pairs should not be treated as mean-reversion candidates under a static full-sample hedge ratio. The EUR/USD ~ GBP/USD borderline result is worth revisiting with Johansen at a later date before writing the pair off entirely.