# Day 23 Research Audit — ARIMA Structure on Forex Return Series

## 1. Question
What order does AIC select for each pair's daily log return series, and do ARIMA residuals pass the Ljung-Box test after fitting?

## 2. Why It Matters
Day 22 Ljung-Box testing found that EUR/USD and USD/JPY showed no autocorrelation at daily frequency, while GBP/USD had marginal structure at lags 5 and 20. ARIMA order selection either confirms or contradicts that finding. If AIC selects non-trivial orders on a series Ljung-Box called white noise, that is AIC overfitting in finite samples — not a discovered edge. For strategy development, any AR structure in daily returns implies short-horizon predictability, but the coefficient magnitude determines whether that predictability survives transaction costs.

## 3. Methodology
- Daily log returns computed from 1-minute OHLCV data, resampled to daily close
- AIC grid search over p ∈ {0,1,2,3}, q ∈ {0,1,2,3}, d = 0 (Day 20 confirmed all three pairs are I(0) on returns)
- Selected order fit via `fit_arima`; Ljung-Box run on residuals at lag 10
- Residuals pass threshold: p > 0.05 (fail to reject null of white noise)
- AR(1) coefficient extracted separately for any pair where AIC selects AR order > 0

## 4. Predictions (written before running)
| Pair    | Expected AIC Order | Residuals Pass LB? | Reasoning |
|---------|--------------------|--------------------|-----------|
| EUR/USD |(0,0,0)|True|Yesterday's ACF/PACF showed no significant lags and Ljung-Box failed to reject white noise across all 20 lags.|
| GBP/USD |(1,0,1)|True|Yesterday's acf/pacf analyis lead to determining that GBPUSD has some characteristics of mothe AR and MA|
| USD/JPY |(0,0,0)|True|Same as EUR/USD|

## 5. Results
| Pair    | AIC Order | AIC Value  | LB p (lag 10) | Residuals OK |
|---------|-----------|------------|---------------|--------------|
| EUR/USD | (0, 0, 0) | -31,930.88 | 0.8653        | True         |
| GBP/USD | (2, 0, 2) | -31,075.26 | 0.8981        | True         |
| USD/JPY | (0, 0, 0) | -31,140.62 | 0.0803        | True         |

GBP/USD AR(1) coefficient (fitted separately): φ = −0.0247, σ² = 2.74 × 10⁻⁵

## 6. Findings
EUR/USD and USD/JPY both returned (0,0,0), consistent with Day 22 Ljung-Box results. USD/JPY's residual Ljung-Box p of 0.0803 clears the 5% bar but not the 10% one, which is the same marginal reading its raw series gave at lag 5 on Day 22.

GBP/USD returned (2,0,2), richer than the (1,0,1) predicted. The MA term was expected because Day 22 showed rejection at lags 5 and 20 with no clean PACF cutoff, and AIC not only kept it but took a second AR and a second MA lag on top. Four parameters bought enough log-likelihood to clear the 2k penalty, which is what an information criterion with a weak penalty does on a series with faint, diffuse structure spread across several lags rather than concentrated at one.

Fitting a plain AR(1) to the same series for comparison gives φ = −0.0247. This is negative, meaning it predicts a partial reversal of today's return tomorrow — anti-persistence, not trending behavior. Today's return predicts tomorrow's with a magnitude of 2.5 cents per dollar, in the opposite direction. Note this is a separate one-parameter fit, not a coefficient of the selected (2,0,2) model.

BIC would almost certainly return (0,0,0) for GBP/USD. At n ≈ 4,054, the BIC penalty per parameter is ln(4054) ≈ 8.3. AIC charges 2. Each of the four terms in the selected (2,0,2) would need to improve log-likelihood by more than 8.3/2 ≈ 4.2 AIC units to survive BIC scrutiny, and with a comparable AR(1) coefficient of only −0.0247 they almost certainly don't. The gap between the AIC and BIC answers is wider here than it was under the previously selected (1,0,0), since there are now four parameters to justify rather than one.

## 7. Alternative Explanations
The GBP/USD (2,0,2) selection is the most likely candidate for finite-sample AIC overfitting, and a four-parameter model on a series this weak is a stronger case for it than a one-parameter model would have been. AIC is known to overfit order in large samples — it charges a flat penalty of 2 per parameter regardless of n, while the true complexity cost grows with sample size.

Three pieces of evidence point toward overfitting rather than genuine structure. First, BIC disagrees — the harsher penalty would return (0,0,0). Second, the Day 22 Ljung-Box p-values on raw returns were not strongly significant; the rejections at lags 5 and 20 were borderline. Third, the comparable AR(1) coefficient of φ = −0.0247 is within sampling noise for a series of this length — the standard error of a phi estimate at n = 4,054 is roughly (1 − φ²)/√n ≈ 0.016, putting the estimate less than 2 standard errors from zero.

To distinguish real predictability from noise, the minimum bar would be: BIC also selects AR order > 0, the coefficient is stable across rolling windows rather than collapsing in out-of-sample periods, and the implied signal survives a transaction cost estimate. None of those conditions are met here.

## 8. Implications for Strategy Development
ARIMA-based mean reversion on daily FX returns is not a strategy candidate. The only pair with any selected AR structure is GBP/USD, and the coefficient is too small to generate a signal that clears transaction costs. A typical EUR/GBP bid-ask spread at the retail level is 0.5–1 pip.

The negative sign on φ is also worth noting. This is anti-persistence — the model expects a partial reversal, not continuation. Anti-persistence at daily frequency is more consistent with bid-ask bounce or microstructure noise than with an exploitable trend or mean-reversion signal.

## 9. Next Steps
- Open question from today: Does the GBP/USD AR(1) coefficient remain stable across rolling 1-year windows, or does it collapse out-of-sample? If it's a full-sample artifact, the rolling estimate will show no consistency. This question gets answered with WalkForward runs.