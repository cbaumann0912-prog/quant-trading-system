# Day 26 Audit — Mean Reversion Half-Life on Cointegrated Spreads

## 1. Question
What is the OU half-life of mean reversion for the EUR/USD on GBP/USD spread, and is it tradeable given transaction costs and the cointegration evidence from Days 24–25?

## 2. Why It Matters
Half-life determines the expected holding period for a mean-reversion trade. A short half-life is irrelevant if the underlying spread relationship is not actually cointegrated — the half-life estimate is only meaningful conditional on a genuine long-run equilibrium existing between the two series. This audit must be read alongside Day 24 and Day 25 findings, not in isolation.

## 3. Methodology
OU parameters estimated via OLS regression of ΔX_t on X_{t-1}

theta = -beta
mu = -alpha / beta
half_life = ln(2) / theta

Implemented as `ou_half_life` in `src/signals/cointegration.py`. Applied to the residual series produced by `engle_granger_test` for all three forex pair combinations.

## 4. Assumptions
- The residual series follows an OU process (linear mean reversion, constant theta, constant sigma).
- `dt = 1` (single bar step, daily resampling); no recalibration for irregular time gaps.
- OLS estimation of theta assumes no autocorrelation in residuals beyond what the AR(1)-style regression already captures.
- Treats each pair's hedge ratio as fixed for the full sample, despite the rolling hedge ratio instability already documented in the Day 24 audit for EUR/USD on GBP/USD (std=0.396, range=−0.11 to +1.57).

## 5. Findings
| Pair | adf_p | theta | mu | sigma | half_life (days) |
|---|---|---|---|---|---|
| EUR/USD~GBP/USD | 0.0621 | 0.003154 | -0.001985 | 0.004432 | 219.79 |
| EUR/USD~USD/JPY | 0.2749 | 0.001984 | 0.010239 | 0.005118 | 349.38 |
| GBP/USD~USD/JPY | 0.4354 | 0.001628 | 0.012313 | 0.006743 | 425.81 |

Bar duration is daily, so half-life in bars equals half-life in calendar days directly.

For EUR/USD on GBP/USD, the strongest of the three: half-life ≈ 220 calendar days (~7.3 months). A mean-reversion trade entered on this spread would expect to wait roughly 7 months before the deviation closes half its distance back to mu. EUR/USD on USD/JPY and GBP/USD on USD/JPY produced longer half-lives of 349 and 426 days respectively.

As a rough estimate, round-trip costs (spread + slippage) typically run on the order of 1–3 bps per trade for EUR/USD and GBP/USD. At a 220-day half-life, a single mean-reversion cycle implies roughly 1.6 round trips per year — at that trade frequency, a 1–3 bps cost is negligible relative to any plausible signal magnitude, since sigma = 0.004432 is more than the estimated cost. Transaction cost is therefore not the binding constraint on this candidate. The binding constraint is the holding period itself: a ~220-day expected holding period is incompatible with a forex execution framework built on 1-minute OHLCV data and designed for short-horizon trade cycles, independent of how cheap or expensive execution is.

Z-score thresholds for EUR/USD on GBP/USD, derived from mu = -0.001985, sigma = 0.004432:
- Entry: spread crosses mu ± 2σ → triggers at approximately -0.01086 or 0.00689
- Exit: spread returns to mu ± 0.5σ → approximately -0.00420 or 0.00023

## 6. Alternative Explanations
The half-life results are consistent with the Day 24 and Day 25 findings: EUR/USD on GBP/USD was borderline at p=0.0621 in Engle-Granger, failing after BH correction, and Johansen found rank 0 system-wide. A 220-day half-life on a spread that already fails cointegration testing at the 5% level does not indicate a tradeable edge — it indicates a regression artifact on a spread without statistically supported long-run equilibrium.

The rolling hedge ratio instability documented in Day 24 also means the EUR/USD on GBP/USD residual series is not a stable object. The OU parameters estimated here use a full-sample, fixed-hedge-ratio spread, which may not generalize to a rolling implementation.

## 7. Next Steps
Carry the half-life finding into the Day 30 strategy candidate decision. Three independent tests — Engle-Granger, Johansen, and OU half-life — now point in the same direction for Multi-Pair Forex Stat Arb. Document the rationale for moving this candidate to invalid status and evaluate replacement candidates (PC2 Carry Regime Signal, or forex scope expansion).