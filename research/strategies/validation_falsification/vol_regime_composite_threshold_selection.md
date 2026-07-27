# Day 43 Research Audit: Regime Composite Score & Threshold Selection

## Question
What threshold(s) on the composite score (rolling vol + rate differential) should define "calm" vs. "turbulent" regimes, and does PCA weighting add value over simple equal-weighting?

## Why it matters
The strategy's falsification criteria require a conditional-IC test within the classified turbulent regime. The threshold trades off regime "purity" against sample size: too strict a threshold starves the test of power regardless of whether a true effect exists.

## Methodology
Full-sample descriptive pass (threshold selection only, not the production computation). For each pair: 78-day rolling realized volatility of log returns + rate differential (2-month-lagged, forward-filled), both z-scored and combined via PCA (1st component, sign-normalized so volatility loading is positive). Computed the share of observations exceeding |composite z| at thresholds 1.00-3.00 in 0.25 increments.

Script: `research/strategies/validation_falsification/vol_regime_composite_threshold_selection.py` (uses the production `DataLoader`, fully reproducible).

**Assumption:** production SignalBuilder must refit z-scoring and PCA inside each walk-forward window using only training data. Full-sample fitting here would leak future correlation structure if it made it into the live signal.

## Findings

| Pair | PC1 loadings (vol, rate_diff) | Explained var | \|z\|>1.00 | \|z\|>1.50 | \|z\|>2.00 | Deadzone 1.0-1.5 |
|---|---|---|---|---|---|---|
| EUR/USD | (0.707, 0.707) | 0.669 | 28.7% (n=1143) | 17.9% (n=713) | 4.8% (n=190) | 10.8% |
| GBP/USD | (0.707, -0.707) | 0.588 | 29.1% (n=1158) | 8.5% (n=337) | 5.2% (n=205) | 20.7% |
| USD/JPY | (0.707, 0.707) | 0.551 | 30.0% (n=1194) | 9.6% (n=383) | 6.8% (n=272) | 20.4% |

**PCA loadings are essentially equal-weight** ((1/sqrt(2), 1/sqrt(2))) for all three pairs, since the two features are only weakly correlated (PC1 explains just 55-67% of joint variance, not the 90%+ that would indicate real re-weighting). PCA isn't adding meaningful value over equal-weighting here.

**GBP/USD's rate-differential loading sign flips** relative to EUR/USD and USD/JPY after sign-normalization: elevated GBP/USD volatility coincides with a *narrower* UK-US differential, the other two pairs show the opposite. Genuine empirical asymmetry, not a computation error.

**Threshold:** at |z| > 1.5, turbulent is 9.2-15.9% of observations (~430-740/pair), enough sample for real test power. At |z| > 2.0 it drops to 3.4-5.0% (~160-235/pair), trading purity for power the falsification criteria need.

## Alternative explanations
- The weak PC1/PC2 correlation could reflect genuine economic independence between vol and carry, or be an artifact of the window/lag choices. Not separately tested.
- GBP/USD's sign flip could be Brexit-era-specific rather than a stable relationship. Not decomposed by sub-period.

## Next steps
- Threshold locked: |composite z| > 1.5 = turbulent, < 1.0 = calm, 1.0-1.5 = deadzone.
- SignalBuilder, put off for a later date, must refit z-scoring/PCA per walk-forward window, not reuse these full-sample statistics.
- Worth checking GBP/USD's sign stability across sub-periods before the eventual paper's limitations section.

---

## Addendum: mean-reversion trigger, conditional forward return

**Question:** what price-deviation magnitude actually predicts reversion (not just rarity), for the calm-regime entry threshold?

**Methodology:** within the calm regime (|composite z| < 1.0), 26-day rolling price z-score vs. sign-adjusted forward return (`-sign(price_z) * forward_return`) at 1wk/2wk/1mo horizons, thresholds 1.00-3.00.

**Findings** (sign-adjusted forward return, z=1.0 -> z=2.5, 1wk/2wk/1mo):
- **EUR/USD:** no clear signal at any threshold, noisy and inconsistent in sign.
- **GBP/USD:** effect increases with threshold at the 2wk and 1mo horizons: (−0.01%, +0.05%, +0.14%) to (−0.12%, +0.16%, +0.32%). The 1wk horizon is negative throughout and gets more negative, so the pattern is horizon-dependent rather than uniform.
- **USD/JPY:** clearest pattern at the two shorter horizons: (+0.07%, +0.07%, −0.05%) to (+0.35%, +0.33%, −0.14%). The 1mo horizon is negative at both ends, so the reversion this picks up is a short-horizon effect that has decayed or reversed by one month.

**Decision:** entry threshold set at z=2.0, where GBP/USD and USD/JPY's effect is non-trivial with workable sample sizes (n=337-347/pair). Note that EUR/USD's conditional forward return at this threshold is negative at every horizon, so the threshold rests on two of the three pairs, not all three.

**Caveat:** this threshold was chosen because it looks strongest in this in-sample descriptive pass, a mild form of look-ahead. It's a defensible design choice, not evidence the strategy works. The actual test is Section 10's out-of-sample walk-forward validation (strategy spec).

**Flag:** EUR/USD's null result alongside GBP/USD and USD/JPY's real ones suggests this strategy may not have a uniform edge across all three pairs. Not a surprise if EUR/USD's Section 10 results come back null.

---

## Addendum (2026-07-19): lockbox leakage in this analysis

The threshold selected here was chosen because it looked strongest in an in-sample descriptive pass. That remains a mild form of look-ahead regardless of the sample it is run on, and it is the reason this audit is a design-choice record rather than evidence the strategy works. The out-of-sample test in Section 10 of the strategy spec is what settles that question.
