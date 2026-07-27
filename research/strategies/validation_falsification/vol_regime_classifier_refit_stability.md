# Day 47 Research Audit: Per-Window Regime Classifier Refit vs. Day 43 Full-Sample Baseline

## Question
Does refitting the regime composite's z-scoring and PCA per walk-forward window (train-only, frozen, applied to test) rather than fitting once on the full sample (Day 43) materially change the resulting turbulent/calm/deadzone labels? And is the fitted vol-carry relationship even stable enough across 5-year windows to trust as a regime signal at all?

## Why it matters
The regime classifier gates which leg of the strategy is active (momentum vs. mean-reversion). Day 43's full-sample fit is leakage by construction — every observation's z-score, including one from 2012, was standardized using statistics computed from the entire 2011-2026 history. If the walk-forward refit produces materially different labels, any backtest run on the Day 43 version overstates what a live classifier could actually have known at each point in time.

## Methodology
`research/strategies/validation_falsification/vol_regime_classifier_refit_stability.py`, using the production `DataLoader`, `WalkForwardValidator` (boundary generation only), and the new `compute_composite_regime_score_walkforward` (`src/signals/regime_refit.py`). Per pair: 5-year rolling training windows, 12-month test windows, 5-day embargo, 10 windows spanning 2011-2025 train starts. Two versions of the composite regime score computed and classified with the pre-registered thresholds (turbulent \|z\|>1.5, calm \|z\|<1.0):

1. **Day 43 baseline** — z-score mean/std and PCA fit once on the full sample.
2. **Day 47 walk-forward** — z-score mean/std and PCA fit fresh on each window's training slice only, frozen, applied to that window's test slice.

**Assumption carried over from Day 43/47's derivation:** with exactly 2 z-scored features, PCA loadings are mathematically pinned to (0.707, ±0.707) regardless of the data (a 2x2 symmetric matrix with equal diagonal entries always has eigenvectors at ±45°). `pc1_vol` is therefore not a stability signal; `rho`, the train-window correlation between vol and rate_diff, is tracked instead since it's the one real degree of freedom.

## Findings

| Pair | Day 43 rho (full-sample) | Per-window rho range | Sign flips (of 9 transitions) | Regime label agreement (WF vs. full-sample) | Bars that flip | WF regime mix (turbulent / calm / deadzone) |
|---|---|---|---|---|---|---|
| EUR/USD | +0.337 | −0.529 to +0.824 | 3 | 54.18% | 1432 / 3125 | 32.7% / 49.3% / 18.0% |
| GBP/USD | −0.186 | −0.559 to +0.389 | 3 | 56.24% | 1367 / 3124 | 31.7% / 48.6% / 19.8% |
| USD/JPY | +0.193 | −0.502 to +0.616 | 3 | 34.34% | 2052 / 3125 | 57.4% / 35.1% / 7.5% |

For reference, the Day 43 baseline's own (full-sample, leaky) regime mix was turbulent 9.2-15.9%, calm ~64.5-71.4%, deadzone 13.6-24.6%.

**The label disagreement is large, not a rounding-error correction.** Roughly half of all out-of-sample days for EUR/USD and GBP/USD get a different regime label depending on whether the classifier is fit correctly or leaked. USD/JPY is worse: only 34.34% of days agree

**The vol-carry correlation is not stable across 5-year windows.** Every pair shows `rho` reversing sign 3 times out of 9 consecutive-window transitions, and the swings aren't small: EUR/USD moves from +0.82 (2015-2020 training window) to −0.53 (2020-2025 window); USD/JPY moves from −0.50 (2015-2021 windows) to +0.62 (2020-2025 window), a sustained multi-window sign reversal, not single-window noise. Since `explained_variance_ratio = (1+|rho|)/2` mostly sits in the 0.50-0.65 range and only occasionally reaches 0.75-0.91, the two features are weakly related most of the time and that weak relationship isn't even consistently signed.

**Every pair's walk-forward turbulent share is far above the Day 43 baseline, and calm share is far below.** This is a systematic pattern in one direction, not scattered noise: EUR/USD calm drops from a baseline ~70.5% to 49.3%; GBP/USD from ~71.4% to 48.6%; USD/JPY from ~64.5% to 35.1%. The leaky full-sample fit was systematically undercounting how often the market would honestly have been classified turbulent.

**USD/JPY trips the strategy's own pre-registered classifier-decay failure condition.** Section 9 of the strategy spec defines failure as live regime proportions drifting such that turbulent days exceed roughly 40%. USD/JPY's walk-forward turbulent share is 57.4% — using historical data already in the sample, not live drift. Combined with the lowest label-agreement rate of the three pairs (34.34%) and the largest sustained rho sign reversal, USD/JPY looks like the pair where the Day 43 thresholds (chosen from the leaky fit) are least trustworthy.

## Alternative explanations
- The rho sign reversals could reflect a genuine, real change in the economic relationship between volatility and carry over time (e.g. a pre/post-2020 low-rate-era shift) rather than an estimation artifact from short 5-year windows. Not separable here without a dedicated sub-period stability study.
- The elevated turbulent share under the walk-forward refit could partly be an artifact of the 5-year/12-month window choice itself (shorter local normalization windows make z-scores more sensitive to local dispersion) rather than a "truer" reading of history. Not tested against alternate `train_years`/`test_months` combinations.

## Next steps
- Decide whether the Day 43 thresholds (turbulent >1.5, calm <1.0) are still appropriate once the classifier is fit correctly, or whether they need to be re-derived using a non-leaky methodology — they were chosen from the same leaky full-sample fit this audit shows produces a different regime mix.
- Given `rho`'s instability (sign flip on 3 of 9 transitions for every pair), worth testing whether a longer training window stabilizes the fit, or whether the vol-carry relationship is too noisy to serve as a reliable structural regime signal — this bears directly on Section 1's falsification hypothesis that a genuine, persistent regime split (Ang & Bekaert) exists at all.