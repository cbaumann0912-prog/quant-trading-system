# Day 47 Research Audit: Per-Window Regime Classifier Refit vs. Day 43 Full-Sample Baseline

## Question
Does refitting the regime composite's z-scoring and PCA per walk-forward window (train-only, frozen, applied to test) rather than fitting once on the full sample (Day 43) materially change the resulting turbulent/calm/deadzone labels? And is the fitted vol-carry relationship even stable enough across 5-year windows to trust as a regime signal at all?

## Why it matters
The regime classifier gates which leg of the strategy is active (momentum vs. mean-reversion). Day 43's full-sample fit is leakage by construction — every observation's z-score, including one from 2012, was standardized using statistics computed from the entire 2011-2023 history. If the walk-forward refit produces materially different labels, any backtest run on the Day 43 version overstates what a live classifier could actually have known at each point in time.

## Methodology
`research/strategies/validation_falsification/vol_regime_classifier_refit_stability.py`, using the production `DataLoader`, `WalkForwardValidator` (boundary generation only), and the new `compute_composite_regime_score_walkforward` (`src/signals/regime_refit.py`). Per pair: 5-year rolling training windows, 12-month test windows, 5-day embargo, 7 windows spanning 2011-2017 train starts. Two versions of the composite regime score computed and classified with the pre-registered thresholds (turbulent \|z\|>1.5, calm \|z\|<1.0):

1. **Day 43 baseline** — z-score mean/std and PCA fit once on the full sample.
2. **Day 47 walk-forward** — z-score mean/std and PCA fit fresh on each window's training slice only, frozen, applied to that window's test slice.

**Assumption carried over from Day 43/47's derivation:** with exactly 2 z-scored features, PCA loadings are mathematically pinned to (0.707, ±0.707) regardless of the data (a 2x2 symmetric matrix with equal diagonal entries always has eigenvectors at ±45°). `pc1_vol` is therefore not a stability signal; `rho`, the train-window correlation between vol and rate_diff, is tracked instead since it's the one real degree of freedom.

## Findings

| Pair | Day 43 rho (full-sample) | Per-window rho range | Sign flips (of 6 transitions) | Regime label agreement (WF vs. full-sample) | Bars that flip | WF regime mix (turbulent / calm / deadzone) |
|---|---|---|---|---|---|---|
| EUR/USD | +0.337 | −0.197 to +0.824 | 2 | 49.41% | 1107 / 2188 | 41.9% / 37.6% / 20.5% |
| GBP/USD | −0.186 | −0.522 to +0.389 | 3 | 55.74% | 968 / 2187 | 45.3% / 39.8% / 14.9% |
| USD/JPY | +0.193 | −0.502 to +0.501 | 3 | 20.52% | 1739 / 2188 | 60.6% / 33.8% / 5.6% |

For reference, the Day 43 baseline's own (full-sample, leaky) regime mix was turbulent 9.2-15.9%, calm ~64.5-71.4%, deadzone 13.6-24.6%.

**The label disagreement is large, not a rounding-error correction.** Roughly half of all out-of-sample days for EUR/USD and GBP/USD get a different regime label depending on whether the classifier is fit correctly or leaked. USD/JPY is far worse: only 20.52% of days agree — the walk-forward and full-sample classifiers disagree about four times out of five

**The vol-carry correlation is not stable across 5-year windows.** `rho` reverses sign on 2 of 6 consecutive-window transitions for EUR/USD and 3 of 6 for the other two pairs, and the swings aren't small: EUR/USD spans −0.20 to +0.82 across its training windows; USD/JPY spans −0.50 to +0.50, crossing zero rather than holding a stable sign. Since `explained_variance_ratio = (1+|rho|)/2` mostly sits in the 0.50-0.65 range and only occasionally reaches 0.75-0.91, the two features are weakly related most of the time and that weak relationship isn't even consistently signed.

**Every pair's walk-forward turbulent share is far above the Day 43 baseline, and calm share is far below.** This is a systematic pattern in one direction, not scattered noise: EUR/USD calm drops from a baseline ~70.5% to 37.6%; GBP/USD from ~71.4% to 39.8%; USD/JPY from ~64.5% to 33.8%. The leaky full-sample fit was systematically undercounting how often the market would honestly have been classified turbulent.

**All three pairs trip the strategy's own pre-registered classifier-decay failure condition.** Section 9 of the strategy spec defines failure as live regime proportions drifting such that turbulent days exceed roughly 40%. Walk-forward turbulent shares are EUR/USD 41.9%, GBP/USD 45.3%, USD/JPY 60.6% — all above the bar, using historical data already in the sample rather than live drift. This is not a single-pair problem to be handled by dropping one pair; the classifier fails its own decay condition across the whole universe the moment it is fit without leakage.

USD/JPY is the worst of the three by a wide margin. A 20.52% label-agreement rate means the walk-forward and full-sample classifiers disagree roughly four days in five, which is worse than what a coin flip between three labels would produce. Combined with the highest turbulent share, USD/JPY is where the Day 43 thresholds are least trustworthy — but the other two are not passing, only failing by less.

## Alternative explanations
- The rho sign reversals could reflect a genuine, real change in the economic relationship between volatility and carry over time (e.g. a pre/post-2020 low-rate-era shift) rather than an estimation artifact from short 5-year windows. Not separable here without a dedicated sub-period stability study.
- The elevated turbulent share under the walk-forward refit could partly be an artifact of the 5-year/12-month window choice itself (shorter local normalization windows make z-scores more sensitive to local dispersion) rather than a "truer" reading of history. Not tested against alternate `train_years`/`test_months` combinations.

## Next steps
- Decide whether the Day 43 thresholds (turbulent >1.5, calm <1.0) are still appropriate once the classifier is fit correctly, or whether they need to be re-derived using a non-leaky methodology — they were chosen from the same leaky full-sample fit this audit shows produces a different regime mix.
- Given `rho`'s instability (sign flip on 2 of 6 transitions for EUR/USD and 3 of 6 for the other two), worth testing whether a longer training window stabilizes the fit, or whether the vol-carry relationship is too noisy to serve as a reliable structural regime signal — this bears directly on Section 1's falsification hypothesis that a genuine, persistent regime split (Ang & Bekaert) exists at all.