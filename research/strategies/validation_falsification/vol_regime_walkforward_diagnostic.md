# Day 46 Research Audit: Preliminary Walk-Forward Diagnostic, Per-Leg OOS IC and Sharpe

## Question
Do the momentum and mean-reversion legs show any out-of-sample relationship with forward returns. Both legs are scored unconditionally here, meaning not filtered to each leg's own regime, using a first walk-forward slicing of the strategy's development data. This slicing is admittedly not final.

## Why it matters
The per-window regime refit and the real conditional interaction-regression test on this strategy are put off for a later date. Before that work happens, it is worth a cheap sanity check that the pipeline produces clean, causal output on real data, and a first look at whether either leg's raw predictor shows any life against forward returns.

## Methodology
- Development data only. DataLoader truncated to 2011-01-01 through 2023-12-31 at load time, so the 2024-2026 lockbox is never loaded, sliced, or scored.
- 7 rolling windows from WalkForwardValidator.generate_windows(), using train_years=5, test_months=12, embargo_days=5. An 8th window does not fit inside the dev range, confirmed by the validator's own boundary check.
- Regime classifier, compute_composite_regime_score plus classify_regime, fit on the full development sample instead of refit per training window. That gap is known, put off for a later date, and not corrected here.
- Both legs scored unconditionally, meaning across every day in each test window regardless of that day's regime label.
- Momentum signal compared against the 26-day forward log return using Spearman IC. Price z-score signal compared against the same forward return, also using Spearman IC.
- A simplified per-window Sharpe also computed for each leg, using exposure at t-1 times the next day's log return, annualized with the empirical observations-per-year of each window's own pnl index. Momentum uses its own plus or minus 1 signal as exposure. Reversion uses a rung-1-style exposure of plus or minus 1 or 0, based on whether the absolute price z-score exceeds 2.0.
- Not vol-targeted per Section 7 and no transaction costs, so this Sharpe is a stand-in, not a real one.

## Findings

| Pair | Momentum OOS IC (mean, n=7) | Reversion OOS IC (mean, n=7) | Momentum Sharpe (mean / std / frac positive) | Reversion Sharpe (mean / std / frac positive) |
|---|---|---|---|---|
| EUR/USD | -0.0858 | -0.1254 | 0.132 / 0.848 / 0.571 | 0.506 / 1.560 / 0.714 |
| GBP/USD | -0.0854 | -0.1207 | 0.180 / 0.229 / 0.714 | -0.033 / 0.838 / 0.429 |
| USD/JPY | -0.1033 | -0.1422 | 0.524 / 0.829 / 0.571 | 0.209 / 0.828 / 0.714 |

The pipeline ran clean on all 3 pairs, with n_obs between 4055 and 4056 and a range of 2011-01-02 to 2023-12-29. All 7 windows scored per pair, with no errors.

Both legs show the same sign direction across all three pairs. That is not the scattered pattern you would expect from pure noise across independent pairs.

Reversion's IC is consistently negative, and that is the directionally correct sign for the mean-reversion hypothesis. Price above its own rolling mean, meaning price_z greater than 0, correlating with lower forward returns is exactly what reversion predicts, even before any regime gating.

Momentum's IC is also consistently negative, which is the wrong sign for continuation. Time-series momentum predicts a positive relationship between trailing trend and forward returns. Read most literally, the unconditional relationship at this 78-day lookback and 26-day horizon looks more like reversal than momentum, in all three pairs.

## Alternative explanations
- Momentum's negative sign could reflect a genuine, if uncomfortable, unconditional reversal effect at this exact lookback and horizon combination. That is not ruled out here.
- A more likely explanation, given the strategy's own hypothesis, is that this diagnostic never filters to momentum's IC only on turbulent days. Section 1's falsification criteria require the conditional IC within the turbulent regime specifically. Pooling across all regime states, turbulent, calm, and deadzone alike, where roughly 85 to 90 percent of observations are not turbulent per Day 43's baseline, could easily dilute or invert an effect that is real and correctly signed only within the turbulent subset. This diagnostic cannot tell these two explanations apart. Only the actual conditional test, put off for a later date, can.
- Consistent cross-pair sign is suggestive but not evidence on its own. It still needs permutation-based significance testing, not eyeballing.
- The Sharpe distributions are not statistically informative at n=7 per pair per leg. The standard deviation frequently exceeds the mean, and GBP/USD's reversion leg has a negative mean Sharpe outright. No inferential weight should be placed on these.

## Known limitations
- The regime classifier is fit on the full development sample instead of being refit per training window. Every test window's turbulent, calm, or deadzone labels are partly informed by data outside that window. This is a real leakage vector and it has not yet been corrected.
- WalkForwardValidator only enforces a train to test embargo gap. It does not purge training rows whose 26-day forward-return label overlaps into the embargo or test period, which is the purging that Lopez de Prado's method actually requires. That has not been implemented yet.
- Both legs were scored unconditionally rather than filtered to their own regime, which does not match Section 10's actual test design.
- The Sharpe calculation is a simplified stand-in. It is not vol-targeted per Section 7, has no transaction costs, and does not use the real position-sizing formula.

## On parameter changes
No strategy parameters, meaning lookbacks, entry or exit thresholds, or regime cutoffs, were changed in response to these numbers, and none will be changed before the actual validation runs. Any parameter adjustment made after seeing labeled out-of-sample results, before the pre-registered test has run, would contaminate those windows for the real validation. It would also implicitly expand the parameter search in a way the project's Benjamini-Hochberg correction across 4 trials does not account for. If a parameter genuinely needs revisiting after that validation, it should be documented as a new variant with its own falsification criteria, not as a silent edit made in response to this diagnostic's output.

## Next steps
- Refit the regime classifier per walk-forward training window, and check refit stability against Day 43's full-sample values.
- Run Section 10's interaction regression, including the condition-number and VIF reliability gate, plus permutation and alternate-window robustness checks. This is the only methodology whose result should inform a pass or fail call, or any parameter reconsideration.
- Once the regime refit exists, it is worth checking whether momentum's IC flips sign when properly conditioned on the turbulent subset. That is the real test of whether today's negative pooled IC is dilution or a genuine problem.