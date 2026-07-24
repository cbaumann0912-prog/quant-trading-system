# Day 53 Research Audit: VaR/CVaR Report, Momentum-Only Pooled Book

## Question
How do parametric VaR, historical VaR, Monte Carlo VaR, and CVaR compare on the momentum-only pooled book's daily pnl, and where do parametric and historical VaR diverge most?

## Scope Note
"All strategies" overstates it. Three of the four candidates (PC2 Carry Regime, Momentum w/ ML Regime, OU Half-Life Mean Reversion) were discarded pre-validation, and the fourth (Volatility Regime Breakout/Mean-Reversion) failed Section 10 on its reversion leg (`day48_two_leg_validation.md`). Only the momentum leg reached validation, reused unchanged here as `momentum_only_pooled_book.md`. This is VaR/CVaR on that one series, not a cross-strategy comparison.

## Methodology 
`day53_var_cvar_report.py` reuses the momentum pipeline unchanged (`momentum_signal`, `compute_composite_regime_score_walkforward`, `classify_regime`, `regime_gated_pnl`), pooled across all three pairs, development window only (2011-2023, lockbox excluded), same pooling convention as Day 50/52. `var_historical`, `var_parametric`, `var_monte_carlo`, and `cvar` (`src/analysis/portfolio.py`) run on that daily pnl at 95% and 99% confidence. Monte Carlo: 100,000 simulations, seed 28.

## Findings
n=1,472, daily mu=0.000041, daily sigma=0.004221, ann_factor=210.27.

| Confidence | VaR historical | VaR parametric | VaR Monte Carlo | CVaR | Divergence (hist − param) |
|---|---|---|---|---|---|
| 95% | 0.006492 | 0.006903 | 0.006859 | 0.010188 | −5.9% |
| 99% | 0.012517 | 0.009780 | 0.009735 | 0.016861 | +28.0% |

## Interpretation
The two confidence levels disagree in direction, which is the actual finding. At 95%, historical VaR sits 5.9% below parametric — if anything, the normal assumption is conservative there. At 99% it flips hard: historical exceeds parametric by 28%. That's the standard fat-tail signature. Excess kurtosis loads probability into the extreme tail while leaving the 95th percentile, still close to the center of the distribution, roughly untouched. A normal-based risk model can look fine at 95% and still be badly wrong at 99%, which is also why Basel's FRTB sizes capital off 97.5%+ rather than 95%.

Day 4 found excess kurtosis in all three pairs (EURUSD 3.04, USDJPY 4.99, GBPUSD 28.35; Student-t df 3.55-5.59), and Day 37 found their absolute returns reject white noise at every Ljung-Box lag. Both point toward fatter tails and more persistent volatility than `var_parametric`'s Gaussian assumption allows, consistent with the 99% divergence above.

`var_monte_carlo` tracks `var_parametric`, not `var_historical`, because as implemented it draws from a fitted normal distribution and inherits the same assumption. CVaR exceeds historical VaR at both confidence levels, which follows directly from averaging the tail beyond the VaR threshold.

## Alternative Explanations
The pooled series is regime-gated (only turbulent days are nonzero), so its tails reflect turbulent-regime dynamics specifically, not the pairs' unconditional distributions cited from Day 4/37. The comparison above is directional, not a like-for-like test on the same series.

## Next Steps
- Run a kurtosis/Student-t fit directly on `momentum_pnl` instead of inferring fat tails from the underlying pairs.
- `var_monte_carlo` currently just restates `var_parametric` by simulation. Sampling from a fitted Student-t or a block-bootstrap draw (Day 37) would make it a genuinely separate estimate.
- These figures are on the unsized signal, not a Kelly-sized position (Day 52). A capital-scaled VaR/CVaR is what would actually inform a risk limit.
