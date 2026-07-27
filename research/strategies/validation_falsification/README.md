# Validation and Falsification

Every test that bore on whether one of the six pre-registered strategies was valid. Pre-registrations live in `../specs/`. Framework module builds and general market studies stay in `research/daily_audit/` and `research/applied_analysis/`.

All six hypotheses were falsified. The 2024–2026 lockbox was never opened.

## Strategy 1 — PC2 Carry Regime
Discarded Day 41 after three independent tests returned null.

| File | Role |
|---|---|
| `pc2_carry_regime_factor_analysis.py` | Factor construction from the Day 19 PCA scores |
| `pc2_carry_regime_permutation_test.md` / `.py` | Test 1: unconditional predictive content. Pooled IC −0.0017, p = 0.951 |
| `pc2_carry_regime_ic_ir_breadth.md` / `.py` | IC/IR and Fundamental Law breadth estimate |
| `pc2_carry_regime_breadth_followup.md` | Correction replacing a naive run-count breadth with a correlation-adjusted estimate |
| `pc2_carry_regime_conditional_ic.md` / `.py` | Test 3: conditional predictability by volatility regime. Null; discarded at condition number 2.27e10 |

## Strategy 2 — Momentum with ML Regime
| File | Role |
|---|---|
| `momentum_ml_regime_falsification.py` | Discarded Day 42. A window-alignment bug was corrected and no base momentum IC survived |

## Strategy 3 — OU Half-Life Mean Reversion
| File | Role |
|---|---|
| `ou_halflife_mean_reversion_falsification.py` | Discarded Day 42. Tests 2, 2b and 2c all failed |

## Strategy 4 — Volatility Regime Breakout / Mean-Reversion
Failed Section 10 on Day 48. Its momentum leg was redeployed as the momentum-only pooled book and later invalidated.

| File | Role |
|---|---|
| `vol_regime_composite_threshold_selection.md` / `.py` | Regime threshold and PCA-vs-equal weighting choice, fixed before testing |
| `vol_regime_pipeline_test.py` | First end-to-end signal run |
| `vol_regime_walkforward_diagnostic.md` / `.py` | Preliminary per-leg OOS IC and Sharpe |
| `vol_regime_classifier_refit_stability.md` / `.py` | Per-window classifier refit replacing a leaky full-sample fit. 34–56% of regime labels flip once the leak is removed |
| `vol_regime_two_leg_section10_validation.md` / `.py` | Section 10 battery. Reversion leg null at p = 0.563 once the lockbox is excluded |
| `vol_regime_signal_report_design.md` | SignalReport construction and multiple-testing summary |
| `vol_regime_signal_report_audit.md` | Per-leg BH verdict. Momentum passes, reversion fails, strategy fails |
| `vol_regime_signal_report_pipeline.py` | Orchestration for the above |
| `momentum_book_regime_conditional_robustness.md` / `.py` | Robustness of the momentum-only book under a GARCH-based regime definition |
| `momentum_book_invalidation.md` | Closes the momentum-only book. Robustness-1 null at 10 pairs, and the original 3-pair interaction ran the wrong direction |

## Strategy 5 — Month-End FX Rebalancing Flow
| File | Role |
|---|---|
| `month_end_fx_flow_validation.py` | H1 primary test |
| `month_end_fx_flow_h1_result.md` | Closed. Significant, wrong sign |

## Strategy 6 — Intraday Overshoot Reversal
| File | Role |
|---|---|
| `intraday_overshoot_signal_definition.md` | Signal definition and look-ahead audit |
| `intraday_overshoot_section10_validation.py` | Full Section 10 battery, rebuilt from raw bars on every run |
| `intraday_overshoot_section10_validation.md` | Verdict: FAIL on threshold monotonicity, per-trade permutation, and H2 |

## Reading order
For the methodology story rather than the chronology, start with `intraday_overshoot_section10_validation.md`, then `momentum_book_invalidation.md`, then `vol_regime_classifier_refit_stability.md`. Those three carry the project's four transferable findings: a controlled leakage demonstration, portfolio aggregation manufacturing significance, two validation criteria with no discriminating power, and the sample-span power ceiling.

## Notes
Scripts run from the repository root and locate it via `parents[3]`, adjusted when these files moved out of `research/applied_analysis/`. `momentum_ml_regime_falsification.py` also carried a `SEED = 28S` typo from Day 42 that made it unrunnable; corrected during the move.

Deliberately left in place: `day39_cv_leakage_comparison` and `day44_pipeline_test` (framework methodology, not strategy verdicts), `day50` / `day52` / `day53` (Monte Carlo drawdown, Kelly, VaR/CVaR — module demonstrations that happen to use strategy returns as input), and `day57_transaction_cost_breakeven` (the transaction-cost module's own deliverable, though it also supplies strategy 6's cost gate).
