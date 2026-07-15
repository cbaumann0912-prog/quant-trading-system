# Strategy Specification: Volatility Regime Breakout/Mean-Reversion

**Date drafted:** Day 42 (2026-07-07), finalized Day 43
**Status:** Complete

## 1. Hypothesis
FX volatility and correlation structure go through genuine regime shifts (Ang & Bekaert, 2002), not continuous scaling of one process. The best trading rule differs by regime: mean-reversion when calm, momentum/breakout when turbulent. A 2-feature classifier (rolling realized volatility + interest rate differential) identifies which regime is active. The vol-index-proxy leg from the original 3-feature design was dropped: no free, historically complete, FX-specific vol index exists.

**Falsification criteria** (pre-registered Day 43):

1. Test statistic: conditional IC of the trading signal within the classifier-identified regime (must be conditional, not unconditional, since the hypothesis is regime-dependence itself).
2. Primary threshold: p < 0.05, uncorrected.
3. Reliability gate (from the Day 41 PC2 failure): interaction regression condition number < ~1e10 (VIF < 10), main effects mean-centered before the interaction term. This strategy needs two such regressions, one per leg (`signal x regime_indicator`), since it has two signals rather than PC2's one.
4. Robustness: primary test + both robustness checks (Section 10) must unanimously agree. Any single null kills the hypothesis.
5. Multiple-testing: final PASS/FAIL must survive Benjamini-Hochberg correction across all 5 strategies tested project-wide. Since the other 4 are already null, a real finding here needs p < 0.01 (BH critical value, rank 1 of 5) to survive.

## 2. Economic Rationale
If the edge exists, it comes from identifying a regime switch before the market fully adjusts. Natural counterparties and why this hasn't been arbitraged away are not addressed in the source write-up. Depends on there being two genuinely distinct regimes, per Ang & Bekaert's state-change claim.

## 3. Data Required
**Instruments:** EUR/USD, GBP/USD, USD/JPY, daily, via existing DataLoader.

**Rate differentials:** FRED OECD 3-month interbank rates (US: IR3TIB01USM156N, Euro Area: IR3TIB01EZM156N, UK: IR3TIB01GBM156N, Japan: IR3TIB01JPM156N), monthly, in `data/{region}_3m_interbank.csv`. Refresh via `fetch_rate_differentials()` in `src/framework/data_loader.py`.

`eurusd_rate_diff = r_EA - r_US`, `gbpusd_rate_diff = r_UK - r_US`, `usdjpy_rate_diff = r_US - r_JP`. Shifted forward 2 months before forward-filling to daily (OECD publication lag, ~6 weeks) to avoid look-ahead. Creates a 59-day warm-up gap; first valid value 2011-03-01.

Full-sample means are not centered near zero or symmetric across pairs (`eurusd_rate_diff` -1.00, `gbpusd_rate_diff` -0.16, `usdjpy_rate_diff` +1.49), which is why the regime threshold below is z-scored rather than an absolute cutoff.

## 4. Signal Logic
A simple rolling-threshold classifier, not a Hamilton-filter Markov-switching model (that's reserved for Days 55-56 per the roadmap). This tests a simplified proxy for Ang & Bekaert's regime concept, not their actual model.

1. **Regime features:** 78-day (1 quarter, `trading_days_per_year // 12 * 3`) rolling realized volatility of log returns, plus the rate differential (2-month-lagged, forward-filled).
2. **Regime combination:** both z-scored and combined via PCA (1st component), sign-normalized so volatility loading is positive. Must be refit inside each walk-forward window in production; the Day 43 script fits full-sample for threshold selection only. Full findings and numbers: `research/daily_audit/day43_regime_composite_threshold_analysis.md`.
3. **Regime classification (hard switch):** `|composite z| > 1.5` = turbulent, `< 1.0` = calm, `1.0-1.5` = deadzone (no trade). Threshold of 1.5 preserves test power (turbulent is 9.2-15.9% of observations vs. 3.4-5.0% at 2.0).
4. **Turbulent rule, time-series momentum** (Moskowitz, Ooi & Pedersen 2012): `signal_t = sign(P_t/P_{t-78} - 1)`.
5. **Calm rule, price z-score mean-reversion:** `z_price_t = (P_t - mean_26(P)) / std_26(P)`, entry `|z_price| > 2.0`. Threshold chosen from conditional-forward-return evidence (audit doc addendum), not rarity alone. GBP/USD and USD/JPY show a real, increasing reversion effect; EUR/USD shows none, a real finding, not something to explain away.

**Caveat:** thresholds above (1.0/1.5/2.0) were chosen from full-sample descriptive analysis, a mild form of look-ahead in the design process. The actual test of whether the strategy works is Section 10's out-of-sample validation.

**Parameters:**

| Parameter | Value |
|---|---|
| `trading_days_per_year` | 312 (FX); 252 for equities |
| Regime window / momentum lookback | 78 trading days |
| Regime thresholds | turbulent > 1.5, calm < 1.0, deadzone 1.0-1.5 |
| Mean-reversion window | 26 trading days |
| Mean-reversion entry threshold | \|price z\| > 2.0 |

## 5. Entry Rule
Regime classifier (Section 4) gates which rule is active; no trade in the deadzone.

**Momentum (turbulent):** `sign(P_t/P_{t-78} - 1)`, re-evaluated daily, no confirmation lag. Literal MOP convention.

**Mean-reversion (calm), 3-rung ladder:** each rung independent, all conditioned on `|composite z| < 1.0`.

| Rung | Trigger |
|---|---|
| 1 (initial) | `\|price z\| > 2.0` |
| 2 (add) | `\|price z\| > 2.5` |
| 3 (add, cap) | `\|price z\| > 3.0` |

Rung 1 is exceedance-triggered (from flat); rungs 2-3 are crossing-triggered (fire once, on first cross above that level, not every day price stays there). The ladder has no direct academic citation, Gatev, Goetzmann & Rouwenhorst (2006) use single-shot entry. This is a practitioner convention that adds 3 free parameters (rung count, spacing, cap) versus the single-threshold version. Sizing per rung is in Section 7.

## 6. Exit Rule
**Deadzone is not a forced-exit zone.** An open position rides through the deadzone; it only force-closes on a flip to the *opposite* regime (turbulent to calm, or vice versa).

**Momentum exit:** sign flip only (mirrors entry, no separate stop-loss at this layer).

**Mean-reversion exit, whichever hits first:**
1. Reversion to target band: `|price z| < 0.5`, closes all rungs together (not staged).
2. Time-stop: 26 trading days from the initial rung-1 entry, force-close regardless of price.

If a position rides through the deadzone and the regime never flips before the data ends, mark-to-market and close at the data boundary (bookkeeping, not a trading decision).

## 7. Position Sizing Rule
**Base method (both legs), ex-ante vol targeting:** `size_t = (target_vol / realized_vol_t) x base_capital`, target vol 40% annualized (MOP's own value, a normalization convention, not a capital recommendation). Uses the same 78-day rolling vol already computed for the classifier rather than MOP's literal EWMA estimate, for parameter parsimony.

**Ladder sizing:** equal weight per rung, `size_t / 3` each, so a fully-built ladder sums to the same `size_t` a momentum position would get at the same vol.

**Hard cap:** required in addition to vol-targeting (which can otherwise blow up as realized vol -> 0). Value set in Section 8.

**Cross-pair:** shared capital pool. If `n` pairs have live positions simultaneously, each position's `size_t` is additionally scaled by `1/n`, rescaled dynamically as positions open/close across pairs.

## 8. Risk Controls
- **Position-size cap:** 2x the vol-targeted size at this strategy's own historical median realized vol (per pair).
- **Max drawdown:** 25% from peak (strategy-level) triggers an automatic halt for manual review. Looser than a typical 15% single-sleeve default, to accommodate this strategy's regime-timing risk.
- **Concentration:** the Section 7 `1/n` scaling is treated as sufficient; no separate net-USD-exposure cap. Known blind spot (doesn't distinguish 3 independent bets from 3 correlated USD bets), accepted for now.
- **Capital allocation:** not set here, deferred to portfolio-construction time. No capital gets committed before Section 10 clears.

## 9. Failure Conditions
**Strategy decay:** realized live Sharpe falling materially below the backtest's deflated-Sharpe confidence band (Day 13 standard). Chosen over a pure IC re-test since it also catches execution/cost decay, not just statistical decay.

**Classifier decay:** live regime proportions drifting from the Day 43 baselines (turbulent ~9.2-15.9%, deadzone ~13.6-24.6%). A classifier producing, say, 40%+ turbulent days means the thresholds no longer match current conditions, a distinct failure from the trading rules themselves breaking.

## 10. Statistical Validation Plan
No separate full-sample screening stage. The primary test runs directly inside each `WalkForwardValidator` test fold, out-of-sample from the start.

**Primary test**, two interaction regressions, pooled across walk-forward out-of-sample folds:
- Momentum: `R_{t+26} = b0 + b1*momentum_signal_t + b2*turbulent_dummy_t + b3*(momentum_signal_t x turbulent_dummy_t) + eps_t`
- Reversion: `R_{t+26} = b0 + b1*price_z_t + b2*calm_dummy_t + b3*(price_z_t x calm_dummy_t) + eps_t`

Both mean-centered before the interaction term. `b3` is the term of interest. Forward horizon: 26 trading days, shared across both legs. Reliability gate: condition number < ~1e10 per leg.

**Robustness checks** (both must agree; either null kills that leg):
1. Alternate regime window: re-run with 156 trading days instead of 78.
2. Permutation test: shuffle regime-dummy labels (preserving base rates), rebuild the null for `b3`, 1000 permutations.

**Verdict:** each leg evaluated independently; strategy-level PASS requires both legs to pass. A momentum-only pass is a different strategy, not this one. Subject to project-wide BH correction (Section 1, item 5) before appearing as a PASS anywhere.

**Lockbox holdout:** a recent slice (e.g. 2024-2026) is reserved and never enters the walk-forward folds at all, development or robustness checks. It is opened once, only if this strategy passes everything above and becomes a genuine deployment candidate. This is a single-use test per hypothesis, not another round of tuning; if the lockbox result disagrees with the walk-forward verdict, that disagreement is reported as-is, not explained away.

## 11. Open Questions / Known Gaps
- PCA adds no real value over equal-weighting (kept for citability/consistency with PC2, not because it's doing work).
- The ladder (Section 5) has no academic citation; adds 3 free parameters versus single-threshold entry.
- Sizing's vol estimator deviates from MOP's literal EWMA convention (parsimony trade-off).
- Cross-pair concentration doesn't distinguish independent bets from correlated USD bets (accepted gap, Section 8).
- EUR/USD shows no reversion signal; Section 10's per-leg (not per-pair) verdict could mask a pair-level failure if EUR/USD alone fails.
- Threshold selection (1.0/1.5/2.0) has a look-ahead caveat: chosen from the same descriptive data the strategy will later be tested against.
- Dynamic `1/n` rescaling and per-window classifier refitting are not yet implemented; both are real work for SignalBuilder (Day 44+).

## 12. Judgmental Adjustments to Statistical Inputs
None of these come from a hypothesis test; each is a documented judgment call.

| Adjustment | Value | Basis |
|---|---|---|
| Position-size hard cap | 2x vol-targeted size at median vol | Risk convention, multiple not tested |
| Max drawdown halt | 25% from peak | Judgmental, accommodates regime-timing risk |
| Ladder rung count/spacing | 3 rungs, z=2.0/2.5/3.0, equal-weighted | Simplicity judgment |
| Cross-pair concentration limit | 1/n scaling only | Accepted blind spot |
| Mean-reversion exit target | \|z\| < 0.5 | Judgmental, trades citability for practicality |
| Fixed time-stop | 26 trading days | Reuses tested horizon, not a derived half-life |
| Shared forward-return horizon (Section 10) | 26 trading days, both legs | Comparability over leg-specific precision |