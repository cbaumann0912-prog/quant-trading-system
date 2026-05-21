# Distribution Assumption Audit — Day 1

**Pairs in scope:** EURUSD, GBPUSD, USDJPY  
**Files inspected:** analytics.py, robustness.py, monte_carlo.py, trailing.py, sim_costs.py

---

## Files Inspected

| # | File | Lines Inspected | Primary Concern |
|---|------|----------------|-----------------|
| 1 | analytics.py | 374–376, 553–556 | Sharpe ratio carries an implicit normality assumption |
| 2 | robustness.py | 67–68 | Same Sharpe pattern, copy-pasted |
| 3 | monte_carlo.py | 54–76, 147–151 | Bootstrap treats trades as i.i.d. |
| 4 | trailing.py | 276–290, 324–328 | ATR uses a simple rolling mean; position sizing assumes R is stable |
| 5 | sim_costs.py | 37–40 | Slippage drawn from a Gaussian |

---

## Assumptions Found

### analytics.py

| Location | Assumption | Validity |
|----------|------------|---------|
| Lines 375–376 | Sharpe annualized with `* np.sqrt(252)` — assumes i.i.d. normally distributed returns to justify square-root-of-time scaling | Wrong on two counts. Forex returns have fat tails and volatility clustering. And `sqrt(252)` is a daily-return convention — per-trade returns on a strategy with variable holding periods do not map cleanly to 252 periods per year. |
| Lines 553–556 | Same Sharpe calculation inside `print_report()` | Same problem, different location. |
| Line 235 | `avg_pnl_pct` stored as a simple mean, implicitly treating trade returns as symmetric | Descriptive only, not used for inference — but if the return distribution is skewed, this number will mislead anyone reading the output. |

### robustness.py

| Location | Assumption | Validity |
|----------|------------|---------|
| Lines 67–68 | `rets.mean() / rets.std() * np.sqrt(252)` — identical formula to analytics.py | Same violations. The deeper problem is that this Sharpe feeds directly into walk-forward PASS/FAIL verdicts at lines 195–206. If Sharpe is overstated because returns are fat-tailed, the verdicts are built on a bad number. |
| Line 195 | Walk-forward thresholds: `sharpe > 1.0` for strong pass, `sharpe > 0.5` for marginal | These cutoffs assume Sharpe is well-estimated, which it is not under fat tails. The thresholds could be too tight or too loose — hard to say without knowing the actual return distribution first. |

### monte_carlo.py

| Location | Assumption | Validity |
|----------|------------|---------|
| Lines 54–76 | Returns pulled from the trade log and treated as exchangeable samples | Bootstrap validity requires trades to be i.i.d. — no serial dependence. Volatility clustering means trades taken during a rough stretch share correlated outcomes. Resampling ignores that entirely. |
| Lines 147–151 | `np.cumprod(1.0 + sampled)` — compounding with no adjustment for serial correlation | If three losing trades happened in the same volatile week, the bootstrap scatters them across random paths and the drawdown distribution comes out too optimistic. |
| Line 167 | Code comment acknowledges: "Assumes trades are exchangeable (no serial dependence)" | Good that it is written down. It has not been tested or corrected for. |

### trailing.py

| Location | Assumption | Validity |
|----------|------------|---------|
| Lines 276–290 | ATR computed as `tr.rolling(ATR_PERIOD).mean()` — equal weight on all past bars | SMA adapts slowly. During a volatility spike, the ATR is still averaging in weeks of calm and will understate the current range. This directly affects stop placement across all three pairs. |
| Lines 324–328 | Position size = `(equity * risk_pct) / stop_distance` — assumes ATR-derived stop distance is a reliable 1R estimate | If ATR understates volatility during a regime shift, position size is too large at exactly the wrong moment. |
| Lines 288–289 | ATR fallback: if ATR is NaN or zero, default to `close * 0.001` | A hardcoded 0.1% of price with no distributional basis. Could be far too tight or too wide depending on the pair and session. |
| Line 83 | `MIN_ATR_RATIO = 0.0003` applied uniformly across all three pairs | USDJPY trades around 150, EURUSD around 1.25. The ratio normalizes for scale, but the 0.03% threshold itself was never calibrated pair-by-pair. |

### sim_costs.py

| Location | Assumption | Validity |
|----------|------------|---------|
| Lines 37–40 | Slippage drawn from `abs(normal(0, slippage_std))` — a half-normal distribution | Real slippage during news or session opens is right-skewed with occasional large gaps. The half-normal misses that tail. The 2x worst-case multiplier may not cover a genuinely bad fill. |
| `_SLIPPAGE_STD` defaults | USDJPY at 0.008 vs EURUSD at 0.00008 — a 100x difference | Correctly scaled for pip size. This one is fine. |

---

## Stylized Facts from Cont (2001) Relevant to This Framework

| Stylized Fact | Relevance |
|--------------|-----------|
| Fat tails / excess kurtosis | Directly undermines the Sharpe ratio in analytics.py and robustness.py. The `sqrt(252)` scaling is derived under normality and breaks down with fat-tailed returns. |
| No autocorrelation in raw returns | Partly supports the bootstrap — there is no linear dependence in the mean. This does not help with the volatility clustering problem. |
| Volatility clustering | The ATR SMA adapts slowly to volatility spikes. The bootstrap destroys temporal clustering. Position sizing can end up too large going into a high-vol regime. |
| Gain/loss asymmetry | Large losses are more common than large gains of the same magnitude. The half-normal slippage model misses the left-tail severity. |
| Aggregational Gaussianity | Returns approach normality at lower frequencies, so the Sharpe assumption gets less wrong as holding periods lengthen. The strategy's multi-hour trades are better behaved than 5m bar returns, though tail risk does not disappear. |

---

## Questions to Resolve

1. **What is the actual kurtosis of per-trade pnl_pct for each pair?**  
   Run `trade_df['pnl_pct'].kurtosis()` on historical output. This matters because excess kurtosis above 1 means the Sharpe ratio is materially off — at which point Sortino or Calmar would be more appropriate replacements.

2. **Is there serial correlation in trade outcomes?**  
   Run `pd.Series(trade_pnls).autocorr(lag=1)` through lag 5. This matters because significant autocorrelation means the bootstrap is underestimating drawdown. A block bootstrap that preserves temporal clusters would be the fix.

3. **Should ATR use EWM instead of SMA?**  
   Compare `tr.ewm(span=ATR_PERIOD).mean()` against the current `tr.rolling(ATR_PERIOD).mean()` across a volatile period like early 2020 or late 2022. The question is whether EWMA adapts fast enough to change stop placement in a way that matters for realized PnL.

4. **Is the Sharpe annualization factor right for this strategy?**  
   Count average trades per year per pair. If it is around 100, annualization should use `sqrt(100)`, not `sqrt(252)`. The current formula overstates Sharpe for any strategy that does not trade daily.

5. **Does MIN_ATR_RATIO need pair-specific calibration?**  
   Check what fraction of bars each pair has filtered at 0.0003. If USDJPY and EURUSD are filtered at very different rates, the threshold is not working equivalently across the portfolio — which means the strategy is effectively more active on one pair than the others for a non-intentional reason.