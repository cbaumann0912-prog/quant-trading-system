# Strategy Specification — FX Momentum with ML Regime Filter

**Strategy name:** FX Momentum with ML Regime Filter
**Date drafted:** Day 42 (2026-07-07)
**Status:** **DISCARDED** — Test A (base momentum effect) failed after correcting a window-alignment bug; Claim B (regime-dependence) never reached, moot

---

## 1. Hypothesis

**Hypothesis:** Currencies exhibiting recent positive (negative) returns tend to continue rising (falling) over a subsequent period — a persistent anomaly (Menkhoff et al.: up to 10% p.a. spread between winners and losers) not explained by standard risk factors. Momentum's predictive power is regime-dependent: it works better in trending macro environments and tends to fail or reverse violently around regime transitions or crowded-trade unwinds. An ML classifier gates exposure based on predicted regime state (trending vs. choppy).

**What would falsify this hypothesis?**

| Test | Question | Result |
|------|----------|--------|
| Test A — Base momentum effect | Is Spearman IC between trailing 26-day return and forward 5-day return significantly positive? (permutation test, tested via two independent methodologies) | **FAILED** — all 3 pairs, both methods |
| Test B — Regime-dependence | Does an ML regime filter improve momentum's predictive power in trending vs. choppy states? | **NOT RUN — moot**, no base effect to condition |

Test A failing was pre-registered as fatal to the base momentum claim, which in turn makes Claim B untestable in any meaningful sense — a regime filter cannot rescue a signal that does not exist unconditionally, and searching for regime-conditional effects in the absence of an unconditional one is a known path to manufacturing spurious findings from noise.

---

## 2. Economic Rationale

**Why should this edge exist?**
Framed by Menkhoff et al. as "limits to arbitrage" — exploiting the momentum anomaly exposes traders to risks (crash risk, funding risk, unpredictable reversals) not captured by simple covariance-based risk measures, which is why professional capital hasn't arbitraged it away. **This project's implementation did not find evidence of the base effect in the tested construction (Section 1); see Section 10 for the full record, including a construction bug that initially produced a spurious positive result.**

**Who are the natural counterparties?** Not explicitly stated in the source write-up.

**Why hasn't this been arbitraged away already?**
Limits to arbitrage — crash risk, funding risk, and unpredictable reversal risk are not captured by simple covariance-based risk measures.

**What known macro/structural regime does this depend on?**
Momentum tends to work better in trending macro environments and tends to fail or reverse violently around regime transitions or crowded-trade unwinds. This was the explicit motivation for the ML regime gate — the classifier was meant to capture exactly the condition under which momentum's underlying mechanism (slow information diffusion, underreaction) is actually operative versus not. **Moot given Test A's failure — there is no base mechanism for a regime filter to condition on.**

---

## 3. Data Required

**Instruments:** EUR/USD, GBP/USD, USD/JPY (all three).

**Data frequency:** Daily close, resampled from 1-minute OHLCV via `.resample('D').last()`.

**Lookback window required:** 26 trading days (1 month, per project's 26-days/month convention, standardized with OU Half-Life strategy work).

**Any external/exogenous data needed?** None used in Test A. Candidate ML classifier features for a regime filter (Claim B) were never specified or sourced, since that stage was never reached.

---

## 4. Signal Logic

**What was computed, step by step (Test A only; live signal never designed):**

1. Compute daily log returns: `log_returns_t = log(price_t / price_{t-1})`
2. Compute cumulative sum of log returns for index-based windowing: `cumsum_t = sum(log_returns_{1..t})`
3. Trailing signal: `trailing_return_t = cumsum_t - cumsum_{t-26}` (26-day cumulative log return)
4. Forward outcome: `forward_return_t = cumsum_{t+5} - cumsum_t` (5-day forward cumulative log return, zero overlap with trailing window)
5. Test A computed Spearman IC between trailing_return and forward_return across all valid rows

**Signal frequency:** Would recompute daily in a live signal; Test A itself was evaluated both as a non-overlapping weekly subsample (Method 1) and as full daily overlapping observations with block-based permutation (Method 2).

**Parameters and values used:**

| Parameter | Value | Justification |
|-----------|-------|----------------|
| Lookback | 26 trading days | 1 month, per project's 26-days/month convention |
| Holding period | 5 trading days | 1 week |
| Significance threshold | α = 0.05, permutation-based | Consistent with framework convention given confirmed fat tails/volatility clustering |

---

## 5. Entry Rule

**Never designed — moot given discard.** No live entry rule was built; Test A operated only on the trailing/forward return relationship, not on an executable signal.

---

## 6. Exit Rule

**Never designed — moot given discard.**

---

## 7. Position Sizing Rule

**Never designed — moot given discard.**

---

## 8. Risk Controls

**Never designed — moot given discard.**

---

## 9. Failure Conditions

**Realized.** Per the pre-registered falsification criterion (Section 1), Test A found no significant positive IC between trailing and forward returns after correcting a construction bug — condition met, strategy discarded.

---

## 10. Statistical Validation Plan — Executed Record

**In-sample testing — results:**

- **Test A** (base momentum effect): Spearman IC between 26-day trailing return and 5-day forward return, tested via two independent methodologies to check robustness to overlapping-window handling.
  - **Initial construction contained a window-alignment bug**: a `.shift(HOLDING-1)` operation was misapplied, causing the "forward return" window to overlap almost entirely with the trailing window (differing by only ~1 day) rather than being genuinely forward-looking. This produced a spuriously large IC (~0.29–0.33, p≈0.00000, all pairs, both methods) that was flagged as implausibly high relative to published FX momentum literature (typically much weaker effects) and investigated before being trusted.
  - **After correction** (explicit cumulative-sum index construction, verified by manual inspection of window date ranges to confirm zero overlap between trailing and forward windows):
    - Method 1 (non-overlapping weekly subsample, n≈946/pair): EUR/USD IC=-0.035 (p=0.281), GBP/USD IC=-0.047 (p=0.150), USD/JPY IC=-0.027 (p=0.410)
    - Method 2 (overlapping daily data, block permutation with block_size=31, n≈4,728/pair): EUR/USD IC=-0.029 (p=0.331), GBP/USD IC=-0.045 (p=0.116), USD/JPY IC=-0.006 (p=0.855)
  - **Both methods agree**: no significant positive IC in any pair. **Test A: FAILED.**

**Multiple-testing correction:** N/A — strategy discarded before reaching cross-candidate correction stage. This candidate's null result should still be counted when computing multiple-testing correction across the full three-candidate shortlist, per project rules.

**Deflated Sharpe Ratio:** Not computed — no return-generating signal was built.

**Confidence intervals:** Not computed for this strategy — Test A used permutation-based significance testing directly, no bootstrap CI was needed at this stage.

**Out-of-sample / walk-forward validation:** Not reached.

**Leakage check:** The *corrected* trailing/forward return construction has zero window overlap, confirmed via manual date-range inspection (Section 4). The *initial, buggy* construction had a de facto leakage-like flaw — the "forward" window improperly reused most of the trailing window's own days, which is functionally similar to a lookahead/overlap leakage bug even though it arose from an indexing error rather than a genuine future-information leak. Worth flagging as a general lesson: an implausibly strong initial result (IC ~0.30, well above literature norms) was the signal that prompted the check that caught this.

**Transaction costs / breakeven:** Not reached.

---

## 11. Open Questions / Known Gaps

- Strategy is discarded; Sections 5–9 were never completed, consistent with not investing further design effort in a strategy whose base premise failed.
- Claim B (ML regime-dependence) was never tested — this was a deliberate methodological choice (testing a regime filter on a non-existent base effect risks manufacturing a spurious regime-conditional finding from noise), not an oversight.
- The initial construction bug (window-alignment error producing spurious IC ~0.30) is a useful cautionary case for future strategy work: an unusually strong result relative to published literature benchmarks should trigger verification before being trusted, not be accepted as a good outcome.
- If momentum is revisited in a future 90-day cycle or as a new candidate, the corrected cumsum-based windowing construction (Section 4) is reusable, validated infrastructure — independent of this specific strategy's failed result.

---

## 12. Judgmental Adjustments to Statistical Inputs

**Adjustment:** None applied. No statistically-derived input was ever adjusted based on forward-looking judgment, since no return-generating signal was built before the strategy was discarded.

**Reasoning:** N/A.