# Strategy Specification — FX Momentum with ML Regime Filter

**Strategy name:** FX Momentum with ML Regime Filter
**Date drafted:** Day 42 (2026-07-07)
**Status:** Draft — hypothesis and economic rationale locked; remaining sections undesigned

---

## 1. Hypothesis

**Hypothesis:** Currencies exhibiting recent positive (negative) returns tend to continue rising (falling) over a subsequent period — a persistent anomaly (Menkhoff et al.: up to 10% p.a. spread between winners and losers) not explained by standard risk factors. Momentum's predictive power is regime-dependent: it works better in trending macro environments and tends to fail or reverse violently around regime transitions or crowded-trade unwinds. An ML classifier gates exposure based on predicted regime state (trending vs. choppy).

**What would falsify this hypothesis?** Not yet defined.

---

## 2. Economic Rationale

**Why should this edge exist?** Framed by Menkhoff et al. as "limits to arbitrage" — exploiting the momentum anomaly exposes traders to risks (crash risk, funding risk, unpredictable reversals) not captured by simple covariance-based risk measures, which is why professional capital hasn't arbitraged it away.

**Who are the natural counterparties?** Not explicitly stated in the source write-up.

**Why hasn't this been arbitraged away already?** Limits to arbitrage — crash risk, funding risk, and unpredictable reversal risk are not captured by simple covariance-based risk measures.

**What known macro/structural regime does this depend on?** Momentum tends to work better in trending macro environments and tends to fail or reverse violently around regime transitions or crowded-trade unwinds. This is the explicit motivation for the ML regime gate — the classifier is meant to capture exactly the condition under which momentum's underlying mechanism (slow information diffusion, underreaction) is actually operative versus not.

## 3. Data Required

**Instruments:** Not yet explicitly specified — presumably EUR/USD, GBP/USD, USD/JPY per framework scope, but not stated for this strategy specifically.

**Data frequency:** Not yet defined.

**Lookback window required:** Not yet defined.

**Any external/exogenous data needed?** Not yet explicitly listed. The ML classifier is described as predicting "is this a trending or choppy regime," but the specific features that classifier would use have not been specified for this strategy (unlike the Volatility Regime Breakout write-up, which explicitly named candidate features).

---

## 4. Signal Logic

**What is computed, step by step:** Not yet defined. No signal construction has been designed.

**Signal frequency:** Not yet defined.

**Parameters used and their values:** Not yet defined.

---

## 5. Entry Rule

**Exact condition:** Not yet defined.

**Does entry depend on confirmation from a second signal/filter?** Yes, by design — an ML regime classifier gates whether momentum exposure is taken, based on predicting trending vs. choppy regime. The specific classifier design, features, and gating threshold have not been defined.

---

## 6. Exit Rule

**Exact condition:** Not yet defined.

**What happens if the position never reaches the exit condition?** Not yet defined.

---

## 7. Position Sizing Rule

**How is position size determined?** Not yet defined.

**Maximum position size / leverage per trade:** Not yet defined.

**How does sizing interact across the three pairs if multiple signals fire simultaneously?** Not yet defined.

---

## 8. Risk Controls

**Stop-loss logic:** Not yet defined.

**Maximum drawdown tolerance:** Not yet defined.

**Correlation/concentration limits:** Not yet defined.

**Maximum capital allocation to this strategy:** Not yet defined.

---

## 9. Failure Conditions

**Under what conditions would this strategy be concluded to have stopped working?** Not yet defined.

**What observable signal would indicate regime change has broken the underlying relationship?** Not yet defined.

---

## 10. Statistical Validation Plan

Not yet defined. No tests have been designed or run for this strategy.

---

## 11. Open Questions / Known Gaps

- Full signal logic, entry/exit rules, sizing, and risk controls are entirely undesigned (Sections 3–9).
- ML classifier's specific architecture, training features, and labeling scheme for "trending vs. choppy regime" have not been specified.
- Falsification criteria (Section 1) have not been written.
- Statistical validation plan (Section 10) has not been started.
- Data requirements (Section 3) have not been specified.

---

## 12. Judgmental Adjustments to Statistical Inputs

**Adjustment:** None — no statistical work has been done on this strategy yet.

**Reasoning:** N/A.