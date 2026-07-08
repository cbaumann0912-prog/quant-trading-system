# Strategy Specification — Volatility Regime Breakout/Mean-Reversion

**Strategy name:** Volatility Regime Breakout/Mean-Reversion
**Date drafted:** Day 42 (2026-07-07)
**Status:** Draft — hypothesis and economic rationale locked; remaining sections undesigned

---

## 1. Hypothesis

**Hypothesis:** Volatility and correlation structure in FX exhibit genuine regime shifts (Ang & Bekaert) rather than continuous scaling of a single process. The optimal trading rule differs by regime — mean-reversion in calm regimes, momentum/breakout in turbulent regimes. An ML classifier identifies regime state using rolling volatility, rate differentials, and a vol index proxy.

**What would falsify this hypothesis?** Not yet defined.

---

## 2. Economic Rationale

**Why should this edge exist?** Ang & Bekaert's regime-shift work formalizes that volatility and correlation structure genuinely change state — not just a "high vol version of the same process." The edge, if it exists, comes from correctly identifying the regime switch before the crowd fully adjusts to the new state.

**Who are the natural counterparties?** Not explicitly stated in the source write-up.

**Why hasn't this been arbitraged away already?** Not explicitly stated beyond the general framing that regime identification is difficult and the edge depends on being early relative to "the crowd."

**What known macro/structural regime does this depend on?** By construction, this strategy depends on there being two genuinely distinct regimes (calm/turbulent) with different optimal trading rules in each — per Ang & Bekaert's core claim (state change, not just volatility scaling of the same process).

## 3. Data Required

**Instruments:** Not yet explicitly specified — presumably EUR/USD, GBP/USD, USD/JPY per framework scope, but not stated for this strategy specifically.

**Data frequency:** Not yet defined.

**Lookback window required:** Not yet defined.

**Any external/exogenous data needed?** Three candidate classifier features explicitly named: rolling volatility, rate differentials, and a vol index proxy. None of these have been sourced, defined precisely, or confirmed available yet.

---

## 4. Signal Logic

**What is computed, step by step:** Not yet defined. Design concept only: an ML classifier determines regime state (calm vs. turbulent), and the trading rule itself switches — mean-reversion logic in the calm regime, momentum/breakout logic in the turbulent regime. No specific statistics, thresholds, or classifier design have been specified.

**Signal frequency:** Not yet defined.

**Parameters used and their values:** Not yet defined.

---

## 5. Entry Rule

**Exact condition:** Not yet defined.

**Does entry depend on confirmation from a second signal/filter?** Yes, by design — the regime classifier itself determines which of two different rule sets (mean-reversion vs. momentum/breakout) is active. Specific classifier output/threshold has not been defined.

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

**What observable signal would indicate regime change has broken the underlying relationship?** Not yet defined. Notably, this strategy's core premise already involves detecting regime change as its primary mechanism, so this failure-condition question is about the *classifier itself* breaking down, which has not been addressed.

---

## 10. Statistical Validation Plan

Not yet defined. No tests have been designed or run for this strategy.

---

## 11. Open Questions / Known Gaps

- Full signal logic, entry/exit rules, sizing, and risk controls are entirely undesigned (Sections 3–9).
- ML classifier's specific architecture, training features (beyond the three named candidates), and regime-labeling scheme have not been specified.
- The two rule sets themselves (mean-reversion logic for calm regime, momentum/breakout logic for turbulent regime) have not been designed — only conceptually described.
- Falsification criteria (Section 1) have not been written.
- Statistical validation plan (Section 10) has not been started.
- Data requirements (Section 3) — three candidate features named (rolling vol, rate differentials, vol index proxy) but not sourced or precisely defined.

---

## 12. Judgmental Adjustments to Statistical Inputs

**Adjustment:** None — no statistical work has been done on this strategy yet.

**Reasoning:** N/A.