# Strategy Specification Template
<!-- Copy this file once per shortlist candidate. Save as
     research/strategy_specs/strategy_{n}_spec.md per Framework Map. -->

**Strategy name:**
**Date drafted:**
**Status:** Draft / Under Review / Locked

---

## 1. Hypothesis

State the core hypothesis in one or two sentences: what pattern or relationship
do you believe exists, and what would have to be true about markets for it to
be tradable?

**Hypothesis:**

**What would falsify this hypothesis?** (Be specific — what result, if you saw
it, would make you discard this strategy rather than tweak its parameters?)

---

## 2. Economic Rationale

This is the section your project instructions flag as most likely to be
challenged. A statistical pattern without an economic story is far more
likely to be a false discovery (multiple-testing artifact, regime-specific
fluke, or data-mined coincidence) than a tradable edge.

**Why should this edge exist?** (Who is on the other side of this trade, and
why would they keep losing to you? What structural, behavioral, or
informational reason explains why the market hasn't already arbitraged this
away?)

**Who are the natural counterparties?** (Hedgers, central banks, retail flow,
other systematic strategies, etc. — be concrete.)

**Why hasn't this been arbitraged away already?** (Capacity constraints,
transaction costs, capital requirements, behavioral biases that persist,
structural features of FX market microstructure, etc.)

**What known macro/structural regime does this depend on?** (E.g., does it
require a particular interest rate regime, risk-on/risk-off pattern, central
bank policy stance? What happens to the edge if that regime changes?)

---

## 3. Data Required

**Instruments:** (Which of EUR/USD, GBP/USD, USD/JPY — all three? Subset?)

**Data frequency:** (Daily close, intraday, etc. — and why that frequency is
appropriate for this strategy's holding period.)

**Lookback window required:** (How much history does the signal need before
it can generate a valid reading — e.g., rolling correlation window, cointegration
estimation window?)

**Any external/exogenous data needed?** (Interest rate differentials, VIX,
macro calendar events, etc. — if none, say so explicitly rather than leaving
blank.)

---

## 4. Signal Logic

**What is actually computed, step by step, to produce a signal?** (Write this
as if explaining to someone who will implement it without you in the room —
exact statistic, exact transformation, exact threshold logic.)

**Signal frequency:** (Does this recompute daily? Continuously? On a rolling
basis?)

**Parameters used and their values:** (List every tunable parameter — z-score
threshold, lookback length, smoothing window, etc. — with the value(s) you
intend to use and brief justification for why that value, not just "this
seemed reasonable.")

---

## 5. Entry Rule

**Exact condition that triggers a new position.** (Numeric threshold,
direction logic — be unambiguous enough that two people implementing this
independently would produce the same trades.)

**Does entry depend on confirmation from a second signal/filter?** (If yes,
specify; if no, say so explicitly.)

---

## 6. Exit Rule

**Exact condition that closes a position.** (Profit target, stop-loss,
time-based exit, mean-reversion target reached, signal decay — specify all
that apply.)

**What happens if the position never reaches the exit condition?** (Is there
a maximum holding period? What if the underlying relationship breaks down
mid-trade — e.g., a cointegration relationship that stops holding?)

---

## 7. Position Sizing Rule

**How is position size determined?** (Fixed fraction, volatility-scaled,
Kelly-derived, signal-strength-proportional — specify the actual formula or
rule, not just the category.)

**Maximum position size / leverage per trade:** (Tie this to your account's
actual leverage caps where relevant — per-pair caps you've already
established, e.g., 50:1.)

**How does sizing interact across the three pairs if multiple signals fire
simultaneously?** (Capital allocation rule across concurrent positions.)

---

## 8. Risk Controls

**Stop-loss logic:** (Per-trade, portfolio-level, or both?)

**Maximum drawdown tolerance:** (At what point does this strategy get paused
or re-evaluated, independent of any single trade's stop?)

**Correlation/concentration limits:** (Given all three pairs share a PC1 USD
factor, does this strategy have any safeguard against unintentionally
concentrated directional USD exposure across positions?)

**Maximum capital allocation to this strategy** (as a fraction of total
managed capital, separate from per-trade leverage).

---

## 9. Failure Conditions

**Under what conditions would you conclude this strategy has stopped
working** (not just had a bad month)? Be specific and pre-commit to this now,
before you have live results that could bias your judgment.

**What observable signal would indicate regime change has broken the
underlying relationship?** (E.g., if cointegration-based: rolling hedge
ratio instability beyond some threshold; if regime-based: PC loading shift
beyond some threshold.)

---

## 10. Statistical Validation Plan

This section defines exactly how you will confirm (or reject) this strategy
using tools already built in the framework, before it goes live.

**In-sample testing:**
- Which test(s) will establish initial statistical significance?
  (t-test on mean return, permutation test, etc.)
- What p-value / significance threshold will you require?

**Multiple-testing correction:**
- Since this is one of (up to) 3 candidates being tested, what correction
  will be applied across all candidates jointly (Bonferroni, Benjamini-Hochberg)?
  Per your project rules, this must be applied to the full candidate set,
  not just the eventual winner.

**Deflated Sharpe Ratio:**
- What is `N` (number of independent trials/configurations) for DSR purposes,
  given how many parameter combinations or variants were explored before
  arriving at this spec?

**Confidence intervals:**
- Will you compute a bootstrap CI on the Sharpe ratio? What block length
  (if block bootstrap) is appropriate given this strategy's autocorrelation
  structure?

**Out-of-sample / walk-forward validation:**
- What is the planned train/test split structure once WalkForwardValidator
  exists? (In-sample fit window, out-of-sample test window, rolling vs.
  expanding.)
- What result, specifically, would constitute walk-forward failure (e.g.,
  Sharpe degradation beyond some threshold from in-sample to out-of-sample)?

**Leakage check:**
- Does any part of this signal use information that would not have been
  available at the time of the trade (e.g., full-sample PCA loadings,
  full-sample cointegration parameters)? If so, how will this be corrected
  for walk-forward (rolling re-estimation)?

**Transaction costs / breakeven:**
- At what estimated transaction cost does this strategy's edge disappear?
  (Even a rough estimate now is useful; refine later with the actual
  transaction cost model.)

---

## 11. Open Questions / Known Gaps

(Anything you're aware you haven't resolved yet — be honest here. An
incomplete spec that flags its own gaps is more useful than one that hides
them.)

---

## 12. Judgmental Adjustments to Statistical Inputs

Per the Markowitz (1952) principle on combining statistical estimation with
judgment: document any place where you are adjusting a statistically-derived
input (expected return, covariance, correlation, hedge ratio, etc.) based on
forward-looking reasoning the historical sample can't capture. State the
adjustment and the reasoning explicitly — never blend judgment into the
statistical step silently.

**Adjustment:**
**Reasoning:**