# Paper Outline

*Day 61. Scope decision: null-results and methodology paper, per Framework Map line 220. Not a strategy paper.*

**Working title:** Underpowered by Design: Six Pre-Registered Null Results in Systematic FX Research

**Target length:** 9,000–11,000 words body, excluding appendices.

**Governing rule for this outline:** no section may report a number that does not exist in the repo today. Where a section needs a number that has not been computed, it is marked `GAP` and either gets written up from an existing script or gets cut. Nothing gets re-run to make the paper look better.

---

## 1. Introduction — Day 62 (~1,200 words)

**Job:** state the question as a binary and give away the result.

- Opening: a systematic research program with a written protocol, six hypotheses, zero survivors, holdout never opened.
- The question: when six pre-registered strategies fail, is that evidence about the strategies or about the design of the study that tested them?
- Answer, stated up front: mostly the latter. Five of six failed at effect sizes the sample could not resolve.
- The four findings, one sentence each, with forward references.
- Positioning against prior work: Harvey/Liu/Zhu on multiple testing, López de Prado on backtest overfitting, Bailey/López de Prado on deflated Sharpe, Lo (2002) on Sharpe annualization under autocorrelation. The gap this fills: those papers describe the failure modes; this one reports a complete pre-registered program where the failure modes were instrumented and measured as they occurred.
- Explicit statement that this is a single-author student project on retail vendor data, and what that does and does not limit.

`GAP` — citations not yet collected. Harvey/Liu/Zhu and Bailey/López de Prado need to be pulled and read before this section is written, not after.

---

## 2. Data — Day 61 draft exists, needs rescoping (~1,500 words)

**Status:** `paper/sections/data.md` exists but is scoped to **three pairs ending 2022-12-31**. The program uses **ten pairs through 2023-12-29**. This section must be rewritten, not edited.

Keep from the existing draft:
- Source, coverage, gap analysis, OHLC validity checks
- The three hard limitations: volume identically zero, no bid/ask recorded, DST timestamp duplication (300 per pair)
- Empirical annualization argument and the 312.3 vs 260.6 vs hardcoded-252 comparison — this is a genuinely good passage, keep it nearly intact
- Embargo and directional purging logic
- Overlapping forward returns and the n/h effective-observation point
- Lockbox definition and the CLI refusal guard

Rewrite:
- Extend row counts and coverage to all ten pairs
- Reconcile the evaluation window: the draft names 2015–2022, the program uses 2011–2023. Pick one, state why.
- The intraday book's 259.44 obs/yr vs the daily 312.30 — both appear in the project, the distinction needs one paragraph

Add:
- The universe design error, forward-referenced to §6. Ten pairs at 13 years was chosen for breadth; effective breadth realized at 2.34. Nine pairs at 21.8 years was available and was not considered.

---

## 3. Research Design — Day 63 (~1,400 words)

**Job:** describe the protocol as a protocol, before any result. This section is what makes the nulls interpretable.

- Pre-registration format: Sections 1–12, fixed before first test. Hypothesis, economic rationale, entry, exit, sizing, risk controls, explicit death conditions.
- The seven-stage pipeline: pre-registration → power gate → signal construction → walk-forward → statistical validation → cost gate → verdict.
- Cumulative trial counting, including failed tests. BH applied across the roster, not within a strategy.
- Lockbox protocol and why it stayed sealed through six failures.
- **Honest process failures, stated here rather than buried:** the Day 30 spec deadline slipped 11 days; Strategy #1 never got a spec at all and was evaluated through daily audits; Strategy #4b's spec is still marked draft on sizing and risk. Report these as departures, not as a rewritten history.
- The power gate became binding only at Day 57 — after five of six candidates had already been tested. Say so plainly. It is the single largest process change the nulls produced and it arrived late.

---

## 4. Strategy Roster and Verdicts — Day 64 (~2,400 words, six subsections)

**Job:** one subsection per hypothesis. Equal structural treatment. No strategy gets extra space for having a prettier number.

Each subsection, same five-part shape:
hypothesis → economic rationale → pre-registered criterion → outcome → what killed it

| # | Strategy | Evidence on hand | Action |
|---|---|---|---|
| 1 | PC2 Carry Regime | 5 docs, richest in project | Write from existing |
| 2 | Momentum + ML Regime | **`.py` only** | `GAP` — write audit `.md` from `momentum_ml_regime_falsification.py` |
| 3 | OU Half-Life Mean Reversion | **`.py` only** | `GAP` — write audit `.md` from `ou_halflife_mean_reversion_falsification.py` |
| 4 | Vol Regime Two-Leg | 6 docs | Write from existing |
| 4b | Momentum-Only Pooled Book | 2 docs | Write from existing; note spec still draft |
| 5 | Month-End FX Flow | 1 doc | Write from existing; note H2/H3 never run |
| 6 | Intraday Overshoot Reversal | 2 docs | Write from existing |

**Do not build a common performance table.** The six criteria are heterogeneous by design — IC tests, nonlinearity operationalizations, two-leg and three-way interaction regressions, a Sharpe/permutation battery. Forcing a shared metric would require re-running each on a basis it was not pre-registered under. The comparison table reports *criterion and outcome*, not a common Sharpe.

`GAP` — two audit write-ups (#2, #3). Estimated one day. No new computation required; the scripts exist and ran.

---

## 5. Four Findings — Days 65–66 (~2,800 words, the paper's contribution)

**Job:** this is why the paper exists. Each finding is a cross-cutting methodological result, anchored to the strategy that surfaced it.

### 5.1 A controlled leakage demonstration
Anchor: Strategy #4. Refitting the volatility-regime classifier inside each training window rather than full-sample changes 44–79% of out-of-sample labels, on identical data and identical code. All three pairs then trip the strategy's own pre-registered 40% turbulent-share decay condition. The leaky fit was systematically undercounting turbulence in one direction, not scattering noise.

`FIX` — the flip percentage appears three ways in the repo: README says 34–56%; `vol_regime_classifier_refit_stability.md` gives agreement 49.41/55.74/20.52% (flips 44–79%); `momentum_book_invalidation.md` gives agreement 32.2–60.8% (flips 39–68%). The last two are the 3-pair and 10-pair runs respectively and are both legitimate. The README figure matches neither and must be corrected.

### 5.2 Aggregation manufacturing significance
Anchor: Strategy #6. Ten signals at ρ = 0.364, effective breadth 2.34 of 10. Pooled book Sharpe 1.043, block-bootstrap p < 0.00001, while no individual pair clears p = 0.05 and largest |t| = 1.69. Independent trades would imply 0.432 — nearly the entire book result is the correlation adjustment, not directional edge. The per-day vs per-trade permutation split (p = 0.0010 vs p = 0.114) is the same fact restated: preserve cross-pair structure and the book looks decisive; strip it and the direction does not clear.

**Framing note:** #6 is not the near-miss. Its Sharpe is the artifact. Write it as the cleanest instance of the finding, never as the strategy that almost worked.

`FIX` — README attributes this finding to `momentum_book_invalidation.md`. The numbers live only in `intraday_overshoot_section10_validation.md`. Correct the citation.

### 5.3 Criteria that do not test their hypothesis
Two instances, both pre-registered, both read as rigorous.

- **Sign untested.** Day 48's rule asked whether an interaction coefficient differed from zero and never asked its sign. The strategy passed with b1 + b3 = −0.0022 — momentum predicting −0.22% over 26 days in precisely the regime it trades. The audit recorded sign consistency across fits as *support* for the pass without noting that the consistent sign was backwards.
- **No power to discriminate.** Day 57's R1 required monotone ordering across three threshold cells whose confidence intervals all span zero. Deciding gap: +0.050 bp against SE 1.014, t = 0.05. Four trades of 1,588 flip the ordering. Monotone in 2 of 11 tradeable years.

The general claim: a criterion with no power to discriminate is worse than no criterion, because it emits a PASS that reads as evidence. Both failures were caught only on re-examination, after the criteria had already been used to issue verdicts.

Consequence adopted: a direction requirement was added to the verdict rule, and Strategy #5 became its first use — failing on sign with every other criterion passed.

### 5.4 The power ceiling is arithmetic
Anchor: program-wide. t ≈ SR × √years, so 13 years requires SR ≈ 0.55 for t = 2. Intraday sampling does not relax this; the binding constraint is calendar span, not observation count. Five of six candidates were untestable at this sample size rather than wrong — a distinction that changes what to do next.

The fix was available the whole time. Dropping NZD/USD (which gates the common start at 2005-08) moves the start to 2002-03, buys 8.8 years, and takes the required Sharpe from 0.555 to 0.428 — 2,147 active days against a 1,871 requirement. The study would have been adequately powered.

The trade is breadth against span, and the realized breadth of 2.34 says the marginal pair was worth far less than the marginal year.

---

## 6. Limitations — Day 66 (~900 words)

Ordered by how much they threaten the conclusions.

1. **Universe design error**, quantified per §5.4 — the central limitation, stated as a specific error and not a vague note about sample size.
2. **No observed spreads.** Every cost figure rests on assumed ECN quotes. Strategy #6 cleared its gate at 4.54× on assumed costs; entry slippage remains unmeasured and 1-minute closes are not tradeable prices.
3. **Serial dependence.** Ljung-Box p = 0.0004 on #6's book returns invalidates the naive Sharpe t-stat. `GAP` — the block bootstrap block length must be stated and justified against the measured ACF. If the block is shorter than the dependence horizon, p < 0.00001 is inflated in the flattering direction.
4. **Single researcher, single data vendor, no replication.**
5. **Strategy #1 has no pre-registration**; #4b's spec is still draft on sizing and risk.
6. **Overlapping forward returns** reduce effective observations to roughly n/h.
7. **Retail vendor data quality**, and the market-structure break that any pre-2005 extension would have to assume across.

---

## 7. Conclusion — Day 67 (~600 words)

- Restate: six hypotheses, zero survivors, lockbox intact.
- The transferable claim is not "these six strategies do not work." It is that a validation protocol can pass a strategy running backwards, that a book-level p-value can be manufactured by correlation, and that most of these hypotheses were never resolvable on the sample chosen to test them.
- What changes for Strategy #7: power calculation before hypothesis, applied to the data window as well as the effect size. Nine pairs from 2002 as the starting design. Direction requirements on every coefficient criterion.
- What the sealed lockbox is worth: one clean out-of-sample shot, unspent after six failures.

---

## Appendices

- A. The six pre-registrations, verbatim and unedited
- B. Framework module inventory and test coverage (30 modules, 372 tests)
- C. What is not implemented from scratch — the statsmodels/scipy table from the README, kept verbatim
- D. Reproduction: Docker image, data mount, `pytest`

---

## Day mapping

| Day | Section | Notes |
|---|---|---|
| 61 | Title, abstract, outline, repo setup | this document |
| 62 | §1 Introduction | requires citations pulled first |
| 63 | §3 Research Design | §2 rescope runs in parallel |
| 64 | §4 Roster and Verdicts | blocked on the two `GAP` audits |
| 65 | §5.1–5.2 | |
| 66 | §5.3–5.4, §6 Limitations | |
| 67 | §7 Conclusion, full revision pass | |
| 68 | Second revision pass | |

## Blocking work before drafting past Day 63

1. Write audit `.md` for Strategy #2 from `momentum_ml_regime_falsification.py`
2. Write audit `.md` for Strategy #3 from `ou_halflife_mean_reversion_falsification.py`
3. Correct the README label-flip percentage and the §5.2 misattribution
4. Rescope `paper/sections/data.md` to ten pairs
5. Resolve the four open items in `paper/sections/signal_construction.md` — standardisation window, the n = 1,638 vs 934 discrepancy, absent exit logic, threshold selection as a trial
6. State and justify the bootstrap block length against measured ACF

None of these requires re-running a closed hypothesis.
