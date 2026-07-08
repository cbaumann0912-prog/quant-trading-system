# Strategy Specification — OU Half-Life Mean-Reversion (Z-Score Threshold)

**Strategy name:** OU Half-Life Mean-Reversion (Z-Score Threshold, PPP-Motivated)
**Date drafted:** Day 42 (2026-07-07)
**Status:** **DISCARDED** — Test 1 passed; Tests 2, 2b, 2c (nonlinearity claim, three independent operationalizations) all failed

---

## 1. Hypothesis

**Hypothesis:** The deviation of EUR/USD, GBP/USD, and USD/JPY's price from its rolling moving average, normalized by rolling volatility (z-score), mean-reverts over time. Large-magnitude z-score deviations are more reliable entry signals than small deviations, because small deviations are statistically indistinguishable from noise around a slowly-drifting equilibrium, while large deviations more plausibly reflect a temporary dislocation likely to correct. MA window, volatility window, and z-threshold are determined via purged-CV grid search on in-sample data (Section 10); working parameters used for hypothesis testing (100-day MA, 100-day vol window, |z| ≥ 1.5 split) were never carried forward to live-signal optimization, since the underlying hypothesis was falsified before that stage.

**What would falsify this hypothesis?**

| Test | Question | Result |
|------|----------|--------|
| Test 1 — Base existence | Does the z-score deviation series show statistically significant mean-reversion? (ADF rejects unit root, KPSS fails to reject stationarity) | **PASSED** — all 3 pairs |
| Test 2 — Nonlinearity (reversion speed) | Is mean reversion time for large-\|z\| excursions (peak ≥ 1.5) statistically distinguishable from small-\|z\| excursions? (permutation test on the difference in mean reversion time) | **FAILED** — all 3 pairs |
| Test 2b — Nonlinearity (forward return magnitude) | Is mean 5-day signed forward return from peak significantly larger for large-\|z\| excursions than small-\|z\| excursions? (permutation test on difference in means) | **FAILED** — all 3 pairs (2 of 3 wrong-signed) |
| Test 2c — Nonlinearity (forward return rank) | Is there a significant positive Spearman IC between peak \|z\| magnitude and signed 5-day forward return? (permutation test on IC) | **FAILED** — all 3 pairs, closest to zero of any variant |

Any one of these tests failing was pre-registered as fatal to the strategy as specified. All three nonlinearity operationalizations failed. Per standing research-integrity rules, this is treated as a closed, discarded result — not retuned or retried further.

---

## 2. Economic Rationale

**Why should this edge exist?**
Deviations from equilibrium exchange rate levels are constrained by transaction costs and capital requirements on international arbitrage — small deviations aren't worth correcting given those costs, but larger dislocations eventually attract capital that pulls price back. This implementation targeted a related but distinct claim: even without modeling true PPP fair value, the *degree* of statistical dislocation from a smoothed price anchor should predict reversion behavior, because large deviations more plausibly reflect genuine order-flow imbalance (forced hedging, temporary liquidity shortfalls) than small deviations, which are more likely routine noise around a slow-moving equilibrium. **This claim was tested three ways and did not hold in any of them (Section 1).**

**Who are the natural counterparties?**
Hedgers and corporates transacting at prevailing rates regardless of short-term deviation (inelastic flow), other systematic mean-reversion strategies competing for the same edge, and momentum-driven flow that pushes price further from equilibrium in the short run (the source of the dislocations this strategy waits out).

**Why hasn't this been arbitraged away already?**
Capacity constraints and the difficulty of confidently distinguishing "large deviation about to revert" from "start of a genuine trend/regime shift" in real time — this was precisely the discrimination problem Tests 2/2b/2c were designed to validate. Transaction costs on frequent small-deviation trading make pure noise-trading unprofitable, which was part of why this strategy explicitly avoided acting on small deviations. **Given the test results, an equally plausible reading is that this discrimination problem is not actually solvable at this magnitude scale using this z-score construction — i.e., there may be no exploitable difference between "large dislocation" and "ordinary noise" for these three pairs at this timescale.**

**What known macro/structural regime does this depend on?**
Requires a regime where price genuinely reverts to a moving-average-defined equilibrium rather than trending persistently (e.g., sustained rate-differential-driven currency trends, as in strong dollar cycles, would violate this). No explicit regime filter was ever built into this strategy — moot now given the strategy's discard, but worth noting as a possible confound: without a regime filter, the excursion sample (Section 4) pools observations across whatever mix of trending and mean-reverting regimes occurred in this data, which could partially explain why the magnitude-conditioning signal didn't emerge cleanly.

---

## 3. Data Required

**Instruments:** EUR/USD, GBP/USD, USD/JPY (all three; framework does not support USD/CHF).

**Data frequency:** Daily close. Resampled from 1-minute OHLCV via `.resample('D').last()`. Justification: MA/vol-window construction and OU half-life estimates (~31–35 days) operate on a multi-week timescale, making daily resolution appropriate; intraday noise would not meaningfully improve signal quality at this holding-period scale.

**Lookback window required:** Minimum ~100 trading days before a valid MA/vol-window reading is available (MA_WINDOW = VOL_WINDOW = 100, working values used for hypothesis testing; never advanced to optimization given the discard).

**Any external/exogenous data needed?** None used. True PPP-based fair value (CPI differentials) was considered and explicitly rejected in favor of a moving-average anchor due to frequency mismatch (monthly CPI vs. daily FX) and added data-acquisition burden not justified for this implementation.

---

## 4. Signal Logic

**What was computed, step by step (for hypothesis-testing purposes; never advanced to a live signal):**

1. Compute rolling moving average of daily close price: `MA_t = mean(price, window=100)`
2. Compute deviation: `deviation_t = price_t - MA_t`
3. Compute rolling volatility of the deviation series: `vol_t = std(deviation, window=100)`
4. Compute z-score: `z_t = deviation_t / vol_t`
5. Track excursions: an excursion begins when `|z_t| ≥ 1.0`, continues while same-sign, tracking a running same-sign peak; a peak-segment ends when magnitude reverts by 1.0 z-unit from the current peak (fixed-absolute reversion rule), or a new peak is set
6. Tests 2/2b/2c each computed a different outcome variable (reversion time, 5-day forward return, IC) conditional on this excursion structure

**Signal frequency:** Recomputed daily, using only trailing data (rolling windows are one-sided/backward-looking — confirmed no lookahead in MA/vol construction).

**Parameters and working values used (hypothesis-testing only, never optimized further):**

| Parameter | Value | Justification |
|-----------|-------|----------------|
| MA window | 100 days | Working value; purged-CV grid search never run given discard |
| Vol window | 100 days | Matched to MA window after diagnosing that a shorter window (originally 20) mechanically undersized the denominator relative to the slow-moving deviation series, inflating z-scores (76–80% of observations above \|z\|=1.0, vs. expected ~32%). Resulting z-score std (~1.7) is elevated above the naive target of 1.0, accepted as an expected property of z-scoring an autocorrelated series (34-day half-life), not treated as a residual bug. |
| Entry threshold | \|z\| ≥ 1.0 | — |
| Large/small pool split | peak \|z\| ≥ 1.5 | Confirmed via threshold-separation script to produce a balanced split (36–38% of observations above, consistent across all three pairs) |
| Reversion-confirmation distance (X) | 1.0 z-unit | Chosen by convention (matches entry threshold scale) after an empirical diagnostic (varying X from 0.3–1.0) showed no plateau, only a monotonic decline attributable to a distance-to-target mechanical confound, not genuine signal |
| Censoring cap | 3× pair-specific OU half-life (~94–105 days) | Computed directly on the z-score series, replacing an earlier, materially longer (~1 year) half-life computed on a different, unrelated price-level deviation object from Day 26 |
| Forward horizon (Test 2b/2c) | 5 trading days | Chosen because it approximated Test 2's observed mean reversion time; **flagged as informed by already-observed data, not independently pre-registered** — logged honestly as a minor deviation from strict pre-registration discipline |

---

## 5. Entry Rule

**Not finalized — moot given discard.** The hypothesis-testing entry threshold (|z| ≥ 1.0) was used only for excursion detection in Tests 1–2c. No live trading entry rule was ever designed, since the strategy was discarded before reaching that stage.

---

## 6. Exit Rule

**Not finalized — moot given discard.** The excursion-end construction (Section 4) was retrospective/analytical only, never translated into a real-time exit rule.

---

## 7. Position Sizing Rule

**Never designed — moot given discard.**

---

## 8. Risk Controls

**Never designed — moot given discard.**

---

## 9. Failure Conditions

**Realized.** Per the pre-registered falsification criteria (Section 1), Tests 2, 2b, and 2c all failed to support the nonlinearity claim central to this strategy's differentiation from vanilla mean-reversion. Per standing rule, a failed pre-registered test is fatal and not subject to retuning — this condition was met and the strategy is discarded accordingly.

---

## 10. Statistical Validation Plan — Executed Record

**In-sample testing — results:**

- **Test 1** (base mean-reversion existence): ADF + KPSS on z-score deviation series. **PASSED** — both tests agree across all three pairs (ADF p ≈ 0.00000, stat -6.76 to -7.51; KPSS p ≈ 0.09–0.10).
- **Test 2** (nonlinearity — reversion time): permutation test, mean reversion time, large (peak ≥1.5) vs. small (peak <1.5) pool. **FAILED.**
  - EUR/USD: n=74, small mean=5.81d, large mean=5.72d, diff=+0.09d, p=0.957
  - GBP/USD: n=63, small mean=5.81d, large mean=8.92d, diff=-3.11d (wrong-signed), p=0.488
  - USD/JPY: n=69, small mean=6.88d, large mean=8.73d, diff=-1.84d (wrong-signed), p=0.649
- **Test 2b** (nonlinearity — forward return magnitude): permutation test, mean 5-day signed forward return from peak, large vs. small pool. **FAILED.**
  - EUR/USD: diff=-0.44% (wrong-signed), p=0.079
  - GBP/USD: diff=+0.11%, p=0.841
  - USD/JPY: diff=-0.15% (wrong-signed), p=0.565
- **Test 2c** (nonlinearity — forward return rank): Spearman IC, peak magnitude vs. signed 5-day forward return, permutation test. **FAILED.**
  - EUR/USD: IC=-0.157, p=0.184
  - GBP/USD: IC=-0.097, p=0.461
  - USD/JPY: IC=0.0003, p=0.997
- Significance threshold used throughout: α = 0.05, permutation-based (not parametric), given confirmed fat tails/volatility clustering in this data (Day 4, Day 37 findings).

**n_trials for this hypothesis family: 3** (Tests 2, 2b, 2c) — to be carried forward if any future strategy references this line of research or reuses this z-score construction.

**Multiple-testing correction:** N/A — strategy discarded before reaching cross-candidate correction stage. This candidate's null result should still be counted when computing multiple-testing correction across the full three-candidate shortlist, per project rules (correction applies to the full set tested, not just survivors).

**Deflated Sharpe Ratio:** Not computed — no return-generating signal was ever built; Tests 1–2c operated on the z-score/excursion structure directly, not on realized strategy returns.

**Confidence intervals:** Block bootstrap CI attempted on OU θ (block_size=20) — **result unreliable**, diagnosed as an artifact of block-bootstrap resampling introducing artificial discontinuities at block boundaries, inflating apparent θ in every replicate (bootstrap CI [0.061, 0.078] did not bracket point estimate 0.020). Not used as the basis for any conclusion. Left as an open technical note for any future series with similar autocorrelation structure — appropriate block length likely needs to be ≥ the series' own half-life (~35 days here), consistent with the Politis & White (2004) block-size-selection question already deferred to buffer days (Day 37 audit).

**Out-of-sample / walk-forward validation:** Not reached — strategy discarded at in-sample hypothesis-testing stage, before `WalkForwardValidator` (Day 45) would have been relevant.

**Leakage check:** The z-score construction itself (rolling MA, rolling vol) used only trailing data — no leakage in that component. The excursion-based tests (2/2b/2c) were inherently retrospective (require observing the future path to identify a peak and measure outcomes) — valid for hypothesis testing as executed, but this construction could never have been lifted directly into real-time signal logic without modification, which is now moot.

**Transaction costs / breakeven:** Not reached.

---

## 11. Open Questions / Known Gaps

- Strategy is discarded; Sections 5–8 were never completed and are not being completed retroactively, consistent with not investing further design effort in a falsified strategy.
- Three independent, pre-registered tests of the nonlinearity claim (reversion time, forward-return magnitude, forward-return rank) all failed — this is treated as a well-evidenced negative result, not an inconclusive one.
- Forward horizon (5 days, Tests 2b/2c) was chosen informed by Test 2's observed reversion times — a minor deviation from strict pre-registration, logged honestly rather than treated as a fully blind choice.
- No regime filter or ML component was ever incorporated (Tony's standing requirement) — moot given the discard, but would have been a gap had the strategy survived.
- Reusable groundwork from this candidate, in case referenced by future strategy work: the corrected z-score construction methodology (MA/vol window sizing diagnosis), the OU half-life-on-z-score-series approach to deriving a censoring cap, and the excursion-detection algorithm itself (Section 4) — all validated as sound infrastructure, independent of this specific strategy's failed hypothesis.
- Two earlier diagnostic dead-ends (MA-window-via-half-life-plateau search; X-confirmation-buffer search) were also negative results, documented in the Rolling Decision Log, and are relevant if similar window/threshold-selection procedures are attempted for other candidates — both failed for a related mechanical-confound reason (distance-to-target scaling dominating the signal being measured).

---

## 12. Judgmental Adjustments to Statistical Inputs

**Adjustment:** None applied. No statistically-derived input was ever adjusted based on forward-looking judgment, since no return-generating signal was built before the strategy was discarded.

**Reasoning:** N/A.