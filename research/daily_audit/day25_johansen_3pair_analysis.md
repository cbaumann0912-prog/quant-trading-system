# Day 25 Research Audit — Johansen Cointegration, 3-Pair System

## 1. Question
Does a Johansen cointegration test on EUR/USD, GBP/USD, and USD/JPY jointly reveal cointegrating relationships that pairwise Engle-Granger testing could miss? How many cointegrating relationships exist among the three series simultaneously, and what does that imply for treating the three pairs as independent signals versus a single shared risk factor?

## 2. Why It Matters
The strategy universe is fixed at three pairs. If EUR/USD, GBP/USD, and USD/JPY share a common stochastic trend, position sizing or signal construction that treats them as independent sources of edge is implicitly over-trading one underlying factor under three different labels. Day 24 found borderline pairwise cointegration. A system-level test is the correct next check before concluding the strategy universe behaves as three distinct opportunities, since pairwise tests can miss relationships that only appear when all three series are considered jointly.

## 3. Methodology
- Data: daily close, EUR/USD, GBP/USD, USD/JPY, resampled from 1-minute OHLCV, inner-joined on shared trading days.
- Test: `johansen_test` (`src/signals/cointegration.py`), det_order=0 (constant in cointegrating equation), k_ar_diff=1.
- Both trace test and max eigenvalue test evaluated at 90/95/99% critical values.
- No parameter tuning against output; lag order and deterministic assumption fixed before running, not adjusted post hoc.

## 4. Assumptions
- det_order=0 assumes no deterministic trend in the cointegrating relationship — reasonable for FX log-price levels absent a structural drift argument.
- k_ar_diff=1 is a default, not validated via lag-order selection (AIC/BIC on the VAR).
- Daily resampling assumes same-day close alignment is meaningful; consistent with Day 4–24 convention.
- Full-sample test assumes the cointegrating relationship, if any, would be stable across the entire 2011–2023 window. Day 19 PCA found a structural break in PC1 variance explained post-2020 — full-sample Johansen could mask a relationship that holds in one sub-period only.

## 5. Findings
**Trace test:**

| Rank hypothesis | Stat | 90% crit | 95% crit | 99% crit |
|---|---|---|---|---|
| r = 0 | 15.7966 | 27.0669 | 29.7961 | 35.4628 |
| r ≤ 1 | 5.4836 | 13.4294 | 15.4943 | 19.9349 |
| r ≤ 2 | 0.6086 | 2.7055 | 3.8415 | 6.6349 |

**Max eigenvalue test:**

| Rank hypothesis | Stat | 90% crit | 95% crit | 99% crit |
|---|---|---|---|---|
| r = 0 | 10.3130 | 18.8928 | 21.1314 | 25.8650 |
| r ≤ 1 | 4.8750 | 12.2971 | 14.2639 | 18.5200 |
| r ≤ 2 | 0.6086 | 2.7055 | 3.8415 | 6.6349 |

**Eigenvalues:** [0.002541, 0.001202, 0.000150]

**rank_trace: 0**
**rank_max_eig: 0**

Both tests agree, and agree decisively. The trace statistic at r = 0 (15.7966) sits below even the 90% critical value (27.0669), not just the 95% threshold — the same is true for the max eigenvalue statistic (10.3130 vs. 18.8928 at 90%). Across the full 13-year sample, there is no detectable cointegrating relationship among EUR/USD, GBP/USD, and USD/JPY, jointly or in any sub-combination.

Because rank = 0, the eigenvectors returned by the test are not economically interpretable as cointegrating relationships. They remain artifacts of the rank-reduction problem, not tradable spreads.

## 6. Alternative Explanations
Day 24's pairwise Engle-Granger test found a borderline result for EUR/USD~GBP/USD that did not survive Benjamini-Hochberg correction. A system-level test failing this decisively, where a pairwise test was at least borderline, strengthens Day 24's conclusion: it rules out the possibility that pairwise testing was underpowered to detect a relationship only visible at the system level.

The structural break identified in the Day 19 PCA audit (PC1 variance explained drifting from ~52% to ~71% post-2020) is a live alternative explanation worth naming: a full-sample test assumes parameter stability across the whole window. If a cointegrating relationship existed only pre-2020 or only post-2020, averaging over the full 13 years could dilute it below detection. This audit does not test that.

## 7. Implications for Portfolio Construction
A rank-0 result across the full system means there is no shared long-run equilibrium tying EUR/USD, GBP/USD, and USD/JPY together that a stat-arb spread could exploit. This does not mean the three pairs are uncorrelated short-term — the Day 19 PCA finding (PC1 USD-strength factor, 59.6% of variance) already established they share a common short-run driver. Cointegration and correlation are distinct: PC1 describes contemporaneous co-movement, not a stationary long-run relationship whose deviations mean-revert. The PCA result and today's Johansen result are consistent with each other, not in tension — the pairs move together (PCA) without that co-movement resolving to a tradable equilibrium (Johansen).

For candidate 1 (Multi-Pair Forex Stat Arb), this is now two independent findings against the cointegration-based version of the idea: Day 24 pairwise and Day 25 system-level. The PC2 carry regime signal (candidate 3) is unaffected by this finding, since it exploits a contemporaneous factor loading rather than a long-run equilibrium spread.

## 8. Next Steps
- Validate k_ar_diff via lag-order selection criterion (AIC/BIC on the underlying VAR) before treating rank = 0 as fully final — current result uses the default lag, not a selected one.
- Consider a sub-period Johansen test (pre/post the Day 19 structural break point) as a follow-up, time permitting, to address the alternative
  explanation in Section 6.
- Carry forward into  strategy spec decision: cointegration-based Multi-Pair Forex Stat Arb (candidate 1) should move toward caution/invalid given two independent null results, absent a specific sub-period argument.