# Research note — session-transition lead-lag check (10-pair universe)

## Question
Does one FX session's return (Asian/London/New York) predict another's, same-day or next-day, across the 10-pair universe?

## Method
10-pair universe, dev window 2011-2023. `src/features/sessions.py` splits raw 1-minute bars into non-overlapping, DST-aware, open-to-open session blocks: Asian [Asian open, London open), London [London open, New York open), New York [New York open, next Asian open). Six transitions per the Day 51 scope: A→L, A→U, L→U (same day), L→ndA, U→ndA, U→ndL (next day).

## Two methodological corrections
The first version defined sessions by each market's own quoted hours, which let London's afternoon and New York's morning share bars during their real overlap. Same-day L→U came back at a spurious r=0.60. Rebuilt as non-overlapping open-to-open blocks, verified by a test confirming no shared bar or gap between sessions.

Pooling all 10 pairs by concatenation also pseudo-replicates each trading day 10 times; the pairs are cross-sectionally correlated, so the effective sample size is far below the raw n. Switched to a cross-pair average (one observation per trading day) as the primary test, with per-pair breakdown as a secondary check.

## Findings
Cross-pair-average Pearson and Spearman correlation, full-session returns:

| Transition | Pearson r | p | Spearman rho | p |
|---|---|---|---|---|
| A → L | -0.036 | 0.040 | 0.006 | 0.711 |
| A → U | 0.004 | 0.836 | 0.022 | 0.208 |
| L → U | 0.061 | 0.0004 | 0.019 | 0.261 |
| L → ndA | -0.027 | 0.170 | 0.026 | 0.174 |
| U → ndA | -0.032 | 0.062 | -0.022 | 0.209 |
| U → ndL | -0.001 | 0.960 | 0.013 | 0.460 |

n = 2,689-3,372. Every transition fails on at least one of Pearson/Spearman; A→L and L→U pass on Pearson but are flatly null on Spearman.

Re-ran with first-hour variants (predictor and/or target restricted to the first 60 minutes of a session, via `load_session_returns_with_first_hour`) — 24 more specifications, same method. None pass both tests with matching sign.

Two results looked promising before these corrections and are worth naming directly. L→ndA's naive pooled significance was almost entirely two pairs, USDCHF (r=-0.207) and EURCHF (r=-0.214); excluding them collapses it to r=0.006, p=0.36. U→ndL's naive significance was pure n-inflation from pooling — corrected, r=-0.001, p=0.96.

## Why a passing p-value here still doesn't count
Two reasons. Pearson is sensitive to a few large-magnitude days; Spearman is rank-based and robust to them. A Pearson pass with a Spearman fail, true of every row above, is the signature of outlier days driving the number rather than a real relationship, confirmed directly for L→ndA by the CHF exclusion. Second, multiple comparisons: 30 specifications were tested in total. At alpha=0.05, chance alone predicts 1-2 false positives, which is roughly what showed up, and none of them survive Spearman or a Bonferroni-consistent threshold (about 0.002 for the 24-test batch).

## Verdict
No evidence that one session predicts another, in any framing tried. Closed as dead.

## Next steps
- CHF-subset recheck for U→ndA is untested but low priority: it was never significant even including CHF.
- `src/features/sessions.py` is built and tested, reusable.
- If a real signal turns up on a future pass: trial #6, with its own written spec, before any implementation.
