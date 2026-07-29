# PC2 Effective Breadth — Follow-On Note (raised Day 40, deferred)

**Status:** Exploratory, not scheduled. Not part of the Day 40 deliverable. Revisit at a
later date (rolling PCA re-estimation window) or later as a standalone audit — do not let
this displace scheduled curriculum days.

## Origin

Day 40 covered IC/IR and the Fundamental Law (`IR ≈ IC × sqrt(BR)`). Applying this to the
PC2 Carry Regime Signal raised a real question: pooled IC ≈ 0 (p = 0.951) from Day 38, but
positive-signal subset IC = 0.017 (p = 0.296) and negative-signal subset IC = 0.057
(p = 0.056) differ. Before any IR number is meaningful for this signal, breadth (N) has to
be defined — and PC2's breadth is not an obvious day-count, because the signal is
autocorrelated in time (regime persistence), not just correlated cross-sectionally.

## What was worked out

- Breadth here is not "one bet per trading day" by default. If the PC2 signal stays on one
  side of zero for extended stretches (regime persistence), consecutive daily readings are
  not independent bets — they're the same information restated.
- The correct tool for this is autocorrelation of the **signal series itself**
  (`corr(x_t, x_{t-k})`) — not cross-pair return correlation, and not `ou_half_life`
  (which measures mean-reversion of a cointegration spread, a different object).
- `rolling_correlation` (Day 5) and `plot_acf_pacf` (Day 22) are the existing tools; no new
  math module is required.
- The estimated autocorrelation `ρ` plugs into `BR_eff = N / (1 + (N-1)ρ)` from today's
  IC/IR notes to get an effective breadth number in place of raw day-count.

## Open questions, not yet resolved

1. **Pooled vs. regime-conditional IR.** Does it make sense to compute a single IR using
   pooled IC and pooled BR_eff, or to split both IC and breadth by regime and report
   regime-conditional IR separately? The Day 38 IC split already exists; a matching breadth
   split does not.
2. **P-hacking risk.** Any regime-conditional breadth/IR analysis is being proposed *after*
   seeing that the pooled result was null and the split-sample results looked more
   promising. If this path is pursued, the post-hoc nature of the regime split must be
   disclosed in the paper limitations section (put off for a later date), regardless of what the analysis shows.
3. **Regime boundary definition.** Regime-conditional breadth requires defining regimes as
   contiguous spans (not just per-day sign of the signal), which needs an explicit rule for
   what counts as "still the same regime" vs. noise crossing zero. Not yet defined.
4. **Threshold arbitrariness.** Any "signal decorrelates by lag k" cutoff needs a principled
   basis (e.g., ±2/√n white-noise band) rather than an arbitrary round number.

## Artifacts already produced

- `research/applied_analysis/day40_pc2_breadth_estimation.py` — draft diagnostic script,
  reconstructs the PC2 signal series and computes lag autocorrelation / candidate `BR_eff`
  values. Contains unverified assumptions about `plot_acf_pacf`'s return signature — needs
  correction against the actual function before it's trusted. Regime-conditional breadth is
  explicitly not implemented in this draft.

## Next steps when resumed

- Verify/fix the script against actual `plot_acf_pacf` and `rolling_correlation` signatures.
- Decide on a regime-boundary definition before attempting regime-conditional breadth.
- Decide pooled-vs-split IR question, with the p-hacking disclosure question settled in
  parallel, not after the fact.