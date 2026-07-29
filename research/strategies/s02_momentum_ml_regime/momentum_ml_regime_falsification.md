# Strategy 2 Falsification — Momentum with ML Regime Filter

## Question
Test A of `spec.md`: is the Spearman IC between a trailing 26-day return and a forward 5-day return significantly positive on EUR/USD, GBP/USD and USD/JPY? The spec pre-registered Test A as fatal — Claim B, the ML regime filter, was never to be tested unless a base effect existed.

## Verdict
**FAIL, all three pairs, both methodologies.** Six tests, zero significant, and five of six point-estimates carry the wrong sign for a momentum hypothesis. Claim B is moot: a regime filter cannot condition an effect that is absent unconditionally, and searching for one anyway is the standard route to manufacturing a finding from noise.

This closes `spec.md` as DISCARDED. Lockbox never opened.

## Methodology
`momentum_ml_regime_falsification.py`, 2011-01-02 to 2023-12-31, three pairs, daily closes resampled from raw 1-minute bars by `resample("D").last()`. Signal and outcome are built on the cumulative-log-return index so the trailing and forward windows share zero days:

    trailing_return_t = cumsum_t − cumsum_{t−26}
    forward_return_t  = cumsum_{t+5} − cumsum_t

Section 1 verifies the alignment directly: `forward_start` falls exactly one day after `trailing_end` for every row. This check exists because an earlier construction overlapped the two windows and produced a spurious IC near +0.30. That bug, not the market, was the original result.

Two independent tests of the same hypothesis:

- **Method 1** — subsample every 5th row so no two observations share a forward window, then a 10,000-iteration permutation test shuffling forward returns.
- **Method 2** — keep all overlapping daily rows and permute in contiguous blocks of 31 (`LOOKBACK + HOLDING`), preserving the serial dependence the overlap induces.

`SEED = 28`, `N_PERMUTATIONS = 10000`, `ALPHA = 0.05`. A pass required `p < 0.05` **and** `IC > 0`.

*Provenance: figures below come from a faithful replay of the script — `compute_signal_outcome` verbatim, identical seed, identical RNG consumption order — with the daily closes read from a cache built by the same load-and-resample chain, because seven repeated 300 MB CSV reads exceeded the available run window. Re-running the script directly should reproduce these exactly and is worth doing once.*

## Findings

Method 1 — non-overlapping subsample, n = 805:

| Pair | IC | permutation p | Verdict |
|---|---|---|---|
| EUR/USD | −0.0393 | 0.2617 | FAIL |
| GBP/USD | −0.0471 | 0.1868 | FAIL |
| USD/JPY | −0.0167 | 0.6361 | FAIL |

Method 2 — all overlapping rows, block permutation, n ≈ 4,024:

| Pair | IC | permutation p | Verdict |
|---|---|---|---|
| EUR/USD | −0.0292 | 0.3501 | FAIL |
| GBP/USD | −0.0453 | 0.1490 | FAIL |
| USD/JPY | +0.0017 | 0.9599 | FAIL |

The two methods agree in sign on EUR/USD and GBP/USD. USD/JPY flips from −0.0167 to +0.0017, both indistinguishable from zero.

## The overlap correction is doing real work

Comparing each permutation p against the naive Spearman p implied by SE(ρ) = 1/√(n−1):

| Pair | n | IC | naive t | naive p | permutation p | inflation |
|---|---|---|---|---|---|---|
| EUR/USD | 805 | −0.0393 | −1.11 | 0.2651 | 0.2617 | 1.0× |
| GBP/USD | 805 | −0.0471 | −1.34 | 0.1817 | 0.1868 | 1.0× |
| USD/JPY | 805 | −0.0167 | −0.47 | 0.6358 | 0.6361 | 1.0× |
| EUR/USD | 4,024 | −0.0292 | −1.85 | 0.0640 | 0.3501 | 5.5× |
| GBP/USD | 4,023 | −0.0453 | −2.87 | **0.0041** | 0.1490 | **36.7×** |
| USD/JPY | 4,024 | +0.0017 | +0.11 | 0.9141 | 0.9599 | 1.1× |

On the non-overlapping subsample the two agree to three decimal places. On the overlapping sample the naive p understates by up to 37×.

GBP/USD is the case worth keeping. Treating 4,023 overlapping observations as independent gives t = −2.87 and p = 0.0041 — conventionally significant. The block permutation, which preserves the dependence induced by consecutive 26-day windows sharing 25 days, returns p = 0.149. The apparent significance is entirely an artifact of counting each of ~130 independent windows about 31 times.

The non-overlapping arm functions as a control: where the dependence is removed by construction, the naive and permutation p-values coincide. That rules out the permutation machinery itself as the source of the gap.

## Interpretation
There is no base momentum effect in this construction at this sample size, and the point estimates lean negative rather than merely null. Five of six ICs are negative; the largest |IC| is 0.0471 against the 0.0705 that Method 1 would need for t = 2.

The negative lean is not isolated. Strategy 4b's interaction regression later found momentum running backwards in the regime it traded (b1 + b3 = −0.0022), and Day 49's independently computed IC agreed at −0.094. Two strategies, different constructions, different test statistics, same direction. Whether that reflects a genuine short-horizon reversal in these pairs or a shared property of the sample is not resolvable here.

The window-alignment bug is the more useful lesson. The original construction overlapped trailing and forward windows and returned IC ≈ +0.30 — an effect size roughly six times anything observed after the fix. A result that large in daily FX should have been treated as evidence of a coding error before it was treated as evidence of an anomaly.

## Alternative Explanations
- **Wrong horizon.** 26-day trailing and 5-day forward is one cell of a grid. Menkhoff et al. document currency momentum over 1–12 month formation and holding periods, mostly cross-sectional rather than time-series. This tests a time-series version at one horizon on three pairs and does not falsify the cross-sectional result.
- **Wrong universe.** Three USD-quoted majors are the most liquid and most arbitraged pairs available. Menkhoff et al.'s spread is driven substantially by higher-yield and less liquid currencies absent here.
- **Underpowered.** At n = 805 the detectable |IC| at t = 2 is 0.0705. An effect of 0.03 — plausible for daily FX — is invisible in this sample regardless of method. This is the same calendar-span ceiling that binds five of the six strategies.

The first two mean this result should not be read as "FX momentum does not exist." It is a null on one construction, and the spec claimed no more than that.

## What Was Not Done
- **Claim B never tested.** No ML classifier was built, no regime features specified or sourced. The spec pre-committed to abandoning it on a Test A failure, and that commitment was honored.
- **No deflated Sharpe.** No return series was ever constructed; Test A operated on the IC directly.
- **No walk-forward.** The strategy was discarded at the in-sample hypothesis-testing stage, before any parameter was fit.
- **Sign not flipped and re-tested.** Five negative ICs invite trading the reverse. Doing so would use the same data to generate and confirm a hypothesis, which is the failure mode pre-registration exists to prevent.

## Next Steps
- Closed. No retuning, no horizon search, no universe expansion on this hypothesis.
- The 37× naive-versus-permutation gap on GBP/USD belongs in the paper alongside the portfolio-aggregation finding. Both are the same mechanism — dependence treated as independence — appearing at different levels of the stack, and this one comes with a built-in control arm.
- Signal construction here is inline and untested, unlike `src/signals/intraday_overshoot.py`. Recorded as a reproducibility gap rather than retrofitted, since retrofitting a closed strategy risks changing the numbers this document reports.
