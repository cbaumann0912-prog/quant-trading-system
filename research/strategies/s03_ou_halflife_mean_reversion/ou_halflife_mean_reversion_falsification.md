# Strategy 3 Falsification — OU Half-Life Mean Reversion

## Question
Two claims from `spec.md`. **Test 1:** does the z-scored deviation of price from its rolling moving average mean-revert at all? **Test 2:** are large-magnitude deviations *more reliable* entry signals than small ones? The spec pre-registered the nonlinearity claim in three independent operationalizations and treated any single failure as fatal.

## Verdict
**Test 1 PASSED. Tests 2, 2b and 2c all FAILED.** The series mean-reverts; the deviation magnitude carries no usable information about how it reverts. Strategy closed, lockbox never opened.

Test 2b returned a *significant* result on EUR/USD at p = 0.0262 — running the opposite direction to the hypothesis.

## Methodology
2011-01-02 to 2023-12-31, EUR/USD, GBP/USD, USD/JPY, daily closes resampled from raw 1-minute bars.

Signal: `z = (price − MA_100) / rolling_std_100`, giving 3,857–3,858 observations per pair. OU parameters fit on the z-series; excursions defined as threshold crossings at |z| ≥ 1.0, pooled by magnitude at |z| = 1.5, and censored at 3× the fitted half-life so that unreverted excursions do not silently become infinite reversion times.

`SEED = 28`, `N_PERMUTATIONS = 10000`, `N_BOOTSTRAP = 1000`, `BLOCK_SIZE = 20`, `ALPHA = 0.05`.

Three operationalizations of the same nonlinearity claim:

- **Test 2** — mean reversion *time*, small pool vs large pool, permutation test
- **Test 2b** — mean 5-day forward *return*, same pools
- **Test 2c** — Spearman IC between |z| at entry and subsequent return

*Provenance, updated Day 72: the script was re-executed end to end with the daily closes rebuilt from the raw 1-minute bars, after confirming that the resampling path used returns a bit-identical daily series to the script's own `to_datetime` + `resample("D").last()` chain — identical index, maximum absolute difference 0.0. Every figure in Tests 1, 2, 2b and 2c below reproduced exactly, including the unreliable bootstrap interval. All analysis code is unmodified.*

## Findings

### Test 1 — base mean reversion: PASS

| Pair | ADF stat | ADF p | KPSS stat | KPSS p | OU θ | Half-life |
|---|---|---|---|---|---|---|
| EUR/USD | −6.1475 | 0.00000 | 0.1510 | 0.100 | 0.01953 | 35.5 d |
| GBP/USD | −6.1831 | 0.00000 | 0.1028 | 0.100 | 0.01966 | 35.3 d |
| USD/JPY | −6.8496 | 0.00000 | 0.4076 | 0.0739 | 0.02187 | 31.7 d |

ADF rejects a unit root and KPSS fails to reject stationarity on all three — the two tests agree, which is the stronger form of the conclusion since they carry opposite null hypotheses.

The block bootstrap CI on θ is unusable and is recorded as such: [0.0604, 0.0781] for EUR/USD against a point estimate of 0.0195. The interval excludes its own point estimate by a factor of three. Diagnosed as a block-boundary artifact — resampling 20-day blocks injects artificial discontinuities at every join, and an OU fit reads those jumps as fast reversion, inflating θ in every replica. The point estimate stands; the interval is discarded.

### Test 2 — reversion time: FAIL

| Pair | excursions | small pool | large pool | diff (small − large) | p |
|---|---|---|---|---|---|
| EUR/USD | 63 | n=17, 5.29 d | n=46, 7.68 d | −2.390 d | 0.6199 |
| GBP/USD | 54 | n=16, 12.05 d | n=38, 7.03 d | +5.023 d | 0.2679 |
| USD/JPY | 58 | n=21, 7.05 d | n=37, 7.05 d | −0.006 d | 1.0000 |

Signs disagree across pairs. USD/JPY's two pools differ by six thousandths of a day.

### Test 2b — 5-day forward return: FAIL, and EUR/USD is significant backwards

| Pair | n | small pool | large pool | diff (large − small) | p |
|---|---|---|---|---|---|
| EUR/USD | 62 | 1.6278% | 0.9785% | **−0.6493%** | **0.0262** |
| GBP/USD | 54 | 1.1387% | 1.3566% | +0.2179% | 0.7073 |
| USD/JPY | 58 | 1.2283% | 0.9822% | −0.2461% | 0.3874 |

The hypothesis predicts large deviations earn more. On EUR/USD they earn 0.65 percentage points *less*, and the permutation test calls that difference significant.

### Test 2c — IC between |z| and subsequent return: FAIL

| Pair | n | IC | p |
|---|---|---|---|
| EUR/USD | 62 | −0.2365 | 0.0643 |
| GBP/USD | 54 | −0.0674 | 0.6270 |
| USD/JPY | 58 | −0.0295 | 0.8280 |

All three negative. EUR/USD again approaches significance in the wrong direction.

## Interpretation

The two claims separate cleanly, and only the first survives. A z-scored deviation from a 100-day moving average is stationary and reverts with a half-life near a month. That is a statement about the construction as much as the market — differencing a series against its own trailing mean will tend to produce something stationary, and ADF rejecting a unit root on such a series is close to guaranteed.

The tradeable claim was the second one, and nothing supports it. Across nine tests the estimates disagree in sign, and where they are consistent they lean negative: larger deviations are followed by *smaller* subsequent returns. Test 2b on EUR/USD reaches conventional significance in that direction.

**This is the third strategy in the roster to produce a significant result pointing the wrong way.** Strategy 4b's interaction ran backwards while passing a significance-only criterion, and strategy 5's H1 was significant with an inverted sign. Here it appears at Day 42, earlier than either. The pattern is consistent enough that a direction requirement should have been in the verdict rule from the start rather than added after 4b exposed the gap.

The binding constraint is sample size. The tests operate on 54 to 63 excursions split into pools of roughly 17 and 45. A difference-in-means at those counts resolves only very large effects, and the honest reading is that Tests 2, 2b and 2c were underpowered before they were run — the same calendar-span ceiling that binds five of the six strategies. The nonlinearity claim was not so much falsified as untestable at this sample size, with the one significant result pointing away from the hypothesis.

## Defect in Section 1 — parameter selection ran at the wrong frequency

Section 1 searched for an MA-window plateau in half-life and reported none, concluding that "half-life scales mechanically with window." That conclusion was disregarded and `MA_WINDOW = 100` was adopted as a working value instead.

Reading the code, the search could not have found a plateau. `GRID_WINDOWS` is built as `months × 26` — 78 to 624 **trading days** — but is applied to `prices = df["Close"]` taken directly from the minute-indexed frame with no resampling. Every other section resamples to daily first. So the grid actually swept rolling windows of 78 to 624 **minutes**, roughly 1.3 to 10.4 hours, and the mechanical scaling it found is the signature of a misapplied window rather than a property of FX.

Section 1 is recorded because a parameter-selection routine silently operating at the wrong frequency is exactly the class of error the project exists to surface. The verdict does not depend on it.

**Day 72 addendum — the search was repeated at daily frequency, and the conclusion holds.** Running the same grid against daily closes rather than minute bars gives the search it was meant to perform. There is still no plateau:

| MA window (days) | 78 | 156 | 234 | 312 | 390 | 468 | 546 | 624 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EUR/USD half-life | 30.1 | 65.4 | 110.3 | 152.7 | 207.4 | 227.2 | 262.6 | 285.1 |
| GBP/USD half-life | 27.5 | 55.8 | 84.2 | 112.9 | 139.4 | 160.5 | 172.0 | 178.3 |
| USD/JPY half-life | 41.1 | 87.9 | 126.9 | 159.3 | 200.1 | 235.9 | 264.7 | 305.8 |

The half-life to window ratio stays inside 0.29 to 0.56 across an eightfold range of windows on all three pairs. Half-life scales with the window at daily frequency exactly as it did at minute frequency, so `MA_WINDOW = 100` remains a working value chosen without empirical support rather than a selected optimum. The frequency defect was real and it did not change the conclusion drawn from the defective run — which is the good case, and not one that could have been asserted without checking.

## Alternative Explanations
- **Wrong pooling threshold.** |z| = 1.5 splits the excursions 17/45. A different cut would change the pools, and none was tested. That the cut is arbitrary is itself part of why the test is weak.
- **Wrong horizon.** Test 2b uses a fixed 5-day forward return against a fitted half-life of 31–35 days. Measuring reversion at one-seventh of its own estimated timescale may simply be too early to see it.
- **The censoring cap interacts with the pools.** Capping at 3× half-life truncates the longest reversion times, and large-|z| excursions are likelier to hit the cap, which biases the large pool's mean reversion time downward.

The third is not benign. It could produce Test 2's result mechanically, independent of any market behaviour, and was not tested for.

## What Was Not Done
- **No deflated Sharpe.** No return-generating signal was built; the tests operate on the z-score and excursion structure directly.
- **No walk-forward or out-of-sample evaluation.** Discarded at the in-sample stage before any parameter was fit.
- ~~The MA/vol window grid search was never repeated at daily frequency.~~ Repeated on Day 72; see the addendum to the Section 1 defect note. No plateau at daily frequency either.
- **Sign not flipped and re-tested.** The consistent negative lean invites trading the reverse. That would use the same data to generate and confirm a hypothesis.

## Next Steps
- Closed. No retuning of the threshold, horizon, or window.
- The Section 1 frequency defect and the block-bootstrap CI failure both belong in the paper's methodology section. The second is directly relevant to the block-length question outstanding for strategy 6 — a 20-day block on a series with a 35-day half-life is shorter than the dependence it is meant to preserve, and the resulting CI was visibly wrong rather than subtly wrong, which is the lucky case.
- The reproducibility gap this document originally recorded is closed. The z-score and excursion logic was inline when the strategy was discarded; it was later extracted to `src/signals/ou_reversion.py` with 10 tests, and the script now imports `zscore_deviation`, `extract_excursions`, `split_pools` and `half_life_from_theta` from there. The Day 72 re-run confirms the extraction did not move any number in this document.
