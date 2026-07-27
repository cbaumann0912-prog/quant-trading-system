# Signal: Intraday Overshoot Fade

Signal definition for `research/strategies/intraday_overshoot_reversal.md`. This document specifies the signal only. Strategy-level entry, exit, sizing and validation live in the strategy spec.

## Definition

For pair `p` on day `d`, let `P_open` be the close of the 09:00 bar in America/New_York local time, and `P_t` the close of the 1-minute bar at minute `t` after 09:00.

Displacement:

```
disp_t = log(P_t / P_open)
```

Signal fires at the first `t` in [0, 180] satisfying:

```
|disp_t| > k * sigma_d
```

Signal value:

```
signal = -sign(disp_t)
```

Zero if no bar crosses the threshold. Maximum one fire per pair per day; no re-entry.

## Threshold scaling

`sigma_d` is a session-horizon volatility estimate known at the start of day `d`:

```
sigma_d = garch_conditional_vol_{d-1} * ratio_{d-1}
ratio_{d-1} = std(session returns through d-1) / std(daily returns through d-1)
```

`garch_conditional_vol` comes from `src.features.garch.fit_garch` on daily log returns. Both terms are lagged one day and `ratio` is computed on an expanding window with a 250-observation minimum, so nothing in `sigma_d` uses information from day `d` or later.

GARCH scales the threshold rather than gating trades. A fixed threshold in price terms fires almost every day in high-volatility periods and almost never in low-volatility periods, which makes both trade count and per-trade risk swing with the regime. Scaling holds the empirical trigger rate near 10% per pair per day across all 10 pairs, which is what a liquidity-provision book needs.

The daily-to-session vol ratio assumes the relationship between daily and intraday volatility is stable. It is not exactly, since intraday volatility has strong time-of-day structure. A session-native volatility model would be a strict improvement and is not implemented.

## Parameters

| Parameter | Value | Source |
|---|---|---|
| Scan window | 09:00-12:00 America/New_York | strategy spec Section 4 |
| Reference price | close of 09:00 bar | strategy spec Section 4 |
| `k` | 2.0 | pre-registered, primary |
| `k` robustness | 1.5, 2.5 | pre-registered, strategy spec Section 10 |
| Vol ratio minimum observations | 250 | fixed before testing |
| Max fires | 1 per pair per day | strategy spec Section 4 |

## Timestamp handling

Raw 1-minute files carry a fixed UTC-5 offset. Convert to true UTC using `FILE_UTC_OFFSET_HOURS` from `src.features.sessions`, then convert DST-aware to America/New_York. The 09:00 reference is a local-clock event and drifts an hour in UTC across the year, so a naive UTC cut smears the session boundary across two windows.

## Look-ahead audit

Every input to the signal at minute `t` on day `d` is observable at or before minute `t` on day `d`:

- `P_open` and `P_t` are realized prices at or before `t`.
- `garch_conditional_vol_{d-1}` is fit on returns through `d-1` and lagged.
- `ratio_{d-1}` is an expanding statistic through `d-1` and lagged.

The signal itself has no fitted parameters, so no walk-forward refit is required for the entry rule. GARCH does estimate parameters and is refit walk-forward per the Day 47 finding that full-sample fitting shifts regime labels by 39-68%.

## Execution note

Entering at the close of the crossing bar overstates realized performance. Roughly 40% of the measured edge disappears within five minutes of the crossing, which is a microstructure component rather than genuine overshoot. Any performance figure computed from crossing-bar entry should be treated as an upper bound, and the +5 minute delayed entry is the honest execution assumption.

## Reference implementation

`research/strategies/validation_falsification/intraday_overshoot_section10_validation.py`. The signal is computed inline in the staging block of that script rather than living in `src/`, because it has not yet cleared out-of-sample validation. If the lockbox test passes, it should be promoted to `src/signals/` with unit tests.
