# Day 35 — AML Ch. 3 (Labeling) & Ch. 4 (Sample Weights): Triple-Barrier Method

## Methodology
Read AML Ch. 3 (Labeling) in full, Ch. 4 §4.1–4.3 (Sample Weights, overlapping outcomes, concurrent labels).

## Findings

**Why fixed-threshold labeling fails.** A fixed-horizon label checks if the return over a set window beats a constant τ. López de Prado's example: τ = 1e-2 applied uniformly, but realized vol might be 1e-4 overnight and 1e-2 at the open. Most observations get labeled 0 by default, even ones that were significant relative to their own regime. The threshold is blind to what conditions the observation happened under. Separately, fixed-horizon labeling only checks start and end points, so it can label a trade a win that would've been stopped out somewhere in the middle — a trade you couldn't have actually held.

**The three barriers.** Upper = profit target, sized off estimated vol, hit first → +1. Lower = stop-loss, same basis, hit first → −1. Vertical = time limit in bars, hit first with no horizontal touch → sign of the return, or 0. Any barrier can be switched off, giving eight configurations. Useful: `[1,1,1]` standard, `[0,1,1]` time exit unless stopped out, `[1,1,0]` hold for profit unless stopped out. Nonsense: `[0,1,0]` (hold until stopped out for no reason) and `[0,0,0]` (no label ever generated).

**Why barriers scale with volatility.** Width is `ptSl[i] * trgt`, and `trgt` is `getDailyVol` — an EW rolling std of daily returns, not a fixed number. Wide barriers when volatile, tight when calm, instead of one number applied everywhere.

**Why that vol estimate has to be causal.** `getDailyVol` only looks backward from the current index. If barrier width used vol computed from inside the labeling window itself, the label would be built on information that didn't exist yet at entry — the model gets told in advance how volatile the coming window is. No live strategy gets that. Barriers have to come from the past because the past is all a real strategy has.

**Path dependency.** Labeling one point needs the whole price path from entry to the vertical barrier, not just the endpoints. A rolling mean needs a fixed window and a formula; this needs to check at every step whether a barrier's been crossed yet, and stop at the first one. Sequential, not vectorizable — the book's implementation walks each event's path directly rather than using one pandas operation.

**Connection to strategy specs.** Upper/lower barriers are the profit-taking and stop-loss levels a real exit rule uses; vertical barrier is the max hold time. Label with raw fixed-horizon returns instead, and you're training on trades that ignore the stop-loss a real strategy would've hit. Triple-barrier forces the labels to match the exit logic already in the spec.

## Interpretation
Barrier widths for the shortlisted strategies need a real causal, rolling vol estimate — not a fixed percent, nothing computed with knowledge of how the window turned out. `h` and `ptSl` aren't knobs to tune until the labels look nice; they have to match what's already in each strategy's exit rule. Overlapping windows are unavoidable at any sane sampling frequency, so whatever validation gets used downstream has to account for that — straight line to purged CV, not optional.

## Open questions for Day 36
- Sign-of-return vs. 0 on vertical touches — needs a real decision before `triple_barrier_labels` gets written.
- `ptSl` and `h` aren't derived anywhere in the chapter — they have to trace back to each candidate's actual spec, not get picked for a nicer label distribution.
- Symmetric vs. asymmetric barriers — learning the side of a bet instead of assuming it needs either no horizontal barriers or symmetric ones. Matters if any shortlisted strategy doesn't have a fixed long/short side going in.