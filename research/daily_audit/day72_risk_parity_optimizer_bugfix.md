# Day 72 — Risk Parity Optimizer Bugfix

**Date:** 2026-08-26
**Scope:** `risk_parity_weights` and `_risk_parity_objective` in `src/analysis/portfolio.py`

---

## Methodology

The reported symptom was that `risk_parity_weights()` returns equal weights for assets with very different volatilities. The investigation had three stages: reproduce the symptom and localise it, isolate which of the two suspected causes is actually responsible by varying each independently, and then verify a fix across a sweep wide enough to include the regimes the original never covered.

Reproduction used synthetic Gaussian panels with a fixed seed, 1,000 observations, and a specified per-asset volatility vector, plus the real 2011–2023 daily FX returns for all ten pairs. Every candidate solution was scored not on the optimizer's own status flag but on the equal-risk condition itself: the spread between the largest and smallest risk-contribution *share*, `wᵢ(Σw)ᵢ / w'Σw`, which is dimensionless and equals zero exactly at ERC.

Three candidate fixes were tested against 400 random correlated panels with random `n` between 2 and 10 and volatilities drawn log-uniformly over three orders of magnitude, and against the real ten-pair covariance.

## Findings

**The symptom is real but narrower than reported, and wider than it looks.** The named failing test, `test_risk_parity_low_vol_gets_higher_weight`, passes in this environment — it returns `[0.857, 0.143]`, not `[0.5, 0.5]`. The same defect fires hard elsewhere: a 60× volatility ratio at n=2 returns exactly `[0.5, 0.5]`; n=5 returns exactly `[0.2]×5`; n=10 returns exactly `[0.1]×10`. In each case the returned weights equal the starting guess to machine precision and `scipy` reports `status=0`, `"Optimization terminated successfully"`.

**The optimizer never took a step.** On the n=10 case, `nfev = 11 = n + 1` — one objective evaluation at the starting point plus the `n` perturbations of a finite-difference gradient. It computed a gradient, proposed a step, rejected it, and exited claiming success without ever moving.

**The cause is the `SCALE = 1e8` factor, and `ftol` is a red herring.** Holding the problem fixed and varying only `ftol` at `SCALE = 1e8`, the solver is stuck at the starting point for every value from `1e-1` to `1e-20`. Tightening the tolerance cannot rescue it. At `SCALE = 1` the same problem converges for every `ftol ≤ 1e-6`. Sweeping `SCALE` with `ftol` fixed at its shipped `1e-9`:

| SCALE | f(x₀) | iterations | outcome |
|---|---:|---:|---|
| 1e0 – 1e7 | 2.2e-04 – 2.2e+03 | 21–70 | converges to ERC |
| **1e8** | 2.2e+04 | 5 | **silently returns x₀, status 0** |
| 1e9 | 2.2e+05 | 7 | status 8 |
| 1e10 – 1e12 | 2.2e+06 – 2.2e+08 | 1 | status 4, "Inequality constraints incompatible" |

The shipped constant sits exactly on the cliff edge. One order of magnitude either way and the bug is either absent or loud.

**Why the magnitude matters at all: the objective was not dimensionless.** The old objective summed squared differences of risk *contributions* `wᵢ(Σw)ᵢ / √(w'Σw)`, which carry the units of volatility. Its value therefore scales with the variance of the input returns. `SCALE = 1e8` was chosen to lift a small number into a range the solver handles well for one dataset — but the appropriate multiplier is a property of the data, not a constant. The same ten-pair panel expressed in a unit 1,000× larger raises `RuntimeError: Inequality constraints incompatible` under the old code while the identical portfolio in decimals solves fine.

**Making the objective dimensionless is necessary and not sufficient.** Normalising by variance instead of volatility gives risk-contribution *shares*, which sum to one and are invariant to units. That removes the scale cliff. It does not fix the problem, because the squared-dispersion objective is non-convex: it has local minima on the boundary where an asset takes zero weight. With long-only bounds and an equal-weight start it still fails 91 of 200 synthetic n=10 panels, and on the real ten-pair FX covariance it parks USD/CAD at exactly zero with a share spread of 0.111. Starting from inverse-volatility weights fixes the synthetic cases but still fails 64 of 400 correlated panels and still fails the real ten-pair covariance.

**The fix is the convex reformulation.** Minimise, over `y > 0`,

    f(y) = ½ y'Σy − Σᵢ ln(yᵢ)

then set `w = y / Σy`. Setting the gradient `Σy − 1/y` to zero gives `yᵢ(Σy)ᵢ = 1` for every asset, which is the equal-risk-contribution condition itself. The quadratic term is convex because `Σ` is positive semi-definite and `−Σ ln(yᵢ)` is strictly convex on the positive orthant, so the minimiser is unique and no starting point can strand the solver in a secondary basin. Solved with L-BFGS-B and the analytic gradient. Results:

| Check | Result |
|---|---|
| 400 random correlated panels, n = 2–10 | 0 failures, worst share spread 1.01e-07 |
| Real FX 2011–2023, n = 3 / 5 / 10 | spread 2.6e-12 / 1.2e-10 / 3.8e-09 |
| `day34` 12 yearly 3-pair windows | spread ~1e-15 throughout |
| Diagonal covariance vs analytic `w ∝ 1/σ` | max error 2.8e-17 at n = 2, 5 and 10 |
| Input units ×1e-4 to ×1e4 | weights identical to 8 decimal places |

**The solution is now verified against the condition, not the flag.** `scipy`'s success flag is what lied in the original bug, so the fix does not trust it in either direction. The returned weights are checked against the equal-risk residual and a spread above `1e-4` raises. L-BFGS-B reports `ABNORMAL_TERMINATION_IN_LNSRCH` on 48 of the 400 panels while sitting on solutions good to 6.4e-08; those are accepted on the residual and logged at debug level.

**Weights are now long-only.** ERC is unique only over the long-only simplex. With shorting allowed, the dimensionless objective finds several distinct weight vectors that all equalise risk shares — n=3 reaches one with a −0.149 weight, n=10 one with −0.543 — and which one is returned depends on the solver path rather than the data.

**No published result in this repository is affected.** On the real ten-pair FX covariance the old code reached a share spread of 1.77e-07 and the same weights as the new code to four decimal places, and none of `day34`'s twelve yearly windows returned the equal-weight starting point. The bug requires large volatility dispersion with weak correlation, higher asset counts, or non-decimal units. Real FX daily returns are none of those. The defect was live and severe in a well-defined regime that this project's own data never entered.

## Interpretation

The instructive part is not the arithmetic, it is that the failure was silent. A solver returned its own starting guess, reported success, and produced a portfolio — equal weights — that is plausible enough to survive a glance. Equal weighting is a real allocation someone might have chosen on purpose. Nothing downstream would have flagged it, because there was no check that the returned weights satisfy the property the function exists to deliver.

The magic constant is the proximate cause and the general lesson is about units. `SCALE = 1e8` was almost certainly tuned until one dataset behaved, which is a reasonable-looking move that quietly makes the function's correctness a property of the data's units rather than of the code. An objective built out of dimensioned quantities cannot have a portable tolerance, and any fixed multiplier attached to it is a calibration to a sample. Normalising to a dimensionless quantity is not tidiness; it is what makes a fixed tolerance mean the same thing on every input.

The second lesson is that a scale fix alone was not enough, and the intermediate fix is instructive precisely because it looked finished. It passed every synthetic test written for the original bug and failed on the actual ten-pair FX covariance the project uses — the non-convexity was invisible until the test set included correlated real data. Choosing the convex formulation is what removes the dependence on a lucky starting point altogether, and it is the standard formulation for this problem for exactly that reason.

The third lesson is the one worth carrying into other modules: an optimizer's success flag reports that the algorithm's internal stopping rule fired, not that the answer is right. Where a closed-form optimality condition exists — and for ERC it does, `wᵢ(Σw)ᵢ` equal across assets — checking it costs one line and converts a silent wrong answer into a loud failure. That check, rather than the reformulation, is the part of this fix most likely to matter elsewhere in `src/`.
