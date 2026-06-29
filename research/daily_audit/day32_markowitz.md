# Day 32 — Markowitz on Forex: Weights Table & Sensitivity

## Methodology
Selected 10 evenly-spaced points from the 50-point target-return sweep (range: global min-variance return up to 1.5x the highest single-pair mean return), extracting markowitz_weights() output at each. Computed weight sensitivity (Δweight/Δtarget_return) between every consecutive pair of the 10 points, and flagged any sign flips across the range. Data: 2011-01-03 to 2026-03-31, daily log returns, n=4758 aligned observations.

## Findings
- Weights table (10 points, target_return 0.000043 to 0.000211): EUR/USD declines from 0.407 to 0.010 (monotonic, stays positive throughout this range). GBP/USD declines from 0.165 to -0.413, crossing zero between target_return 0.000080 (weight +0.035) and 0.000098 (weight -0.024). USD/JPY rises from 0.428 to 1.403, exceeding 100% allocation (leveraged) past roughly the midpoint of the range, staying positive throughout.
- Sensitivity: identical slope at every one of the 9 consecutive intervals — EUR/USD -2359.90, GBP/USD -3431.23, USD/JPY +5791.13 (Δweight per unit Δtarget_return). No variation across the range.
- This constant slope is a structural consequence of the closed-form solution: x = (1/2)Σ⁻¹(λp̄ - νones) is linear in target_return for fixed Σ and p̄, since λ and ν themselves solve a linear 2x2 system with target_return only on the right-hand side. The GBP/USD sign flip is therefore not a threshold effect or instability onset — it is the deterministic point where an already-fixed straight line crosses zero.
- Only GBP/USD flips sign in this range; EUR/USD and USD/JPY remain positive throughout.

## Interpretation
The constant weight sensitivity is mathematically expected and makes the optimizer's behavior more predictable, but it does not make the resulting allocations more trustworthy. Rather than indicating instability, the linear relationship shows that increasingly aggressive short and leveraged positions are a deterministic consequence of the unconstrained Markowitz solution when target return is increased. The real concern is therefore not unpredictable optimization behavior, but the optimizer's systematic amplification of noisy expected-return estimates into economically extreme portfolios, reinforcing the need for out-of-sample validation, shrinkage, and practical portfolio constraints.