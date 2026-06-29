# Day 33 — Efficient Frontier, EUR/USD, GBP/USD, USD/JPY

## Methodology
Swept target_return across the standard convention range [p̄.min(), p̄.max()], solving the unconstrained-short Markowitz QP via efficient_frontier() at 50 grid points. Global min-variance computed via minimum_variance_portfolio(). Equal-weight benchmark computed directly. As a secondary check, also confirmed the result holds under a 50:1 per-pair leverage cap (uniform across EUR/USD, GBP/USD, USD/JPY) via leverage_bounded_return_range() — see Findings. Data: 2011-01-03 to 2026-03-31, daily log returns, n=4758 aligned observations.

## Findings
- Global min-variance: return 0.000043/day, variance 0.00000734, weights EUR 0.407 / GBP 0.165 / JPY 0.428 (all long) — matches the prior inline closed-form calculation to displayed precision, confirming minimum_variance_portfolio() is correct.
- Max-Sharpe (grid): return 0.000126/day, Sharpe 0.4290, weights EUR 0.210 / GBP -0.123 / JPY 0.913 — short GBP/USD, leveraged long USD/JPY, largest single weight 0.91.
- Equal-weight (1/3, 1/3, 1/3): return 0.000026/day, variance 0.00000816, volatility 0.002856. Strictly dominated by the global min-variance portfolio on both axes — lower return and higher variance (0.00000816 vs. 0.00000734).
- Leverage check: feasible range under a 50:1 per-pair cap is [-0.008665, 0.008603] — far wider than the actual swept range — and the max-Sharpe weight (0.91) is nowhere near binding (cap of 50).

## Interpretation
The leverage check confirms this result isn't an artifact of an unconstrained sweep — even under a realistic 50:1 margin cap, the optimizer's natural solution stays far inside the limit (largest weight 0.91 vs. a feasible range that would tolerate up to roughly ±50). Whether tighter leverage limits or a different asset/strategy universe (e.g., capital allocation across multiple strategies rather than raw pairs) would bring the constraint into play is an open question this exercise doesn't resolve. The max-Sharpe portfolio's short GBP/long JPY tilt likely still reflects estimation-error sensitivity in unconstrained Markowitz rather than a robust signal, independent of leverage.

## Limitations
This analysis is in-sample only — the max-Sharpe and global min-variance portfolios were both fit and evaluated on the same 2011–2026 data, so the 0.4271 Sharpe figure has not been tested for statistical significance and may not hold out-of-sample. No multiple-testing or deflated Sharpe correction was applied, despite that machinery already existing. The result also says nothing about the actual strategy candidates under consideration (stat arb, vol straddle, PC2 carry) — this was portfolio theory on raw pair returns, not a test of any specific trading hypothesis.