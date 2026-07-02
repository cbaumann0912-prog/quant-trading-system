# Week 5 Review — Optimization (Days 31–35)

## Methodology
`gradient_descent`, `constrained_optimize` (SLSQP) — `src/stats/optimization.py`. `markowitz_weights`, `efficient_frontier`, `risk_parity_weights` — `src/analysis/portfolio.py`.

## Findings
- Test suite: 23 passed, 1 skipped, 0 failed (`test_optimization.py`: 3/3; `test_portfolio.py`: 20/21).
- Skipped: `test_near_singular_sigma_conditioning` — no conditioning check on near-singular Sigma implemented. Relevant given Week 4's PC1 concentration findings (up to 76%); near-singular Sigma is realistic on this dataset, not synthetic.
- FX-scale covariance (~1e-4–1e-5) caused SLSQP convergence failures under default tolerances. Fix: SCALE=1e6 on objective/Jacobian, ftol=1e-9, maxiter=2000. Risk parity: SCALE=1e8.

## Interpretation
Getting SLSQP to converge on FX-scale covariances required rescaling the objective and Jacobian by 1e6 — a numerical detail, but one that would've silently produced garbage weights if left unfixed, which is worth remembering for every future module touching this data's native scale. Markowitz, efficient frontier, and risk parity are now all built on the same scipy-based constrained optimizer rather than three separate closed-form implementations, which keeps the portfolio math consistent across the framework. Near-singular Sigma conditioning remains an open TODO.