# Summer 2026 — Quantitative Trading Research

## The Story

In January 2026, during the second semester of my freshman year at Cornell, I built a systematic intraday FX strategy from scratch. The strategy — a multi-timeframe BOS/FVG reversal system — was backtested across 15 years of 1-minute data on EURUSD, GBPUSD, and USDJPY, producing a Sharpe ratio of 1.60 and a max drawdown of -5.36% under realistic execution conditions.

The results looked compelling. But I had a problem: I didn't really understand them.

I could produce an equity curve, a walk-forward validation, a parameter robustness heatmap. I could talk about spread modeling and shared capital constraints. What I couldn't do was explain, from first principles, *why* a Sharpe of 1.60 is meaningful, what the p-value on that t-stat actually means, or whether the effect size was economically significant beyond being statistically so. The code worked. The math behind it was a black box.

This repository documents the summer I spent fixing that.

---

## What This Repo Is

This is a structured, self-directed quant finance curriculum built around two goals:

1. **Deep understanding** — building the mathematical and statistical foundation underneath the tools I was already using: distributions, hypothesis testing, correlation, effect size, power, and beyond.

2. **Career development** — building the credentials, code, and research output needed to pursue quantitative trading and systematic research professionally.

The work is organized as a 90-day program running through Summer 2026, with three blocks per day: math, Python implementation, and applied research on real strategy data.

---

## Longer-Term Vision

Beyond the summer, the goal is twofold: break into a quantitative research or trading role, and build toward an independent systematic wealth management practice — developing, testing, validating, and live-deploying automated strategies using this framework as the foundation.

---

## Repository Structure

```
summer2026/
├── src/
│   ├── stats/              # Statistical library built from scratch
│   │   ├── distributions.py
│   │   ├── hypothesis_tests.py
│   │   ├── correlation.py
│   │   ├── portfolio_stats.py
│   │   └── clt_simulator.py
│   └── analysis/           # Strategy backtesting and simulation engine
├── research/
│   ├── audit/              # Daily research outputs and findings
│   ├── stats/              # Applied statistical analysis on strategy data
│   ├── Papers/             # Reading notes and paper summaries
│   └── weekly_reviews/     # Weekly reflection and progress tracking
├── tests/                  # Full test suite (36 tests and growing)
├── docs/                   # Original strategy research papers
└── results/                # Backtest outputs and performance charts
```

---

## Original Strategy Results

The BOS/FVG Reversal strategy was the starting point for this work. Results below are from the original backtest on 15 years of intraday FX data (2011–2026).

**Realistic Execution:**
- Return: +136.71%
- Sharpe Ratio: 1.60
- Max Drawdown: -5.36%

**Worst-Case Execution (2× spread + slippage):**
- Return: +62.97%
- Sharpe Ratio: 0.867
- Max Drawdown: -8.74%

Full methodology in [`docs/full_paper.pdf`](docs/full_paper.pdf).

---

## Limitations and Open Questions

- Statistical significance: 216 trades over the backtest period.
  The t-stat on the Sharpe is 1.6, giving [Y]% confidence the edge is non-zero.
  This analysis is in progress.

- Overfitting risk: Parameters were selected on in-sample data.
  Walk-forward results are presented in full paper, but data snooping
  bias from the initial signal design cannot be fully ruled out.

- Execution assumptions: "Realistic" execution uses modeled spread data.
  True transaction costs in live trading could differ materially.

- Signal theory: BOS/FVG logic lacks microstructure justification.
  Whether the observed pattern reflects a genuine, persistent edge or
  a data artifact is an open research question.

---

## Progress

| Day | Math | Python | Research |
|-----|------|--------|----------|
| 1 | Probability foundations | Portfolio stats | Distribution assumptions audit |
| 2 | Covariance & correlation | Covariance matrix | Sharpe audit |
| 3 | CLT & sampling distributions | CLT simulator | AQR momentum notes |
| 4 | Student-t, exponential, bivariate | Distributions module | Return distribution analysis |
| 5 | Correlation & regime dependence | Correlation module | Correlation regimes |
| 8 | Hypothesis testing | t-test implementation | Strategy significance testing |
| 9 | p-values & effect size | p-value interpretation + Cohen's d | p-value table all strategies |
| ... | | | |