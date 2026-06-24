# Quant Trading System

*Systematic FX Research — Cornell Engineering, Summer 2026*

---

In January 2026, during the second semester of my freshman year at Cornell, I built a systematic intraday FX strategy from scratch — a multi-timeframe BOS/FVG reversal system, backtested across 15 years of 1-minute data on EUR/USD, GBP/USD, and USD/JPY. Under realistic execution conditions it produced a Sharpe ratio of 1.60 and a max drawdown of -5.36%.

The results looked compelling. But I had a problem: I didn't really understand them. I could produce an equity curve, a walk-forward validation, a parameter robustness heatmap. What I couldn't do was explain, from first principles, why a Sharpe of 1.60 is meaningful, what the p-value on that t-stat actually means, or whether the effect size was economically significant beyond being statistically so. The code ran. The math behind it was a black box.

This repository is what happened when I decided that wasn't good enough.

---

## What This Is

An all-in-one research pipeline. Feed it a strategy's logic, and it backtests it, analyzes the results, runs the statistical machinery needed to say with actual rigor whether the strategy holds up, optimizes its parameters, and — if it survives all of that — leaves it ready for live implementation.

Every piece of statistical machinery in that pipeline — hypothesis testing, regression, stationarity testing, cointegration, PCA, factor models, portfolio construction, volatility modeling — is implemented from scratch rather than imported as a library call. The point isn't just to get a verdict on a strategy; it's to be able to defend, derive, or rebuild every step that produced that verdict.

The output for any given strategy is a statistical case: validated or invalidated, with the reasoning shown, not just a Sharpe ratio asserted at the end of a backtest.

---

## Data

| Pair | Timeframe | History | Used As |
|---|---|---|---|
| EUR/USD | 1-minute OHLCV | 15 years | Signal construction; resampled to daily log returns for statistical analysis |
| GBP/USD | 1-minute OHLCV | 15 years | Signal construction; resampled to daily log returns for statistical analysis |
| USD/JPY | 1-minute OHLCV | 15 years | Signal construction; resampled to daily log returns for statistical analysis |

Raw data lives outside this repo at a local path referenced via configurable path constants.

---

## Repository Structure

```
archive/                       # earlier work, kept for reference

research/
├── applied_analysis/          # flat exploratory analysis scripts
├── audit_images/              # plots and figures referenced in audits
├── daily_audit/               # daily research audit documents
├── papers/                    # reference papers and reading notes
├── portfolio/                 # portfolio-construction research
├── signals/                   # signal-specific research notes
├── weekly_reviews/            # weekly review documents
└── framework_map.md           # roadmap and module-by-module plan

results/
└── new_results/

src/
├── __init__.py
├── analysis/
│   ├── __init__.py
│   ├── performance_analyzer.py   # Sharpe, Sortino, drawdown, DSR, run_report
│   └── portfolio_stats.py        # covariance, portfolio variance/return
├── data/
│   ├── __init__.py
│   ├── stationarity.py           # ADF, KPSS, ACF/PACF, Ljung-Box
│   └── time_series.py            # ARIMA fitting and order selection
├── evaluation/
│   ├── __init__.py
│   ├── bootstrap_ci.py           # bootstrap confidence intervals
│   └── significance.py           # Bonferroni, Benjamini-Hochberg, permutation tests
├── features/
│   ├── __init__.py
│   └── pca.py                    # eigendecomposition, PCA from scratch
├── signals/
│   ├── __init__.py
│   └── cointegration.py          # Engle-Granger, Johansen, OU half-life
└── stats/
    ├── __init__.py
    ├── correlation.py            # rolling correlation, regime breaks
    ├── distributions.py          # normal, log-normal, Student-t
    ├── hypothesis_tests.py       # t-test, p-values, Cohen's d, power analysis
    └── regression.py             # OLS, residual diagnostics

tests/
├── __init__.py
├── test_cointegration.py
├── test_correlation.py
├── test_distributions.py
├── test_hypothesis_tests.py
├── test_pca.py
├── test_performance_analyzer.py
├── test_portfolio_stats.py
├── test_regression.py
├── test_significance.py
├── test_stationarity.py
└── test_time_series.py

.gitignore
README.md
requirements.txt
```

This is a living structure — `signals/`, `features/`, and `evaluation/` will keep growing as research findings turn into strategy specs, strategy specs turn into signal code, and signal code feeds backtest, stress-test, and parameter-optimization scripts.

---

## Validation Pipeline

Every strategy that goes in comes out with a verdict, not a vibe:

1. **Input** — the strategy's logic and a written hypothesis: entry, exit, sizing, and the economic reason an edge should exist.
2. **Backtest** — full-history performance under realistic and worst-case execution assumptions.
3. **Statistical validation** — hypothesis testing, bootstrap confidence intervals, deflated Sharpe, multiple testing correction applied across every strategy tested, not just the one that looks best.
4. **Stress test** — performance under different volatility regimes, correlation breaks, and parameter perturbations, to see what actually breaks the edge.
5. **Optimization** — parameters and position sizing tuned to the conditions where the edge holds, not the conditions that happened to produce the best backtest.
6. **Decision** — validated and live-implementation-ready, or invalidated and documented as to why.

A strategy that can't survive this isn't a strategy. It's a curve fit with good marketing.

---

## Research Outputs

Applied findings are logged to `research/daily_audit/`, with supporting figures in `research/audit_images/`. Each document connects a theoretical concept to real data — the goal is a paper trail that shows not just *what* the code produces but *what it means*, including the times the data disagreed with what I expected going in.

---

## Setup

```bash
git clone https://github.com/cbaumann0912-prog/summer2026.git
cd summer2026
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `pytest`

```bash
python -m pytest
```

---

## Development Notes

Everything in `src/` is written from scratch — no inherited logic, no functions carried forward that I can't derive independently.