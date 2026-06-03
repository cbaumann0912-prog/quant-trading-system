# Quant Trading System

*Systematic FX Research — Cornell Engineering, Summer 2026*

---

In January 2026, during the second semester of my freshman year at Cornell, I built a systematic intraday FX strategy from scratch. The strategy — a multi-timeframe BOS/FVG reversal system — was backtested across 15 years of 1-minute data on EUR/USD, GBP/USD, and USD/JPY, producing a Sharpe ratio of 1.60 and a max drawdown of -5.36% under realistic execution conditions.

The results looked compelling. But I had a problem: I didn't really understand them.

I could produce an equity curve, a walk-forward validation, a parameter robustness heatmap. What I couldn't do was explain, from first principles, *why* a Sharpe of 1.60 is meaningful, what the p-value on that t-stat actually means, or whether the effect size was economically significant beyond being statistically so. The code ran. The math behind it was a black box.

This repository documents the summer I spent fixing that.

---

## What This Is

A 90-day, self-directed curriculum built around one goal: understand the mathematics underneath the tools I was already using.

The curriculum covers distributions, hypothesis testing, regression, stationarity testing, cointegration, PCA, factor models, portfolio construction, and volatility modeling. Each topic is implemented from scratch in Python and applied directly to the three currency pairs from the original backtest — every concept has a corresponding empirical audit on real data in `research/audit/`. No black boxes. 

By the end of summer, the deliverables are a complete, CLI-accessible, Dockerized systematic research framework and a research paper written to the standards of published systematic strategy literature.

The framework is designed around a specific research workflow: a hypothesis emerges from the reading, gets formalized into a one-page strategy spec before any code is written, then runs through the full validation pipeline — walk-forward, purged cross-validation, multiple testing correction. Three candidate strategies developed over the course of the summer are the test cases for that pipeline.

---

## Original Strategy Results

The BOS/FVG Reversal strategy is the starting point and primary research object. All figures are from the original backtest on 15 years of 1-minute intraday data (2011–2026).

| Execution Scenario | Return | Sharpe | Max Drawdown |
|---|---|---|---|
| Realistic (modeled spread) | +136.71% | 1.60 | -5.36% |
| Worst-case (2× spread + slippage) | +62.97% | 0.867 | -8.74% |

Full methodology in [`archive/docs/full_paper.pdf`](/archive/docs/full_paper.pdf). Whether these numbers reflect a genuine, persistent edge or a well-dressed artifact is the central question this summer is built to answer — the [Limitations](#limitations) section documents what's known, what's uncertain, and what would need to be true for the results to hold.

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
src/
├── analysis/
│   ├── factor_models.py          # CAPM, PCA factor decomposition
│   ├── performance_analyzer.py   # + deflated Sharpe, IC/IR, transaction costs
│   ├── portfolio.py              # Markowitz, risk parity, Kelly, VaR, CVaR
│   └── signal_report.py          # Structured strategy output reports
├── data/
│   ├── loader.py                 # OHLCV ingestion, resampling, log returns
│   ├── stationarity.py           # ADF, KPSS, ACF/PACF, Ljung-Box
│   └── time_series.py            # ARIMA fitting
├── evaluation/
│   ├── bootstrap_ci.py
│   ├── cross_validation.py       # Purged K-fold (embargo-aware)
│   ├── significance.py           # Bonferroni, Benjamini-Hochberg, permutation tests
│   └── walk_forward.py           # Walk-forward validation engine
├── features/
│   ├── pca.py                    # Eigendecomposition, PCA from scratch
│   └── volatility.py             # GARCH(1,1), volatility regime classifier
├── signals/
│   ├── cointegration.py          # Engle-Granger, Johansen, OU half-life
│   ├── signal_builder.py         # Unified signal construction interface
│   └── triple_barrier.py         # Triple barrier labeling (de Prado)
└── stats/
    ├── optimization.py           # Gradient descent, SLSQP
    ├── regression.py             # OLS, ridge, lasso, diagnostics
    └── stochastic.py             # Random walk, GBM, OU process

paper/
└── forex_systematic_research.md

research/
├── audit/
├── strategy_specs/               # One-page signal specs for 3 strategies
└── deployment/
    └── live_deployment_plan.md
```

---

## Research Outputs

Applied findings are logged daily to `research/audit/`. Each document connects a theoretical concept to real data from the strategy — the goal is a paper trail that shows not just *what* the code produces but *what it means*.

---

## Limitations

Open questions guiding the research.

**Sample size.** The backtest covers 216 trades. The t-statistic on the mean return is the primary basis for assessing edge significance; power analysis and confidence interval work is documented progressively in `research/audit/` as the hypothesis testing curriculum unfolds.

**Overfitting risk.** Parameters were selected on in-sample data. Walk-forward results are presented in the original paper, but data-snooping bias from the initial signal design cannot be fully ruled out. Multiple testing correction — deflated Sharpe, Benjamini-Hochberg — will be applied across all candidate strategies and documented explicitly before any is selected for live deployment.

**Execution assumptions.** "Realistic" execution uses modeled spread data. Live transaction costs could differ materially. A transaction cost breakeven analysis is planned for Day 57.

**Signal theory.** BOS/FVG logic lacks microstructure justification. Whether the observed pattern reflects a genuine, persistent edge or a well-specified artifact is the central open question — and one of the primary motivations for the cointegration and factor model work ahead.

---

## Setup

```bash
git clone https://github.com/<your-handle>/quant-trading-system.git
cd quant-trading-system
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scipy`, `pytest`

```bash
python -m pytest
```

---

## Development Notes

The original BOS/FVG backtesting framework has been archived. Everything in src/ is written from scratch as part of this curriculum — no inherited logic, no functions carried forward that I can't derive independently.