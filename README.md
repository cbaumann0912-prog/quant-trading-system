# Quant Trading System

*Systematic FX Research — Cornell Engineering, Summer 2026*

---

In January 2026, during the second semester of my freshman year at Cornell, I built a systematic intraday FX strategy from scratch — a multi-timeframe BOS/FVG reversal system, backtested across 15 years of 1-minute data on EUR/USD, GBP/USD, and USD/JPY. Under realistic execution conditions it produced a Sharpe ratio of 1.60 and a max drawdown of -5.36%.

The results looked compelling. But I had a problem: I didn't really understand them. I could produce an equity curve, a walk-forward validation, a parameter robustness heatmap. What I couldn't do was explain, from first principles, why a Sharpe of 1.60 is meaningful, what the p-value on that t-stat actually means, or whether the effect size was economically significant beyond being statistically so. The code ran. The math behind it was a black box.

This repository is what happened when I decided that wasn't good enough.

---

## What This Is

An all-in-one research pipeline. Feed it a pre-registered hypothesis, and it constructs the signal, runs walk-forward validation with leakage controls, applies the statistical machinery needed to say with actual rigor whether the effect survives, prices in transaction costs, and returns a verdict.

The statistical machinery in that pipeline is derived and implemented from first principles rather than imported: OLS and regularized regression, eigendecomposition and PCA, GARCH by maximum likelihood, purged cross-validation, block bootstrap, permutation testing, multiple-testing correction, deflated Sharpe, IC/IR, Markowitz and risk parity, Kelly, VaR/CVaR, GBM and OU simulation. A short list of standard test statistics is called through `statsmodels` rather than rebuilt, and is named explicitly below. The point isn't to get a verdict; it's to be able to derive, defend, or rebuild every step that produced it.

The output for a strategy is a statistical case: validated or invalidated, with the reasoning shown.

So far the pipeline has returned six verdicts. All six are invalidated. That result is the subject of the sections below, and it is the reason this repository is worth reading.

---

## Current State — July 2026

| | |
|---|---|
| Framework | 6,588 lines across 30 modules in `src/` |
| Tests | 372 tests across 29 files, 4,838 lines |
| Research record | 39 daily audits, 34 reproducible analysis scripts, 30 falsification documents |
| Universe | 10 FX pairs of 1-minute OHLCV, plus 8 three-month interbank rate series |
| Strategies pre-registered | 6 |
| Strategies surviving validation | 0 |
| Lockbox (2024-01 to 2026-05) | Never opened |
| Commits | 202 |

The lockbox is the part I'd point to first. It is a reserved slice of unseen data that the spec permits opening only on a PASS. Six candidates have failed without it being touched, which means the project still holds one clean shot at an out-of-sample test. Opening it to confirm a failure would spend that for nothing, and would create a live temptation to revive a dead strategy if the result came back positive by chance.

---

## Strategy Roster

Six hypotheses, each pre-registered in writing before any test was run. Specs live in `research/strategies/specs/` and are unedited from when they were written; only the status line is amended on closure. Every test that bore on a verdict is in `research/strategies/validation_falsification/`, indexed by strategy.

| # | Strategy | Closed | Verdict |
|---|---|---|---|
| 1 | PC2 Carry Regime | Day 41 | Three independent tests null. Pooled IC −0.0017, p = 0.951. Conditional predictability discarded at condition number 2.27e10 |
| 2 | Momentum with ML Regime Filter | Day 42 | No base momentum IC survived once a window-alignment bug was corrected |
| 3 | OU Half-Life Mean Reversion | Day 42 | Tests 2, 2b and 2c all failed |
| 4 | Volatility Regime Breakout / Mean-Reversion | Day 49 | Reversion leg null at p = 0.563 once the lockbox is correctly excluded from the walk-forward windows. Section 10 requires both legs |
| 4b | Momentum-Only Pooled Book *(leg #4 redeployed, not a new trial)* | Day 57 | Robustness-1 null at 10 pairs, p = 0.399. The original 3-pair pass had the interaction running the wrong direction, b1+b3 = −0.0022 |
| 5 | Month-End FX Rebalancing Flow | Day 57 | H1 significant, wrong sign |
| 6 | Intraday Overshoot Reversal | Day 57 | Section 10 FAIL on R1 threshold monotonicity, R3 per-trade permutation (p = 0.114), and H2. Passed its cost gate at 4.54x |

Strategy #1 has no spec file. It originated in the Day 18–19 PCA work and was evaluated through daily audits rather than the spec-first process, which is a documented departure from the intended discipline rather than a rewrite of it.

---

## What The Nulls Produced

The contribution here is the machinery that killed six strategies, and four findings that machinery surfaced. Each is reproducible from the scripts in this repo.

**A controlled leakage demonstration.** The same hypothesis passes under a full-sample GARCH fit and fails under a walk-forward one, on identical data and identical code. Moving the volatility-regime classifier refit inside each training window flips 34–56% of the regime labels. This is the cleanest before/after leakage example I have, because nothing changes except where the fit happens.
→ `vol_regime_classifier_refit_stability.md`

**Portfolio aggregation manufacturing significance.** Ten pairs at mean cross-correlation ρ = 0.36 give an effective breadth of 2.34, not 10. The pooled book posts Sharpe +1.043 at p < 0.00001 while no individual component is significant and the cross-pair mean IR confidence interval contains zero. The book-level p-value is an artifact of treating correlated series as independent draws.
→ `momentum_book_invalidation.md`

**Two validation criteria that looked rigorous and were not.** Day 48's rule tested whether an interaction coefficient differed from zero but never tested its sign — the strategy passed while running backwards. Day 57's R1 required ordering three cells whose confidence intervals all span zero; the deciding gap carried t = 0.05, and 4 trades out of 1,588 flip the verdict. A criterion with no power to discriminate is worse than no criterion, because it produces a PASS that reads as evidence.
→ `intraday_overshoot_section10_validation.md`

**The power ceiling is arithmetic, not opinion.** Thirteen years of daily data requires a true Sharpe near 0.55 to reach t = 2. Intraday sampling does not relax this, because the binding constraint is calendar span rather than observation count. Five of the six candidates were untestable at this sample size rather than merely wrong — a distinction that changes what you should do next. Dropping NZD/USD, which gates the common window at 2005-08, buys 8.8 years and moves the required Sharpe from 0.555 to 0.428.
→ `research/notes/data_span_power_constraint.md`

---

## Data

| Pairs | Series | History | Use |
|---|---|---|---|
| EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY, EUR/CHF | 1-minute OHLCV | ~13 years development (2011-2023) | Signal construction; resampled to daily log returns for statistical work |
| 8 three-month interbank rates | Daily | matched | Carry and rate-differential factors |

Development window is 2011–2023. The 2024-01-01 to 2026-05-01 slice is held as a lockbox and has never been read. Raw data lives outside the repo at a configurable path.

Annualization is computed empirically everywhere from each series' own DatetimeIndex, never hardcoded to 252. The daily series measures 312.30 observations per year because the vendor buckets the Sunday FX open as its own date; the intraday session book measures 259.44. Both are correct for their own series, and the discrepancy is the kind of thing that quietly corrupts a Sharpe ratio if you assume a constant.

No session caching exists anywhere in the project. Every analysis rebuilds from raw 1-minute bars on every run.

---

## Validation Pipeline

A strategy gets a verdict, not a vibe.

1. **Pre-registration.** A written spec before any code: hypothesis, economic rationale, entry, exit, sizing, risk controls, and the explicit conditions under which the strategy is declared dead. Sections 1–12, fixed before the first test.
2. **Power gate.** `compute_required_sample_size` and `compute_achieved_power` run before data is touched. If the sample cannot resolve the effect size the hypothesis implies, the hypothesis is not tested. This became a binding rule at Day 57 and is the single largest process change to come out of six nulls.
3. **Signal construction.** `SignalBuilder` takes any `signal_fn(data, lookback) -> pd.Series` under a causal-by-convention contract, with `validate_no_lookahead` as a mechanical leakage check.
4. **Walk-forward validation.** Expanding or rolling windows through `WalkForwardValidator`, with every fit — including regime classifiers and volatility models — refit inside the training window. Purged cross-validation with overlap purging and embargo is the R&D tool; walk-forward is the deployment-realistic backtest. They are not substitutes.
5. **Statistical validation.** Permutation tests, block bootstrap confidence intervals that respect serial dependence, deflated Sharpe, and Benjamini-Hochberg correction applied across every strategy tested rather than the one that looks best. Trial count is cumulative and includes failed tests.
6. **Cost gate.** Round-trip spread, rollover, and implied trade count against a breakeven Sharpe. Strategy #6 cleared this at 4.54x, and still failed on the statistics.
7. **Verdict.** Documented either way, with the failure mode named. The lockbox opens only on a PASS.

A strategy that can't survive this isn't a strategy. It's a curve fit with good marketing.

---

## What's Implemented

**Statistics** — `src/stats/`
Normal, log-normal, Student-t densities and tail-mass comparison. Wald/t-tests, p-value interpretation, Cohen's d, required-sample-size and achieved-power calculations. OLS via normal equations, R²/adjusted R², residual diagnostics, ridge, lasso by numerical minimization of the L1-penalized objective, interaction regression with VIF and centering. Rolling correlation and a bootstrap-calibrated CUSUM regime-shift detector. Gradient descent and an SLSQP constrained optimizer validated against the closed-form KKT solution. GBM and Ornstein-Uhlenbeck simulation with OU parameter fitting and Monte Carlo stress testing.

**Evaluation** — `src/evaluation/`
Bootstrap confidence intervals and block bootstrap for autocorrelated series. Purged k-fold cross-validation with three-condition overlap purging and embargo. Permutation tests including paired-sign and interaction-coefficient variants. Bonferroni and Benjamini-Hochberg correction.

**Signals** — `src/signals/`
`SignalBuilder` with IC, rolling IC, forward returns, and lookahead validation. Momentum, price z-score and ladder mean-reversion, regime-gated composites, per-window regime refit, vol-scaled triple-barrier labels with first-touch tie-breaking, and the intraday overshoot session builder. Engle-Granger, Johansen, cointegration spreads, OU half-life.

**Features** — `src/features/`
Eigendecomposition and PCA from first principles with SVD-based matrix inversion. GARCH(1,1) fit by maximum likelihood, with a 1-D k-means volatility regime classifier. Session construction with correct local-open-to-UTC handling.

**Analysis** — `src/analysis/`
`PerformanceAnalyzer`: empirical annualization, Sharpe, deflated Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor, t-stat, Jarque-Bera, Ljung-Box, tracking error, regime-conditional attribution. Transaction cost model: pip sizing, bps conversion, round-trip cost, rollover, breakeven return and Sharpe, max viable spread. Markowitz closed-form and numerical, efficient frontier, minimum-variance and ERC risk parity, Kelly and fractional Kelly, historical/parametric/Monte Carlo VaR and CVaR. CAPM decomposition, PCA factor extraction, IC/IR. `SignalReport` as a pure formatter over the above.

**Framework** — `src/framework/`
`DataLoader` with composition-root embargo and leakage-safe train/test splitting. `WalkForwardValidator` with window generation and slicing.

### What is not from scratch

Stated plainly, because a reader can check in thirty seconds and a blanket claim would not survive it. These call an established implementation:

| Component | Source | Why |
|---|---|---|
| ADF, KPSS | `statsmodels.tsa.stattools` | Critical-value tables are interpolated from published simulations; reimplementing reproduces a lookup, not the theory |
| Johansen trace test | `statsmodels.tsa.vector_ar.vecm` | Same reason — the eigenvalue problem is hand-derived in `pca.py`, the critical values are not |
| ARIMA fitting | `statsmodels.tsa.arima` | Order selection logic and AIC comparison are mine; the state-space MLE is not |
| Ljung-Box, Jarque-Bera | `statsmodels`, `scipy.stats` | Standard test statistics |
| Distribution CDFs, `spearmanr` | `scipy.stats` | Numerical primitives |
| SLSQP, MLE optimizer | `scipy.optimize` | The objective functions — GARCH log-likelihood, lasso penalty, risk-parity ERC — are hand-written; the solver is not |

Two of these were originally scoped for first-principles implementation and ended up here instead. They are listed rather than quietly reclassified.

---

## Repository Structure

```
src/
├── analysis/
│   ├── factor_models.py          # CAPM, PCA factor extraction
│   ├── performance_analyzer.py   # metrics, DSR, regime attribution, cost model
│   ├── portfolio.py              # Markowitz, frontier, risk parity, Kelly, VaR/CVaR
│   ├── portfolio_stats.py        # covariance, portfolio variance/return
│   └── signal_report.py          # per-leg stats formatter with BH summary
├── data/
│   ├── stationarity.py           # ADF, KPSS, ACF/PACF, Ljung-Box
│   └── time_series.py            # ARIMA fitting, AIC order selection
├── evaluation/
│   ├── bootstrap.py              # bootstrap CI, block bootstrap
│   ├── cross_validation.py       # purged k-fold with embargo
│   └── significance.py           # Bonferroni, BH, permutation tests
├── features/
│   ├── garch.py                  # GARCH(1,1) MLE, k-means vol regimes
│   ├── pca.py                    # eigendecomposition, PCA, SVD inverse
│   ├── regime_classifier.py      # composite regime score
│   └── sessions.py               # session return construction
├── framework/
│   ├── data_loader.py            # leakage-safe loading and splitting
│   └── walk_forward.py           # walk-forward window engine
├── signals/
│   ├── cointegration.py          # Engle-Granger, Johansen, OU half-life
│   ├── intraday_overshoot.py     # strategy #6 session/trade builder
│   ├── mean_reversion.py         # z-score and ladder signals
│   ├── momentum.py
│   ├── regime_gated.py
│   ├── regime_refit.py           # per-window classifier refit
│   ├── signal_builder.py         # signal-agnostic scoring engine
│   └── triple_barrier.py
└── stats/
    ├── correlation.py            # rolling correlation, CUSUM regime shifts
    ├── distributions.py
    ├── hypothesis_tests.py       # t-test, effect size, power
    ├── optimization.py           # gradient descent, SLSQP
    ├── regression.py             # OLS, ridge, lasso, interactions, VIF
    └── stochastic.py             # GBM, OU, stress testing

research/
├── daily_audit/                  # 39 audits, one per day with a research deliverable
├── applied_analysis/             # 34 reproducible scripts backing those audits
├── strategies/
│   ├── specs/                    # 6 pre-registrations, unedited post-hoc
│   └── validation_falsification/ # 30 files, every test bearing on a verdict
├── notes/                        # block-length selection, PCA universe, power constraint
└── audit_images/                 # figures referenced in audits

tests/                            # 372 tests, one file per module
```

Audits are dated to the research day that produced them and are not reconstructed after the fact. They include the runs where the data contradicted what I expected going in, which is most of them.

---

## Setup

```bash
git clone https://github.com/cbaumann0912-prog/summer2026.git
cd summer2026
pip install -r requirements.txt
python -m pytest
```

Dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `pytest`. Pinned in `requirements.txt`.

Analysis scripts in `research/applied_analysis/` and `research/strategies/validation_falsification/` are flat and run from the repository root. They rebuild from raw bars on every run, so they are slow and reproducible rather than fast and cached.
