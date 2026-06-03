# Framework Map — 90-Day Quant Acceleration Plan


## Current Framework Structure

```
src/
├── analysis/
│   └── performance_analyzer.py
|   └── portfolio_stats.py 
├── data/
├── evaluation/
│   └── bootstrap_ci.py
├── features/
├── signals/
└── stats/
    ├── distributions.py
    ├── hypothesis_tests.py
    └── correlation.py
```

---

## Phase 1: Statistical Foundation (Days 1–30)

### Week 1 — Probability Fundamentals

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 1–7 | Probability distributions, Student-t, log returns | `fit_student_t`, `log_returns`, `excess_kurtosis` | `src/stats/distributions.py` | ✅ Built |

### Week 2 — Hypothesis Testing

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 8 | t-tests and z-tests | `t_test_mean` | `src/stats/hypothesis_tests.py` | ✅ Built |
| 9 | p-values | `p_value_interpretation` | `src/stats/hypothesis_tests.py` | ✅ Built |
| 10 | Type I & II error, power analysis | `compute_effect_size_cohens_d`, `compute_required_sample_size`, `compute_achieved_power` | `src/stats/hypothesis_tests.py` | ✅ Built |
| 11 | Confidence intervals | `bootstrap_confidence_interval` | `src/evaluation/bootstrap_ci.py` | ✅ Built |
| 12 | Multiple testing: Bonferroni correction | `bonferroni_correction` | `src/evaluation/significance.py` | ⏳ Planned |
| 13 | Benjamini-Hochberg procedure | `benjamini_hochberg` | `src/evaluation/significance.py` | ⏳ Planned |
| 14 | Week 2 review + significance tests module | — | `src/evaluation/significance.py` | ⏳ Planned |

### Weeks 3–4 — Regression & Linear Algebra

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 15 | OLS — normal equations | `fit_ols` | `src/stats/regression.py` | ⏳ Planned |
| 16 | R² and adjusted R² | `r_squared`, `adj_r_squared` | `src/stats/regression.py` | ⏳ Planned |
| 17 | Residual analysis and diagnostics | `residual_diagnostics` | `src/stats/regression.py` | ⏳ Planned |
| 18 | Ridge and Lasso regularization | `fit_ridge`, `fit_lasso` | `src/stats/regression.py` | ⏳ Planned |
| 19 | PCA from scratch | `pca` | `src/features/pca.py` | ⏳ Planned |
| 20 | Linear algebra / eigenvectors | `eigen_decomposition` | `src/features/pca.py` | ⏳ Planned |
| 21 | Week 3 review + regression module | — | `src/stats/regression.py` | ⏳ Planned |

### Week 4 — Time Series

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 22 | Stationarity | `check_stationarity` | `src/data/stationarity.py` | ⏳ Planned |
| 23 | ADF and KPSS tests | `adf_test`, `kpss_test` | `src/data/stationarity.py` | ⏳ Planned |
| 24 | ACF and PACF | `plot_acf_pacf` | `src/data/stationarity.py` | ⏳ Planned |
| 25 | ARIMA | `fit_arima` | `src/data/time_series.py` | ⏳ Planned |
| 26 | Cointegration — Engle-Granger | `engle_granger_test`, `cointegration_spread` | `src/signals/cointegration.py` | ⏳ Planned |
| 27 | Time series review + applied notebook | — | `src/data/time_series.py` | ⏳ Planned |

### Stochastic Intro — Interview Fundamentals

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 28 | Random walks | `simulate_random_walk` | `src/stats/stochastic.py` | ⏳ Planned |
| 29 | Gambler's Ruin | `gamblers_ruin_probability` | `src/stats/stochastic.py` | ⏳ Planned |
| 30 | Markov chains | `markov_transition_matrix`, `stationary_distribution` | `src/stats/stochastic.py` | ⏳ Planned |

---

## Phase 2: Research Engine (Days 31–60)

### Optimization

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 31 | Gradient descent | `gradient_descent` | `src/stats/optimization.py` | ⏳ Planned |
| 32 | Convex optimization | `convex_optimize` | `src/stats/optimization.py` | ⏳ Planned |
| 33 | Markowitz mean-variance | `markowitz_weights` | `src/analysis/portfolio.py` | ⏳ Planned |
| 34 | Efficient frontier | `efficient_frontier` | `src/analysis/portfolio.py` | ⏳ Planned |
| 35 | scipy.optimize integration | `constrained_optimize` | `src/stats/optimization.py` | ⏳ Planned |
| 36 | Optimization applied to strategy | `optimize_strategy_params` | `src/analysis/portfolio.py` | ⏳ Planned |
| 37 | Review + optimization module | — | `src/stats/optimization.py` | ⏳ Planned |

### Advanced Statistics + LdP Deep Dive

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 38 | Bootstrap resampling | `bootstrap_resample` | `src/evaluation/bootstrap_ci.py` | ⏳ Planned |
| 39 | Permutation tests | `permutation_test` | `src/evaluation/significance.py` | ⏳ Planned |
| 40 | Purged cross-validation | `purged_cross_validation` | `src/evaluation/cross_validation.py` | ⏳ Planned |
| 41 | Combinatorial purged CV | `combinatorial_purged_cv` | `src/evaluation/cross_validation.py` | ⏳ Planned |
| 42 | Deflated Sharpe ratio | `deflated_sharpe_ratio` | `src/analysis/performance_analyzer.py` | ⏳ Planned |
| 43 | IC and IR decomposition | `information_coefficient`, `information_ratio` | `src/analysis/performance_analyzer.py` | ⏳ Planned |
| 44 | Review + apply to strategy candidates | — | — | ⏳ Planned |

### Factor Models — Compressed

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 45 | CAPM | `capm_expected_return` | `src/analysis/factor_models.py` | ⏳ Planned |
| 46 | Fama-French three-factor | `fama_french_three_factor` | `src/analysis/factor_models.py` | ⏳ Planned |
| 47 | PCA factors and alpha decomposition | `pca_factor_decomposition` | `src/analysis/factor_models.py` | ⏳ Planned |

### Framework Rebuild — Dedicated Block

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 48 | Architecture design | — | All `src/` modules | ⏳ Planned |
| 49 | Data pipeline | `load_ohlcv`, `resample_to_daily`, `compute_log_returns` | `src/data/loader.py` | ⏳ Planned |
| 50 | Feature engineering module | `compute_features` | `src/features/engineer.py` | ⏳ Planned |
| 51 | Purged CV integration | `purged_cross_validation` | `src/evaluation/cross_validation.py` | ⏳ Planned |
| 52 | Walk-forward validation engine | `walk_forward_validate` | `src/evaluation/walk_forward.py` | ⏳ Planned |
| 53 | Performance attribution and reporting | `performance_report` | `src/analysis/performance_analyzer.py` | ⏳ Planned |
| 54 | Full test suite | — | `tests/` | ⏳ Planned |

### Signal Research Framework Sprint

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 55–60 | Run all 3 candidate strategies through rebuilt framework | Strategy-specific | `src/signals/` | ⏳ Planned |

---

## Phase 3: Institutional Output (Days 61–90)

### Portfolio Construction

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 61 | Kelly criterion | `kelly_fraction`, `fractional_kelly` | `src/analysis/portfolio.py` | ⏳ Planned |
| 62 | Value at Risk | `var_historical`, `var_parametric`, `var_monte_carlo` | `src/analysis/portfolio.py` | ⏳ Planned |
| 63 | CVaR / Expected Shortfall | `cvar` | `src/analysis/portfolio.py` | ⏳ Planned |
| 64 | Position sizing integration | `position_sizing` | `src/analysis/portfolio.py` | ⏳ Planned |
| 65 | Drawdown analysis | `max_drawdown`, `drawdown_series` | `src/analysis/performance_analyzer.py` | ⏳ Planned |
| 66 | Portfolio construction applied | — | `src/analysis/portfolio.py` | ⏳ Planned |
| 67 | Review + portfolio module | — | `src/analysis/portfolio.py` | ⏳ Planned |

### Volatility Modeling

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 68 | Realized volatility | `realized_volatility` | `src/features/volatility.py` | ⏳ Planned |
| 69 | GARCH(1,1) | `fit_garch` | `src/features/volatility.py` | ⏳ Planned |
| 70 | GARCH extensions | `fit_egarch`, `fit_gjr_garch` | `src/features/volatility.py` | ⏳ Planned |
| 71 | Volatility regimes | `markov_switching_volatility` | `src/features/volatility.py` | ⏳ Planned |
| 72 | Volatility forecasting | `forecast_volatility` | `src/features/volatility.py` | ⏳ Planned |
| 73 | Volatility signals | `volatility_signal` | `src/signals/volatility_signal.py` | ⏳ Planned |
| 74 | Review + volatility module | — | `src/features/volatility.py` | ⏳ Planned |

### Microstructure Deep Dive

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 75 | Bid-ask spread decomposition | `spread_decomposition` | `src/features/microstructure.py` | ⏳ Planned |
| 76 | Adverse selection | `adverse_selection_component` | `src/features/microstructure.py` | ⏳ Planned |
| 77 | Order flow toxicity / VPIN | `vpin` | `src/features/microstructure.py` | ⏳ Planned |
| 78 | Limit order book dynamics | `lob_imbalance` | `src/features/microstructure.py` | ⏳ Planned |
| 79 | Price impact models | `price_impact` | `src/features/microstructure.py` | ⏳ Planned |
| 80 | Microstructure in FX | `fx_microstructure_features` | `src/features/microstructure.py` | ⏳ Planned |
| 81 | Interview prep | — | `research/papers/` | ⏳ Planned |

### Research Paper Finalization + GitHub Polish

| Day | Topic | Deliverable | Status |
|-----|-------|-------------|--------|
| 82 | Full draft review | Trimmed paper draft | ⏳ Planned |
| 83 | Results section | Final charts and tables | ⏳ Planned |
| 84 | Methodology section | Precision-edited methodology | ⏳ Planned |
| 85 | Limitations and future work | Honest limitations section | ⏳ Planned |
| 86 | GitHub repo cleanup | Clean README, docstrings | ⏳ Planned |
| 87 | CLI and end-to-end run test | Working CLI entry point | ⏳ Planned |
| 88 | Peer review pass | Recruiter-ready paper | ⏳ Planned |
| 89 | Final edits and formatting | Formatted final draft | ⏳ Planned |
| 90 | Final commit | Paper published, assessment written | ⏳ Planned |

---

## Target Framework Structure at Day 90

```
src/
├── analysis/
│   ├── factor_models.py        ← Days 45–47
│   ├── performance_analyzer.py ← Days 11, 42, 43, 53, 65
│   └── portfolio.py            ← Days 33–34, 61–66
├── data/
│   ├── loader.py               ← Day 49
│   ├── stationarity.py         ← Days 22–24
│   └── time_series.py          ← Days 25, 27
├── evaluation/
│   ├── bootstrap_ci.py         ← Day 11 ✅
│   ├── cross_validation.py     ← Days 40–41, 51
│   ├── significance.py         ← Days 12–13, 39
│   └── walk_forward.py         ← Day 52
├── features/
│   ├── engineer.py             ← Day 50
│   ├── microstructure.py       ← Days 75–80
│   ├── pca.py                  ← Days 19–20
│   └── volatility.py           ← Days 68–74
├── signals/
│   ├── cointegration.py        ← Day 26
│   └── volatility_signal.py    ← Day 73
└── stats/
    ├── distributions.py        ← Days 1–7 ✅
    ├── hypothesis_tests.py     ← Days 8–10 ✅
    ├── optimization.py         ← Days 31–32, 35
    ├── regression.py           ← Days 15–18
    └── stochastic.py           ← Days 28–30
```