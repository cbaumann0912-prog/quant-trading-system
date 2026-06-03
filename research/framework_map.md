# Framework Map — 90-Day Quant Acceleration Plan


## Current Framework Structure

```
src/
├── analysis/
│   ├── performance_analyzer.py
│   └── portfolio_stats.py
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

> ⚠️ **Parallel Track — Strategy Hypothesis Development**
> Running alongside all of Phase 1, in margin time only (not replacing any scheduled work):
> Develop a written one-page specification for each of your ≤3 candidate strategies.
> Each spec must define: signal logic, entry rule, exit rule, position sizing rule, data required, and the economic reason the edge should exist.
> These documents must be complete by Day 30. They are what you hand to SignalBuilder at Day 44.
> Do not begin any coding of signal logic during Phase 1. Written specification only.

### Week 1 — Probability Fundamentals

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 1 | Normal & log-normal distributions — distributions module. Research: framework distribution audit | `normal_pdf`, `normal_cdf`, `lognormal_pdf`, `simulate_log_returns`, `simulate_price_path` | `src/stats/distributions.py` | ✅ Built |
| 2 | Expectation, variance & covariance — portfolio statistics module. Research: Sharpe ratio audit | `compute_covariance_matrix`, `compute_portfolio_variance`, `compute_portfolio_return` | `src/analysis/portfolio_stats.py` | ✅ Built |
| 3 | CLT & law of large numbers — CLT simulator module. Research: AQR momentum paper methodology | — (CLT simulator, archived) | `archive/` | ✅ Built |
| 4 | Student-t & exponential distributions — tail analysis module. Research: return distribution analysis | `student_t_pdf`, `tail_mass_comparison` | `src/stats/distributions.py` | ✅ Built |
| 4 | Return distribution analysis — EUR/USD, GBP/USD, USD/JPY log returns, excess kurtosis, Student-t fit | — (research) | `research/audit/day04_return_distribution_analysis.md` | ✅ Built |
| 5 | Correlation matrices — correlation module. Research: rolling correlation regimes | `rolling_correlation`, `detect_regime_breaks` | `src/stats/correlation.py` | ✅ Built |
| 6 | Sharpe (1994) paper — PerformanceAnalyzer scaffold. Research: Sharpe (1994) paper read | `compute_sharpe`, `compute_max_drawdown` | `src/analysis/performance_analyzer.py` | ✅ Built |
| 7 | Week 1 review — Sharpe & max drawdown continued. Research: week 1 audit | — (review) | `research/audit/day07_week1_review.md` | ✅ Built |

### Week 2 — Hypothesis Testing

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 8 | Wald test, p-values & hypothesis testing — t-test module. Research: strategy significance test | `t_test_mean` | `src/stats/hypothesis_tests.py` | ✅ Built |
| 9 | p-value interpretation + Cohen's d. Research: p-value table all strategies | `p_value_interpretation`, `compute_effect_size_cohens_d` | `src/stats/hypothesis_tests.py` | ✅ Built |
| 10 | Permutation test & likelihood ratio test — power analysis module. Research: power analysis all strategies | `compute_required_sample_size`, `compute_achieved_power` | `src/stats/hypothesis_tests.py` | ✅ Built |
| 11 | Confidence intervals — bootstrap CI module. Research: Sharpe CI on top strategy | `bootstrap_confidence_interval` | `src/evaluation/bootstrap_ci.py` | ✅ Built |
| 12 | Multiple testing correction — multiple testing module. Research: multiple testing on strategies | `bonferroni_correction` | `src/evaluation/significance.py` | ⏳ Planned |
| 13 | Deflated Sharpe ratio — DSR module. Research: DSR for all strategies | `deflated_sharpe_ratio` | `src/analysis/performance_analyzer.py` | ⏳ Planned |
| 14 | Week 2 review — significance suite integration. Research: week 2 review | — | `src/evaluation/significance.py` | ⏳ Planned |

### Week 3 — Regression & Linear Algebra

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 15 | OLS normal equations — OLS from scratch. Research: OLS EUR/USD vs GBP/USD | `fit_ols` | `src/stats/regression.py` | ⏳ Planned |
| 16 | Adjusted R² & residual diagnostics — diagnostics module. Research: residual diagnostics | `r_squared`, `adj_r_squared`, `residual_diagnostics` | `src/stats/regression.py` | ⏳ Planned |
| 17 | Ridge & Lasso regularization. Research: ridge on EUR/USD prediction | `fit_ridge`, `fit_lasso` | `src/stats/regression.py` | ⏳ Planned |
| 18 | Eigendecomposition — linear algebra module. Research: eigendecomposition forex | `eigen_decomposition` | `src/features/pca.py` | ⏳ Planned |
| 19 | PCA from first principles — PCA module. Research: PCA on forex pairs | `pca` | `src/features/pca.py` | ⏳ Planned |
| 20 | Stationarity: ADF & KPSS — stationarity module. Research: stationarity all pairs | `check_stationarity`, `adf_test`, `kpss_test` | `src/data/stationarity.py` | ⏳ Planned |
| 21 | Week 3 review — regression test suite. Research: week 3 review | — | `src/stats/regression.py` | ⏳ Planned |

### Week 4 — Time Series & Cointegration

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 22 | ACF & PACF — ACF/PACF/Ljung-Box module. Research: ACF/PACF on EUR/USD | `plot_acf_pacf`, `ljung_box_test` | `src/data/stationarity.py` | ⏳ Planned |
| 23 | ARMA/ARIMA structure — ARIMA module. Research: ARIMA on forex pairs | `fit_arima` | `src/data/time_series.py` | ⏳ Planned |
| 24 | Cointegration: Engle-Granger — cointegration module. Research: cointegration all pair combinations | `engle_granger_test`, `cointegration_spread` | `src/signals/cointegration.py` | ⏳ Planned |
| 25 | Johansen cointegration — Johansen test module. Research: Johansen 3-pair analysis | `johansen_test` | `src/signals/cointegration.py` | ⏳ Planned |
| 26 | Half-life of mean reversion — OU half-life module. Research: half-life & z-score thresholds | `ou_half_life` | `src/signals/cointegration.py` | ⏳ Planned |
| 27 | Time series review — PerformanceAnalyzer sprint day 1. Research: first analyzer run | — | `src/analysis/performance_analyzer.py` | ⏳ Planned |

### Days 28–30 — Month 1 Checkpoint

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 28 | Week 4 self-assessment — PerformanceAnalyzer sprint day 2. Research: full analyzer run all strategies | — | `src/analysis/performance_analyzer.py` | ⏳ Planned |
| 29 | Stochastic intro: random walks & Gambler's ruin — README & requirements.txt. Research: month 1 assessment | `simulate_random_walk`, `gamblers_ruin_probability` | `src/stats/stochastic.py` | ⏳ Planned |
| 30 | Month 1 final test suite — GitHub push. Research: strategy shortlist (≤3 candidates) ⚠️ Hard deadline — written one-page specification for each candidate strategy also due today. Specifications must be complete before entering Phase 2. | — | `tests/`, `research/strategy_specs/` | ⏳ Planned |

---

## Phase 2: Research Engine (Days 31–51)

### Days 31–37 — Optimization

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 31 | Convex optimization fundamentals — gradient descent module. Research: AML Ch.2 data bars | `gradient_descent` | `src/stats/optimization.py` | ⏳ Planned |
| 32 | Markowitz optimization — Markowitz optimizer. Research: Markowitz on forex | `markowitz_weights` | `src/analysis/portfolio.py` | ⏳ Planned |
| 33 | Efficient frontier — efficient frontier module. Research: efficient frontier forex | `efficient_frontier` | `src/analysis/portfolio.py` | ⏳ Planned |
| 34 | Risk parity — risk parity optimizer. Research: Markowitz vs risk parity | `risk_parity_weights` | `src/analysis/portfolio.py` | ⏳ Planned |
| 35 | scipy.optimize & SLSQP — Markowitz via scipy. Research: AML Ch.4 triple barrier | `constrained_optimize` | `src/stats/optimization.py` | ⏳ Planned |
| 36 | Triple barrier worked example — triple barrier labels. Research: week 5 review + optimization tests | `triple_barrier_labels` | `src/signals/triple_barrier.py` | ⏳ Planned |
| 37 | Bootstrap methods — block bootstrap. Research: block bootstrap Sharpe CI | `bootstrap_resample` | `src/evaluation/bootstrap_ci.py` | ⏳ Planned |

### Days 38–44 — Advanced Stats + Framework Build

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 38 | Permutation tests — permutation test module. Research: permutation vs t-test | `permutation_test` | `src/evaluation/significance.py` | ⏳ Planned |
| 39 | Purged cross-validation — PurgedKFold. Research: CV leakage comparison | `purged_cross_validation` | `src/evaluation/cross_validation.py` | ⏳ Planned |
| 40 | IC/IR & fundamental law — IC/IR module. Research: IC/IR on all signals | `information_coefficient`, `information_ratio` | `src/analysis/performance_analyzer.py` | ⏳ Planned |
| 41 | CAPM & factor betas — CAPM module. Research: CAPM decomposition forex | `capm_expected_return` | `src/analysis/factor_models.py` | ⏳ Planned |
| 42 | PCA factor extraction — PCA factor model. Research: week 6 review | `pca_factor_decomposition` | `src/analysis/factor_models.py` | ⏳ Planned |
| 43 | Framework rebuild: architecture design — DataLoader class. Research: IC analysis EUR/USD momentum | `load_ohlcv`, `resample_to_daily`, `compute_log_returns` | `src/data/loader.py` | ⏳ Planned |
| 44 | SignalBuilder design — SignalBuilder class. Research: DataLoader + SignalBuilder pipeline test | `SignalBuilder` | `src/signals/signal_builder.py` | ⏳ Planned |

### Days 45–51 — Signal Construction Sprint + Stochastic Processes

> ⚠️ **Restructured Block — Signal Construction Is Now The Primary Deliverable**
> Original Days 45–49 have been resequenced to prioritize strategy signal construction.
> WalkForwardValidator is moved earlier (Day 45) to unblock signal runs by Day 49.
> Polars migration and parallel WalkForward are non-critical-path optimizations — both deferred to the buffer block (Days 72–77).
> Each strategy signal should arrive here fully specified on paper from Day 30. Day 45 onward is execution only.

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 45 | WalkForward design — WalkForwardValidator skeleton. Research: USD/JPY mean-reversion hypothesis | `walk_forward_validate` | `src/evaluation/walk_forward.py` | ⏳ Planned |
| 46 | Strategy 1 signal construction — implement Strategy 1 signal logic through SignalBuilder. Research: Strategy 1 first pipeline run | `SignalBuilder` (Strategy 1 implementation) | `src/signals/signal_builder.py` | ⏳ Planned |
| 47 | Strategy 2 signal construction — implement Strategy 2 signal logic through SignalBuilder. Research: Strategy 2 first pipeline run | `SignalBuilder` (Strategy 2 implementation) | `src/signals/signal_builder.py` | ⏳ Planned |
| 48 | Strategy 3 signal construction — implement Strategy 3 signal logic through SignalBuilder. Research: Strategy 3 first pipeline run | `SignalBuilder` (Strategy 3 implementation) | `src/signals/signal_builder.py` | ⏳ Planned |
| 49 | Run all 3 strategies through WalkForward — SignalReport design + initial outputs for all candidates. Research: week 7 review + strategy comparison table | `SignalReport`, `walk_forward_validate` | `src/analysis/signal_report.py`, `src/evaluation/walk_forward.py` | ⏳ Planned |
| 50 | Brownian motion & GBM — GBM simulator. Research: Monte Carlo drawdown analysis | `simulate_brownian_motion`, `simulate_gbm` | `src/stats/stochastic.py` | ⏳ Planned |
| 51 | OU process simulation — OU simulator & stress test. Research: month 2 assessment + strategy candidate selection decision | `simulate_ou` | `src/stats/stochastic.py` | ⏳ Planned |

---

## Phase 3: Institutional Output (Days 52–90)

### Days 52–60 — Portfolio Construction + Volatility Modeling

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 52 | Kelly criterion — Kelly position sizing. Research: Kelly sizing on validated signals | `kelly_fraction`, `fractional_kelly` | `src/analysis/portfolio.py` | ⏳ Planned |
| 53 | VaR & CVaR theory — VaR & CVaR module. Research: VaR/CVaR all strategies | `var_historical`, `var_parametric`, `var_monte_carlo`, `cvar` | `src/analysis/portfolio.py` | ⏳ Planned |
| 54 | Correlation regime shifts — regime shift detector. Research: regime shifts all pairs | `regime_shift_detector` | `src/stats/correlation.py` | ⏳ Planned |
| 55 | GARCH(1,1) theory — GARCH module. Research: GARCH on all forex pairs | `fit_garch` | `src/features/volatility.py` | ⏳ Planned |
| 56 | Volatility regimes — vol regime classifier. Research: regime-conditional performance | `vol_regime_classifier` | `src/features/volatility.py` | ⏳ Planned |
| 57 | Transaction costs & microstructure — transaction cost model. Research: transaction cost breakeven | `transaction_cost_model` | `src/analysis/performance_analyzer.py` | ⏳ Planned |
| 58 | CLI design with Click — CLI runner. Research: CLI run all pairs both signals | `cli_runner` | `run_research.py` | ⏳ Planned |
| 59 | Docker theory — Dockerfile & .dockerignore. Research: week 8 review | — | `Dockerfile` | ⏳ Planned |
| 60 | Performance profiling — profile & optimize. Research: GitHub framework v1 push | — | `src/` | ⏳ Planned |

### Days 61–71 — Paper Writing Track

> **Note on the paper block:** 8 days is sufficient given that writing the paper is a review mechanism —
> you are articulating methodology and results from work already completed across 60 days of builds.
> The paper is not being written cold. It is being written by someone who built every line of the framework
> it describes. Each section should come quickly as a result.
> Apply multiple testing correction (deflated Sharpe, Benjamini-Hochberg) to the full set of strategies
> tested — not only the winner — and document the selection methodology explicitly in the paper.
> This is what separates a defensible research document from a p-hacking narrative.

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 61 | Paper abstract — paper repo setup. Research: paper data section | — | `paper/forex_systematic_research.md` | ⏳ Planned |
| 62 | Paper introduction. Research: paper Python pass — update code to match paper | — | `paper/forex_systematic_research.md` | ⏳ Planned |
| 63 | Paper methodology part 1 — signal construction. Research: paper performance tables | — | `paper/forex_systematic_research.md` | ⏳ Planned |
| 64 | Paper results — in-sample. Research: paper results OOS | — | `paper/forex_systematic_research.md` | ⏳ Planned |
| 65 | Paper risk analysis. Research: paper plot suite | — | `paper/forex_systematic_research.md` | ⏳ Planned |
| 66 | Paper limitations — include explicit section on strategy selection methodology and multiple testing correction applied. Research: paper transaction cost table | — | `paper/forex_systematic_research.md` | ⏳ Planned |
| 67 | Paper full revision pass. Research: framework README polish | — | `paper/forex_systematic_research.md` | ⏳ Planned |
| 68 | Paper second revision pass. Research: quant-stats README polish | — | `paper/forex_systematic_research.md` | ⏳ Planned |
| 69 | LinkedIn update — Cornell alumni networking outreach drafts | — | `research/networking/day69_outreach_drafts.md` | ⏳ Planned |
| 70 | Internship target research — application materials draft | — | `research/` | ⏳ Planned |
| 71 | Final framework audit (docstrings, type hints, flake8) — week 10 review + deployment plan finalize | — | `research/deployment/live_deployment_plan.md` | ⏳ Planned |

### Days 72–77 — Buffer + Overflow Sprint ⚙️

> Priority order for this block has been updated to include deferred items from the Days 45–51 restructure.

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 72–77 | Flexible buffer block — priority order: (1) paper weak sections; (2) parallel WalkForward (deferred from Day 46); (3) Parquet migration (deferred from Day 48); (4) Polars migration (deferred from Day 49, lowest priority — only if time permits); (5) live deployment monitoring dashboard; (6) additional networking outreach | — | — | ⏳ Planned |

### Days 79–89 — Live Monitoring + Next Phase Prep ⚙️

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 79–89 | Flexible operational block — monitor deployment (P&L, Sharpe, drawdown weekly), respond to networking replies, read Month 2 candidate papers, scope next 90-day plan | — | — | ⏳ Planned |

### Day 90 — Final Assessment + Final Commit

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 90 | Read Day 1 journal → write 90-day final assessment → final commit across all repos → full test suite → CLI end-to-end run EUR/USD, GBP/USD, USD/JPY → confirm Docker builds. Commit: `v1.0.0 — 90-day quant acceleration complete` | — | `research/90day_final_assessment.md` | ⏳ Planned |

---

## Target Framework Structure at Day 90

```
src/
├── analysis/
│   ├── factor_models.py        ← Days 41–42
│   ├── performance_analyzer.py ← Days 6–7, 13, 40, 57
│   ├── portfolio.py            ← Days 32–34, 52–53
│   ├── portfolio_stats.py      ← Day 2 ✅
│   └── signal_report.py        ← Day 49
├── data/
│   ├── loader.py               ← Days 43, 48–49 (Parquet: buffer)
│   ├── stationarity.py         ← Days 20, 22
│   └── time_series.py          ← Day 23
├── evaluation/
│   ├── bootstrap_ci.py         ← Days 11, 37 ✅
│   ├── cross_validation.py     ← Day 39
│   ├── significance.py         ← Days 12, 14, 38
│   └── walk_forward.py         ← Days 45, 49 (parallel: buffer)
├── features/
│   ├── pca.py                  ← Days 18–19, 42
│   └── volatility.py           ← Days 55–56
├── signals/
│   ├── cointegration.py        ← Days 24–26
│   ├── signal_builder.py       ← Days 44–48
│   └── triple_barrier.py       ← Day 36
└── stats/
    ├── correlation.py          ← Days 5, 54 ✅
    ├── distributions.py        ← Days 1, 4 ✅
    ├── hypothesis_tests.py     ← Days 8–10 ✅
    ├── optimization.py         ← Days 31, 35
    ├── regression.py           ← Days 15–17
    └── stochastic.py           ← Days 29, 50–51

run_research.py                 ← Day 58 (CLI entry point)
Dockerfile                      ← Day 59
paper/
└── forex_systematic_research.md ← Days 61–68
research/
├── audit/                      ← Days 4, 7 ✅
├── deployment/
│   └── live_deployment_plan.md ← Day 71
├── networking/
│   └── day69_outreach_drafts.md ← Day 69
├── strategy_specs/             ← Day 30 ⚠️ Hard deadline
│   ├── strategy_1_spec.md
│   ├── strategy_2_spec.md
│   └── strategy_3_spec.md
└── 90day_final_assessment.md   ← Day 90
```

---