# Framework Map — 90-Day Quant Acceleration Plan

**Execution philosophy**: Derive everything yourself. Understand everything yourself. Implement efficiently.

Unless explicitly stated otherwise, every day in this framework follows the same execution philosophy: mathematics is the primary learning objective; research interpretation is the primary deliverable; Python implementation is an engineering task that should be completed efficiently, using AI whenever appropriate; time saved through implementation is always redirected toward deeper mathematical understanding or stronger research. This paragraph applies globally across all 90 days without requiring each day to restate it.

## Principle — Understanding Is the Scarce Resource

In modern quantitative research, implementation speed is no longer a durable competitive advantage. Mathematical intuition, experimental design, statistical reasoning, and research judgment remain scarce. Therefore this framework always prioritizes, in order:

1. Mathematical understanding
2. Research methodology
3. Scientific interpretation
4. Implementation

AI may accelerate implementation, but it may never replace understanding. If mathematical understanding is complete, implementation should be completed as efficiently as possible so that additional time can be invested into deeper research.

## Implementation Taxonomy

Every module in this framework falls into one of two categories. The distinction is recorded here as a design decision, not a per-module label. The governing principle: **mathematical understanding determines whether something must be learned from first principles; implementation determines only how efficiently that understanding is expressed in code.** AI may assist implementation whenever doing so does not reduce mathematical understanding.

**Implement yourself** — modules where the bug you hit, the dimension mismatch you fix, or the numerical edge case you encounter encodes the learning. These are also the topics a quant interviewer will probe by asking you to derive or reconstruct on a whiteboard. Writing them yourself is non-negotiable.

Covers: all of `src/stats/` (OLS, bootstrap CI, hypothesis tests, distributions, permutation test, DSR, power analysis, Bonferroni, BH); all signal logic in `src/signals/` (cointegration suite, triple barrier labels, signal functions inside SignalBuilder); `eigen_decomposition` and `pca` in `src/features/pca.py`; all of `src/stats/stochastic.py` (GBM, OU — the simulation code is the mathematical object); walk-forward validation and leakage-prevention mechanics in `src/evaluation/`; purged cross-validation; all portfolio math (Markowitz, Kelly, VaR/CVaR, risk parity, efficient frontier); GARCH; IC/IR; CAPM decomposition; regime detection and volatility regime classifier; gradient descent; `gamblers_ruin_probability`.

**AI assistance acceptable** — infrastructure where the implementation adds nothing beyond what you already understand conceptually, and where no interview will ever ask you to reconstruct it, as well as any mathematically-derived module once understanding has been fully demonstrated. The non-negotiable condition for any AI-assisted module: review every line before committing. If you cannot explain a line under questioning, rewrite it yourself until you can.

Covers: `DataLoader.load`, `.split_train_test`, `.get_returns` in `src/framework/data_loader.py` (pure I/O plumbing, Day 43 — the embargo/leakage-prevention *logic* inside `split_train_test`/`get_window` was implement-yourself; the class scaffolding and CSV-parsing plumbing around it was AI-assisted); `plot_acf_pacf` visualization wrapper (Day 22 — understand the ACF/PACF math and Ljung-Box statistic in the math block; the matplotlib code is not the learning); `fit_arima` statsmodels API wrapper (Day 23 — understand ARIMA structure and AIC-based order selection deeply; the API call is three lines); `constrained_optimize` scipy wrapper (Day 35 — understand SLSQP and KKT conditions from the math block; the scipy.optimize call is not the learning); `SignalReport` output formatting (Day 49); `transaction_cost_model` arithmetic implementations (Day 57 — understand the breakeven economics; the formulas are arithmetic); `cli_runner` with Click (Day 58); `Dockerfile` (Day 59); cProfile and line_profiler boilerplate (Day 60 — the vectorization fix stays hand-implemented); README and `requirements.txt` (Day 29).

**Reclaimed time rule**: time reclaimed through AI-assisted implementation — whether infrastructure or mathematically derived modules after understanding has been demonstrated — is always reinvested into deeper mathematical study, additional robustness analysis, research interpretation, literature review, or scientific writing. The objective is never to reduce work; it is to shift effort toward higher-leverage cognitive tasks. Specific redirects are documented in the calendar event descriptions for each affected day.

---

## Current Framework Structure (as of Day 43 — actual repo state, not the Day-11 snapshot this section originally held)

```
src/
├── analysis/
│   ├── factor_models.py        ← Days 41–42 (CAPM, interaction regression, PCA factor extraction)
│   ├── performance_analyzer.py ← Days 6–7, 13, 39 (IC/IR pulled forward), 40 (breadth refinement)
│   ├── portfolio.py             ← Days 32–34 (Markowitz, efficient frontier, risk parity)
│   └── portfolio_stats.py      ← Day 2
├── data/
│   ├── stationarity.py         ← Days 20, 22
│   └── time_series.py          ← Day 23
├── evaluation/
│   ├── bootstrap.py            ← Days 11, 37 (built as bootstrap.py, not bootstrap_ci.py as originally planned)
│   ├── cross_validation.py     ← Day 39
│   └── significance.py         ← Days 12, 14, 38
├── features/
│   └── pca.py                  ← Days 18–19
├── framework/
│   └── data_loader.py          ← Day 43 ✅ (class-based rebuild — not the src/data/loader.py function-based plan)
├── signals/
│   ├── cointegration.py        ← Days 24–26 (Engle-Granger, Johansen, OU half-life)
│   └── triple_barrier.py       ← Day 36
└── stats/
    ├── correlation.py          ← Day 5
    ├── distributions.py        ← Days 1, 4
    ├── hypothesis_tests.py     ← Days 8–10
    ├── optimization.py         ← Days 31, 35
    └── regression.py           ← Days 15–17

NOT YET BUILT (contrary to the original Day 29/50/51 plan): src/stats/stochastic.py — Day 29's
core deliverable (simulate_random_walk, gamblers_ruin_probability) was never written. Only the
AI-assisted infra half of Day 29 (README/requirements.txt) happened.
```

---

## Phase 1: Statistical Foundation (Days 1–30)

> ⚠️ **Parallel Track — Strategy Hypothesis Development — ACTUAL OUTCOME, not the original plan**
> The original plan: develop a written one-page spec for each of ≤3 candidate strategies, in margin time, complete by Day 30, handed to SignalBuilder at Day 44.
> **What actually happened:** the Day 30 deadline slipped. Formal spec-writing for the two named candidates (Momentum w/ ML Regime, OU Half-Life Mean Reversion) didn't happen until the Day 41 "Backlog Sprint," roughly 11 days late — both were tested and discarded the very next day (Day 42). The third candidate, PC2 Carry Regime, never got a formal one-page spec at all; it was evaluated directly through a sequence of daily audits (Days 18–19 origin, Days 38/40/41 formal tests) instead of the planned spec-first process. **All three original candidates failed statistical validation.** A fourth strategy, Volatility Regime Breakout/Mean-Reversion, not part of the original Day 30 shortlist, was designed from scratch on Day 43 to have a live candidate at all going into Phase 2's signal-construction sprint. This is recorded here plainly because it's a real process failure worth learning from, not something to paper over: the "written spec before any testing" discipline is sound, but it did not survive contact with an 11-day schedule slip in practice.

### Week 1 — Probability Fundamentals

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 1 | Normal & log-normal distributions — distributions module. Research: framework distribution audit | `normal_pdf`, `normal_cdf`, `lognormal_pdf`, `simulate_log_returns`, `simulate_price_path` | `src/stats/distributions.py` | ✅ Built |
| 2 | Expectation, variance & covariance — portfolio statistics module. Research: Sharpe ratio audit | `compute_covariance_matrix`, `compute_portfolio_variance`, `compute_portfolio_return` | `src/analysis/portfolio_stats.py` | ✅ Built |
| 3 | CLT & law of large numbers — CLT simulator module (later archived, not part of the production `src/` tree). Research: AQR momentum paper methodology notes | — (CLT simulator, archived) | `archive/` | ✅ Built |
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
| 11 | Confidence intervals — bootstrap CI module. Research: Sharpe CI on top strategy | `bootstrap_confidence_interval` | `src/evaluation/bootstrap.py` (built as `bootstrap.py`, not `bootstrap_ci.py` as originally planned) | ✅ Built |
| 12 | Multiple testing correction — significance module. Research: multiple testing on strategies | `bonferroni_correction`, `benjamini_hochberg_correction` | `src/evaluation/significance.py` | ✅ Built |
| 13 | Deflated Sharpe ratio — DSR module. Research: DSR for all strategies | `deflated_sharpe_ratio` | `src/analysis/performance_analyzer.py` | ✅ Built |
| 14 | Week 2 review — significance suite integration. Research: week 2 review | — | `src/evaluation/significance.py` | ✅ Built |

### Week 3 — Regression & Linear Algebra

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 15 | OLS normal equations — OLS from scratch, R²/residual diagnostics implemented alongside `fit_ols` in the same pass rather than deferred to Day 16. Research: OLS EUR/USD vs GBP/USD (hedge ratio 0.5596, R² 0.376, kurtosis 3.77) | `fit_ols` | `src/stats/regression.py` | ✅ Built |
| 16 | Adjusted R² & residual diagnostics — formalized into dedicated functions + test suite (`test_regression.py`) on top of Day 15's initial pass. Research: residual diagnostics EUR/USD ~ GBP/USD | `r_squared`, `adj_r_squared`, `residual_diagnostics` | `src/stats/regression.py` | ✅ Built |
| 17 | Ridge & Lasso regularization. Also: a repo-wide docstring pass across all of `src/` (not originally scheduled here). Research: ridge on EUR/USD next-day returns | `fit_ridge`, `fit_lasso` | `src/stats/regression.py` | ✅ Built |
| 18 | Eigendecomposition — numerically stable matrix ops module. Research: eigendecomposition on forex covariance matrix — early doubts about PC2's interpretability first flagged here (resolved/hedged pending Day 19's score time-series check) | `eigen_decomposition` | `src/features/pca.py` | ✅ Built |
| 19 | PCA from first principles — PCA module + full test suite. Research: PC score time-series analysis, yearly eigenstructure stability 2011–2025 — this is where PC2 as a research object actually originates (later formally discarded Day 41) | `pca` | `src/features/pca.py` | ✅ Built |
| 20 | Stationarity: ADF & KPSS — stationarity module, plus an Engle-Granger pre-test across all pairs pulled forward from Day 24. Research: stationarity all pairs | `check_stationarity`, `adf_test`, `kpss_test` | `src/data/stationarity.py` | ✅ Built |
| 21 | Week 3 review — regression test suite. Research: week 3 review | — | `src/stats/regression.py` | ✅ Built |

### Week 4 — Time Series & Cointegration

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 22 | ACF & PACF — ACF/PACF/Ljung-Box module + tests. Research: ACF/PACF on EUR/USD | `plot_acf_pacf`, `ljung_box_test` | `src/data/stationarity.py` | ✅ Built |
| 23 | ARMA/ARIMA structure — ARIMA module, AIC-based order selection. Research: ARIMA on forex pairs | `fit_arima` | `src/data/time_series.py` | ✅ Built |
| 24 | Cointegration: Engle-Granger — cointegration module (pre-tested early on Day 20). Research: cointegration all pair combinations | `engle_granger_test`, `cointegration_spread` | `src/signals/cointegration.py` | ✅ Built |
| 25 | Johansen cointegration — Johansen test module + tests. Research: Johansen 3-pair analysis | `johansen_test` | `src/signals/cointegration.py` | ✅ Built |
| 26 | Half-life of mean reversion — OU half-life module + tests. Research: half-life & z-score thresholds. This module later becomes the OU Half-Life Mean Reversion strategy candidate (discarded Day 42) | `ou_half_life` | `src/signals/cointegration.py` | ✅ Built |
| 27 | Time series review — PerformanceAnalyzer sprint day 1, core performance metrics implemented. Research: first analyzer run on raw pair returns | — | `src/analysis/performance_analyzer.py` | ✅ Built |

### Days 28–30 — Month 1 Checkpoint

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 28 | Week 4 self-assessment — PerformanceAnalyzer sprint day 2 (Jarque-Bera, Ljung-Box, tracking error added). Research: full analyzer run, raw return baselines for all 3 pairs | — | `src/analysis/performance_analyzer.py` | ✅ Built |
| 29 | Stochastic intro: random walks & Gambler's ruin — **not built.** Only the AI-assisted infra half of this day happened (README/`requirements.txt` update); the implement-yourself math (`simulate_random_walk`, `gamblers_ruin_probability`) was never written and `src/stats/stochastic.py` does not exist yet. This is a real, currently-unresolved gap, not a planning error — it needs to be picked up before Day 50 (GBM) and Day 51 (OU simulation) can build on the same module. | `simulate_random_walk`, `gamblers_ruin_probability` | `src/stats/stochastic.py` | ❌ Not built |
| 30 | Month 1 final test suite — GitHub push (done: full test suite exists for every module built through Day 28). Research: strategy shortlist ⚠️ hard deadline for a written one-page spec per candidate — **this deadline was missed.** No spec work is dated to Day 30 in the repo history; the actual spec-writing (for 2 of the 3 candidates) didn't happen until the Day 41 backlog sprint, ~11 days late (see the Parallel Track note above). | — | `tests/` | ⚠️ Partially built (tests ✅, strategy specs slipped to Day 41) |

---

## Phase 2: Research Engine (Days 31–51)

### Days 31–37 — Optimization

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 31 | Convex optimization fundamentals — gradient descent module. Research: AML Ch.2 data bars | `gradient_descent` | `src/stats/optimization.py` | ✅ Built |
| 32 | Markowitz optimization — Markowitz optimizer + tests. Research: Markowitz weights/sensitivity on forex | `markowitz_weights` | `src/analysis/portfolio.py` | ✅ Built |
| 33 | Efficient frontier — `efficient_frontier`, `minimum_variance_portfolio`, leverage-bounded sweep range + tests. Research: efficient frontier + GMV analysis on all 3 pairs (leverage non-binding check) | `efficient_frontier` | `src/analysis/portfolio.py` | ✅ Built |
| 34 | Risk parity — ERC risk parity optimizer + tests. Research: Markowitz vs risk parity comparison | `risk_parity_weights` | `src/analysis/portfolio.py` | ✅ Built |
| 35 | scipy.optimize & SLSQP — Markowitz ported to `scipy.optimize.minimize`, validated against the closed-form KKT solution. Research: AML Ch.3–4, triple-barrier method + overlapping-outcomes tradeoffs documented | `constrained_optimize` | `src/stats/optimization.py` | ✅ Built |
| 36 | Triple barrier worked example — vol-scaled triple-barrier labels, first-touch tie-breaking, + tests. Also: weekly reviews 1–5 backfilled to a consistent brief three-section format (they hadn't been written contemporaneously each week). Research: week 5 review + optimization tests | `triple_barrier_labels` | `src/signals/triple_barrier.py` | ✅ Built |
| 37 | Bootstrap methods — block bootstrap (preserves autocorrelation) + tests. Also: the empirical annualization factor was corrected/unified across the portfolio and evaluation modules. Research: block bootstrap Sharpe/std CI across all 3 pairs | `bootstrap_resample` | `src/evaluation/bootstrap.py` | ✅ Built |

### Days 38–44 — Advanced Stats + Framework Build

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 38 | Permutation tests — permutation test module (empirical p-values, directional testing) + tests. Research: validated the PC2 factor score against forward returns — **PC2 test #1: null** (pooled IC not significant), logged with multiple-testing correction | `permutation_test` | `src/evaluation/significance.py` | ✅ Built |
| 39 | Purged cross-validation — `purged_cross_validation` (3-condition overlap purging + embargo per AML Ch.7) + `paired_sign_permutation_test` + tests. `information_coefficient`/`information_ratio` were also built here, a day early (originally Day 40). Research: CV leakage comparison — **PC2 test #2: no leakage-inflation artifact found, still null** | `purged_cross_validation`, `information_coefficient`, `information_ratio` (pulled forward) | `src/evaluation/cross_validation.py`, `src/analysis/performance_analyzer.py` | ✅ Built |
| 40 | IC/IR & fundamental law — core functions were already built Day 39; this day's actual work was a methodology correction (run-level correlation-adjusted breadth estimate, replacing a naive raw run-count breadth) plus a follow-up note. Research: IC/IR on all signals | — (refinement of Day 39's module) | `src/analysis/performance_analyzer.py`, `research/notes/day40_pc2_breadth_followup.md` | ✅ Built |
| 41 | CAPM & factor betas — CAPM alpha/beta decomposition + interaction regression module, both with OLS diagnostics + tests. Research: **PC2 test #3 — conditional predictability closed as a null result across three independent tests; PC2 formally discarded.** Backlog sprint: strategy specs finally written (Momentum w/ ML Regime, OU Half-Life Mean Reversion) — ~11 days after the original Day 30 deadline | `capm_expected_return`, `interaction_regression` | `src/analysis/factor_models.py` | ✅ Built |
| 42 | PCA factor extraction — `pca_factor_decomposition` + tests; test coverage on `performance_analyzer.py`/`time_series.py` raised to 98%/100%. Research: week 6 review, plus same-day applied analysis on both remaining candidates — **Momentum w/ ML Regime discarded** (window-alignment bug corrected, no base momentum IC) and **OU Half-Life Mean Reversion discarded** (Tests 2/2b/2c failed). All 3 original candidates are now null. | `pca_factor_decomposition` | `src/analysis/factor_models.py` | ✅ Built |
| 43 | Framework rebuild: architecture design — DataLoader class (composition-root embargo, `get_window` for WalkForward) + tests. Research: expanding-vs-rolling permutation test; with all 3 original candidates dead, a 4th strategy (Volatility Regime Breakout/Mean-Reversion) was designed from scratch and fully specified (Sections 1–12) so Phase 2 has a live candidate | `DataLoader.load`, `.split_train_test`, `.get_window`, `.get_returns` | `src/framework/data_loader.py` | ✅ Built |
| 44 | SignalBuilder design + implementation — signal-agnostic class (`signal_fn(data, lookback) -> pd.Series` contract, causal by convention), IC/rolling-IC scoring delegated to `information_coefficient`, `validate_no_lookahead` mechanical leakage check. Found and worked around a real gap in `information_coefficient` (doesn't drop NaNs before `spearmanr`/`pearsonr`, which propagate NaN on any single missing pair) rather than silently patching it at the source. Research: DataLoader + SignalBuilder 3-pair pipeline test (`src/signals/momentum.py`, `research/applied_analysis/day44_pipeline_test.py`) surfaced two further real findings — (1) `compute_rolling_ic` was silently NaN-padding constant-signal windows (34% of 60-bar windows for a 78-day-lookback momentum signal), fixed to skip them and covered by `test_rolling_ic_skips_constant_signal_windows`; (2) mean rolling IC is systematically negative even after that fix, traced to `holding_period=26` forward returns being ~0.95 lag-1 autocorrelated (overlapping-outcomes problem), so naive rolling IC there isn't a trustworthy diagnostic without a breadth correction — full writeup in `research/daily_audit/day44_pipeline_test.md`. | `SignalBuilder`, `momentum_signal` | `src/signals/signal_builder.py`, `src/signals/momentum.py`, `tests/test_signal_builder.py`, `research/applied_analysis/day44_pipeline_test.py`, `research/daily_audit/day44_pipeline_test.md` | ✅ Built |

### Days 45–51 — Signal Construction Sprint + Stochastic Processes

> ⚠️ **Restructured Block — Signal Construction Is Now The Primary Deliverable**
> Original Days 45–49 have been resequenced to prioritize strategy signal construction.
> WalkForwardValidator is moved earlier (Day 45) to unblock signal runs by Day 49.
> Polars migration and parallel WalkForward are non-critical-path optimizations — both deferred to the buffer block (Days 72–77).
> Day 45 onward is execution only. Of the 4 candidate strategies developed across the project (the original 3 slipped to Days 41–42, not Day 30 as planned — see the Parallel Track note above), only one survived statistical validation: PC2 Carry Regime (Day 41), Momentum w/ ML Regime, and OU Half-Life Mean Reversion are all formally discarded. The Volatility Regime Breakout/Mean-Reversion strategy (spec finalized Day 43, `research/strategies/volatility_regime_breakout_mean_revert.md`) is the sole candidate arriving at SignalBuilder. Days 46–48, originally three separate strategy-build days, are repurposed: Day 46 builds the one surviving strategy, Days 47–48 cover its two unresolved engineering gaps (per-window regime refit, two-leg validation regression) rather than sitting empty.

| Day | Topic | Function(s) | Location | Status |
|-----|-------|-------------|----------|--------|
| 45 | WalkForward design — WalkForwardValidator skeleton. Research: USD/JPY mean-reversion hypothesis (already satisfied by the Day 43 spec — repurposed as buffer/read-ahead) | `walk_forward_validate` | `src/evaluation/walk_forward.py` | ⏳ Planned |
| 46 | Vol Regime strategy signal construction — implement both legs (momentum + mean-reversion ladder) through SignalBuilder. Research: first pipeline run (trial #5 for project-wide multiple-testing correction; reserve lockbox, do not touch) | `SignalBuilder` (Vol Regime Breakout/Mean-Reversion) | `src/signals/signal_builder.py` | ⏳ Planned |
| 47 | Per-window regime classifier refit — z-scoring/PCA refit inside each walk-forward training window (Section 4 gap). Research: check refit stability vs. Day 43 full-sample values | `SignalBuilder` (regime refit) | `src/signals/signal_builder.py` | ⏳ Planned |
| 48 | Two-leg interaction regression validation (Section 10) — condition-number/VIF gate, alternate-window + permutation robustness checks. Research: first validation run + per-leg verdict | (validation logic) | `src/evaluation/` | ⏳ Planned |
| 49 | Run Vol Regime strategy (both legs) through WalkForward — SignalReport design + outputs. Research: week 7 review + debug buffer (no cross-strategy comparison table — one candidate, not three) | `SignalReport`, `walk_forward_validate` | `src/analysis/signal_report.py`, `src/evaluation/walk_forward.py` | ⏳ Planned |
| 50 | Brownian motion & GBM — GBM simulator. Research: Monte Carlo drawdown analysis | `simulate_brownian_motion`, `simulate_gbm` | `src/stats/stochastic.py` | ⏳ Planned |
| 51 | OU process simulation — OU simulator & stress test. Research: month 2 assessment + validation go/no-go checkpoint (not a candidate-selection decision — only one candidate reached validation) | `simulate_ou` | `src/stats/stochastic.py` | ⏳ Planned |

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
│   ├── factor_models.py        ← Days 41–42 ✅
│   ├── performance_analyzer.py ← Days 6–7, 13, 39–40, 57 ✅ (through Day 40; Day 57 pending)
│   ├── portfolio.py            ← Days 32–34 ✅, 52–53 pending
│   ├── portfolio_stats.py      ← Day 2 ✅
│   └── signal_report.py        ← Day 49
├── data/
│   ├── stationarity.py         ← Days 20, 22 ✅
│   └── time_series.py          ← Day 23 ✅
├── evaluation/
│   ├── bootstrap.py            ← Days 11, 37 ✅ (built as bootstrap.py, not bootstrap_ci.py)
│   ├── cross_validation.py     ← Day 39 ✅
│   ├── significance.py         ← Days 12, 14, 38 ✅ (bonferroni + BH added Day 12)
│   └── walk_forward.py         ← Days 45, 49 (parallel: buffer)
├── features/
│   ├── pca.py                  ← Days 18–19 ✅, 42's PCA factor extraction actually lives in factor_models.py
│   └── volatility.py           ← Days 55–56
├── framework/
│   └── data_loader.py          ← Day 43 ✅ (class-based; Parquet migration deferred to buffer, Days 72–77)
├── signals/
│   ├── cointegration.py        ← Days 24–26 ✅
│   ├── signal_builder.py       ← Days 44–48
│   └── triple_barrier.py       ← Day 36 ✅
└── stats/
    ├── correlation.py          ← Days 5, 54 ✅ (Day 5; Day 54 pending)
    ├── distributions.py        ← Days 1, 4 ✅
    ├── hypothesis_tests.py     ← Days 8–10 ✅
    ├── optimization.py         ← Days 31, 35 ✅
    ├── regression.py           ← Days 15–17 ✅
    └── stochastic.py           ← Days 29, 50–51 — NOT STARTED (Day 29's core deliverable was skipped; see note above)

run_research.py                 ← Day 58 (CLI entry point)
Dockerfile                      ← Day 59
paper/
└── forex_systematic_research.md ← Days 61–68
research/                        ← actual structure differs from the original plan below
├── daily_audit/                 ← Days 1–43 ✅ (not `research/audit/` as originally named — one .md per day with a research deliverable; several days have none: 3, 6–7, 14, 21, 29–30, 36, 42, since those were week-reviews or genuinely skipped — see gaps noted above)
├── applied_analysis/            ← Days 4–43 ✅ (reproducible .py script per research day, not originally planned as its own directory)
├── weekly_reviews/               ← week01–week06.md ✅ (backfilled retroactively on Day 36, not written contemporaneously)
├── notes/                       ← ad hoc follow-ups (e.g. day40_pc2_breadth_followup.md, bootstrap block-length selection) — not originally planned as a directory
├── strategies/                  ← Day 41–42 (not `research/strategy_specs/` or `strategy_N_spec.md` as originally planned)
│   ├── momentum_w_ML_regime.md         ← discarded Day 42
│   ├── ou_halflife_mean_reversion.md   ← discarded Day 42
│   └── volatility_regime_breakout_mean_revert.md ← Day 43, sole surviving candidate
