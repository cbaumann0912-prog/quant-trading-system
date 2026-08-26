# Day 71 — Final Framework Audit

**Date:** 2026-08-05
**Scope:** All 42 modules under `src/` (167 functions, 7 classes)

---

## 1. What question was investigated?

Does the codebase meet the standard a professional reviewer would apply to an
unfamiliar research repository? Five criteria were checked: docstring and
type-hint completeness, logging coverage, absence of hardcoded paths and magic
numbers, absence of dead code, and error handling on all I/O.

## 2. Why does the question matter?

The framework's outputs are the evidence base for the paper. A result is only as
trustworthy as the process that produced it, and two of the criteria above are
research-integrity controls rather than style preferences: without logging, a run
cannot be audited after the fact; without centralized seeding, a result cannot be
reproduced from a single recorded value.

## 3. Methodology

Static analysis rather than reading by eye, so the pass is repeatable:

- `flake8` for style and unused imports.
- A custom AST walker for docstring coverage, type-hint coverage, I/O call sites,
  numeric default arguments, and `print` vs. logging usage.
- `grep` for seed literals and logging imports.
- Baseline test run captured before any edit; targeted reruns after.

## 4. Assumptions

- flake8 defaults with `--max-line-length=120` represent acceptable house style.
  Line length is the one arbitrary parameter.
- The AST walker's I/O detection is name-based (`read_csv`, `to_csv`, `open`,
  `savefig`, `load`, `dump`). It false-positives on any method named `load`,
  including `DataLoader.load`, so its output was treated as a candidate list
  requiring manual confirmation, not as a finding.
- Test pass/fail is taken as evidence that behaviour was preserved. Coverage was
  not measured, so this is weaker evidence than it appears.

## 5. Findings

### P0 — Logging coverage was zero

No `logging` import existed anywhere in 42 modules. One `print()` in
`data/time_series.py` was the only observability in the framework, and it
reported ARIMA convergence failures — the most decision-relevant warning in that
module — to a stdout stream nothing captures.

**Resolution.** Added `src/utils/logging_config.py` implementing the standard
library/application split: library code calls `get_logger(__name__)` only, and
`configure_logging()` is called once by the entry point. All 34 non-`__init__`
modules instrumented. `capture_warnings=True` routes `statsmodels` and `scipy`
convergence warnings into the run log, next to the result they may have
contaminated.

The instrumentation immediately surfaced something previously invisible: on a
two-pair 2015–2020 panel, the aligned data contains **1 missing value** that no
prior run reported.

### P0 — RNG seeds were scattered and mutually inconsistent

Seeds were hardcoded as function defaults in six modules and disagreed: `42` in
`evaluation/bootstrap.py`, `evaluation/significance.py`, and
`distributions.simulate_price_path`; `28` in `distributions.simulate_log_returns`
and all three functions in `stats/stochastic.py`.

Reproducibility was therefore per-function, not per-run. More seriously, a seed
buried in a default argument is an undocumented researcher degree of freedom:
nothing prevents a seed from being adjusted until a bootstrap p-value crosses a
threshold, and nothing in the output records that it happened.

**Resolution.** Added `src/utils/random_state.py` with one `DEFAULT_SEED`
(overridable via `QUANT_SEED`), a `get_rng()` factory, and `resolve_seed()` with
a type guard. All 12 seed sites converted to `seed: int | None = None`. `get_rng`
returns a fresh `Generator` per call, so a result does not depend on the order in
which functions were invoked.

### P1 — Nine unguarded I/O operations

`framework/data_loader.py` held six, including `pd.read_csv` against a live FRED
URL with no timeout, no retry, and no exception handling. Because pandas inherits
urllib's default of blocking indefinitely, a FRED outage would have hung a
research run rather than failed it.

**Resolution.** All nine guarded with specific exception types and diagnostic
messages naming the pair and path. The FRED fetch now distinguishes transient
failures (retried, 3 attempts, linear backoff) from non-transient ones, and
raises if any region fails — a partial refresh leaves the rate panel in a
mixed-vintage state, and a carry signal computed on it would be silently wrong.
`plot_acf_pacf` now closes its figure in a `finally` block, so a write failure
mid-loop cannot leak figures into matplotlib's global registry.

### P1 — Docstring gaps

Function-level coverage was already strong (150/160). Gaps were concentrated in
`signals/signal_builder.py` (5 public methods), `analysis/signal_report.py` (3),
plus one each in `data/stationarity.py` and `signals/regime_gated.py`. Separately,
**0 of 42 modules had a module-level docstring**.

**Resolution.** All 10 function docstrings written and all 42 module docstrings
added. Where a function encodes a research decision, the docstring records the
reasoning — why `compute_rolling_ic` steps by `window` rather than by 1, why
`validate_no_lookahead` is a falsification test whose pass does not certify
absence of leakage, why Spearman is the IC default.

### P2 — flake8: 152 issues → 0

145 cosmetic issues cleared with `autopep8`. Seven fixed by hand: one unused
`numpy` import, two lambda assignments converted to named functions, one
over-long signature, one bracket alignment, two long-line splits.

### P2 — Dead code

`reorganize_signals.py` at repo root is a one-shot migration script whose own
docstring instructs deleting it after use. Referenced nowhere. **Not removed** —
`git rm` was blocked by filesystem permissions. Pending manual removal.

---

## 6. Alternative explanations and challenges to the task spec

**The "no magic numbers — all in config YAML" criterion is too broad as written.**
The audit found 42 numeric defaults, but they are not one category:

- *Strategy parameters* — `build_overshoot_sessions(250, 500)`,
  `mean_reversion_ladder(0.5, 26)`, `classify_regime(1.5)`,
  `detect_correlation_regime_shifts(60)`. These belong in config.
- *Statistical and numerical constants* — `alpha=0.05`, `tol=1e-10`,
  `n_bootstrap=1000`, `max_iter=100`. These do **not**. Placing a significance
  level in a strategy config invites tuning alpha as a parameter, which is
  precisely the p-hacking surface this project exists to close.

Recommended three-way split: strategy parameters → YAML (Block 2); seeds → one
global run-level seed (**done**); statistical constants → named module constants
with a comment justifying the value.

**Passing tests are weaker evidence than they look.** Coverage was never measured.
The suite passing after ~250 edits shows nothing was obviously broken; it does not
show the logging statements are correct or that the new exception branches behave
as intended. The FRED retry path has **no test** and was verified only by reading.

**Two test-suite defects surfaced but were not fixed** (out of scope for Block 1):

1. `test_cointegration.py::test_johansen_real_data_three_pairs` and three
   `test_portfolio.py` tests parse ~900 MB of real minute CSVs inside a unit test.
   Unit tests should run against fixtures; real-data tests belong behind a marker.
2. The suite has no timeout plugin and no fast/slow markers, so a hang is
   indistinguishable from slowness.

---

## 7. Verification

| Criterion | Before | After |
|---|---|---|
| Missing function docstrings | 10 | **0** |
| Modules without module docstring | 42 / 42 | **0 / 42** |
| Missing/partial type hints | 2 | **0** |
| Modules with logging | 0 / 34 | **34 / 34** |
| Unguarded I/O operations | 9 | **0** |
| flake8 issues | 152 | **0** |
| Scattered RNG seeds | 12 sites, 2 values | **1 `DEFAULT_SEED`** |

**Tests:** 500+ passing across all 35 test files. One test
(`test_select_arima_order_skips_failed_combinations`) was updated from `capsys`
to `caplog`, since the warning it asserts on now routes through logging rather
than stdout.

**Regression control:** the `test_portfolio.py` slowness was confirmed
pre-existing, not introduced, by extracting `HEAD` to a scratch directory and
running the same file against original and current code under identical no-data
conditions. Both produced 30 passed / 1 skipped / 3 errors in 1.35s.

**Docker CLI** (`research/run_research.py --help`) still runs.

---

## 7b. Addendum — seed set to 28, comments removed, and a defect found while re-running

Two follow-up changes were requested after the main pass: remove all comments
project-wide, and set the seed to 28.

**Comments.** All 72 removed across `src/`, `tests/`, and `research/`. None were
inline comments in the PEP 8 sense (code followed by `#`); all were standalone
block comments. No functional comments (`noqa`, `type:`, shebangs, encoding
declarations) existed, so removal was mechanically safe. Verified by tokenizer:
0 remain. `src/` remains flake8-clean and all tests pass.

**Seed.** `DEFAULT_SEED` is now `28`.

### New P0 finding: `bootstrap_confidence_interval` was never reproducible

Tracing which results the seed change would move surfaced a defect the main audit
missed. `evaluation/bootstrap.py::bootstrap_confidence_interval` had **no seed
parameter at all** and drew from the legacy process-global generator via
`np.random.choice`. Every confidence interval it has ever produced is
unreproducible from the code.

This escaped the first pass because that pass searched for the string `seed`. A
function whose defect is the *absence* of a seed cannot be found that way — the
correct query is the set of call sites that consume randomness, not the set that
mentions it. Methodological note for future audits: enumerate RNG consumers, not
seed mentions.

**Resolution.** Signature extended with `seed: int | None = None`, routed through
`get_rng`. Verified reproducible; explicit overrides still work.

**Affected call sites:** `day11_sharpe_confidence_intervals.py:31`,
`day37_block_bootstrap_sharpe_ci.py:58,71`, and
`s06_intraday_overshoot/intraday_overshoot_section10_validation.py:277` — the
per-pair mean IR confidence interval for the primary validated strategy.

### Re-run: s01 conditional IC permutation tests

These four calls omitted `seed` and so moved from the old default of 42 to 28.
Re-run on identical data:

| Test | n | observed | p (seed 42) | p (seed 28) | Δ |
|---|---:|---:|---:|---:|---:|
| high_vol_regime | 415 | 0.06925 | 0.15485 | 0.15285 | −0.00200 |
| low_vol_regime | 519 | 0.00043 | 0.98901 | 0.98601 | −0.00300 |
| median_split_high | 454 | 0.08661 | 0.06693 | 0.06993 | +0.00300 |
| median_split_low | 455 | −0.04771 | 0.31369 | 0.33467 | +0.02098 |

**No conclusion changes.** Largest movement is 0.021. `median_split_high` is the
only p-value near a threshold and it moves *away* from 0.05 (0.0669 → 0.0699), so
it remains non-significant under either seed.

### Seed sensitivity of the bootstrap CI

Because the old CIs are unreproducible, the meaningful question is how much a CI
varies across seeds. Measured on a 10-element resampling base matching the
per-pair IR case, across 40 seeds:

| n_bootstrap | CI-low range | CI-high range |
|---|---:|---:|
| 3,000 (s06 setting) | 0.0088 | 0.0080 |
| 10,000 | 0.0042 | 0.0039 |

The published IR bounds are stable to roughly ±0.004 — small enough that no
qualitative claim depends on the seed, but the interval was never exactly
reproducible, and s06 uses the noisier 3,000-resample setting. Raising
`N_BOOTSTRAP` to 10,000 halves the Monte Carlo error at negligible cost.

### Not completed

The **full s06 section-10 re-run was not executed.** It requires loading all ten
pairs of raw minute data (~2.9 GB), which exceeds the per-command time limit
available in this environment. The exact published IR CI bounds have therefore
not been regenerated — only the magnitude of their seed sensitivity has been
bounded. This must be re-run locally before the paper is finalized.

Method note for the s01 re-run: to fit the time limit, daily closes were derived
with the string-sort/groupby path from `data_loader` rather than the research
script's `to_datetime` + `resample("D").last()`. Equivalence was verified before
use — on a 400,000-row slice both paths produced identical indexes and identical
values (346 daily observations, exact match).

---

## 8. Next steps

1. Re-run `s06_intraday_overshoot/intraday_overshoot_section10_validation.py`
   locally and update the IR CI wherever it is cited. Also re-run `day11` and
   `day37`, whose CIs were likewise unseeded.
2. Raise `N_BOOTSTRAP` in s06 from 3,000 to 10,000.
3. `git rm reorganize_signals.py` (blocked by permissions during this pass).
4. Block 2: `config/strategy_config.yaml` — strategy parameters only, per the
   three-way split in section 6.
5. Call `configure_logging()` in `research/run_research.py`, writing to a per-run
   log file, and record `DEFAULT_SEED` in the results manifest so a saved result
   carries the seed that produced it.
6. Add a test for the FRED retry and partial-failure path, mocking the endpoint.
   Currently unverified by anything but inspection.
7. Move the four real-data tests behind a `@pytest.mark.slow` marker and add
   `pytest-timeout`.
8. Audit any other function that consumes randomness without exposing a seed —
   the `bootstrap_confidence_interval` defect suggests the enumeration should be
   done by RNG consumption, not by grepping for `seed`.
