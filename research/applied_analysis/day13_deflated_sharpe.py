"""Deflated Sharpe Ratio applied to current walk-forward strategy output.

Replaces the earlier version, which was pointed at the archived
FVG_BoS_Reversal results (66-79 non-zero daily observations spread over 15
years) and had not been executable since `PerformanceAnalyzer.__init__`
dropped its `ann_factor` argument.

Method
------
For every report in results/, the out-of-sample bar-level P&L series is
rebuilt by replaying the exact pipeline `run_research.py` used to produce that
report. Parameters are read back out of the JSON rather than re-specified, so
the reconstruction cannot silently drift from the published run. Per-window
P&L is `exposure.shift(1) * log_returns` restricted to the test slice, matching
`run_research.window_sharpe`, and the windows are concatenated into one pooled
out-of-sample series per configuration.

The DSR is then evaluated against the number of configurations actually
searched. Six reports exist (3 pairs x 2 signals), so n_trials = 6 is the
documented grid. That is a LOWER BOUND on the true trial count: lookback
values, the reversion entry threshold, and window geometry were all chosen at
some point and none of that exploration is recorded. A sensitivity sweep over
n_trials is printed for that reason.

Annualization
-------------
`PerformanceAnalyzer.deflated_sharpe_ratio` de-annualizes its input via
`compute_ann_factor()`, which divides observation count by the calendar span of
the returns index. The pooled series is discontinuous -- test windows are
separated by training and embargo regions -- so its raw span would understate
bars per year and inflate the de-annualized Sharpe. The analyzer is therefore
handed a synthetic contiguous index calibrated to the sample-level
annualization factor recorded in the report. The round trip is asserted.

All figures are gross of transaction costs and unsized, consistent with the
underlying reports. They are signal-level diagnostics, not tradable
performance.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from research.run_research import SIGNAL_REGISTRY
from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.framework.data_loader import DataLoader
from src.framework.walk_forward import WalkForwardValidator

RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = Path(os.environ.get("QUANT_DATA_DIR", REPO_ROOT.parent / "data"))
TRIAL_SWEEP = (1, 6, 25, 100, 500)


def load_reports() -> list[dict]:
    """Read every report in results/, sorted for deterministic output."""
    paths = sorted(RESULTS_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No reports in {RESULTS_DIR}. Run run_research.py first."
        )
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def pooled_oos_pnl(report: dict, price_cache: dict[str, pd.Series]) -> pd.Series:
    """Rebuild the pooled out-of-sample P&L series for one report.

    Replays the report's own parameters through the same signal, exposure and
    walk-forward machinery `run_research.py` used, then concatenates the
    per-window test-slice P&L.
    """
    pair = report["pair"]
    params = report["parameters"]
    spec = SIGNAL_REGISTRY[report["signal"]]

    if pair not in price_cache:
        loader = DataLoader(
            pairs=[pair],
            start=str(pd.Timestamp(params["train_start"]).date()),
            end=str(pd.Timestamp(params["train_end"]).date()),
            embargo_days=params["embargo_days"],
            data_dir=str(DATA_DIR),
        )
        price_cache[pair] = loader.load()[pair]

    prices = price_cache[pair]
    frame = prices.to_frame(name="price")
    log_returns = np.log(prices / prices.shift(1))

    signal = spec["signal_fn"](frame, params["lookback"])
    exposure = spec["exposure_fn"](signal)

    validator = WalkForwardValidator(
        signal_fn=spec["signal_fn"],
        data=frame,
        n_windows=params["n_windows"],
        train_years=params["train_years"],
        test_months=params["test_months"],
        embargo_days=params["embargo_days"],
    )

    chunks = []
    for window in validator.generate_windows():
        mask = (signal.index >= window["test_start"]) & (
            signal.index < window["test_end"]
        )
        test_index = signal.index[mask]
        pnl = (
            exposure.reindex(test_index).shift(1) * log_returns.reindex(test_index)
        ).dropna()
        chunks.append(pnl)

    return pd.concat(chunks)


def calibrated_analyzer(pnl: pd.Series, ann_factor: float) -> PerformanceAnalyzer:
    """Wrap pooled P&L in an index whose empirical ann factor equals the target.

    The pooled series is discontinuous, so its own index cannot be used for
    annualization. A synthetic evenly spaced index spanning
    `len(pnl) / ann_factor` years reproduces the intended factor.
    """
    n = len(pnl)
    span_days = (n / ann_factor) * 365.25
    index = pd.Timestamp("2000-01-01") + pd.to_timedelta(
        np.linspace(0.0, span_days, n), unit="D"
    )
    analyzer = PerformanceAnalyzer(returns=pd.Series(pnl.to_numpy(), index=index))

    recovered = analyzer.compute_ann_factor()
    if not np.isclose(recovered, ann_factor, rtol=1e-3):
        raise AssertionError(
            f"Annualization round trip failed: target {ann_factor:.4f}, "
            f"recovered {recovered:.4f}."
        )
    return analyzer


def main() -> None:
    reports = load_reports()
    n_trials_grid = len(reports)
    price_cache: dict[str, pd.Series] = {}
    rows = []

    for report in reports:
        pnl = pooled_oos_pnl(report, price_cache)
        ann = report["sample"]["annualization_factor"]

        sr_period = pnl.mean() / pnl.std()
        sr_annual = sr_period * np.sqrt(ann)
        skewness = float(stats.skew(pnl))
        kurtosis = float(stats.kurtosis(pnl))

        analyzer = calibrated_analyzer(pnl, ann)
        dsr = {
            n: analyzer.deflated_sharpe_ratio(
                observed_sharpe=sr_annual,
                n_trials=n,
                n_obs=len(pnl),
                skewness=skewness,
                kurtosis=kurtosis,
            )
            for n in TRIAL_SWEEP
        }

        rows.append(
            {
                "config": f"{report['pair']} {report['signal']}",
                "n_obs": len(pnl),
                "sharpe": sr_annual,
                "skew": skewness,
                "kurt": kurtosis,
                "dsr": dsr,
                "window_mean": report["sharpe_summary"]["mean"],
            }
        )

    print(
        "Deflated Sharpe Ratio -- pooled out-of-sample P&L from results/\n"
        f"Documented grid: {n_trials_grid} configurations "
        "(lower bound on true trial count)\n"
    )
    header = (
        f"{'Config':<26} {'n_obs':>6} {'Sharpe':>8} {'Skew':>7} {'Kurt':>8}  "
        + "".join(f"{'DSR N=' + str(n):>11}" for n in TRIAL_SWEEP)
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['config']:<26} {r['n_obs']:>6} {r['sharpe']:>8.3f} "
            f"{r['skew']:>7.3f} {r['kurt']:>8.3f}  "
            + "".join(f"{r['dsr'][n]:>11.4f}" for n in TRIAL_SWEEP)
        )

    print(
        "\nPooled Sharpe is computed on the concatenated test slices and is not\n"
        "the mean of per-window Sharpes; both are shown for cross-reference."
    )
    print(f"\n{'Config':<26} {'pooled SR':>10} {'mean window SR':>16}")
    print("-" * 54)
    for r in rows:
        print(f"{r['config']:<26} {r['sharpe']:>10.3f} {r['window_mean']:>16.3f}")

    survivors = [r["config"] for r in rows if r["dsr"][n_trials_grid] >= 0.95]
    print(
        f"\nConfigurations with DSR >= 0.95 at n_trials={n_trials_grid}: "
        f"{survivors if survivors else 'none'}"
    )


if __name__ == "__main__":
    main()
