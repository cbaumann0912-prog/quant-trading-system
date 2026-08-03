from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import click
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from research.run_research import (
    DELAY, EMBARGO_DAYS, END, EXIT_MIN, GARCH_MIN_TRAIN, K, SCAN_CLOSE, SCAN_OPEN,
    START, TEST_MONTHS, TRAIN_YEARS, VOL_RATIO_MIN_OBS, max_fitting_n_windows,
)
from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.analysis.portfolio import cvar, var_historical
from src.evaluation.significance import benjamini_hochberg_correction
from src.features.garch import fit_garch
from src.framework.data_loader import DataLoader
from src.framework.walk_forward import WalkForwardValidator
from src.signals.intraday_overshoot import build_overshoot_sessions, overshoot_trades
from src.stats.hypothesis_tests import t_test_mean

GARCH_DEV_START = "2015-01-01"
GARCH_DEV_END = "2022-12-31"
VAR_CONFIDENCE_LEVELS = (0.95, 0.99)
CVAR_CONFIDENCE = 0.95

STATS_SCHEMA_VERSION = 2

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("QUANT_DATA_DIR", REPO_ROOT.parent / "data"))
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
STATS_CACHE_PATH = TABLES_DIR / ".day63_pair_stats.json"

N_TRIALS_DOCUMENTED = 10
BH_ALPHA = 0.05


def load_reports() -> list[dict]:
    paths = sorted(glob.glob(str(RESULTS_DIR / "*_intraday-overshoot.json")))
    if not paths:
        raise FileNotFoundError(
            f"No *_intraday-overshoot.json reports in {RESULTS_DIR}. "
            "Run research/run_research_overshoot.py --pairs all first."
        )
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def _rebuild_sessions_and_trades(pair: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = build_overshoot_sessions(
        pair=pair, data_dir=DATA_DIR, start=START, end=END,
        ks=[K], entry_delays=[DELAY],
        scan_open=SCAN_OPEN, scan_close=SCAN_CLOSE, exit_min=EXIT_MIN,
        vol_ratio_min_obs=VOL_RATIO_MIN_OBS, garch_min_train=GARCH_MIN_TRAIN,
    )
    trades = overshoot_trades({pair: sessions}, k=K, delay=DELAY).set_index("date").sort_index()
    return sessions, trades


def _spearman(x: pd.Series, y: pd.Series) -> float:
    aligned = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
    if len(aligned) < 2 or aligned["x"].nunique() < 2 or aligned["y"].nunique() < 2:
        return float("nan")
    ic, _ = scipy_stats.spearmanr(aligned["x"].to_numpy(), aligned["y"].to_numpy())
    return float(ic)


def compute_pair_stats(pair: str) -> dict:
    """Full-sample (Table 1) and pooled-OOS (Table 3) stats for one pair.
    Both share the one expensive step: rebuilding sessions and trades."""
    sessions, trades = _rebuild_sessions_and_trades(pair)
    disp_col = f"disp_{K}"

    full_ic = _spearman(sessions[disp_col], trades["ret"].reindex(sessions.index))
    full_sharpe = (
        PerformanceAnalyzer(trades["ret"]).compute_sharpe() if len(trades) >= 2 else float("nan")
    )

    n_windows = max_fitting_n_windows(sessions[["sigma"]], TRAIN_YEARS, TEST_MONTHS, EMBARGO_DAYS)
    validator = WalkForwardValidator(
        signal_fn=None, data=sessions[["sigma"]], n_windows=n_windows,
        train_years=TRAIN_YEARS, test_months=TEST_MONTHS, embargo_days=EMBARGO_DAYS,
    )
    chunks = []
    for w in validator.generate_windows():
        mask = (trades.index >= w["test_start"]) & (trades.index < w["test_end"])
        chunks.append(trades.loc[mask, "ret"])
    pooled_oos = pd.concat(chunks) if chunks else pd.Series(dtype=float)

    if len(pooled_oos) >= 2:
        t_result = t_test_mean(pooled_oos, null_mean=0.0, confidence=1 - BH_ALPHA)
        ann = sessions.index
        ann_factor = len(ann) / ((ann.max() - ann.min()).days / 365.25)
        sr_period = pooled_oos.mean() / pooled_oos.std()
        pooled_sharpe = float(sr_period * np.sqrt(ann_factor))
        skewness = float(scipy_stats.skew(pooled_oos))
        kurtosis = float(scipy_stats.kurtosis(pooled_oos))
        span_days = (len(pooled_oos) / ann_factor) * 365.25
        synth_index = pd.Timestamp("2000-01-01") + pd.to_timedelta(
            np.linspace(0.0, span_days, len(pooled_oos)), unit="D"
        )
        analyzer = PerformanceAnalyzer(returns=pd.Series(pooled_oos.to_numpy(), index=synth_index))
        dsr = analyzer.deflated_sharpe_ratio(
            observed_sharpe=pooled_sharpe, n_trials=N_TRIALS_DOCUMENTED,
            n_obs=len(pooled_oos), skewness=skewness, kurtosis=kurtosis,
        )
        t_stat, p_value = float(t_result["t_stat"]), float(t_result["p_value"])

        var_95 = var_historical(pooled_oos, confidence=VAR_CONFIDENCE_LEVELS[0])
        var_99 = var_historical(pooled_oos, confidence=VAR_CONFIDENCE_LEVELS[1])
        cvar_95 = cvar(pooled_oos, confidence=CVAR_CONFIDENCE)
        max_dd = analyzer.compute_max_drawdown()["value"]
    else:
        pooled_sharpe = t_stat = p_value = dsr = float("nan")
        var_95 = var_99 = cvar_95 = max_dd = float("nan")

    loader = DataLoader(pairs=[pair], start=GARCH_DEV_START, end=GARCH_DEV_END, data_dir=str(DATA_DIR))
    daily_prices = loader.load()[pair]
    daily_log_ret = np.log(daily_prices / daily_prices.shift(1)).dropna()
    garch_persistence = float(fit_garch(daily_log_ret)["persistence"])

    return {
        "schema_version": STATS_SCHEMA_VERSION,
        "pair": pair,
        "full_ic": full_ic,
        "full_sharpe": full_sharpe,
        "n_trades_full": int(len(trades)),
        "n_obs_oos": int(len(pooled_oos)),
        "pooled_oos_sharpe": pooled_sharpe,
        "t_stat": t_stat,
        "p_value": p_value,
        "dsr": dsr,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "max_drawdown": max_dd,
        "garch_persistence": garch_persistence,
    }


def load_stats_cache() -> dict:
    if STATS_CACHE_PATH.exists():
        return json.loads(STATS_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_stats_cache(cache: dict) -> None:
    STATS_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def build_table2(reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        sharpes = [
            w["sharpe"] for w in report["window_results"]
            if w["sharpe"] is not None and not (isinstance(w["sharpe"], float) and np.isnan(w["sharpe"]))
        ]
        arr = np.array(sharpes, dtype=float)
        s = report["sharpe_summary"]
        rows.append({
            "pair": report["pair"],
            "n_windows": s["n"],
            "mean": s["mean"],
            "std": s["std"],
            "min": float(arr.min()) if arr.size else float("nan"),
            "max": float(arr.max()) if arr.size else float("nan"),
            "frac_positive": s["frac_positive"],
        })
    return rows


def render_table1(rows: list[dict]) -> str:
    lines = [
        "### Table 1 -- Full-sample metrics (2011-2023, unsplit, NOT out-of-sample)",
        "",
        "| Pair | IC (full sample) | Sharpe (full sample, annualized) | n trades |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['pair']} | {r['full_ic']:.4f} | {r['full_sharpe']:.4f} | {r['n_trades_full']} |")
    lines.append("")
    lines.append(
        "IC = Spearman(disp_k, trade_return) over every trade the pair produced, 2011-2023. "
        "Sharpe is the annualized Sharpe of the full, unsplit trade-return series -- raw, unsized, "
        "gross of costs. This table pools training and test periods and exists for comparability "
        "only; Tables 2 and 3 hold the out-of-sample numbers."
    )
    return "\n".join(lines)


def render_table2(rows: list[dict]) -> str:
    lines = [
        "### Table 2 -- OOS walk-forward Sharpe distribution",
        "",
        "| Pair | n windows | Mean | Std | Min | Max | % positive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['pair']} | {r['n_windows']} | {r['mean']:.4f} | {r['std']:.4f} | "
            f"{r['min']:.4f} | {r['max']:.4f} | {r['frac_positive']:.0%} |"
        )
    lines.append("")
    lines.append(
        "Per-window annualized Sharpe of that window's triggered trades only. "
        "5y train / 3mo test / 5-day embargo, k=2.0, entry_delay=5min."
    )
    return "\n".join(lines)


def render_table4(rows: list[dict]) -> str:
    lines = [
        "### Table 4 -- Risk metrics (Day 65)",
        "",
        "| Pair | VaR 95% | VaR 99% | CVaR 95% | Max drawdown | GARCH(1,1) persistence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['pair']} | {r['var_95']:.4f} | {r['var_99']:.4f} | {r['cvar_95']:.4f} | "
            f"{r['max_drawdown']:.2%} | {r['garch_persistence']:.4f} |"
        )
    lines.append("")
    lines.append(
        "VaR/CVaR are historical, computed on the pooled-OOS trade-return series from Table 3 "
        "(per-trade log return, not daily). Max drawdown uses that same series' cumulative return "
        "path, test windows concatenated in chronological order. The path isn't continuous in "
        "calendar time since training and embargo periods are excluded, so read it as a "
        "trade-sequence drawdown, not a calendar-time one. GARCH(1,1) persistence (alpha+beta) is "
        "fit on the pair's own daily FX log returns, 2015-2022 -- a property of the price series, "
        "unrelated to the overshoot signal. Cross-reference against day62's GARCH figure."
    )
    return "\n".join(lines)


def render_table3(rows: list[dict]) -> str:
    lines = [
        "### Table 3 -- Significance (pooled out-of-sample trades only)",
        "",
        f"n_trials = {N_TRIALS_DOCUMENTED} (all 10 pairs actually run at k=2.0, delay=5min). "
        f"BH alpha = {BH_ALPHA}.",
        "",
        "| Pair | n obs (OOS) | Pooled OOS Sharpe | t-stat | p-value | BH-significant | DSR |",
        "|---|---:|---:|---:|---:|:---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['pair']} | {r['n_obs_oos']} | {r['pooled_oos_sharpe']:.4f} | "
            f"{r['t_stat']:.4f} | {r['p_value']:.5f} | {'Yes' if r['bh_significant'] else 'No'} | "
            f"{r['dsr']:.4f} |"
        )
    lines.append("")
    lines.append(
        "Pooled OOS Sharpe concatenates only trades inside a walk-forward test window, never a "
        "training window, across all windows for that pair. It is not the mean of Table 2's "
        "per-window Sharpes, and not Table 1's full-sample Sharpe."
    )
    return "\n".join(lines)


@click.command()
@click.option("--pairs", default="all", show_default=True, help="Comma-separated pairs, or 'all'.")
def main(pairs: str) -> None:
    reports = load_reports()
    all_pairs = [r["pair"] for r in reports]
    requested = all_pairs if pairs.lower() == "all" else [p.strip().upper() for p in pairs.split(",")]

    cache = load_stats_cache()
    for pair in requested:
        cached = cache.get(pair)
        if cached is not None and cached.get("schema_version") == STATS_SCHEMA_VERSION:
            continue
        reason = "no cache" if cached is None else f"stale schema v{cached.get('schema_version')}"
        click.echo(f"{pair} ... rebuilding trade ledger ({reason}, ~20-25s)", nl=False)
        cache[pair] = compute_pair_stats(pair)
        save_stats_cache(cache)
        click.echo(" done")

    missing = [
        p for p in all_pairs
        if p not in cache or cache[p].get("schema_version") != STATS_SCHEMA_VERSION
    ]
    if missing:
        click.echo(
            f"\n{len(cache) - len(missing)}/{len(all_pairs)} pairs cached at current schema "
            f"(v{STATS_SCHEMA_VERSION}) in {STATS_CACHE_PATH.relative_to(REPO_ROOT)}. "
            f"Still missing/stale: {missing}. Re-run with --pairs {','.join(missing)} "
            f"(or --pairs all) to finish."
        )
        return

    t1_rows = [cache[p] for p in all_pairs]
    t3_prelim = [cache[p] for p in all_pairs]
    p_values = [r["p_value"] for r in t3_prelim]
    bh_rejected = benjamini_hochberg_correction(p_values, alpha=BH_ALPHA)
    for row, rej in zip(t3_prelim, bh_rejected):
        row["bh_significant"] = bool(rej)

    t4_rows = [cache[p] for p in all_pairs]
    t2_rows = build_table2(reports)

    body = "\n\n".join([
        "# Day 63 -- Results Tables (Intraday Overshoot, k=2.0, entry_delay=5min)",
        f"Generated from {len(reports)} reports in `results/`.",
        render_table1(t1_rows),
        render_table2(t2_rows),
        render_table3(t3_prelim),
        render_table4(t4_rows),
    ])

    out_path = TABLES_DIR / "day63_results_tables.md"
    out_path.write_text(body + "\n", encoding="utf-8")
    print(body)
    print(f"\nWritten: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
