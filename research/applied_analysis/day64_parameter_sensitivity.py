from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.run_research import (
    EMBARGO_DAYS, END, EXIT_MIN, GARCH_MIN_TRAIN, SCAN_CLOSE, SCAN_OPEN,
    START, TEST_MONTHS, TRAIN_YEARS, VOL_RATIO_MIN_OBS, max_fitting_n_windows,
)
from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.framework.data_loader import SUPPORTED_PAIRS
from src.framework.walk_forward import WalkForwardValidator
from src.signals.intraday_overshoot import build_overshoot_sessions, overshoot_trades

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("QUANT_DATA_DIR", REPO_ROOT.parent / "data"))
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = REPO_ROOT / "paper" / "tables"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
GRID_CACHE_PATH = CACHE_DIR / ".day64_pair_grid.json"

KS = [1.0, 1.5, 2.0, 2.5, 3.0]
DELAYS = [0, 1, 5, 15, 30]
DPI = 300


def pooled_oos_sharpe(trades: pd.DataFrame, windows: list[dict]) -> float:
    """Pooled OOS Sharpe for one (k, delay): concatenate every test window's
    triggered trades (never training-window trades) and take the Sharpe of
    the pooled series. `trades` is already specific to one (k, delay),
    built by the caller via overshoot_trades(..., k=k, delay=delay).
    Mirrors day63's Table 3 methodology."""
    chunks = []
    for w in windows:
        mask = (trades.index >= w["test_start"]) & (trades.index < w["test_end"])
        chunks.append(trades.loc[mask, "ret"])
    pooled = pd.concat(chunks) if chunks else pd.Series(dtype=float)
    if len(pooled) < 2:
        return float("nan")
    try:
        return PerformanceAnalyzer(pooled).compute_sharpe()
    except ValueError:
        return float("nan")


def compute_pair_grid(pair: str) -> dict:
    """One pair's full (k, delay) grid of pooled-OOS Sharpe values."""
    sessions = build_overshoot_sessions(
        pair=pair, data_dir=DATA_DIR, start=START, end=END,
        ks=KS, entry_delays=DELAYS,
        scan_open=SCAN_OPEN, scan_close=SCAN_CLOSE, exit_min=EXIT_MIN,
        vol_ratio_min_obs=VOL_RATIO_MIN_OBS, garch_min_train=GARCH_MIN_TRAIN,
    )

    n_windows = max_fitting_n_windows(sessions[["sigma"]], TRAIN_YEARS, TEST_MONTHS, EMBARGO_DAYS)
    validator = WalkForwardValidator(
        signal_fn=None, data=sessions[["sigma"]], n_windows=n_windows,
        train_years=TRAIN_YEARS, test_months=TEST_MONTHS, embargo_days=EMBARGO_DAYS,
    )
    windows = validator.generate_windows()

    grid: dict[str, dict[str, float]] = {}
    for k in KS:
        grid[str(k)] = {}
        for delay in DELAYS:
            trades = overshoot_trades({pair: sessions}, k=k, delay=delay).set_index("date").sort_index()
            grid[str(k)][str(delay)] = pooled_oos_sharpe(trades, windows)

    return {"pair": pair, "n_windows": n_windows, "grid": grid}


def load_cache() -> dict:
    if GRID_CACHE_PATH.exists():
        return json.loads(GRID_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    GRID_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def render_heatmap(cache: dict) -> Path:
    """Average pooled-OOS Sharpe across all 10 pairs, one cell per (k, delay)."""
    pairs = sorted(cache)
    mean_grid = np.full((len(KS), len(DELAYS)), np.nan)
    for i, k in enumerate(KS):
        for j, delay in enumerate(DELAYS):
            vals = [cache[p]["grid"][str(k)][str(delay)] for p in pairs]
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            mean_grid[i, j] = np.mean(vals) if vals else np.nan

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mean_grid, cmap="RdYlGn", vmin=-np.nanmax(np.abs(mean_grid)), vmax=np.nanmax(np.abs(mean_grid)))
    ax.set_xticks(range(len(DELAYS)))
    ax.set_xticklabels([f"{d}min" for d in DELAYS])
    ax.set_yticks(range(len(KS)))
    ax.set_yticklabels([f"k={k}" for k in KS])
    ax.set_xlabel("Entry delay")
    ax.set_ylabel("Crossing threshold (k x session sigma)")
    ax.set_title(
        f"Intraday Overshoot -- Mean Pooled-OOS Sharpe across {len(pairs)} pairs\n"
        "(walk-forward test windows only, 5y train / 3mo test / 5-day embargo)"
    )
    for i in range(len(KS)):
        for j in range(len(DELAYS)):
            v = mean_grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color="black" if abs(v) < 0.6 * np.nanmax(np.abs(mean_grid)) else "white")

    prod_k_idx = KS.index(2.0)
    prod_d_idx = DELAYS.index(5)
    ax.add_patch(plt.Rectangle((prod_d_idx - 0.5, prod_k_idx - 0.5), 1, 1, fill=False, edgecolor="blue", linewidth=3))

    fig.colorbar(im, ax=ax, label="Mean pooled-OOS Sharpe")
    plt.tight_layout()
    out = FIGURES_DIR / "day64_parameter_sensitivity_heatmap.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


@click.command()
@click.option("--pairs", default="all", show_default=True, help="Comma-separated pairs, or 'all'.")
def main(pairs: str) -> None:
    all_pairs = sorted(SUPPORTED_PAIRS)
    requested = all_pairs if pairs.lower() == "all" else [p.strip().upper() for p in pairs.split(",")]

    cache = load_cache()
    for pair in requested:
        if pair in cache:
            continue
        click.echo(f"{pair} ... building {len(KS)}x{len(DELAYS)} grid (no cache, ~20-25s)", nl=False)
        cache[pair] = compute_pair_grid(pair)
        save_cache(cache)
        click.echo(" done")

    missing = [p for p in all_pairs if p not in cache]
    if missing:
        click.echo(
            f"\n{len(cache)}/{len(all_pairs)} pairs cached in {GRID_CACHE_PATH.relative_to(REPO_ROOT)}. "
            f"Still missing: {missing}. Re-run with --pairs {','.join(missing)} (or --pairs all) to finish."
        )
        return

    out = render_heatmap(cache)
    print(f"Heatmap written: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
