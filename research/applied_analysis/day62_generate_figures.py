from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.portfolio import efficient_frontier, minimum_variance_portfolio
from src.features.garch import fit_garch
from src.framework.data_loader import DataLoader, SUPPORTED_PAIRS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = sorted(SUPPORTED_PAIRS)
DEV_START = "2015-01-01"
DEV_END = "2022-12-31"
DPI = 300
GRID_ROWS, GRID_COLS = 5, 2


def _load_overshoot_reports() -> list[dict]:
    paths = sorted(glob.glob(str(RESULTS_DIR / "*_intraday-overshoot.json")))
    if not paths:
        raise FileNotFoundError(
            f"No *_intraday-overshoot.json reports in {RESULTS_DIR}. "
            "Run research/run_research_overshoot.py --pairs all first."
        )
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def _load_all_prices() -> dict[str, pd.Series]:
    """Load daily close prices for all 10 pairs. Rebuilds from raw 1-minute
    CSVs every run (no caching), consistent with the rest of the project."""
    prices = {}
    for pair in PAIRS:
        loader = DataLoader(pairs=[pair], start=DEV_START, end=DEV_END, data_dir=str(DATA_DIR))
        prices[pair] = loader.load()[pair]
    return prices


def plot_window_ic(reports: list[dict]) -> Path:
    """Figure 1: walk-forward OOS IC per window, one line per pair."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for report in reports:
        wr = report["window_results"]
        dates = [pd.Timestamp(w["test_start"]) for w in wr]
        ic = [w["ic"] if w["ic"] is not None else np.nan for w in wr]
        ax.plot(dates, ic, marker="o", markersize=3, linewidth=1.0, alpha=0.8, label=report["pair"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Intraday Overshoot -- Walk-Forward OOS IC per Window (Spearman(disp_k, trade_return))")
    ax.set_xlabel("Test window start")
    ax.set_ylabel("IC")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "day62_rolling_ic_by_signal.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_walkforward_sharpe(reports: list[dict]) -> Path:
    """Figure 2: OOS Sharpe per walk-forward window, one bar chart per pair."""
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(13, 3.0 * GRID_ROWS), sharex=False)
    axes_flat = axes.flatten()

    for ax, report in zip(axes_flat, reports):
        wr = report["window_results"]
        idx = [w["window_idx"] for w in wr]
        sharpes = [w["sharpe"] for w in wr]
        missing = [s is None for s in sharpes]
        vals = [s if s is not None else 0.0 for s in sharpes]
        colors = ["lightgray" if m else ("seagreen" if v >= 0 else "firebrick") for m, v in zip(missing, vals)]
        ax.bar(idx, vals, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{report['pair']}", fontsize=10)
        ax.set_ylabel("Sharpe", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, axis="y")

    for ax in axes_flat[len(reports):]:
        ax.axis("off")

    fig.suptitle(
        "Intraday Overshoot -- Walk-Forward OOS Sharpe per Window, by Pair "
        "(gray = missing/unscored window)",
        y=1.0, fontsize=12,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "day62_walkforward_sharpe_distribution.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_efficient_frontier(prices_by_pair: dict[str, pd.Series]) -> Path:
    """Figure 3: efficient frontier across all 10 pairs."""
    rets = {}
    for pair in PAIRS:
        prices = prices_by_pair[pair]
        rets[pair] = np.log(prices / prices.shift(1)).dropna()
    returns = pd.concat(rets, axis=1, join="inner")
    returns.columns = PAIRS

    ann_factor = len(returns) / ((returns.index.max() - returns.index.min()).days / 365.25)
    mv = minimum_variance_portfolio(returns)
    frontier = efficient_frontier(returns, ann_factor=ann_factor, n_points=50)
    max_sharpe_idx = int(np.argmax(frontier["sharpes"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(frontier["volatilities"], frontier["returns"], color="steelblue", linewidth=2, label="Efficient Frontier")
    ax.scatter(
        frontier["volatilities"][max_sharpe_idx], frontier["returns"][max_sharpe_idx],
        color="crimson", marker="*", s=200, zorder=5, label="Max-Sharpe (grid)",
    )
    ax.scatter(
        np.sqrt(mv["variance"]), mv["return"],
        color="black", marker="x", s=80, zorder=5, label="Global Min-Variance",
    )
    ax.set_xlabel("Volatility (std. dev., daily)")
    ax.set_ylabel("Expected Return (daily)")
    ax.set_title(f"Efficient Frontier -- all 10 pairs ({DEV_START} to {DEV_END})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "day62_efficient_frontier.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_garch_overlay(prices_by_pair: dict[str, pd.Series]) -> Path:
    """Figure 4: GARCH(1,1) conditional vol overlaid on price, one panel per pair."""
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(13, 3.2 * GRID_ROWS), sharex=False)
    axes_flat = axes.flatten()

    for ax, pair in zip(axes_flat, PAIRS):
        prices = prices_by_pair[pair]
        log_ret = np.log(prices / prices.shift(1)).dropna()
        g = fit_garch(log_ret)
        cond_vol = g["conditional_vol"]
        aligned_prices = prices.reindex(cond_vol.index)

        ax.plot(aligned_prices.index, aligned_prices.values, color="steelblue", linewidth=0.9)
        ax.set_ylabel("Price", color="steelblue", fontsize=8)
        ax.tick_params(axis="y", labelcolor="steelblue", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)

        ax2 = ax.twinx()
        ax2.plot(cond_vol.index, cond_vol.values, color="firebrick", alpha=0.75, linewidth=0.9)
        ax2.set_ylabel("Cond. vol", color="firebrick", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="firebrick", labelsize=7)

        ax.set_title(f"{pair} (persistence={g['persistence']:.3f})", fontsize=9)
        ax.grid(alpha=0.25)

    for ax in axes_flat[len(PAIRS):]:
        ax.axis("off")

    fig.suptitle("Price vs. GARCH(1,1) Conditional Volatility, by Pair", y=1.0, fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / "day62_garch_conditional_vol_overlay.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    reports = _load_overshoot_reports()
    prices_by_pair = _load_all_prices()

    outputs = [
        plot_window_ic(reports),
        plot_walkforward_sharpe(reports),
        plot_efficient_frontier(prices_by_pair),
        plot_garch_overlay(prices_by_pair),
    ]
    print("Figures written:")
    for path in outputs:
        size_kb = path.stat().st_size / 1024
        print(f"  {path.relative_to(REPO_ROOT)}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
