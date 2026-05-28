# =============================================================================
# run_portfolio.py  —  Portfolio Backtest
# =============================================================================
#
# HOW TO RUN
# ----------
#   cd C:\Users\clayb\OneDrive\Desktop\quant\backtest
#   python run_portfolio.py
#
# Portfolio config: 5.5% portfolio cap, per-pair risk from RISK_PCT below.
#
# OUTPUT  →  results/portfolio/realistic/   and   results/portfolio/worst_case/
#   trades.csv        — combined trade log (all pairs)
#   equity.csv        — clean time/equity curve
#   equity_detail.csv — equity curve with event type + n_open
#   yearly.csv        — annual PnL breakdown
#   pairs.csv         — per-pair contribution summary
#   summary.csv       — scalar performance metrics
#   overlap.csv       — simultaneous-position statistics
#
# CHARTS → results/portfolio/charts/
#   figure_1_equity_realistic.png        — equity curve (portfolio + 3 pairs, realistic)
#   figure_2_equity_comparison.png       — realistic vs worst-case overlay  (WORST_CASE=True only)
#   figure_3_drawdown_realistic.png      — % drawdown from peak (realistic)
#   figure_4_yearly_returns.png          — annual returns bar chart (realistic)
#   figure_5_pair_contribution.png       — pair PnL contribution (realistic)
#   figure_6_trade_pnl_distribution.png  — per-trade PnL histogram (realistic)
# =============================================================================

import os
import sys
import io

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for box-drawing chars)
# line_buffering=True ensures prints appear immediately instead of being held in a buffer.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import matplotlib
matplotlib.use("Agg")   # non-interactive; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BACKTEST_DIR = os.path.dirname(os.path.abspath(__file__))   # .../quant/backtest/
_PROJECT_ROOT = os.path.dirname(_BACKTEST_DIR)                # .../quant/
if _BACKTEST_DIR not in sys.path:
    sys.path.insert(0, _BACKTEST_DIR)

from portfolio import run_portfolio_backtest, PORTFOLIO_STARTING_EQUITY
from sim_costs import reset_rng


# =============================================================================
# CONFIGURATION  —  edit here
# =============================================================================

# ---- Data files ----
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

PAIRS_CONFIG = [
    {
        "pair":       "EURUSD",
        "file_path":  os.path.join(DATA_DIR, "EURUSD.csv"),
        "start_date": None,
        "end_date":   None,
    },
    {
        "pair":       "GBPUSD",
        "file_path":  os.path.join(DATA_DIR, "GBPUSD.csv"),
        "start_date": None,
        "end_date":   None,
    },
    {
        "pair":       "USDJPY",
        "file_path":  os.path.join(DATA_DIR, "USDJPY.csv"),
        "start_date": None,
        "end_date":   None,
    },
]

# ---- Portfolio cap settings ----
PORTFOLIO_CAP = 0.055      # 5.5% total open-risk cap

RISK_PCT = {
    "EURUSD": 0.015,   # 1.5%
    "GBPUSD": 0.01,    # 1%
    "USDJPY": 0.015,   # 1.5%
}

# ---- Account ----
STARTING_EQUITY = 100_000.0

# ---- Output ----
_REALISTIC_DIR  = os.path.join(_PROJECT_ROOT, "results", "portfolio", "realistic")
_WORST_CASE_DIR = os.path.join(_PROJECT_ROOT, "results", "portfolio", "worst_case")
_CHART_DIR      = os.path.join(_PROJECT_ROOT, "results", "portfolio", "charts")

# ---- Verbosity ----
VERBOSE = True

# ---- Pair line colours (consistent across all charts) ----
_PAIR_COLORS = {
    "EURUSD": "#ff7f0e",   # orange
    "GBPUSD": "#2ca02c",   # green
    "USDJPY": "#9467bd",   # purple
}
_PORTFOLIO_REALISTIC_COLOR = "#1f77b4"   # blue
_PORTFOLIO_WORST_CASE_COLOR = "#d62728"  # red


# =============================================================================
# CHART HELPERS
# =============================================================================

def _portfolio_equity_series(equity_curve: list) -> tuple:
    """Return (times_array, equities_array) from a portfolio equity_curve list."""
    eq_df = pd.DataFrame(equity_curve)
    eq_df["time"] = pd.to_datetime(eq_df["time"])
    eq_df = eq_df.drop_duplicates(subset="time", keep="last").sort_values("time")
    return eq_df["time"].values, eq_df["equity"].values


def _pair_equity_series(per_pair_results: dict, starting_equity: float = 100_000.0) -> dict:
    """
    Extract (times, equities) for each pair from per_pair_results.
    Normalises each curve so the first equity value equals starting_equity.
    Since trailing.py uses STARTING_EQUITY = 100_000 the normalisation is
    essentially a no-op for standard runs.
    """
    curves = {}
    for pair, res in per_pair_results.items():
        eq_df = res["equity_df"]
        if eq_df is None or eq_df.empty:
            continue
        eq_df = eq_df.copy()
        eq_df["time"] = pd.to_datetime(eq_df["time"])
        eq_df = eq_df.drop_duplicates(subset="time", keep="last").sort_values("time")
        first_val = float(eq_df["equity"].iloc[0])
        norm = starting_equity / first_val if first_val > 0 else 1.0
        curves[pair] = (eq_df["time"].values, eq_df["equity"].values * norm)
    return curves


def _extend_equity_to(times: np.ndarray, equities: np.ndarray,
                       end_time: pd.Timestamp) -> tuple:
    """Extend an equity curve with a flat tail so it reaches end_time.

    The portfolio equity curve records one point per trade event, so it
    stops at the last trade closure.  The per-pair equity curves record
    every 5-minute bar and reach the end of the dataset.  Without this
    extension the portfolio line appears 'cut off' on the charts.

    Parameters
    ----------
    times    : sorted array of datetime64 timestamps
    equities : equity values aligned to times
    end_time : target end timestamp (typically the last bar of the data)

    Returns
    -------
    (times, equities) with a single extra point appended if needed
    """
    if len(times) == 0:
        return times, equities
    last_t = pd.Timestamp(times[-1])
    end_t  = pd.Timestamp(end_time)
    if last_t < end_t:
        times    = np.append(times,    np.datetime64(end_t))
        equities = np.append(equities, equities[-1])
    return times, equities


# ---------------------------------------------------------------------------
# Figure 1  —  Equity Curve: Realistic Portfolio + 3 Pairs
# ---------------------------------------------------------------------------

def plot_figure1_equity_realistic(
    portfolio_times,
    portfolio_equities,
    pair_curves: dict,
    out_path: str,
    starting_equity: float = 100_000.0,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        portfolio_times, portfolio_equities,
        label="Portfolio (Realistic)",
        color=_PORTFOLIO_REALISTIC_COLOR,
        linewidth=2.5, zorder=5,
    )

    for pair, (times, equities) in pair_curves.items():
        ax.plot(
            times, equities,
            label=pair,
            color=_PAIR_COLORS.get(pair, "gray"),
            linewidth=1.5, alpha=0.85, linestyle="--",
        )

    ax.axhline(starting_equity, color="black", linewidth=0.7, linestyle=":", alpha=0.4)

    ax.set_title("Portfolio Equity Curve — Realistic Execution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Equity ($)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Figure 2  —  Equity Curve: Realistic vs Worst-Case Overlay
# ---------------------------------------------------------------------------

def plot_figure2_equity_comparison(
    real_portfolio_times,
    real_portfolio_equities,
    real_pair_curves: dict,
    worst_portfolio_times,
    worst_portfolio_equities,
    worst_pair_curves: dict,
    out_path: str,
    starting_equity: float = 100_000.0,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    # Worst-case pair lines (drawn first so they sit behind the portfolio lines)
    for pair, (times, equities) in worst_pair_curves.items():
        ax.plot(
            times, equities,
            color=_PAIR_COLORS.get(pair, "gray"),
            linewidth=1.2, alpha=0.45, linestyle="--",
        )

    # Realistic pair lines
    for pair, (times, equities) in real_pair_curves.items():
        ax.plot(
            times, equities,
            color=_PAIR_COLORS.get(pair, "gray"),
            linewidth=1.2, alpha=0.55, linestyle=":",
        )

    # Worst-case portfolio line
    ax.plot(
        worst_portfolio_times, worst_portfolio_equities,
        label="Portfolio (Worst-Case)",
        color=_PORTFOLIO_WORST_CASE_COLOR,
        linewidth=2.5, linestyle="-.", zorder=5,
    )

    # Realistic portfolio line
    ax.plot(
        real_portfolio_times, real_portfolio_equities,
        label="Portfolio (Realistic)",
        color=_PORTFOLIO_REALISTIC_COLOR,
        linewidth=2.5, zorder=6,
    )

    # Pair legend proxies
    for pair, color in _PAIR_COLORS.items():
        ax.plot([], [], color=color, linewidth=1.5, linestyle="--",
                label=f"{pair} (Realistic --  / Worst-Case -- )")

    ax.axhline(starting_equity, color="black", linewidth=0.7, linestyle=":", alpha=0.4)

    ax.set_title(
        "Portfolio Equity Curve — Worst-Case Execution  (overlaid with Realistic)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Equity ($)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Figure 3  —  Drawdown Curve (Realistic)
# ---------------------------------------------------------------------------

def plot_figure3_drawdown(
    portfolio_times,
    portfolio_equities,
    out_path: str,
) -> None:
    equities = np.asarray(portfolio_equities, dtype=float)
    peak     = np.maximum.accumulate(equities)
    drawdown = (equities - peak) / peak * 100   # negative values

    max_dd_idx = int(np.argmin(drawdown))
    max_dd_val = float(drawdown[max_dd_idx])
    max_dd_time = pd.Timestamp(portfolio_times[max_dd_idx])

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(portfolio_times, drawdown, 0,
                    color="#d62728", alpha=0.30, label="Drawdown")
    ax.plot(portfolio_times, drawdown, color="#d62728", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

    # Annotate max drawdown
    ax.annotate(
        f"Max DD: {max_dd_val:.2f}%\n{max_dd_time.strftime('%Y-%m-%d')}",
        xy=(max_dd_time, max_dd_val),
        xytext=(max_dd_time, max_dd_val - 0.8),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=9, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_title("Portfolio Drawdown Curve — Realistic Execution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown from Peak (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Figure 4  —  Yearly Returns Bar Chart
# ---------------------------------------------------------------------------

def plot_figure4_yearly_returns(yearly_df: pd.DataFrame, out_path: str) -> None:
    if yearly_df.empty:
        return

    years   = yearly_df["year"].values
    returns = yearly_df["return_pct"].values
    colors  = ["#2ca02c" if r >= 0 else "#d62728" for r in returns]

    fig, ax = plt.subplots(figsize=(max(10, len(years) * 1.1), 6))

    bars = ax.bar(years, returns, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)

    # Value labels on bars
    for bar, ret in zip(bars, returns):
        y_offset = 0.25 if ret >= 0 else -0.25
        va       = "bottom" if ret >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ret + y_offset,
            f"{ret:+.1f}%",
            ha="center", va=va, fontsize=8.5, fontweight="bold",
        )

    ax.set_title("Annual Returns by Year — Realistic Execution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Calendar Year", fontsize=11)
    ax.set_ylabel("Annual Return (%)", fontsize=11)
    ax.set_xticks(years)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Figure 5  —  Pair Contribution Horizontal Bar Chart
# ---------------------------------------------------------------------------

def plot_figure5_pair_contribution(pair_df: pd.DataFrame, out_path: str) -> None:
    if pair_df.empty:
        return

    pairs     = pair_df["pair"].values
    pnls      = pair_df["total_pnl"].values.astype(float)
    pcts      = pair_df["pct_of_total_pnl"].values.astype(float)
    trades    = pair_df["trades"].values
    win_rates = pair_df["win_rate_pct"].values.astype(float)

    colors = [_PAIR_COLORS.get(p, "steelblue") for p in pairs]

    fig, ax = plt.subplots(figsize=(12, max(4, len(pairs) * 1.5)))

    bars = ax.barh(pairs, pnls, color=colors, edgecolor="white", linewidth=0.5, height=0.55)
    ax.axvline(0, color="black", linewidth=0.8)

    # Annotation: dollar PnL | % of total | trade count | win rate
    x_range = max(abs(pnls)) if len(pnls) else 1.0
    pad     = x_range * 0.015

    for bar, pnl_val, pct, n, wr in zip(bars, pnls, pcts, trades, win_rates):
        label = f"  ${pnl_val:,.0f}  ({pct:+.1f}% of total)  —  {n} trades  |  WR {wr:.0f}%"
        x_pos = pnl_val + pad if pnl_val >= 0 else pnl_val - pad
        ha    = "left"       if pnl_val >= 0 else "right"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha=ha, va="center", fontsize=9,
        )

    ax.set_title("PnL Contribution by Pair — Realistic Execution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total PnL ($)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Figure 6  —  Trade PnL Distribution Histogram
# ---------------------------------------------------------------------------

def plot_figure6_pnl_distribution(portfolio_trade_log: list, out_path: str) -> None:
    if not portfolio_trade_log:
        return

    tdf  = pd.DataFrame(portfolio_trade_log)
    pnls = tdf["total_portfolio_pnl"].astype(float).values

    fig, ax = plt.subplots(figsize=(12, 6))

    n_bins  = min(50, max(20, len(pnls) // 5))
    _, bins, patches = ax.hist(pnls, bins=n_bins, edgecolor="white", linewidth=0.4)

    # Colour bins by sign
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor("#2ca02c" if left >= 0 else "#d62728")

    ax.axvline(0, color="black", linewidth=1.8, linestyle="--", label="Zero PnL", zorder=5)

    n_pos  = int((pnls > 0).sum())
    n_neg  = int((pnls < 0).sum())
    n_zero = int((pnls == 0).sum())
    ax.text(
        0.97, 0.95,
        f"Trades: {len(pnls)}\nPositive: {n_pos}  |  Negative: {n_neg}  |  Zero: {n_zero}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    ax.set_title("Trade PnL Distribution — Realistic Execution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Per-Trade PnL ($)", fontsize=11)
    ax.set_ylabel("Trade Count", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Verify data files exist before loading anything
    for cfg in PAIRS_CONFIG:
        if not os.path.exists(cfg["file_path"]):
            raise FileNotFoundError(
                f"Data file not found: {cfg['file_path']}\n"
                f"Expected at: {DATA_DIR}"
            )

    os.makedirs(_CHART_DIR, exist_ok=True)

    print("=" * 64)
    print("  PORTFOLIO BACKTEST  —  REALISTIC + WORST-CASE")
    print(f"  Account  : ${STARTING_EQUITY:,.0f}")
    print(f"  Pairs    : {', '.join(c['pair'] for c in PAIRS_CONFIG)}")
    print(f"  Cap      : {PORTFOLIO_CAP:.1%}  |  "
          + "  ".join(f"{p} {r:.2%}" for p, r in RISK_PCT.items()))
    print("=" * 64)

    # ------------------------------------------------------------------
    # 1. Realistic scenario  (spread x1, slippage x1)
    # RNG is reset to seed=42 before every run so that fill prices are
    # identical across runs — the only source of non-determinism in this
    # codebase is the np.random.normal() slippage draw in sim_costs.py.
    # ------------------------------------------------------------------
    print("\n  >>> REALISTIC scenario")
    reset_rng(42)
    real_results = run_portfolio_backtest(
        pairs_config      = PAIRS_CONFIG,
        risk_pct_by_pair  = RISK_PCT,
        portfolio_cap_pct = PORTFOLIO_CAP,
        starting_equity   = STARTING_EQUITY,
        spread_mult       = 1.0,
        slip_mult         = 1.0,
        out_dir           = _REALISTIC_DIR,
        verbose           = VERBOSE,
    )

    # ------------------------------------------------------------------
    # 2. Worst-case scenario  (spread x2, slippage x2)
    # Independent seed reset — worst-case results are reproducible on
    # their own regardless of what ran before this call.
    # ------------------------------------------------------------------
    print("\n  >>> WORST-CASE scenario  (2x spread + 2x slippage)")
    reset_rng(42)
    worst_results = run_portfolio_backtest(
        pairs_config      = PAIRS_CONFIG,
        risk_pct_by_pair  = RISK_PCT,
        portfolio_cap_pct = PORTFOLIO_CAP,
        starting_equity   = STARTING_EQUITY,
        spread_mult       = 2.0,
        slip_mult         = 2.0,
        out_dir           = _WORST_CASE_DIR,
        verbose           = VERBOSE,
    )

    # ------------------------------------------------------------------
    # Build equity series for charting
    # ------------------------------------------------------------------
    real_ptimes,  real_pequities  = _portfolio_equity_series(real_results["equity_curve"])
    worst_ptimes, worst_pequities = _portfolio_equity_series(worst_results["equity_curve"])

    real_pair_curves  = _pair_equity_series(real_results["per_pair_results"],  STARTING_EQUITY)
    worst_pair_curves = _pair_equity_series(worst_results["per_pair_results"], STARTING_EQUITY)

    # The portfolio equity curve records one point per trade event; it
    # stops at the last trade closure.  Per-pair equity curves record
    # every 5m bar through the end of the dataset, so the portfolio line
    # appeared "cut off" when plotted alongside them.  Extend all curves
    # to the same end-of-data timestamp so every line reaches the right
    # edge of the chart.
    _all_end_times = [times[-1] for curves in (real_pair_curves, worst_pair_curves)
                      for times, _ in curves.values() if len(times)]
    if _all_end_times:
        _data_end     = max(pd.Timestamp(t) for t in _all_end_times)
        real_ptimes,  real_pequities  = _extend_equity_to(real_ptimes,  real_pequities,  _data_end)
        worst_ptimes, worst_pequities = _extend_equity_to(worst_ptimes, worst_pequities, _data_end)
        real_pair_curves  = {p: _extend_equity_to(t, e, _data_end)
                             for p, (t, e) in real_pair_curves.items()}
        worst_pair_curves = {p: _extend_equity_to(t, e, _data_end)
                             for p, (t, e) in worst_pair_curves.items()}

    # ------------------------------------------------------------------
    # Generate all charts
    # ------------------------------------------------------------------
    print(f"\n  Generating charts -> {_CHART_DIR}")

    # Figure 1 — Realistic equity: portfolio + 3 pairs
    plot_figure1_equity_realistic(
        real_ptimes, real_pequities,
        real_pair_curves,
        out_path        = os.path.join(_CHART_DIR, "figure_1_equity_realistic.png"),
        starting_equity = STARTING_EQUITY,
    )

    # Figure 2 — Worst-case overlaid with realistic
    plot_figure2_equity_comparison(
        real_ptimes,  real_pequities,  real_pair_curves,
        worst_ptimes, worst_pequities, worst_pair_curves,
        out_path        = os.path.join(_CHART_DIR, "figure_2_equity_comparison.png"),
        starting_equity = STARTING_EQUITY,
    )

    # Figure 3 — Drawdown curve (realistic)
    plot_figure3_drawdown(
        real_ptimes, real_pequities,
        out_path = os.path.join(_CHART_DIR, "figure_3_drawdown_realistic.png"),
    )

    # Figure 4 — Yearly returns bar chart (realistic)
    plot_figure4_yearly_returns(
        real_results["yearly_df"],
        out_path = os.path.join(_CHART_DIR, "figure_4_yearly_returns.png"),
    )

    # Figure 5 — Pair contribution (realistic)
    plot_figure5_pair_contribution(
        real_results["pair_df"],
        out_path = os.path.join(_CHART_DIR, "figure_5_pair_contribution.png"),
    )

    # Figure 6 — Trade PnL distribution (realistic)
    plot_figure6_pnl_distribution(
        real_results["portfolio_trade_log"],
        out_path = os.path.join(_CHART_DIR, "figure_6_trade_pnl_distribution.png"),
    )

    print(f"\n  All charts saved to: {_CHART_DIR}")
    print("  Portfolio backtest complete.")
    return real_results, worst_results


if __name__ == "__main__":
    main()
