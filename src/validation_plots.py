# =============================================================================
# validation_plots.py — Publication-quality charts for validate.py
# =============================================================================
# One function per section of validate.py.  Each function takes the data the
# validation run already produces, saves a small set of paper-ready PNGs, and
# returns the list of written paths.
#
# Style is intentionally restrained: muted palette, no frames on top/right,
# dashed light grid, consistent figure sizes, savefig at 220 DPI.
# =============================================================================

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

_STYLE = {
    "figure.dpi":        110,
    "savefig.dpi":       220,
    "font.size":         11,
    "axes.titlesize":    12.5,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.7,
    "legend.frameon":    False,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
}

C_PRIMARY   = "#1f4e79"   # deep blue
C_SECONDARY = "#b45f06"   # burnt orange
C_ACCENT    = "#4f7a3a"   # sage green
C_RED       = "#9c3a3a"
C_GREY      = "#7a7a7a"

PAIR_COLORS = {
    "EURUSD": "#d17a1a",   # warm amber
    "GBPUSD": "#2f7a43",   # forest green
    "USDJPY": "#6a4a99",   # muted purple
}


def _apply_style() -> None:
    for k, v in _STYLE.items():
        plt.rcParams[k] = v


def _save(fig: plt.Figure, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _dollar_fmt(x, _p):
    return f"${x:,.0f}"


def _pct_fmt(x, _p):
    return f"{x:+.1f}%"


# =============================================================================
# 1. BASELINE
# =============================================================================

def plot_baseline(
    trade_df:     pd.DataFrame,
    equity_curve: list,
    yearly_df:    pd.DataFrame,
    pair_df:      pd.DataFrame,
    out_dir:      str,
) -> list[str]:
    _apply_style()
    written: list[str] = []

    # ---- equity curve ----
    eq = pd.DataFrame(equity_curve)
    if not eq.empty:
        eq["time"] = pd.to_datetime(eq["time"])
        eq = eq.drop_duplicates("time", keep="last").sort_values("time")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eq["time"], eq["equity"], color=C_PRIMARY, linewidth=1.8)
        ax.axhline(eq["equity"].iloc[0], color=C_GREY, linestyle="--",
                   linewidth=1, alpha=0.6, label="Start equity")
        ax.set_title("Portfolio Equity Curve — Realistic Validation Run")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
        ax.legend()
        fig.tight_layout()
        written.append(_save(fig, out_dir, "baseline_equity_curve.png"))

        # ---- drawdown curve ----
        eq_v = eq["equity"].to_numpy(dtype=float)
        peak = np.maximum.accumulate(eq_v)
        dd   = (eq_v - peak) / peak * 100.0

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.fill_between(eq["time"], dd, 0, color=C_RED, alpha=0.30)
        ax.plot(eq["time"], dd, color=C_RED, linewidth=1.2)
        ax.set_title("Portfolio Drawdown Curve — Realistic Validation Run")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1f}%"))
        fig.tight_layout()
        written.append(_save(fig, out_dir, "baseline_drawdown_curve.png"))

    # ---- yearly returns ----
    if yearly_df is not None and not yearly_df.empty and "return_pct" in yearly_df.columns:
        years = yearly_df["year"].astype(str).to_list()
        rets  = yearly_df["return_pct"].to_numpy(dtype=float)
        colors = [C_ACCENT if v >= 0 else C_RED for v in rets]

        fig, ax = plt.subplots(figsize=(9, 4.8))
        bars = ax.bar(years, rets, color=colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, v in zip(bars, rets):
            ax.text(bar.get_x() + bar.get_width()/2, v + (0.5 if v >= 0 else -1.3),
                    f"{v:+.1f}%", ha="center", fontsize=9, color=C_GREY)
        ax.set_title("Annual Returns — Portfolio Validation")
        ax.set_xlabel("Year")
        ax.set_ylabel("Return (%)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1f}%"))
        fig.tight_layout()
        written.append(_save(fig, out_dir, "baseline_yearly_returns.png"))

    # ---- pair contribution ----
    if pair_df is not None and not pair_df.empty and "total_pnl" in pair_df.columns:
        pairs = pair_df["pair"].to_list()
        pnls  = pair_df["total_pnl"].to_numpy(dtype=float)
        colors = [PAIR_COLORS.get(p, C_PRIMARY) for p in pairs]

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        bars = ax.bar(pairs, pnls, color=colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, v in zip(bars, pnls):
            ax.text(bar.get_x() + bar.get_width()/2, v,
                    f"  ${v:,.0f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=10)
        ax.set_title("PnL Contribution by Pair")
        ax.set_xlabel("Currency Pair")
        ax.set_ylabel("Total PnL (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
        fig.tight_layout()
        written.append(_save(fig, out_dir, "baseline_pair_contribution.png"))

    # ---- per-trade pnl_pct histogram ----
    if trade_df is not None and not trade_df.empty and "pnl_pct" in trade_df.columns:
        vals = trade_df["pnl_pct"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.hist(vals, bins=50, color=C_PRIMARY, alpha=0.80, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.9)
        ax.axvline(vals.mean(), color=C_SECONDARY, linestyle="--",
                   linewidth=1.5, label=f"Mean {vals.mean():+.2f}%")
        ax.axvline(np.median(vals), color=C_ACCENT, linestyle=":",
                   linewidth=1.5, label=f"Median {np.median(vals):+.2f}%")
        ax.set_title("Per-Trade PnL Distribution")
        ax.set_xlabel("Per-trade PnL (%)")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        written.append(_save(fig, out_dir, "baseline_trade_pnl_distribution.png"))

    return written


# =============================================================================
# 2. WALK-FORWARD
# =============================================================================

def plot_walkforward(
    walk_result:     dict,
    trade_df:        pd.DataFrame,
    train_start:     str,
    train_end:       str,
    test_start:      str,
    test_end:        str,
    starting_equity: float,
    out_dir:         str,
) -> list[str]:
    _apply_style()
    written: list[str] = []

    tr = walk_result.get("TRAIN", {}) or {}
    te = walk_result.get("TEST", {})  or {}

    # ---- grouped metric bar chart ----
    metrics = [
        ("Total Return (%)", "total_return"),
        ("Max Drawdown (%)", "max_drawdown"),
        ("Sharpe",           "sharpe"),
        ("Profit Factor",    "profit_factor"),
        ("Win Rate (%)",     "win_rate"),
    ]
    labels     = [m[0] for m in metrics]
    train_vals = [float(tr.get(m[1], np.nan)) for m in metrics]
    test_vals  = [float(te.get(m[1], np.nan)) for m in metrics]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    b1 = ax.bar(x - w/2, train_vals, w, color=C_PRIMARY,   label="Train (in-sample)",    edgecolor="white")
    b2 = ax.bar(x + w/2, test_vals,  w, color=C_SECONDARY, label="Test (out-of-sample)", edgecolor="white")
    for bars, vals in ((b1, train_vals), (b2, test_vals)):
        for bar, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            ax.text(bar.get_x() + bar.get_width()/2, v,
                    f"  {v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9, color=C_GREY)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Walk-Forward Validation — In-Sample vs Out-of-Sample")
    ax.set_ylabel("Metric value")
    ax.legend()
    fig.tight_layout()
    written.append(_save(fig, out_dir, "walkforward_metric_comparison.png"))

    # ---- rebased equity curves ----
    if (trade_df is not None and not trade_df.empty
        and "entry_time" in trade_df.columns and "pnl_pct" in trade_df.columns):
        df = trade_df.copy()
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        r = df["pnl_pct"].to_numpy(dtype=float) / 100.0

        def _slice_curve(t0: str, t1: str):
            m = (df["entry_time"] >= pd.Timestamp(t0)) & (df["entry_time"] <= pd.Timestamp(t1))
            idx = np.where(m.to_numpy())[0]
            if idx.size == 0:
                return None, None
            sub = r[idx]
            eq  = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + sub)])
            tt  = pd.concat(
                [pd.Series([pd.Timestamp(t0)]), df["entry_time"].iloc[idx]]
            ).reset_index(drop=True)
            return tt, eq

        tr_t, tr_e = _slice_curve(train_start, train_end)
        te_t, te_e = _slice_curve(test_start,  test_end)

        if tr_e is not None or te_e is not None:
            fig, ax = plt.subplots(figsize=(10.5, 5))
            if tr_e is not None:
                ax.plot(tr_t, tr_e, color=C_PRIMARY, linewidth=1.7,
                        label=f"Train ({train_start[:4]}–{train_end[:4]})")
            if te_e is not None:
                ax.plot(te_t, te_e, color=C_SECONDARY, linewidth=1.7,
                        label=f"Test  ({test_start[:4]}–{test_end[:4]})")
            ax.axhline(100, color=C_GREY, linestyle="--", linewidth=1, alpha=0.7)
            ax.set_title("Walk-Forward Equity Curves (rebased to 100)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Equity (rebased)")
            ax.legend()
            fig.tight_layout()
            written.append(_save(fig, out_dir, "walkforward_equity_curves_rebased.png"))

    # ---- TP hit rates ----
    vals = [tr.get("tp1_hit_rate"), te.get("tp1_hit_rate"),
            tr.get("tp2_hit_rate"), te.get("tp2_hit_rate")]
    if all(v is not None and np.isfinite(v) for v in vals):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        x = np.arange(2)
        w = 0.38
        b1 = ax.bar(x - w/2, [vals[0], vals[2]], w, color=C_PRIMARY,   label="Train", edgecolor="white")
        b2 = ax.bar(x + w/2, [vals[1], vals[3]], w, color=C_SECONDARY, label="Test",  edgecolor="white")
        for bars, v in ((b1, [vals[0], vals[2]]), (b2, [vals[1], vals[3]])):
            for bar, val in zip(bars, v):
                ax.text(bar.get_x() + bar.get_width()/2, val,
                        f"  {val:.1f}%", ha="center", va="bottom",
                        fontsize=10, color=C_GREY)
        ax.set_xticks(x)
        ax.set_xticklabels(["TP1", "TP2"])
        ax.set_title("TP Hit Rate — Train vs Test")
        ax.set_ylabel("Hit rate (%)")
        ax.legend()
        fig.tight_layout()
        written.append(_save(fig, out_dir, "walkforward_tp_hit_rates.png"))

    return written


# =============================================================================
# 3. PARAMETER ROBUSTNESS
# =============================================================================

def plot_robustness(
    param_df:  pd.DataFrame,
    out_dir:   str,
    base_tp1:  float = 0.75,
    base_tp2:  float = 1.50,
) -> list[str]:
    if param_df is None or param_df.empty:
        return []
    _apply_style()
    written: list[str] = []

    heatmaps = [
        ("total_return",  "Total Return (%)",  "RdYlGn",   "robustness_tp_heatmap_return.png",         "{:+.2f}"),
        ("sharpe",        "Sharpe Ratio",      "RdYlGn",   "robustness_tp_heatmap_sharpe.png",         "{:.2f}"),
        ("max_drawdown",  "Max Drawdown (%)",  "RdYlGn_r", "robustness_tp_heatmap_drawdown.png",       "{:.2f}"),
        ("profit_factor", "Profit Factor",     "RdYlGn",   "robustness_tp_heatmap_profit_factor.png",  "{:.2f}"),
    ]

    for col, title, cmap, fname, fmt in heatmaps:
        if col not in param_df.columns:
            continue
        piv  = (param_df
                .pivot_table(index="tp1_r", columns="tp2_r", values=col, aggfunc="mean")
                .sort_index().sort_index(axis=1))
        data = piv.to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_yticks(range(len(piv.index)))
        ax.set_xticklabels([f"{v:.2f}" for v in piv.columns])
        ax.set_yticklabels([f"{v:.2f}" for v in piv.index])
        ax.set_xlabel("TP2 (R multiple)")
        ax.set_ylabel("TP1 (R multiple)")
        ax.set_title(f"TP Robustness Grid — {title}")

        for i, tp1 in enumerate(piv.index):
            for j, tp2 in enumerate(piv.columns):
                v = data[i, j]
                if not np.isfinite(v):
                    continue
                is_base = abs(tp1 - base_tp1) < 1e-6 and abs(tp2 - base_tp2) < 1e-6
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="black", fontsize=10,
                        fontweight="bold" if is_base else "normal")
                if is_base:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=2.2,
                    ))

        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.ax.tick_params(labelsize=9)
        fig.tight_layout()
        written.append(_save(fig, out_dir, fname))

    # ---- return vs drawdown scatter ----
    if {"total_return", "max_drawdown"}.issubset(param_df.columns):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for _, row in param_df.iterrows():
            is_base = bool(row.get("is_baseline", False))
            color = C_SECONDARY if is_base else C_PRIMARY
            size  = 150 if is_base else 60
            ax.scatter(row["max_drawdown"], row["total_return"],
                       c=color, s=size, edgecolor="white", zorder=3)
            ax.annotate(f"({row['tp1_r']:.2f}/{row['tp2_r']:.2f})",
                        (row["max_drawdown"], row["total_return"]),
                        textcoords="offset points", xytext=(6, 5),
                        fontsize=8, color=C_GREY)
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_ylabel("Total Return (%)")
        ax.set_title("TP Grid — Return vs Drawdown  (baseline highlighted)")
        fig.tight_layout()
        written.append(_save(fig, out_dir, "robustness_return_vs_drawdown.png"))

    return written


# =============================================================================
# 4. MONTE CARLO (bootstrap)
# =============================================================================

def plot_monte_carlo(
    mc_result:          dict,
    out_dir:            str,
    drawdown_threshold: float = 10.0,
) -> list[str]:
    if not mc_result:
        return []
    _apply_style()
    written: list[str] = []

    curves   = mc_result["equity_curves"]
    final_eq = mc_result["final_equities"]
    tot_ret  = mc_result["total_returns"]
    dd       = mc_result["drawdowns"]
    start    = float(mc_result["starting_equity"])
    s        = mc_result["summary"]

    # ---- fan chart ----
    rng = np.random.default_rng(0)
    n   = curves.shape[0]
    x   = np.arange(curves.shape[1])
    sample_idx = rng.choice(n, size=min(80, n), replace=False)
    p5   = np.percentile(curves, 5,  axis=0)
    p95  = np.percentile(curves, 95, axis=0)
    med  = np.median(curves, axis=0)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i in sample_idx:
        ax.plot(x, curves[i], color=C_PRIMARY, alpha=0.10, linewidth=0.8)
    ax.fill_between(x, p5, p95, color=C_PRIMARY, alpha=0.18,
                    label="5th–95th percentile band")
    ax.plot(x, med, color="black", linewidth=2.0, label="Median path")
    ax.axhline(start, color=C_RED, linestyle="--", linewidth=1.2,
               label=f"Start equity (${start:,.0f})")
    ax.set_title("Bootstrap Monte Carlo — Equity Fan Chart")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
    ax.legend(loc="upper left")
    fig.tight_layout()
    written.append(_save(fig, out_dir, "monte_carlo_equity_fan_chart.png"))

    # ---- final equity histogram ----
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(final_eq, bins=50, color=C_ACCENT, alpha=0.80, edgecolor="white")
    ax.axvline(start, color=C_RED, linestyle="--", linewidth=1.3, label="Start")
    ax.axvline(s["median_final_equity"], color="black", linewidth=1.5,
               label=f"Median ${s['median_final_equity']:,.0f}")
    ax.axvline(s["p5_final_equity"], color=C_SECONDARY, linestyle=":",
               linewidth=1.5, label=f"5th pct ${s['p5_final_equity']:,.0f}")
    ax.axvline(s["p95_final_equity"], color=C_SECONDARY, linestyle=":",
               linewidth=1.5, label=f"95th pct ${s['p95_final_equity']:,.0f}")
    ax.set_title("Bootstrap Monte Carlo — Final Equity Distribution")
    ax.set_xlabel("Final equity (USD)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
    ax.legend()
    fig.tight_layout()
    written.append(_save(fig, out_dir, "monte_carlo_final_equity_hist.png"))

    # ---- total return histogram ----
    ret_pct = tot_ret * 100.0
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(ret_pct, bins=50, color=C_PRIMARY, alpha=0.80, edgecolor="white")
    ax.axvline(0, color=C_RED, linestyle="--", linewidth=1.3, label="Break-even")
    ax.axvline(s["median_total_return"]*100, color="black", linewidth=1.5,
               label=f"Median {s['median_total_return']*100:+.2f}%")
    ax.axvline(s["p5_total_return"]*100,  color=C_SECONDARY, linestyle=":",
               linewidth=1.5, label=f"5th pct  {s['p5_total_return']*100:+.2f}%")
    ax.axvline(s["p95_total_return"]*100, color=C_SECONDARY, linestyle=":",
               linewidth=1.5, label=f"95th pct {s['p95_total_return']*100:+.2f}%")
    ax.set_title("Bootstrap Monte Carlo — Total Return Distribution")
    ax.set_xlabel("Total return (%)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.legend()
    fig.tight_layout()
    written.append(_save(fig, out_dir, "monte_carlo_total_return_hist.png"))

    # ---- drawdown histogram ----
    dd_pct = dd * 100.0
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(dd_pct, bins=50, color=C_RED, alpha=0.75, edgecolor="white")
    ax.axvline(dd_pct.mean(), color="black", linewidth=1.5,
               label=f"Mean {dd_pct.mean():.2f}%")
    ax.axvline(drawdown_threshold, color=C_SECONDARY, linestyle="--",
               linewidth=1.5, label=f"{drawdown_threshold:.0f}% threshold")
    ax.set_title("Bootstrap Monte Carlo — Max Drawdown Distribution")
    ax.set_xlabel("Max drawdown (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    written.append(_save(fig, out_dir, "monte_carlo_drawdown_hist.png"))

    # ---- percentile summary ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    groups = [
        ("Final Equity (USD)",
            [s["p5_final_equity"], s["median_final_equity"],
             s["mean_final_equity"], s["p95_final_equity"]],
            ["5th pct", "Median", "Mean", "95th pct"],
            "${:,.0f}"),
        ("Total Return (%)",
            [s["p5_total_return"]*100, s["median_total_return"]*100,
             s["mean_total_return"]*100, s["p95_total_return"]*100],
            ["5th pct", "Median", "Mean", "95th pct"],
            "{:+.2f}%"),
        ("Max Drawdown (%)",
            [s["mean_drawdown"]*100, s["median_drawdown"]*100,
             s["worst_drawdown"]*100],
            ["Mean", "Median", "Worst"],
            "{:.2f}%"),
    ]
    palette = [C_SECONDARY, C_PRIMARY, C_ACCENT, C_PRIMARY]
    for ax, (title, vals, lbls, fmt) in zip(axes, groups):
        colors = palette[:len(vals)]
        bars = ax.barh(lbls, vals, color=colors, edgecolor="white")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        xmax = max(abs(v) for v in vals) or 1.0
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.01 * xmax,
                    bar.get_y() + bar.get_height()/2,
                    " " + fmt.format(v), va="center", fontsize=10)
    fig.suptitle("Bootstrap Monte Carlo — Percentile Summary",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    written.append(_save(fig, out_dir, "monte_carlo_percentile_summary.png"))

    # ---- return CDF ----
    ret_sorted = np.sort(ret_pct)
    cdf        = np.linspace(0, 1, len(ret_sorted))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(ret_sorted, cdf, color=C_PRIMARY, linewidth=1.8)
    ax.axvline(0, color=C_RED, linestyle="--", linewidth=1.2, label="Break-even")
    ax.axhline(0.05, color=C_GREY, linestyle=":", linewidth=1)
    ax.axhline(0.95, color=C_GREY, linestyle=":", linewidth=1)
    ax.set_title("Bootstrap Monte Carlo — Total Return CDF")
    ax.set_xlabel("Total return (%)")
    ax.set_ylabel("Cumulative probability")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.legend()
    fig.tight_layout()
    written.append(_save(fig, out_dir, "monte_carlo_return_cdf.png"))

    return written


# =============================================================================
# 5. FINAL VALIDATION SUMMARY
# =============================================================================

def plot_validation_summary(
    baseline_metrics: dict,
    walk_result:      dict,
    param_df:         pd.DataFrame,
    mc_result:        dict,
    dd_limit:         float,
    out_dir:          str,
) -> list[str]:
    _apply_style()
    bm = baseline_metrics or {}
    tr = (walk_result or {}).get("TRAIN", {}) or {}
    te = (walk_result or {}).get("TEST",  {}) or {}

    fig = plt.figure(figsize=(13.5, 8))
    gs  = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.35)

    # ---- panel 1: baseline ----
    ax = fig.add_subplot(gs[0, 0])
    names = ["Return (%)", "Max DD (%)", "Sharpe"]
    vals  = [bm.get("total_return_pct", 0.0),
             bm.get("max_drawdown_pct", 0.0),
             bm.get("sharpe", 0.0)]
    colors = [C_ACCENT if vals[0] >= 0 else C_RED, C_RED, C_PRIMARY]
    bars = ax.bar(names, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v,
                f"  {v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Baseline")

    # ---- panel 2: walk-forward ----
    ax = fig.add_subplot(gs[0, 1])
    labels = ["Return (%)", "Sharpe"]
    tr_vals = [tr.get("total_return", 0.0), tr.get("sharpe", 0.0)]
    te_vals = [te.get("total_return", 0.0), te.get("sharpe", 0.0)]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w/2, tr_vals, w, color=C_PRIMARY,   label="Train")
    ax.bar(x + w/2, te_vals, w, color=C_SECONDARY, label="Test")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Walk-Forward")
    ax.legend(fontsize=9)

    # ---- panel 3: robustness return spread ----
    ax = fig.add_subplot(gs[0, 2])
    if param_df is not None and not param_df.empty and "total_return" in param_df.columns:
        returns = param_df["total_return"].to_numpy(dtype=float)
        ax.hist(returns, bins=max(6, min(10, len(returns))),
                color=C_ACCENT, edgecolor="white", alpha=0.85)
        ax.axvline(returns.mean(), color="black", linestyle="--",
                   linewidth=1.3, label=f"Mean {returns.mean():.2f}%")
        ax.legend(fontsize=9)
    ax.set_title("Robustness — Return Spread")
    ax.set_xlabel("Return (%)")

    # ---- panel 4: MC final equity ----
    ax = fig.add_subplot(gs[1, 0])
    if mc_result:
        ax.hist(mc_result["final_equities"], bins=40,
                color=C_PRIMARY, edgecolor="white", alpha=0.85)
        ax.axvline(mc_result["starting_equity"], color=C_RED, linestyle="--",
                   label="Start")
        ax.axvline(mc_result["summary"]["p5_final_equity"],
                   color=C_SECONDARY, linestyle=":", label="5th pct")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, p: f"${v/1000:.0f}k"))
        ax.legend(fontsize=9)
    ax.set_title("Monte Carlo — Final Equity")

    # ---- panel 5: MC drawdown ----
    ax = fig.add_subplot(gs[1, 1])
    if mc_result:
        dd_pct = mc_result["drawdowns"] * 100.0
        ax.hist(dd_pct, bins=40, color=C_RED, edgecolor="white", alpha=0.80)
        ax.axvline(dd_limit, color="black", linestyle="--",
                   label=f"{dd_limit:.0f}% limit")
        ax.legend(fontsize=9)
    ax.set_title("Monte Carlo — Max Drawdown")
    ax.set_xlabel("Drawdown (%)")

    # ---- panel 6: scorecard ----
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    base_pass = (bm.get("total_return_pct", 0) > 0
                 and bm.get("max_drawdown_pct", 100) <= dd_limit)
    walk_pass = te.get("total_return", -1) > 0
    robust_stable = (param_df is not None and not param_df.empty
                     and (param_df["total_return"] > 0).mean() >= 0.6)

    rows = [
        ("Baseline",     "PASS"     if base_pass     else "REVIEW"),
        ("Walk-forward", "PASS"     if walk_pass     else "REVIEW"),
        ("Robustness",   "STABLE"   if robust_stable else "SENSITIVE"),
    ]
    if mc_result:
        s = mc_result["summary"]
        rows.append(("MC  5th pct return",
                     f"{s['p5_total_return']*100:+.1f}%"))
        rows.append((f"MC  P(DD > {dd_limit:.0f}%)",
                     f"{s['prob_dd_exceeds_threshold']*100:.1f}%"))

    y = 0.95
    ax.text(0.02, y, "Validation Scorecard",
            fontsize=12, fontweight="bold", transform=ax.transAxes)
    y -= 0.13
    for k, v in rows:
        ax.text(0.02, y, k, fontsize=10.5, transform=ax.transAxes)
        ax.text(0.98, y, v, fontsize=10.5, ha="right",
                fontweight="bold", transform=ax.transAxes)
        y -= 0.11

    fig.suptitle("Portfolio Validation Summary",
                 fontsize=14.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return [_save(fig, out_dir, "validation_summary_dashboard.png")]
