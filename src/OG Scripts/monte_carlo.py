# =============================================================================
# monte_carlo.py — Bootstrap Monte Carlo for Portfolio Validation
# =============================================================================
# Consumes a completed trade log and simulates alternative histories by
# sampling trades WITH REPLACEMENT (bootstrap).  Produces real dispersion in
# final equity / return / drawdown -- the distribution we actually care about
# for assessing outcome uncertainty.
#
# (The old "shuffle / without replacement" mode has been removed: under
# multiplicative compounding it is final-equity invariant and was not useful.)
#
# PUBLIC API
# ----------
# run_monte_carlo(trade_df, n_simulations, starting_equity,
#                 drawdown_threshold, seed, plot, verbose, pnl_scale) -> dict
# summary_to_frame(result) -> pandas.DataFrame   (tidy row for CSV export)
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Return extraction (robust to column-naming differences)
# ---------------------------------------------------------------------------

def _extract_returns(
    trade_df: pd.DataFrame,
    pnl_scale: str = "auto",
) -> np.ndarray:
    """
    Derive a fraction-form return per trade from whatever columns the trade
    log carries.

    Preference order:
        1. `pnl_pct`  (single-pair and portfolio engines both emit this;
                       stored in percent form -- 0.575 == 0.575%)
        2. `total_portfolio_pnl / portfolio_equity_at_entry`   (portfolio)
        3. `total_pnl / equity_at_entry`                        (single-pair)
    """
    if "pnl_pct" in trade_df.columns:
        pnl_pct = trade_df["pnl_pct"].to_numpy(dtype=float)
        scale = pnl_scale if pnl_scale != "auto" else _detect_pnl_scale(trade_df, pnl_pct)
        if scale == "percent":
            return pnl_pct / 100.0
        return pnl_pct

    if {"total_portfolio_pnl", "portfolio_equity_at_entry"}.issubset(trade_df.columns):
        num = trade_df["total_portfolio_pnl"].to_numpy(dtype=float)
        den = trade_df["portfolio_equity_at_entry"].to_numpy(dtype=float)
        return np.where(den != 0, num / den, 0.0)

    if {"total_pnl", "equity_at_entry"}.issubset(trade_df.columns):
        num = trade_df["total_pnl"].to_numpy(dtype=float)
        den = trade_df["equity_at_entry"].to_numpy(dtype=float)
        return np.where(den != 0, num / den, 0.0)

    raise ValueError(
        "trade_df lacks recognized return columns. Need 'pnl_pct' OR "
        "('total_portfolio_pnl' + 'portfolio_equity_at_entry') OR "
        "('total_pnl' + 'equity_at_entry')."
    )


def _detect_pnl_scale(trade_df: pd.DataFrame, pnl_pct: np.ndarray) -> str:
    if "equity_after" in trade_df.columns and len(trade_df) >= 2:
        eq = trade_df["equity_after"].to_numpy(dtype=float)
        observed = eq[1:] / eq[:-1] - 1.0
        reported = pnl_pct[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.nanmedian(
                np.abs(observed) / np.where(reported != 0, np.abs(reported), np.nan)
            )
        if np.isfinite(ratio):
            return "percent" if ratio < 0.1 else "fraction"

    return "percent" if np.nanmax(np.abs(pnl_pct)) > 3.0 else "fraction"


# ---------------------------------------------------------------------------
# Bootstrap simulation (vectorized)
# ---------------------------------------------------------------------------

def _simulate_bootstrap(
    returns: np.ndarray,
    n_simulations: int,
    starting_equity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw n_simulations alternative histories by sampling trades WITH
    replacement (each sim draws n_trades samples).

    Returns
    -------
    equity_curves : (n_sims, n_trades + 1)   column 0 = starting_equity
    final_equity  : (n_sims,)
    max_drawdown  : (n_sims,)                positive fractions (0.15 == 15% DD)
    """
    n_trades = returns.size
    idx      = rng.integers(0, n_trades, size=(n_simulations, n_trades))
    sampled  = returns[idx]

    growth = np.cumprod(1.0 + sampled, axis=1)
    equity_curves = np.concatenate(
        [np.full((n_simulations, 1), starting_equity), starting_equity * growth],
        axis=1,
    )

    final_equity = equity_curves[:, -1]
    running_peak = np.maximum.accumulate(equity_curves, axis=1)
    dd           = (equity_curves - running_peak) / running_peak
    max_drawdown = -dd.min(axis=1)

    return equity_curves, final_equity, max_drawdown


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def _summarize(
    final_equity: np.ndarray,
    max_drawdown: np.ndarray,
    starting_equity: float,
    drawdown_threshold_pct: float,
) -> dict:
    total_return = final_equity / starting_equity - 1.0
    thr_frac     = drawdown_threshold_pct / 100.0

    return {
        "mean_final_equity":   float(np.mean(final_equity)),
        "median_final_equity": float(np.median(final_equity)),
        "p5_final_equity":     float(np.percentile(final_equity, 5)),
        "p95_final_equity":    float(np.percentile(final_equity, 95)),
        "mean_total_return":   float(np.mean(total_return)),
        "median_total_return": float(np.median(total_return)),
        "p5_total_return":     float(np.percentile(total_return, 5)),
        "p95_total_return":    float(np.percentile(total_return, 95)),
        "mean_drawdown":       float(np.mean(max_drawdown)),
        "median_drawdown":     float(np.median(max_drawdown)),
        "worst_drawdown":      float(np.max(max_drawdown)),
        "prob_loss":                  float(np.mean(final_equity < starting_equity)),
        "prob_dd_exceeds_threshold":  float(np.mean(max_drawdown > thr_frac)),
        "drawdown_threshold_pct":     float(drawdown_threshold_pct),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _safe_bins(values: np.ndarray, target: int = 50):
    """Guard against degenerate ranges that matplotlib.hist rejects."""
    vmin, vmax = float(np.min(values)), float(np.max(values))
    span       = vmax - vmin
    center     = 0.5 * (vmin + vmax)
    degenerate = (
        not np.isfinite(vmin) or not np.isfinite(vmax)
        or span <= 0
        or span < max(abs(center), 1.0) * 1e-6
    )
    if degenerate:
        anchor = center if np.isfinite(center) else 0.0
        pad    = max(abs(anchor) * 1e-6, 1.0)
        return np.array([anchor - pad, anchor + pad])
    return target


def _plot_results(
    equity_curves: np.ndarray,
    final_equity: np.ndarray,
    max_drawdown: np.ndarray,
    starting_equity: float,
    drawdown_threshold_pct: float,
    n_simulations: int,
    n_fan: int = 50,
    rng: np.random.Generator | None = None,
) -> None:
    rng    = rng or np.random.default_rng()
    n_sims = equity_curves.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Monte Carlo — BOOTSTRAP  ({n_simulations:,} simulations)",
        fontsize=13, fontweight="bold",
    )

    # 1. Fan chart
    ax = axes[0]
    sample_idx = rng.choice(n_sims, size=min(n_fan, n_sims), replace=False)
    for i in sample_idx:
        ax.plot(equity_curves[i], color="steelblue", alpha=0.12, linewidth=0.8)
    ax.plot(np.median(equity_curves, axis=0), color="black",
            linewidth=2.2, label="Median curve")
    ax.axhline(starting_equity, color="red", linestyle="--", linewidth=1,
               alpha=0.7, label=f"Start ({starting_equity:,.0f})")
    ax.set_title("Equity Curve Fan")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # 2. Final equity dist
    ax = axes[1]
    ax.hist(final_equity, bins=_safe_bins(final_equity),
            color="seagreen", alpha=0.75, edgecolor="white")
    ax.axvline(starting_equity, color="red", linestyle="--",
               linewidth=1.2, label="Start equity")
    ax.axvline(np.median(final_equity), color="black", linestyle="-",
               linewidth=1.5, label="Median")
    ax.axvline(np.percentile(final_equity, 5), color="darkorange", linestyle=":",
               linewidth=1.5, label="5th pct")
    ax.axvline(np.percentile(final_equity, 95), color="darkorange", linestyle=":",
               linewidth=1.5, label="95th pct")
    ax.set_title("Final Equity Distribution")
    ax.set_xlabel("Final equity")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Max drawdown dist
    ax = axes[2]
    dd_pct = max_drawdown * 100.0
    ax.hist(dd_pct, bins=_safe_bins(dd_pct),
            color="firebrick", alpha=0.75, edgecolor="white")
    ax.axvline(np.mean(dd_pct), color="black", linestyle="-",
               linewidth=1.5, label=f"Mean {np.mean(dd_pct):.1f}%")
    ax.axvline(drawdown_threshold_pct, color="darkorange", linestyle="--",
               linewidth=1.5, label=f"{drawdown_threshold_pct:.0f}% threshold")
    ax.set_title("Max Drawdown Distribution")
    ax.set_xlabel("Max drawdown (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

_INTERPRETATION_LINES = [
    "Alternative histories drawn WITH REPLACEMENT from the realized trade",
    "distribution.  Equity/return dispersion reflects sampling variability",
    "around the observed edge; P(loss) and P(dd > threshold) are empirical",
    "tail probabilities for underperforming the realized historical result.",
    "Assumes trades are exchangeable (no serial dependence).",
]


def _print_report(
    summary: dict,
    starting_equity: float,
    n_simulations: int,
) -> None:
    W   = 76
    SEP = "=" * W
    DIV = "  " + "-" * (W - 2)

    print(f"\n{SEP}")
    print(f"  MONTE CARLO — BOOTSTRAP")
    print(f"  Simulations  : {n_simulations:,}")
    print(f"  Start equity : ${starting_equity:,.2f}")
    print(SEP)

    print("  FINAL EQUITY ($)")
    print(f"    mean    {summary['mean_final_equity']:>14,.2f}   "
          f"median {summary['median_final_equity']:>14,.2f}")
    print(f"    5th pct {summary['p5_final_equity']:>14,.2f}   "
          f"95th   {summary['p95_final_equity']:>14,.2f}")

    print(DIV)
    print("  TOTAL RETURN (%)")
    print(f"    mean    {summary['mean_total_return']*100:>+14.2f}   "
          f"median {summary['median_total_return']*100:>+14.2f}")
    print(f"    5th pct {summary['p5_total_return']*100:>+14.2f}   "
          f"95th   {summary['p95_total_return']*100:>+14.2f}")

    print(DIV)
    print("  MAX DRAWDOWN (%)")
    print(f"    mean    {summary['mean_drawdown']*100:>14.2f}   "
          f"median {summary['median_drawdown']*100:>14.2f}")
    print(f"    worst   {summary['worst_drawdown']*100:>14.2f}")

    print(DIV)
    print("  TAIL PROBABILITIES")
    thr = summary['drawdown_threshold_pct']
    print(f"    P(final < start)   : {summary['prob_loss']*100:>6.2f}%")
    print(f"    P(drawdown > {thr:>4.1f}%) : "
          f"{summary['prob_dd_exceeds_threshold']*100:>6.2f}%")

    print(DIV)
    print("  OUTCOME-UNCERTAINTY ANALYSIS")
    for ln in _INTERPRETATION_LINES:
        print(f"    {ln}")

    print(f"{SEP}\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_monte_carlo(
    trade_df: pd.DataFrame,
    n_simulations: int = 1000,
    starting_equity: float = 100_000.0,
    drawdown_threshold: float = 10.0,
    seed: int | None = 42,
    plot: bool = True,
    verbose: bool = True,
    pnl_scale: str = "auto",
) -> dict:
    """
    Bootstrap Monte Carlo on a completed trade log.

    Samples trades WITH REPLACEMENT to build n_simulations alternative
    histories, then summarizes the resulting distribution of final equity,
    total return, and max drawdown.
    """
    if trade_df is None or trade_df.empty:
        raise ValueError("trade_df is empty — nothing to simulate.")

    returns = _extract_returns(trade_df, pnl_scale=pnl_scale)
    rng     = np.random.default_rng(seed)

    equity_curves, final_equity, max_drawdown = _simulate_bootstrap(
        returns, n_simulations, starting_equity, rng=rng,
    )
    summary = _summarize(final_equity, max_drawdown, starting_equity,
                         drawdown_threshold_pct=drawdown_threshold)

    result = {
        "mode":                   "bootstrap",
        "n_simulations":          n_simulations,
        "starting_equity":        starting_equity,
        "drawdown_threshold_pct": drawdown_threshold,
        "final_equities":         final_equity,
        "total_returns":          final_equity / starting_equity - 1.0,
        "drawdowns":              max_drawdown,
        "equity_curves":          equity_curves,
        "summary":                summary,
    }

    if verbose:
        _print_report(summary, starting_equity, n_simulations)

    if plot:
        _plot_results(
            equity_curves, final_equity, max_drawdown,
            starting_equity=starting_equity,
            drawdown_threshold_pct=drawdown_threshold,
            n_simulations=n_simulations, rng=rng,
        )

    return result


# ---------------------------------------------------------------------------
# Tidy summary -> DataFrame (for CSV export)
# ---------------------------------------------------------------------------

def summary_to_frame(result: dict) -> pd.DataFrame:
    s = result["summary"]
    row = {
        "mode":            result["mode"],
        "n_simulations":   result["n_simulations"],
        "starting_equity": result["starting_equity"],
        **s,
    }
    return pd.DataFrame([row])
