# =============================================================================
# analytics.py
# Research-grade analytics for the forex backtesting project.
#
# PUBLIC API
# ----------
# compute_mfe_mae(trade_df, df_5m)          -> trade_df with MFE/MAE columns
# build_yearly_summary(trade_df, start_eq)  -> yearly summary DataFrame
# build_pair_summary(results, pair, start)  -> one-row pair summary dict
# build_continuation_stats(trade_df)        -> continuation analysis DataFrame
# print_analytics(trade_df, pair, strat)    -> full console printout
# print_edge_diagnostics(trade_df, pair, strat) -> MFE/MAE interpretation
# export_all(trade_df, pair, strat, out_dir)-> writes all CSVs
# =============================================================================

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STARTING_EQUITY = 100_000.0


# ===========================================================================
# 1. PER-TRADE MFE / MAE
# ===========================================================================
def compute_mfe_mae(trade_df: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Maximum Favourable Excursion (MFE) and Maximum Adverse Excursion (MAE)
    for every trade by scanning the 5m bars between entry_time and exit_time.

    Columns added to trade_df
    -------------------------
    mfe_price  : largest favourable price move from entry before exit
    mae_price  : largest adverse  price move from entry before exit
    mfe_R      : mfe_price / R_price  (NaN if R_price not in trade_df)
    mae_R      : mae_price / R_price

    Direction convention
    --------------------
    Long  (bull): MFE = max High - entry,  MAE = entry - min Low
    Short (bear): MFE = entry - min Low,   MAE = max High - entry
    """
    if trade_df.empty:
        return trade_df

    df = trade_df.copy()
    mfe_list, mae_list = [], []

    highs  = df_5m["High"].values
    lows   = df_5m["Low"].values
    idx    = df_5m.index

    has_R = "R_price" in df.columns

    for _, row in df.iterrows():
        entry_time = pd.Timestamp(row["entry_time"])
        exit_time  = pd.Timestamp(row["exit_time"])
        entry_p    = float(row["entry_price"])
        direction  = row["direction"]

        # Find bar indices that span the trade
        i_start = idx.searchsorted(entry_time,  side="left")
        i_end   = idx.searchsorted(exit_time,   side="right")
        i_end   = min(i_end, len(idx))

        if i_start >= i_end:
            mfe_list.append(np.nan)
            mae_list.append(np.nan)
            continue

        seg_high = highs[i_start:i_end]
        seg_low  = lows[i_start:i_end]

        if direction == "bull":
            mfe = max(0.0, float(np.max(seg_high)) - entry_p)
            mae = max(0.0, entry_p - float(np.min(seg_low)))
        else:  # bear
            mfe = max(0.0, entry_p - float(np.min(seg_low)))
            mae = max(0.0, float(np.max(seg_high)) - entry_p)

        mfe_list.append(round(mfe, 6))
        mae_list.append(round(mae, 6))

    df["mfe_price"] = mfe_list
    df["mae_price"] = mae_list

    if has_R:
        R = df["R_price"].replace(0, np.nan)
        df["mfe_R"] = (df["mfe_price"] / R).round(3)
        df["mae_R"] = (df["mae_price"] / R).round(3)

    return df


# ===========================================================================
# 2. CONTINUATION ANALYSIS  (how far do trades run after signal / after TP1)
# ===========================================================================
def build_continuation_stats(trade_df: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    For each trade, measure how many R-multiples price reached during the trade
    and whether it retraced to breakeven after TP1.

    Requires columns in trade_df: entry_price, direction, R_price, exit_time
    trailing.py trades also have tp1_hit, tp2_hit.

    Returns a DataFrame with one row per trade:
        trade_idx, direction, reached_1R, reached_1_5R, reached_2R,
        retraced_to_be_after_tp1, max_R_reached, post_tp1_continuation_R
    """
    if trade_df.empty or "R_price" not in trade_df.columns:
        return pd.DataFrame()

    rows = []
    highs = df_5m["High"].values
    lows  = df_5m["Low"].values
    idx   = df_5m.index

    for i, row in trade_df.iterrows():
        entry_p   = float(row["entry_price"])
        direction = row["direction"]
        R         = float(row.get("R_price", np.nan))
        if np.isnan(R) or R == 0:
            continue

        entry_time = pd.Timestamp(row["entry_time"])
        exit_time  = pd.Timestamp(row["exit_time"])
        i_start = idx.searchsorted(entry_time, side="left")
        i_end   = min(idx.searchsorted(exit_time, side="right"), len(idx))

        seg_high = highs[i_start:i_end]
        seg_low  = lows[i_start:i_end]

        if direction == "bull":
            fav_move = float(np.max(seg_high)) - entry_p if len(seg_high) else 0
            adv_move = entry_p - float(np.min(seg_low))  if len(seg_low)  else 0
        else:
            fav_move = entry_p - float(np.min(seg_low))  if len(seg_low)  else 0
            adv_move = float(np.max(seg_high)) - entry_p if len(seg_high) else 0

        max_R   = fav_move / R
        r1      = max_R >= 1.0
        r1_5    = max_R >= 1.5
        r2      = max_R >= 2.0

        # Retracement to BE after TP1 (only meaningful for trailing strategy)
        tp1_hit = bool(row.get("tp1_hit", False))
        retraced_be = False
        post_tp1_continuation = np.nan

        if tp1_hit and r1:
            # After hitting TP1 (0.75R), did adverse move reach back to entry (0R)?
            # Proxy: did Low/High come back to entry after 0.75R target was passed?
            tp1_price = (entry_p + 0.75 * R) if direction == "bull" else (entry_p - 0.75 * R)
            i_tp1 = idx.searchsorted(entry_time, side="left")
            # Find bar where TP1 was first touched
            if direction == "bull":
                for j in range(i_start, i_end):
                    if highs[j] >= tp1_price:
                        i_tp1 = j
                        break
            else:
                for j in range(i_start, i_end):
                    if lows[j] <= tp1_price:
                        i_tp1 = j
                        break

            post_high = highs[i_tp1:i_end]
            post_low  = lows[i_tp1:i_end]
            if len(post_high):
                if direction == "bull":
                    retraced_be = float(np.min(post_low)) <= entry_p
                    post_tp1_continuation = (float(np.max(post_high)) - (entry_p + 0.75*R)) / R
                else:
                    retraced_be = float(np.max(post_high)) >= entry_p
                    post_tp1_continuation = ((entry_p - 0.75*R) - float(np.min(post_low))) / R

        rows.append({
            "trade_idx":               i,
            "entry_time":              row["entry_time"],
            "direction":               direction,
            "reached_1R":              r1,
            "reached_1_5R":            r1_5,
            "reached_2R":              r2,
            "max_R_reached":           round(max_R, 3),
            "tp1_hit":                 tp1_hit,
            "retraced_to_be_after_tp1": retraced_be,
            "post_tp1_continuation_R": round(post_tp1_continuation, 3)
                                       if not np.isnan(post_tp1_continuation) else np.nan,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# 3. YEARLY PERFORMANCE BREAKDOWN
# ===========================================================================
def build_yearly_summary(trade_df: pd.DataFrame, starting_equity: float = STARTING_EQUITY) -> pd.DataFrame:
    """
    Aggregate per-year performance from a trade log.

    Requires columns: entry_time, total_pnl (or pnl), pnl_pct, equity_after
    """
    if trade_df.empty:
        return pd.DataFrame()

    df = trade_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["year"] = df["entry_time"].dt.year

    # Resolve PnL column name (trailing uses total_pnl, 2to1 uses pnl)
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl"

    rows = []
    eq_start = starting_equity

    for year, grp in df.groupby("year"):
        n      = len(grp)
        wins   = (grp[pnl_col] > 0).sum()
        wr     = wins / n * 100 if n else 0
        pnl    = grp[pnl_col].sum()
        eq_end = grp["equity_after"].iloc[-1]
        ret    = (eq_end - eq_start) / eq_start * 100

        # Max drawdown within the year
        eq_vals = np.concatenate([[eq_start], grp["equity_after"].values])
        peak    = np.maximum.accumulate(eq_vals)
        dd      = (eq_vals - peak) / peak * 100
        max_dd  = float(dd.min())

        avg_pnl = grp["pnl_pct"].mean() if "pnl_pct" in grp.columns else np.nan

        rows.append({
            "year":          year,
            "start_equity":  round(eq_start, 2),
            "end_equity":    round(eq_end, 2),
            "return_pct":    round(ret, 2),
            "n_trades":      n,
            "wins":          int(wins),
            "win_rate_pct":  round(wr, 1),
            "total_pnl":     round(pnl, 2),
            "avg_pnl_pct":   round(float(avg_pnl), 4),
            "max_drawdown_pct": round(max_dd, 2),
        })
        eq_start = eq_end   # carry forward

    return pd.DataFrame(rows)


# ===========================================================================
# 4. PAIR SUMMARY  (one row per strategy-pair run, for batch comparison)
# ===========================================================================
def build_pair_summary(
    trade_df:       pd.DataFrame,
    equity_df:      pd.DataFrame,
    final_equity:   float,
    pair:           str,
    strategy_name:  str,
    start_date:     Optional[str] = None,
    end_date:       Optional[str] = None,
    starting_equity: float = STARTING_EQUITY,
) -> dict:
    """
    Produce a single summary dict for this pair × strategy combination.
    Suitable for appending to a CSV batch report.
    """
    pnl_col = "total_pnl" if "total_pnl" in trade_df.columns else "pnl"

    if trade_df.empty:
        return {
            "pair": pair, "strategy": strategy_name,
            "start_date": start_date, "end_date": end_date,
            "n_trades": 0, "final_equity": round(final_equity, 2),
        }

    n       = len(trade_df)
    wins    = (trade_df[pnl_col] > 0).sum()
    losses  = (trade_df[pnl_col] < 0).sum()
    wr      = wins / n * 100 if n else 0

    total_ret = (final_equity - starting_equity) / starting_equity * 100

    # CAGR
    cagr = np.nan
    if start_date and end_date:
        try:
            years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
            if years > 0:
                cagr = ((final_equity / starting_equity) ** (1 / years) - 1) * 100
        except Exception:
            pass

    eq_vals = np.concatenate([[starting_equity], trade_df["equity_after"].values])
    peak    = np.maximum.accumulate(eq_vals)
    dd      = (eq_vals - peak) / peak * 100
    max_dd  = float(dd.min())

    gp = trade_df.loc[trade_df[pnl_col] > 0, pnl_col].sum()
    gl = abs(trade_df.loc[trade_df[pnl_col] < 0, pnl_col].sum())
    pf = gp / gl if gl > 0 else np.inf

    avg_win  = trade_df.loc[trade_df[pnl_col] > 0, pnl_col].mean() if wins  > 0 else 0.0
    avg_loss = trade_df.loc[trade_df[pnl_col] < 0, pnl_col].mean() if losses > 0 else 0.0
    expect   = (wr / 100) * avg_win + (1 - wr / 100) * avg_loss

    avg_mfe = trade_df["mfe_price"].mean() if "mfe_price" in trade_df.columns else np.nan
    avg_mae = trade_df["mae_price"].mean() if "mae_price" in trade_df.columns else np.nan

    # Average holding time
    avg_hold = np.nan
    if "entry_time" in trade_df.columns and "exit_time" in trade_df.columns:
        hold = (pd.to_datetime(trade_df["exit_time"]) -
                pd.to_datetime(trade_df["entry_time"])).dt.total_seconds() / 3600
        avg_hold = round(float(hold.mean()), 2)

    return {
        "pair":             pair,
        "strategy":         strategy_name,
        "start_date":       start_date,
        "end_date":         end_date,
        "n_trades":         n,
        "wins":             int(wins),
        "losses":           int(losses),
        "win_rate_pct":     round(wr, 2),
        "total_return_pct": round(total_ret, 2),
        "cagr_pct":         round(float(cagr), 2) if not np.isnan(cagr) else np.nan,
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor":    round(pf, 3),
        "expectancy_$":     round(float(expect), 2),
        "avg_win_$":        round(float(avg_win), 2),
        "avg_loss_$":       round(float(avg_loss), 2),
        "avg_mfe_price":    round(float(avg_mfe), 6) if not np.isnan(avg_mfe) else np.nan,
        "avg_mae_price":    round(float(avg_mae), 6) if not np.isnan(avg_mae) else np.nan,
        "avg_hold_hrs":     avg_hold,
        "final_equity":     round(final_equity, 2),
    }


# ===========================================================================
# 5. CONSOLE ANALYTICS PRINTOUT
# ===========================================================================
def print_analytics(
    trade_df:      pd.DataFrame,
    final_equity:  float,
    pair:          str,
    strategy_name: str,
    starting_equity: float = STARTING_EQUITY,
) -> None:
    """Print the full research analytics to console."""
    pnl_col = "total_pnl" if "total_pnl" in trade_df.columns else "pnl"
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  ANALYTICS  —  {pair}  [{strategy_name}]")
    print(sep)

    if trade_df.empty:
        print("  No trades.")
        return

    n      = len(trade_df)
    wins   = (trade_df[pnl_col] > 0).sum()
    total_ret = (final_equity - starting_equity) / starting_equity * 100

    eq_vals = np.concatenate([[starting_equity], trade_df["equity_after"].values])
    peak    = np.maximum.accumulate(eq_vals)
    dd      = (eq_vals - peak) / peak * 100
    max_dd  = float(dd.min())

    rets   = trade_df["pnl_pct"].values / 100 if "pnl_pct" in trade_df.columns else np.zeros(n)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    gp = trade_df.loc[trade_df[pnl_col] > 0, pnl_col].sum()
    gl = abs(trade_df.loc[trade_df[pnl_col] < 0, pnl_col].sum())
    pf = gp / gl if gl > 0 else np.inf

    print(f"  Trades          : {n}")
    print(f"  Wins / Losses   : {wins} / {n - wins}")
    print(f"  Win rate        : {wins/n*100:.1f}%")
    print(f"  Total return    : {total_ret:+.2f}%")
    print(f"  Final equity    : ${final_equity:,.2f}")
    print(f"  Max drawdown    : {max_dd:.2f}%")
    print(f"  Sharpe (ann.)   : {sharpe:.3f}")
    print(f"  Profit factor   : {pf:.3f}")

    if "mfe_price" in trade_df.columns:
        print(f"\n  MFE / MAE")
        print(f"  Avg MFE (price) : {trade_df['mfe_price'].mean():.6f}")
        print(f"  Avg MAE (price) : {trade_df['mae_price'].mean():.6f}")
        if "mfe_R" in trade_df.columns:
            print(f"  Avg MFE (R)     : {trade_df['mfe_R'].mean():.3f}R")
            print(f"  Avg MAE (R)     : {trade_df['mae_R'].mean():.3f}R")

    if "tp1_hit" in trade_df.columns:
        tp1 = trade_df["tp1_hit"].mean() * 100
        tp2 = trade_df["tp2_hit"].mean() * 100 if "tp2_hit" in trade_df.columns else np.nan
        print(f"\n  TP Hit Rates")
        print(f"  TP1 (0.75R) hit : {tp1:.1f}%")
        if not np.isnan(tp2): print(f"  TP2 (1.50R) hit : {tp2:.1f}%")

    print(sep)


def print_edge_diagnostics(
    trade_df: pd.DataFrame,
    pair: str,
    strategy_name: str,
) -> None:
    """
    Interpret MFE/MAE data to diagnose where edge is being lost.

    Rules (Phase 2.3):
      Avg MFE > 2R AND unprofitable  → Problem = exit model
      Avg MFE < 1R                   → Problem = entry signal
    """
    pnl_col = "total_pnl" if "total_pnl" in trade_df.columns else "pnl"
    if trade_df.empty or "mfe_R" not in trade_df.columns:
        return

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  EDGE DIAGNOSTICS  —  {pair}  [{strategy_name}]")
    print(sep)

    mfe_R = trade_df["mfe_R"].dropna()
    mae_R = trade_df["mae_R"].dropna() if "mae_R" in trade_df.columns else pd.Series(dtype=float)
    profitable = trade_df[pnl_col].sum() > 0

    avg_mfe = float(mfe_R.mean()) if len(mfe_R) else np.nan
    avg_mae = float(mae_R.mean()) if len(mae_R) else np.nan

    print(f"  Avg MFE (R)     : {avg_mfe:.3f}R" if not np.isnan(avg_mfe) else "  Avg MFE (R)     : n/a")
    print(f"  Avg MAE (R)     : {avg_mae:.3f}R" if not np.isnan(avg_mae) else "  Avg MAE (R)     : n/a")

    if len(mfe_R):
        print(f"\n  % of trades whose MFE reached:")
        for r_level in [1.0, 1.5, 2.0]:
            pct = (mfe_R >= r_level).mean() * 100
            print(f"    {r_level}R : {pct:.1f}%")

    print(f"\n  DIAGNOSIS:")
    if not np.isnan(avg_mfe):
        if avg_mfe > 2.0 and not profitable:
            print(f"  [!] Avg MFE = {avg_mfe:.2f}R > 2R but strategy is UNPROFITABLE.")
            print(f"      -> ROOT CAUSE: Exit model. Trades move far enough — exits are")
            print(f"         too early (leaving profit) or stops moved to BE prematurely.")
        elif avg_mfe < 1.0:
            print(f"  [!] Avg MFE = {avg_mfe:.2f}R < 1R.")
            print(f"      -> ROOT CAUSE: Entry signal. Price barely moves favourably after")
            print(f"         entry. Review liquidity sweep + BOS conditions.")
        elif profitable and avg_mfe >= 1.0:
            print(f"  [OK] Avg MFE = {avg_mfe:.2f}R >= 1R and strategy is PROFITABLE.")
            print(f"       Edge appears statistically valid.")
        else:
            print(f"  [?]  Avg MFE = {avg_mfe:.2f}R. Marginal edge — review stop tightness")
            print(f"       relative to typical adverse excursion ({avg_mae:.2f}R MAE).")

    if not np.isnan(avg_mfe) and not np.isnan(avg_mae):
        ratio = avg_mae / avg_mfe if avg_mfe > 0 else np.nan
        if not np.isnan(ratio) and ratio > 0.75:
            print(f"  [!] MAE/MFE ratio = {ratio:.2f} (> 0.75).")
            print(f"      -> Trades frequently retrace close to entry before running.")
            print(f"         Consider tighter stop or waiting for deeper pullback entry.")

    print(sep)


def print_yearly_summary(yearly_df: pd.DataFrame, pair: str, strategy_name: str) -> None:
    if yearly_df.empty:
        return
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  YEARLY BREAKDOWN  —  {pair}  [{strategy_name}]")
    print(sep)
    print(yearly_df.to_string(index=False))
    print(sep)


def print_continuation_summary(cont_df: pd.DataFrame, pair: str, strategy_name: str) -> None:
    if cont_df.empty:
        return
    sep = "=" * 64
    n = len(cont_df)
    print(f"\n{sep}")
    print(f"  CONTINUATION ANALYSIS  —  {pair}  [{strategy_name}]")
    print(sep)
    print(f"  % reaching 1R  : {cont_df['reached_1R'].mean()*100:.1f}%")
    if "reached_1_5R" in cont_df.columns:
        print(f"  % reaching 1.5R: {cont_df['reached_1_5R'].mean()*100:.1f}%")
    print(f"  % reaching 2R  : {cont_df['reached_2R'].mean()*100:.1f}%")
    tp1_trades = cont_df[cont_df["tp1_hit"]]
    if not tp1_trades.empty:
        be_pct = tp1_trades["retraced_to_be_after_tp1"].mean() * 100
        avg_cont = tp1_trades["post_tp1_continuation_R"].mean()
        print(f"\n  Of trades that hit TP1 ({len(tp1_trades)} trades):")
        print(f"  % retracing to BE after TP1 : {be_pct:.1f}%")
        if not np.isnan(avg_cont):
            print(f"  Avg post-TP1 continuation   : {avg_cont:.3f}R")
    print(f"  Avg max R reached           : {cont_df['max_R_reached'].mean():.3f}R")
    print(sep)


# ===========================================================================
# 6. UNIFIED REPORT  (replaces the 4 individual print calls in run_stress_test)
# ===========================================================================
def print_report(
    trade_df:        pd.DataFrame,
    equity_df:       pd.DataFrame,
    final_equity:    float,
    yearly_df:       pd.DataFrame,
    cont_df:         pd.DataFrame,
    pair:            str,
    strategy_name:   str,
    scenario:        str  = "",
    spread:          float = 0.0,
    slippage_std:    float = 0.0,
    risk_pct:        float = 0.0075,
    exit_mode:       str  = "dual_tp",
    starting_equity: float = STARTING_EQUITY,
) -> None:
    """Single-entry-point console report covering all 6 output sections."""
    pnl_col = "total_pnl" if "total_pnl" in trade_df.columns else "pnl"
    W   = 66
    SEP = "=" * W
    DIV = "  " + "─" * (W - 2)

    # ── 1. RUN HEADER ────────────────────────────────────────────────────────
    scn_tag = f"  {scenario.upper()}" if scenario else ""
    print(f"\n{SEP}")
    print(f"  {pair}  —  {strategy_name}{scn_tag}")
    print(f"  Risk: {risk_pct:.2%}  |  Spread: {spread:.5f}  "
          f"|  Slip: {slippage_std:.5f}  |  Exit: {exit_mode}")
    print(SEP)

    if trade_df.empty:
        print("  No trades recorded.")
        print(SEP)
        return

    # ── Core metrics (used across multiple sections) ──────────────────────
    n         = len(trade_df)
    wins      = (trade_df[pnl_col] > 0).sum()
    wr        = wins / n * 100
    total_pnl = float(trade_df[pnl_col].sum())
    total_ret = (final_equity - starting_equity) / starting_equity * 100

    eq_vals = np.concatenate([[starting_equity], trade_df["equity_after"].values])
    peak    = np.maximum.accumulate(eq_vals)
    max_dd  = float(((eq_vals - peak) / peak * 100).min())

    rets   = trade_df["pnl_pct"].values / 100 if "pnl_pct" in trade_df.columns else np.zeros(n)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    gp     = float(trade_df.loc[trade_df[pnl_col] > 0, pnl_col].sum())
    gl     = abs(float(trade_df.loc[trade_df[pnl_col] < 0, pnl_col].sum()))
    pf     = gp / gl if gl > 0 else np.inf

    # ── 2. PERFORMANCE SUMMARY ───────────────────────────────────────────────
    L = 22
    print(f"\n  PERFORMANCE SUMMARY")
    print(f"  {'Trades':<{L}} {n}")
    print(f"  {'Wins / Losses':<{L}} {wins} / {n - wins}   ({wr:.1f}% win rate)")
    print(f"  {'Total PnL':<{L}} ${total_pnl:>10,.2f}")
    print(f"  {'Total return':<{L}} {total_ret:>+10.2f}%")
    print(f"  {'Final equity':<{L}} ${final_equity:>10,.2f}")
    print(f"  {'Max drawdown':<{L}} {max_dd:>+10.2f}%")
    print(f"  {'Sharpe (ann.)':<{L}} {sharpe:>10.3f}")
    print(f"  {'Profit factor':<{L}} {pf:>10.3f}")

    # ── 3. EDGE SUMMARY ──────────────────────────────────────────────────────
    has_mfe = "mfe_R" in trade_df.columns and trade_df["mfe_R"].notna().any()
    if has_mfe:
        avg_mfe = float(trade_df["mfe_R"].dropna().mean())
        avg_mae = (float(trade_df["mae_R"].dropna().mean())
                   if "mae_R" in trade_df.columns else np.nan)
        print(f"\n  EDGE SUMMARY")
        print(f"  {'Avg MFE':<{L}} {avg_mfe:.3f}R")
        if not np.isnan(avg_mae):
            print(f"  {'Avg MAE':<{L}} {avg_mae:.3f}R")
        if not cont_df.empty:
            for lvl, col in [(1.0, "reached_1R"), (1.5, "reached_1_5R"), (2.0, "reached_2R")]:
                if col in cont_df.columns:
                    print(f"  {'% reaching '+str(lvl)+'R':<{L}} {cont_df[col].mean()*100:.1f}%")
            print(f"  {'Avg max R reached':<{L}} {cont_df['max_R_reached'].mean():.3f}R")

    # ── 4. EXIT SUMMARY ──────────────────────────────────────────────────────
    if "tp1_hit" in trade_df.columns:
        tp1_rate = trade_df["tp1_hit"].mean() * 100
        tp2_rate = (trade_df["tp2_hit"].mean() * 100
                    if "tp2_hit" in trade_df.columns else np.nan)
        print(f"\n  EXIT SUMMARY")
        print(f"  {'TP1 (0.75R) hit':<{L}} {tp1_rate:.1f}%")
        if not np.isnan(tp2_rate):
            print(f"  {'TP2 (1.50R) hit':<{L}} {tp2_rate:.1f}%")
        if not cont_df.empty and "tp1_hit" in cont_df.columns:
            tp1_rows = cont_df[cont_df["tp1_hit"]]
            if not tp1_rows.empty:
                be_pct    = tp1_rows["retraced_to_be_after_tp1"].mean() * 100
                avg_cont  = tp1_rows["post_tp1_continuation_R"].mean()
                print(f"  {'% BE retrace (TP1)':<{L}} {be_pct:.1f}%")
                if not np.isnan(avg_cont):
                    print(f"  {'Post-TP1 continuation':<{L}} {avg_cont:.3f}R")

    # ── 5. YEARLY BREAKDOWN ──────────────────────────────────────────────────
    if not yearly_df.empty:
        print(f"\n  YEARLY BREAKDOWN")
        hdr = (f"  {'Year':<5}  {'Start':>10}  {'End':>10}  "
               f"{'Ret%':>6}  {'Trades':>6}  {'WR%':>5}  {'PnL':>10}  {'MaxDD%':>7}")
        print(hdr)
        print(DIV)
        for _, yr in yearly_df.iterrows():
            print(f"  {int(yr['year']):<5}  "
                  f"${yr['start_equity']:>9,.0f}  "
                  f"${yr['end_equity']:>9,.0f}  "
                  f"{yr['return_pct']:>+5.1f}%  "
                  f"{int(yr['n_trades']):>6}  "
                  f"{yr['win_rate_pct']:>4.1f}%  "
                  f"${yr['total_pnl']:>9,.0f}  "
                  f"{yr['max_drawdown_pct']:>+6.1f}%")

    # ── 6. FINAL DECISION ────────────────────────────────────────────────────
    dd_headroom = 10.0 + max_dd   # max_dd is negative, e.g. -3.68 → 6.32 remaining
    if max_dd >= -5.0:
        status = "ACCEPTABLE"
        lever  = "increase risk — DD headroom available"
    elif max_dd >= -10.0:
        status = "ACCEPTABLE"
        lever  = "maintain current risk"
    elif max_dd >= -15.0:
        status = "REVIEW"
        lever  = "reduce risk or tighten exits"
    else:
        status = "REJECT"
        lever  = "reduce risk or change exit model"

    print(f"\n{DIV}")
    print(f"  STATUS         : {status}")
    print(f"  NEXT LEVER     : {lever}")
    print(f"  DD HEADROOM    : {dd_headroom:.2f}%  (target ≤ 10%)")
    print(f"{SEP}\n")


# ===========================================================================
# 7. EXPORT ALL CSVs
# ===========================================================================
def export_all(
    trade_df:       pd.DataFrame,
    equity_df:      pd.DataFrame,
    yearly_df:      pd.DataFrame,
    cont_df:        pd.DataFrame,
    pair_summary:   dict,
    pair:           str,
    strategy_name:  str,
    out_dir:        str = ".",
) -> None:
    """
    Write all analytics CSVs to out_dir/PAIR/.

    Output structure
    ----------------
    out_dir/
      PAIR/
        PAIR_trades.csv       — full trade log
        PAIR_equity.csv       — bar-level equity curve
        PAIR_yearly.csv       — annual performance breakdown
        PAIR_continuation.csv — post-TP1 continuation analysis
        PAIR_summary.csv      — single-row scalar metrics

    Files are prefixed with the pair name so they remain identifiable
    if files from different subfolders are ever compared or merged.
    """
    pair_dir = os.path.join(out_dir, pair)
    os.makedirs(pair_dir, exist_ok=True)

    files = []

    if not trade_df.empty:
        p = os.path.join(pair_dir, f"{pair}_trades.csv")
        trade_df.to_csv(p, index=False)
        files.append(p)

    if not equity_df.empty:
        p = os.path.join(pair_dir, f"{pair}_equity.csv")
        equity_df.to_csv(p, index=False)
        files.append(p)

    if not yearly_df.empty:
        p = os.path.join(pair_dir, f"{pair}_yearly.csv")
        yearly_df.to_csv(p, index=False)
        files.append(p)

    if not cont_df.empty:
        p = os.path.join(pair_dir, f"{pair}_continuation.csv")
        cont_df.to_csv(p, index=False)
        files.append(p)

    if pair_summary:
        p = os.path.join(pair_dir, f"{pair}_summary.csv")
        pd.DataFrame([pair_summary]).to_csv(p, index=False)
        files.append(p)

    print(f"\n  Exports written to: {os.path.abspath(pair_dir)}")
    for f in files:
        print(f"    {os.path.basename(f)}")
