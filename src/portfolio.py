# =============================================================================
# portfolio.py  —  Multi-Pair Portfolio Backtest  (Step 1, Deployment Pipeline)
# =============================================================================
#
# PURPOSE
# -------
# Combine validated single-pair backtests (EURUSD, GBPUSD, USDJPY) into one
# portfolio simulation with a single shared $100,000 account.  All three pairs
# trade simultaneously; overlapping trades draw from the same capital base and
# update one combined equity curve.
#
# This is NOT three separate runs summed together.  It replays trade events
# chronologically and updates shared equity at each leg closure.
#
# HOW PnL IS SCALED
# -----------------
# Each per-pair backtest sizes positions as:
#   units = (per_pair_equity_at_entry × risk_pct) / stop_distance
#
# In the portfolio the same formula applies but with portfolio equity:
#   portfolio_units = (portfolio_equity_at_entry × risk_pct) / stop_distance
#
# Therefore:
#   portfolio_pnl_for_leg = per_pair_pnl_for_leg
#                         × (portfolio_equity_at_entry / per_pair_equity_at_entry)
#
# This scale factor is constant for all legs of the same trade (TP1, TP2, STOP)
# because it depends only on equity at entry, which is fixed once the trade opens.
# The same price-change × units arithmetic applies to all three pairs regardless
# of how the engine internally denominates the currency.
#
# RISK MODEL
# ----------
# Each trade sizes using its configured per-pair risk_pct.  Before opening a
# new trade, the engine checks:
#     (total_open_risk_$ + new_trade_risk_$) / portfolio_equity <= PORTFOLIO_CAP_PCT
# If the cap would be breached, the trade is rejected (not entered) and all
# subsequent exit events for that trade are silently skipped.
#
# OPEN-RISK ACCOUNTING
# --------------------
# At entry   : open_risk_$ = portfolio_equity_at_entry × risk_pct[pair]
# After TP1  : stop moves to breakeven → remaining open_risk_$ set to 0.
#              Assumption: slippage at the breakeven stop is small relative to
#              the original risk and is treated as 0 for cap-accounting purposes.
#              This is the conservative (slightly optimistic) choice.  If you want
#              stricter accounting, set BREAKEVEN_RESIDUAL_RISK_PCT to a small
#              fraction (e.g. 0.1 × original risk) in the config below.
# After STOP : position closed, open_risk_$ removed.
# After TP2  : position closed, open_risk_$ (already 0) confirmed removed.
#
# SAME-TIMESTAMP EVENT ORDERING
# ------------------------------
# When multiple events share the same timestamp, this module processes them in
# this fixed order:
#   1. Full closes  (TP2, STOP, BE_STOP, EOD)  — updates equity first
#   2. Partial closes (TP1)                    — updates equity + clears risk
#   3. New entries  (ENTRY)                    — sizes on freshly updated equity
# Within each group, events are sorted by (pair, trade_id) for determinism.
# This ensures that a close and a new entry on the same bar use the most current
# equity for sizing, which is the most realistic assumption.
#
# PUBLIC API
# ----------
# run_per_pair_backtests(pairs_config, risk_pct_by_pair) -> per_pair_results
# build_event_stream(per_pair_results) -> events
# simulate_portfolio(events, ...) -> (trade_log, equity_curve, rejected)
# compute_portfolio_metrics(trade_log, equity_curve) -> metrics dict
# build_yearly_breakdown(trade_log) -> DataFrame
# build_pair_contributions(trade_log) -> DataFrame
# build_overlap_analysis(equity_curve, trade_log) -> dict
# print_portfolio_report(metrics, yearly_df, pair_df, overlap, rejected)
# export_portfolio(trade_log, equity_curve, yearly_df, pair_df, metrics, ...)
# run_portfolio_backtest(pairs_config, ...) -> results dict
# =============================================================================

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup: allow importing from the backtest directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from trailing import run_backtest, STARTING_EQUITY as _SINGLE_PAIR_STARTING_EQUITY
from sim_costs import get_default_costs
from data_loader import load_local_data


# =============================================================================
# PORTFOLIO CONFIGURATION  (edit these to change behaviour)
# =============================================================================

# Starting equity for the shared portfolio account
PORTFOLIO_STARTING_EQUITY: float = 100_000.0

# Per-pair risk fractions
RISK_PCT_BY_PAIR: dict = {
    "EURUSD": 0.0150,   # 1.50%
    "GBPUSD": 0.0100,   # 1.00%
    "USDJPY": 0.0150,   # 1.50%
}

# Portfolio open-risk cap (fraction of equity).  Trades are rejected when
# projected total open risk would exceed this.
PORTFOLIO_CAP_PCT: float = 0.055   # 5.5%

# After TP1 the stop moves to breakeven.  We treat remaining open risk as
# this fraction of the original risk (0.0 = fully zero out after TP1).
# Set to a small value (e.g. 0.10) for more conservative cap accounting.
BREAKEVEN_RESIDUAL_RISK_FRACTION: float = 0.0

# Within same-timestamp events: lower sort key = processed first
# Full closes → TP1 partial → new entries
_EVENT_TYPE_ORDER: dict = {
    "TP2":     0,
    "STOP":    0,
    "BE_STOP": 0,
    "EOD":     0,
    "TP1":     1,
    "ENTRY":   2,
}


# =============================================================================
# STEP 1  —  RUN PER-PAIR BACKTESTS
# =============================================================================

def run_per_pair_backtests(
    pairs_config: list,
    risk_pct_by_pair: dict = None,
    spread_mult: float = 1.0,
    slip_mult:   float = 1.0,
    verbose: bool = True,
    engine_kwargs: dict = None,
) -> dict:
    """
    Run the validated single-pair backtest engine for each pair and capture
    both the trade log and the granular exit-events DataFrame.

    Parameters
    ----------
    pairs_config : list of dicts, each containing:
        {
          "pair":       str   — e.g. "EURUSD"
          "file_path":  str   — absolute path to the 1m CSV
          "start_date": str   — "YYYY-MM-DD" or None (use full dataset)
          "end_date":   str   — "YYYY-MM-DD" or None
        }
    risk_pct_by_pair : per-pair risk override; defaults to RISK_PCT_BY_PAIR
    spread_mult      : multiplier applied to default spread (1.0 = realistic,
                       2.0 = worst-case doubling)
    slip_mult        : multiplier applied to default slippage_std (same scale)
    verbose          : print progress to console

    Returns
    -------
    dict  keyed by pair label:
        {
          "trade_df":       DataFrame  — full trade log (22 columns)
          "exit_events_df": DataFrame  — leg-level events (TP1/TP2/STOP/EOD)
          "equity_df":      DataFrame  — per-bar equity curve from single-pair run
          "final_equity":   float
          "risk_pct":       float      — risk % used for this pair
        }
    """
    if risk_pct_by_pair is None:
        risk_pct_by_pair = RISK_PCT_BY_PAIR
    engine_kwargs = engine_kwargs or {}

    results = {}
    for cfg in pairs_config:
        pair = cfg["pair"]
        if verbose:
            print(f"\n{'='*64}")
            print(f"  Per-pair backtest: {pair}")
            print(f"{'='*64}")

        frames = load_local_data(
            cfg["file_path"],
            pair=pair,
            start_date=cfg.get("start_date"),
            end_date=cfg.get("end_date"),
            chart_tf="1h",
        )

        base_spread, base_slip = get_default_costs(pair)
        spread    = base_spread * spread_mult
        slippage  = base_slip   * slip_mult
        risk_pct  = risk_pct_by_pair.get(pair, 0.015)

        trade_df, exit_events_df, equity_df, final_equity = run_backtest(
            frames,
            spread=spread,
            slippage_std=slippage,
            use_session_filter=True,
            use_pullback_entry=True,
            use_structure_stop=True,
            risk_pct=risk_pct,
            verbose=verbose,
            **engine_kwargs,   # e.g. tp1_r / tp2_r for param-robustness sweeps
        )

        results[pair] = {
            "trade_df":       trade_df,
            "exit_events_df": exit_events_df,
            "equity_df":      equity_df,
            "final_equity":   final_equity,
            "risk_pct":       risk_pct,
        }

        if verbose:
            print(f"  {pair}: {len(trade_df)} trades  "
                  f"final equity ${final_equity:,.2f}  "
                  f"risk_pct={risk_pct:.2%}")

    return results


# =============================================================================
# STEP 2  —  BUILD UNIFIED CHRONOLOGICAL EVENT STREAM
# =============================================================================

def build_event_stream(per_pair_results: dict) -> list:
    """
    Convert per-pair trade logs and exit-events DataFrames into a single flat
    list of events sorted chronologically.

    Event types produced:
        ENTRY   — trade opens; carries entry metadata for position sizing
        TP1     — 60 % partial close; stop moves to breakeven
        TP2     — remaining 40 % close (full exit at profit)
        STOP    — full close at stop loss (before TP1)
        BE_STOP — full close at breakeven stop (after TP1, loss ≈ 0)
        EOD     — end-of-data force close

    Each event dict contains everything the portfolio simulator needs to
    process it without looking back at the original DataFrames.

    Tie-breaking within identical timestamps:
        (timestamp, _EVENT_TYPE_ORDER[type], pair, trade_id)
    Full closes are processed before partial closes, which are processed
    before new entries — so equity is always current when a trade opens.
    """
    events = []

    for pair, result in per_pair_results.items():
        trade_df  = result["trade_df"]
        exit_df   = result["exit_events_df"]
        risk_pct  = result["risk_pct"]

        if trade_df.empty or exit_df.empty:
            continue

        # Build fast lookup: trade_id (int) → trade row Series
        trade_lookup = {int(r["trade_id"]): r for _, r in trade_df.iterrows()}

        # ------------------------------------------------------------------
        # ENTRY events  (one per trade row)
        # ------------------------------------------------------------------
        for _, row in trade_df.iterrows():
            tid   = int(row["trade_id"])
            tkey  = f"{pair}_{tid}"

            # Recover equity at entry: equity_after includes this trade's PnL,
            # so equity_at_entry = equity_after - total_pnl.
            per_pair_equity_at_entry = float(row["equity_after"]) - float(row["total_pnl"])

            events.append({
                "time":                     pd.Timestamp(row["entry_time"]),
                "type":                     "ENTRY",
                "pair":                     pair,
                "trade_key":                tkey,
                "trade_id":                 tid,
                "direction":                row["direction"],
                "entry_price":              float(row["entry_price"]),
                "orig_stop_price":          float(row["orig_stop_price"]),
                "R":                        float(row["R_price"]),
                "per_pair_equity_at_entry": per_pair_equity_at_entry,
                "risk_pct":                 risk_pct,
                "exit_mode":                str(row.get("exit_mode", "dual_tp")),
                # EXIT-event fields not applicable at ENTRY
                "per_pair_pnl": 0.0,
                "price":        float(row["entry_price"]),
            })

        # ------------------------------------------------------------------
        # EXIT events  (from exit_events_df: TP1, TP2, STOP, BE_STOP, EOD)
        # ------------------------------------------------------------------
        for _, row in exit_df.iterrows():
            tid   = int(row["trade_id"])
            tkey  = f"{pair}_{tid}"
            trow  = trade_lookup.get(tid)
            if trow is None:
                continue

            per_pair_equity_at_entry = float(trow["equity_after"]) - float(trow["total_pnl"])

            events.append({
                "time":                     pd.Timestamp(row["time"]),
                "type":                     str(row["event"]),
                "pair":                     pair,
                "trade_key":                tkey,
                "trade_id":                 tid,
                "direction":                str(trow["direction"]),
                "entry_price":              float(trow["entry_price"]),
                "orig_stop_price":          float(trow["orig_stop_price"]),
                "R":                        float(trow["R_price"]),
                "per_pair_equity_at_entry": per_pair_equity_at_entry,
                "risk_pct":                 risk_pct,
                "exit_mode":                str(trow.get("exit_mode", "dual_tp")),
                "per_pair_pnl":             float(row["pnl"]),
                "price":                    float(row["price"]),
            })

    # Sort: timestamp → event-type priority → pair → trade_id
    events.sort(key=lambda e: (
        e["time"],
        _EVENT_TYPE_ORDER.get(e["type"], 99),
        e["pair"],
        e["trade_id"],
    ))

    return events


# =============================================================================
# STEP 3  —  PORTFOLIO SIMULATION
# =============================================================================

def simulate_portfolio(
    events: list,
    risk_pct_by_pair: dict = None,
    portfolio_cap_pct: float = PORTFOLIO_CAP_PCT,
    breakeven_residual: float = BREAKEVEN_RESIDUAL_RISK_FRACTION,
    starting_equity: float = PORTFOLIO_STARTING_EQUITY,
    verbose: bool = True,
) -> tuple:
    """
    Replay the unified event stream against one shared equity account.
    Trades are rejected when the projected total open risk would exceed
    `portfolio_cap_pct` of current equity.

    Parameters
    ----------
    events            : output of build_event_stream()
    risk_pct_by_pair  : per-pair risk fractions used to size each entry
    portfolio_cap_pct : max total open risk / equity fraction
    breakeven_residual: fraction of original risk remaining after TP1/BE
                        (0.0 = fully zero out, default conservative approach)
    starting_equity   : initial account balance
    verbose           : print cap rejections to console

    Returns
    -------
    portfolio_trade_log : list of dicts — one record per completed trade
    equity_curve        : list of dicts — one record per event
    rejected_keys       : list of str  — trade_keys rejected by the cap
    """
    if risk_pct_by_pair is None:
        risk_pct_by_pair = RISK_PCT_BY_PAIR

    portfolio_equity = starting_equity
    open_positions   = {}     # trade_key → position state dict
    total_open_risk  = 0.0    # running sum of open risk in $
    rejected_keys    = set()  # trade_keys rejected by the portfolio cap

    portfolio_trade_log: list = []
    equity_curve: list = [{
        "time":          events[0]["time"] if events else pd.Timestamp("1970-01-01"),
        "equity":        starting_equity,
        "event":         "START",
        "pair":          "",
        "trade_key":     "",
        "n_open":        0,
        "open_risk_pct": 0.0,
    }]

    # Accumulating dict for in-flight trade records (completed on final close)
    in_progress: dict = {}

    for ev in events:
        t     = ev["time"]
        etype = ev["type"]
        pair  = ev["pair"]
        tkey  = ev["trade_key"]
        tid   = ev["trade_id"]

        # ==================================================================
        # ENTRY
        # ==================================================================
        if etype == "ENTRY":
            # Skip if already rejected (shouldn't happen but guard)
            if tkey in rejected_keys:
                continue

            risk_pct   = risk_pct_by_pair.get(pair, 0.015)
            new_risk_usd = portfolio_equity * risk_pct

            # --- Portfolio cap check ---
            # Open risk accounting:
            #   - new_risk_usd  = portfolio_equity × risk_pct[pair]  (full R at entry)
            #   - After TP1, stop moves to breakeven → remaining risk set to 0
            #     (controlled by BREAKEVEN_RESIDUAL_RISK_FRACTION, default 0.0)
            #   - A trade is rejected if (current_open_risk + new_risk) / equity > cap
            projected_risk = (total_open_risk + new_risk_usd) / portfolio_equity
            if projected_risk > portfolio_cap_pct:
                rejected_keys.add(tkey)
                if verbose:
                    print(
                        f"  [CAP REJECT] {tkey} @ {t}  "
                        f"open={total_open_risk/portfolio_equity:.2%}  "
                        f"new={risk_pct:.2%}  "
                        f"projected={projected_risk:.2%} > cap={portfolio_cap_pct:.2%}"
                    )
                continue

            # Scale factor: rescales per-pair PnL to portfolio equity.
            # All legs of this trade use the same scale factor (fixed at entry).
            per_pair_eq  = ev["per_pair_equity_at_entry"]
            scale_factor = (portfolio_equity / per_pair_eq) if per_pair_eq > 0 else 1.0

            open_positions[tkey] = {
                "pair":                      pair,
                "trade_id":                  tid,
                "direction":                 ev["direction"],
                "entry_time":                t,
                "entry_price":               ev["entry_price"],
                "orig_stop_price":           ev["orig_stop_price"],
                "R":                         ev["R"],
                "risk_pct":                  risk_pct,
                "risk_usd":                  new_risk_usd,   # updated after TP1
                "scale_factor":              scale_factor,
                "portfolio_equity_at_entry": portfolio_equity,
                "per_pair_equity_at_entry":  per_pair_eq,
                "tp1_hit":                   False,
                "exit_mode":                 ev.get("exit_mode", "dual_tp"),
            }

            in_progress[tkey] = {
                "trade_key":                 tkey,
                "pair":                      pair,
                "trade_id":                  tid,
                "entry_time":                t,
                "entry_price":               ev["entry_price"],
                "orig_stop_price":           ev["orig_stop_price"],
                "R":                         ev["R"],
                "direction":                 ev["direction"],
                "portfolio_equity_at_entry": portfolio_equity,
                "per_pair_equity_at_entry":  per_pair_eq,
                "scale_factor":              scale_factor,
                "risk_usd":                  new_risk_usd,
                "risk_pct":                  risk_pct,
                "total_portfolio_pnl":       0.0,
                "tp1_hit":                   False,
                "tp2_hit":                   False,
                "exit_time":                 None,
                "exit_price":                None,
                "exit_event":                None,
                "exit_mode":                 ev.get("exit_mode", "dual_tp"),
            }

            total_open_risk += new_risk_usd

        # ==================================================================
        # TP1  (partial close — 60 %; stop moves to breakeven)
        # ==================================================================
        elif etype == "TP1":
            if tkey in rejected_keys or tkey not in open_positions:
                continue

            pos   = open_positions[tkey]
            pnl   = ev["per_pair_pnl"] * pos["scale_factor"]
            portfolio_equity += pnl

            # After TP1 the stop is at breakeven.  Reduce open risk by
            # (1 - residual_fraction) of original risk.
            removed_risk = pos["risk_usd"] * (1.0 - breakeven_residual)
            total_open_risk  -= removed_risk
            pos["risk_usd"]   = pos["risk_usd"] * breakeven_residual   # residual (default 0)
            pos["tp1_hit"]    = True

            rec = in_progress[tkey]
            rec["total_portfolio_pnl"] += pnl
            rec["tp1_hit"]             = True

        # ==================================================================
        # FULL CLOSE  (TP2, STOP, BE_STOP, EOD)
        # ==================================================================
        elif etype in ("TP2", "STOP", "BE_STOP", "EOD"):
            if tkey in rejected_keys or tkey not in open_positions:
                continue

            pos  = open_positions.pop(tkey)
            pnl  = ev["per_pair_pnl"] * pos["scale_factor"]
            portfolio_equity += pnl

            # Remove whatever open risk remains (0 if TP1 was already hit)
            total_open_risk -= pos["risk_usd"]
            total_open_risk  = max(total_open_risk, 0.0)  # floating-point guard

            # Finalise the trade record
            rec = in_progress.pop(tkey, {})
            rec["total_portfolio_pnl"] += pnl
            rec["exit_time"]   = t
            rec["exit_price"]  = ev["price"]
            rec["tp2_hit"]     = (etype == "TP2")
            rec["exit_event"]  = etype
            rec["equity_after"] = round(portfolio_equity, 2)
            peq = rec.get("portfolio_equity_at_entry", starting_equity)
            rec["pnl_pct"] = rec["total_portfolio_pnl"] / peq * 100 if peq else 0.0
            rec["total_portfolio_pnl"] = round(rec["total_portfolio_pnl"], 2)

            portfolio_trade_log.append(rec)

        # ==================================================================
        # Record equity curve point after every processed event
        # ==================================================================
        n_open        = len(open_positions)
        open_risk_pct = (total_open_risk / portfolio_equity * 100
                         if portfolio_equity > 0 else 0.0)

        equity_curve.append({
            "time":          t,
            "equity":        round(portfolio_equity, 2),
            "event":         etype,
            "pair":          pair,
            "trade_key":     tkey,
            "n_open":        n_open,
            "open_risk_pct": round(open_risk_pct, 4),
        })

    return portfolio_trade_log, equity_curve, list(rejected_keys)


# =============================================================================
# STEP 4  —  ANALYTICS
# =============================================================================

def compute_portfolio_metrics(
    portfolio_trade_log: list,
    starting_equity: float = PORTFOLIO_STARTING_EQUITY,
) -> dict:
    """
    Aggregate performance metrics for the combined portfolio.

    Sharpe ratio is annualised using per-trade returns.  The annualisation
    factor is computed from the actual date range and trade count so it adapts
    to the backtest length automatically.
    """
    if not portfolio_trade_log:
        return {}

    tdf = pd.DataFrame(portfolio_trade_log)
    tdf["total_portfolio_pnl"] = tdf["total_portfolio_pnl"].astype(float)
    tdf["portfolio_equity_at_entry"] = tdf["portfolio_equity_at_entry"].astype(float)

    n_trades  = len(tdf)
    wins      = int((tdf["total_portfolio_pnl"] > 0).sum())
    losses    = n_trades - wins
    win_rate  = wins / n_trades * 100 if n_trades else 0.0
    total_pnl = float(tdf["total_portfolio_pnl"].sum())

    # Build equity series: starting equity + running cumulative PnL
    eq_series = np.concatenate([[starting_equity], tdf["equity_after"].astype(float).values])
    final_eq  = float(eq_series[-1])
    total_ret = (final_eq - starting_equity) / starting_equity * 100

    # Maximum drawdown
    peak   = np.maximum.accumulate(eq_series)
    max_dd = float(((eq_series - peak) / peak * 100).min())

    # Sharpe ratio (per-trade, annualised)
    returns = tdf["total_portfolio_pnl"] / tdf["portfolio_equity_at_entry"]
    try:
        t0    = pd.Timestamp(tdf["entry_time"].min())
        t1    = pd.Timestamp(tdf["exit_time"].max())
        years = max((t1 - t0).days / 365.25, 1 / 365.25)
        trades_per_year = n_trades / years
    except Exception:
        trades_per_year = 25.0
    ann_factor = float(np.sqrt(trades_per_year))
    sharpe = (float(returns.mean()) / float(returns.std()) * ann_factor
              if len(returns) > 1 and returns.std() > 0 else 0.0)

    # Profit factor
    gross_wins   = float(tdf.loc[tdf["total_portfolio_pnl"] > 0, "total_portfolio_pnl"].sum())
    gross_losses = float(abs(tdf.loc[tdf["total_portfolio_pnl"] < 0, "total_portfolio_pnl"].sum()))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    avg_win  = float(tdf.loc[tdf["total_portfolio_pnl"] > 0, "total_portfolio_pnl"].mean()) if wins  else 0.0
    avg_loss = float(tdf.loc[tdf["total_portfolio_pnl"] < 0, "total_portfolio_pnl"].mean()) if losses else 0.0

    return {
        "n_trades":     n_trades,
        "wins":         wins,
        "losses":       losses,
        "win_rate_pct": round(win_rate, 2),
        "total_pnl":    round(total_pnl, 2),
        "final_equity": round(final_eq, 2),
        "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe":       round(sharpe, 3),
        "profit_factor": round(profit_factor, 3),
        "avg_win_$":    round(avg_win, 2),
        "avg_loss_$":   round(avg_loss, 2),
    }


def build_yearly_breakdown(
    portfolio_trade_log: list,
    starting_equity: float = PORTFOLIO_STARTING_EQUITY,
) -> pd.DataFrame:
    """
    Annual performance breakdown for the combined portfolio.
    Equity at the start of each year is the equity at the end of the prior year.
    """
    if not portfolio_trade_log:
        return pd.DataFrame()

    tdf = pd.DataFrame(portfolio_trade_log)
    tdf["year"]        = pd.to_datetime(tdf["exit_time"]).dt.year
    tdf["won"]         = tdf["total_portfolio_pnl"] > 0
    tdf["equity_after"] = tdf["equity_after"].astype(float)
    tdf["total_portfolio_pnl"] = tdf["total_portfolio_pnl"].astype(float)

    rows = []
    running_eq = starting_equity
    for year, grp in tdf.groupby("year"):
        n         = len(grp)
        wins      = int(grp["won"].sum())
        pnl       = float(grp["total_portfolio_pnl"].sum())
        start_eq  = running_eq
        running_eq = float(grp["equity_after"].iloc[-1])
        ret        = pnl / start_eq * 100 if start_eq else 0.0

        gross_w = float(grp.loc[grp["won"],  "total_portfolio_pnl"].sum())
        gross_l = float(abs(grp.loc[~grp["won"], "total_portfolio_pnl"].sum()))
        pf      = gross_w / gross_l if gross_l > 0 else float("inf")

        rows.append({
            "year":          year,
            "trades":        n,
            "wins":          wins,
            "losses":        n - wins,
            "win_rate_pct":  round(wins / n * 100 if n else 0.0, 1),
            "total_pnl":     round(pnl, 2),
            "return_pct":    round(ret, 2),
            "profit_factor": round(pf, 3),
            "equity_end":    round(running_eq, 2),
        })

    return pd.DataFrame(rows)


def build_pair_contributions(portfolio_trade_log: list) -> pd.DataFrame:
    """
    Per-pair PnL contribution, trade count, win rate, and profit factor
    within the combined portfolio.
    """
    if not portfolio_trade_log:
        return pd.DataFrame()

    tdf = pd.DataFrame(portfolio_trade_log)
    tdf["won"] = tdf["total_portfolio_pnl"] > 0
    tdf["total_portfolio_pnl"] = tdf["total_portfolio_pnl"].astype(float)

    rows = []
    for pair, grp in tdf.groupby("pair"):
        n       = len(grp)
        wins    = int(grp["won"].sum())
        pnl     = float(grp["total_portfolio_pnl"].sum())
        gross_w = float(grp.loc[grp["won"],  "total_portfolio_pnl"].sum())
        gross_l = float(abs(grp.loc[~grp["won"], "total_portfolio_pnl"].sum()))
        pf      = gross_w / gross_l if gross_l > 0 else float("inf")
        rows.append({
            "pair":          pair,
            "trades":        n,
            "wins":          wins,
            "losses":        n - wins,
            "win_rate_pct":  round(wins / n * 100 if n else 0.0, 1),
            "total_pnl":     round(pnl, 2),
            "profit_factor": round(pf, 3),
            "pct_of_total_pnl": None,   # filled below
        })

    df    = pd.DataFrame(rows)
    total = float(df["total_pnl"].sum())
    if total != 0:
        df["pct_of_total_pnl"] = (df["total_pnl"] / total * 100).round(1)
    else:
        df["pct_of_total_pnl"] = 0.0

    return df


def build_overlap_analysis(
    equity_curve: list,
    portfolio_trade_log: list,
) -> dict:
    """
    Analyse how often multiple trades were open simultaneously.

    Uses two complementary methods:
      1. Counts from equity_curve n_open field at ENTRY events (how crowded was
         the account when each new trade started)
      2. Interval-overlap sweep on the trade log for max simultaneous positions
         and max simultaneous open risk
    """
    if not portfolio_trade_log or not equity_curve:
        return {}

    eq_df = pd.DataFrame(equity_curve)
    eq_df = eq_df[eq_df["event"] != "START"].copy()

    if eq_df.empty:
        return {}

    # --- Snapshot stats from equity curve ---
    entry_snaps = eq_df[eq_df["event"] == "ENTRY"].copy()
    if entry_snaps.empty:
        return {}

    # n_open at ENTRY events: 1 = just this trade, 2 = this + 1 other, etc.
    n_solo      = int((entry_snaps["n_open"] == 1).sum())
    n_two       = int((entry_snaps["n_open"] == 2).sum())
    n_three_plus = int((entry_snaps["n_open"] >= 3).sum())
    max_n_open   = int(eq_df["n_open"].max())
    max_open_risk_pct = float(eq_df["open_risk_pct"].max())

    # --- Interval sweep on trade log ---
    tdf = pd.DataFrame(portfolio_trade_log)
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["exit_time"]  = pd.to_datetime(tdf["exit_time"])
    tdf["risk_usd"]   = tdf["risk_usd"].astype(float)
    tdf["portfolio_equity_at_entry"] = tdf["portfolio_equity_at_entry"].astype(float)

    max_concurrent      = 0
    max_concurrent_risk = 0.0
    trades_with_overlap = 0

    for i, row in tdf.iterrows():
        # Trades open AT THE SAME TIME as this one (excluding itself)
        concurrent = tdf[
            (tdf["entry_time"] <= row["entry_time"]) &
            (tdf["exit_time"]  >  row["entry_time"]) &
            (tdf.index != i)
        ]
        n_sim = len(concurrent) + 1   # include this trade
        if n_sim > max_concurrent:
            max_concurrent = n_sim
            sim_risk = (float(row["risk_usd"]) + float(concurrent["risk_usd"].sum()))
            entry_eq = float(row["portfolio_equity_at_entry"])
            max_concurrent_risk = sim_risk / entry_eq * 100 if entry_eq else 0.0

        if len(concurrent) > 0:
            trades_with_overlap += 1

    return {
        "max_concurrent_positions":   max_concurrent,
        "max_concurrent_open_risk_pct": round(max_concurrent_risk, 2),
        "trades_with_any_overlap":    trades_with_overlap,
        "entries_only_position":      n_solo,
        "entries_with_1_other_open":  n_two,
        "entries_with_2plus_others":  n_three_plus,
        "max_n_open_at_any_event":    max_n_open,
        "peak_open_risk_pct":         round(max_open_risk_pct, 2),
    }


# =============================================================================
# PRINTING
# =============================================================================

def print_portfolio_report(
    metrics:           dict,
    yearly_df:         pd.DataFrame,
    pair_df:           pd.DataFrame,
    overlap:           dict,
    rejected:          list,
    portfolio_cap_pct: float = PORTFOLIO_CAP_PCT,
    risk_pct_by_pair:  dict  = None,
    spread_mult:       float = 1.0,
    slip_mult:         float = 1.0,
) -> None:
    """Pretty-print the full portfolio report to stdout."""
    if risk_pct_by_pair is None:
        risk_pct_by_pair = RISK_PCT_BY_PAIR

    SEP = "=" * 64

    cost_label = ("WORST-CASE  (2× spread + 2× slippage)"
                  if spread_mult >= 2.0
                  else f"realistic  (spread×{spread_mult:.1f}  slip×{slip_mult:.1f})"
                  if spread_mult != 1.0 or slip_mult != 1.0
                  else "realistic")

    print(f"\n{SEP}")
    print(f"  PORTFOLIO BACKTEST  —  Cap = {portfolio_cap_pct:.1%}")
    print(f"  Cost scenario   : {cost_label}")
    print(f"{SEP}")
    print(f"  Per-pair risk settings:")
    for p, r in risk_pct_by_pair.items():
        print(f"    {p}: {r:.2%}")
    print(f"  Starting equity: ${PORTFOLIO_STARTING_EQUITY:,.0f}")
    print(f"{SEP}")

    print(f"\n  ─── Combined Portfolio Performance ───")
    m = metrics
    print(f"  Trades          : {m.get('n_trades', 0)}")
    print(f"  Wins / Losses   : {m.get('wins', 0)} / {m.get('losses', 0)}")
    print(f"  Win Rate        : {m.get('win_rate_pct', 0):.1f}%")
    print(f"  Total PnL       : ${m.get('total_pnl', 0):>12,.2f}")
    print(f"  Total Return    : {m.get('total_return_pct', 0):>+8.2f}%")
    print(f"  Final Equity    : ${m.get('final_equity', 0):>12,.2f}")
    print(f"  Max Drawdown    : {m.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Sharpe Ratio    : {m.get('sharpe', 0):.3f}")
    print(f"  Profit Factor   : {m.get('profit_factor', 0):.3f}")
    print(f"  Avg Win         : ${m.get('avg_win_$', 0):>10,.2f}")
    print(f"  Avg Loss        : ${m.get('avg_loss_$', 0):>10,.2f}")

    if rejected:
        print(f"\n  Cap rejections: {len(rejected)} trade(s) skipped (cap={portfolio_cap_pct:.1%})")

    print(f"\n  ─── Yearly Breakdown ───")
    if not yearly_df.empty:
        print(yearly_df.to_string(index=False))
    else:
        print("  (no data)")

    print(f"\n  ─── Pair Contributions ───")
    if not pair_df.empty:
        print(pair_df.to_string(index=False))
    else:
        print("  (no data)")

    print(f"\n  ─── Overlap Analysis ───")
    print(f"  Max simultaneous open positions : {overlap.get('max_concurrent_positions', 0)}")
    print(f"  Max simultaneous open risk      : {overlap.get('max_concurrent_open_risk_pct', 0):.2f}%")
    print(f"  Peak open risk (any event)      : {overlap.get('peak_open_risk_pct', 0):.2f}%")
    print(f"  Trades with any overlap         : {overlap.get('trades_with_any_overlap', 0)}")
    print(f"  Entries: only position          : {overlap.get('entries_only_position', 0)}")
    print(f"  Entries: 1 other open           : {overlap.get('entries_with_1_other_open', 0)}")
    print(f"  Entries: 2+ others open         : {overlap.get('entries_with_2plus_others', 0)}")

    print(f"{SEP}\n")


# =============================================================================
# EXPORT
# =============================================================================

def export_portfolio(
    portfolio_trade_log: list,
    equity_curve:        list,
    yearly_df:           pd.DataFrame,
    pair_df:             pd.DataFrame,
    metrics:             dict,
    overlap:             dict,
    out_dir:             str,
) -> None:
    """
    Save all portfolio outputs to CSV files in out_dir.

    The folder already carries the scenario context (realistic / worst_case),
    so filenames are kept short and free of redundant prefixes.

    Files produced:
        trades.csv        — combined trade log
        equity.csv        — clean time/equity curve  (deduplicated, for charting)
        equity_detail.csv — equity curve with event type, pair, n_open, open_risk
        yearly.csv        — annual breakdown
        pairs.csv         — per-pair contribution
        summary.csv       — scalar metrics
        overlap.csv       — overlap statistics
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Trade log
    if portfolio_trade_log:
        tdf = pd.DataFrame(portfolio_trade_log)
        path = os.path.join(out_dir, "trades.csv")
        tdf.to_csv(path, index=False)
        print(f"  Saved: trades.csv  ({len(tdf)} trades)")

    # 2. Equity curve — one row per event; also save a clean time/equity version
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve)
        # Detailed version (includes event type, pair, n_open, risk)
        path_full = os.path.join(out_dir, "equity_detail.csv")
        eq_df.to_csv(path_full, index=False)
        # Clean version: deduplicated time/equity for charting
        path_clean = os.path.join(out_dir, "equity.csv")
        eq_df[["time", "equity"]].drop_duplicates(subset="time", keep="last").to_csv(
            path_clean, index=False
        )
        print(f"  Saved: equity.csv  ({len(eq_df)} event points)")

    # 3. Yearly breakdown
    if not yearly_df.empty:
        path = os.path.join(out_dir, "yearly.csv")
        yearly_df.to_csv(path, index=False)
        print(f"  Saved: yearly.csv")

    # 4. Pair contributions
    if not pair_df.empty:
        path = os.path.join(out_dir, "pairs.csv")
        pair_df.to_csv(path, index=False)
        print(f"  Saved: pairs.csv")

    # 5. Summary metrics
    if metrics:
        rows = [{"metric": k, "value": v} for k, v in metrics.items()]
        path = os.path.join(out_dir, "summary.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Saved: summary.csv")

    # 6. Overlap stats
    if overlap:
        rows = [{"metric": k, "value": v} for k, v in overlap.items()]
        path = os.path.join(out_dir, "overlap.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Saved: overlap.csv")


# =============================================================================
# MAIN PIPELINE  (callable by run_portfolio.py)
# =============================================================================

def run_portfolio_backtest(
    pairs_config:      list,
    risk_pct_by_pair:  dict  = None,
    portfolio_cap_pct: float = PORTFOLIO_CAP_PCT,
    starting_equity:   float = PORTFOLIO_STARTING_EQUITY,
    spread_mult:       float = 1.0,
    slip_mult:         float = 1.0,
    out_dir:           str   = None,
    verbose:           bool  = True,
    engine_kwargs:     dict  = None,
) -> dict:
    """
    Full pipeline:
        1. Run per-pair backtests
        2. Build unified event stream
        3. Simulate shared-equity portfolio (with open-risk cap)
        4. Compute analytics
        5. Print report
        6. Export CSVs  (if out_dir is given)

    Parameters
    ----------
    pairs_config      : list of {pair, file_path, start_date, end_date}
    risk_pct_by_pair  : per-pair risk fractions
    portfolio_cap_pct : max total open risk / equity fraction
    starting_equity   : initial account balance
    spread_mult       : multiplier on default spread per pair  (1.0 = realistic,
                        2.0 = worst-case doubling from sim_costs.SCENARIO_WORST_CASE)
    slip_mult         : multiplier on default slippage std (same scale)
    out_dir           : directory for CSV output; None = skip export
    verbose           : print progress + report

    Returns
    -------
    dict with keys:
        metrics, yearly_df, pair_df, overlap,
        portfolio_trade_log, equity_curve, rejected,
        per_pair_results
    """
    if risk_pct_by_pair is None:
        risk_pct_by_pair = RISK_PCT_BY_PAIR

    # ---- Step 1: per-pair runs ----
    per_pair_results = run_per_pair_backtests(
        pairs_config, risk_pct_by_pair,
        spread_mult=spread_mult, slip_mult=slip_mult,
        verbose=verbose,
        engine_kwargs=engine_kwargs,
    )

    total_trades  = sum(len(r["trade_df"])       for r in per_pair_results.values())
    total_events  = sum(len(r["exit_events_df"]) for r in per_pair_results.values())
    if verbose:
        print(f"\n  Per-pair totals: {total_trades} trades  |  {total_events} exit events")

    # ---- Step 2: event stream ----
    events = build_event_stream(per_pair_results)
    if verbose:
        print(f"  Unified event stream: {len(events)} events")

    # ---- Step 3: simulate ----
    portfolio_trade_log, equity_curve, rejected = simulate_portfolio(
        events,
        risk_pct_by_pair=risk_pct_by_pair,
        portfolio_cap_pct=portfolio_cap_pct,
        starting_equity=starting_equity,
        verbose=verbose,
    )
    if verbose:
        n_rej = len(rejected)
        print(f"  Portfolio trades: {len(portfolio_trade_log)}  "
              f"| Rejected by cap: {n_rej}")

    # ---- Step 4: analytics ----
    metrics   = compute_portfolio_metrics(portfolio_trade_log, starting_equity)
    yearly_df = build_yearly_breakdown(portfolio_trade_log, starting_equity)
    pair_df   = build_pair_contributions(portfolio_trade_log)
    overlap   = build_overlap_analysis(equity_curve, portfolio_trade_log)

    # ---- Step 5: report ----
    if verbose:
        print_portfolio_report(
            metrics, yearly_df, pair_df, overlap, rejected,
            portfolio_cap_pct=portfolio_cap_pct,
            risk_pct_by_pair=risk_pct_by_pair,
            spread_mult=spread_mult,
            slip_mult=slip_mult,
        )

    # ---- Step 6: export ----
    if out_dir:
        if verbose:
            print(f"\n  Exporting to: {out_dir}")
        export_portfolio(
            portfolio_trade_log, equity_curve, yearly_df, pair_df,
            metrics, overlap, out_dir,
        )

    return {
        "metrics":             metrics,
        "yearly_df":           yearly_df,
        "pair_df":             pair_df,
        "overlap":             overlap,
        "portfolio_trade_log": portfolio_trade_log,
        "equity_curve":        equity_curve,
        "rejected":            rejected,
        "per_pair_results":    per_pair_results,
    }
