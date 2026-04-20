# =============================================================================
# robustness.py — Validation & Robustness Suite
# =============================================================================
# Tasks 2–5, 7, 8
#
# PUBLIC API
# ----------
# run_walk_forward(frames, pair, run_backtest_fn, ...)  -> dict
# run_rolling_windows(frames, pair, run_backtest_fn, ...) -> DataFrame
# run_mfe_mae_by_year(trade_df, pair, ...)             -> DataFrame
# run_param_robustness(frames, pair, run_backtest_fn, ...) -> DataFrame
# compute_rolling_performance(trade_df, pair, ...)     -> DataFrame
# print_final_decision(trade_df, final_equity, ...)    -> None
# run_full_robustness_suite(frames, pair, ...)         -> dict
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from analytics import compute_mfe_mae, build_yearly_summary, STARTING_EQUITY
except ImportError:
    STARTING_EQUITY = 100_000.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _filter_frames_by_date(frames: dict, start_date: str, end_date: str) -> dict:
    """Slice every timeframe DataFrame to [start_date, end_date]."""
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
    filtered = {}
    for tf, df in frames.items():
        mask = (df.index >= s) & (df.index <= e)
        filtered[tf] = df.loc[mask].copy()
    return filtered


def _metrics(trade_df: pd.DataFrame, final_equity: float,
             starting_equity: float = STARTING_EQUITY) -> dict:
    """Return a standard metrics dict from a trade_df."""
    if trade_df.empty:
        return {
            "trades": 0, "win_rate": 0.0, "total_return": 0.0,
            "final_equity": starting_equity, "max_drawdown": 0.0,
            "sharpe": 0.0, "profit_factor": 0.0,
        }
    pnl_col = "total_pnl" if "total_pnl" in trade_df.columns else "pnl"
    n    = len(trade_df)
    wins = (trade_df[pnl_col] > 0).sum()
    wr   = wins / n * 100

    tot  = (final_equity - starting_equity) / starting_equity * 100

    eq   = np.concatenate([[starting_equity], trade_df["equity_after"].values])
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak * 100).min())

    rets   = trade_df["pnl_pct"].values / 100 if "pnl_pct" in trade_df.columns else np.zeros(n)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    gp = float(trade_df.loc[trade_df[pnl_col] > 0, pnl_col].sum())
    gl = abs(float(trade_df.loc[trade_df[pnl_col] < 0, pnl_col].sum()))
    pf = gp / gl if gl > 0 else float("inf")

    return {
        "trades": n, "win_rate": round(wr, 1),
        "total_return": round(tot, 2),
        "final_equity": round(final_equity, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "profit_factor": round(pf, 3),
    }


def _nan(v) -> bool:
    """Safe NaN check for float and non-float types."""
    try:
        return v != v
    except Exception:
        return False


# =============================================================================
# TASK 2 — WALK-FORWARD VALIDATION
# =============================================================================

def run_walk_forward(
    frames: dict,
    pair: str,
    run_backtest_fn,
    spread: float = 0.0,
    slippage_std: float = 0.0,
    train_start: str = "2021-01-01",
    train_end: str   = "2023-12-31",
    test_start: str  = "2024-01-01",
    test_end: str    = "2025-12-31",
    risk_pct: float  = 0.015,
    starting_equity: float = STARTING_EQUITY,
    **kwargs,
) -> dict:
    """
    Walk-forward validation.

    TRAIN: train_start – train_end   (parameters fitted to this period)
    TEST:  test_start  – test_end    (SAME parameters, no re-fitting)

    Returns
    -------
    dict with keys "TRAIN" and "TEST", each containing a metrics dict.
    """
    W   = 72
    SEP = "=" * W
    DIV = "  " + "─" * (W - 2)

    results = {}

    for label, t0, t1 in [
        ("TRAIN", train_start, train_end),
        ("TEST",  test_start,  test_end),
    ]:
        sub = _filter_frames_by_date(frames, t0, t1)
        if sub["5m"].empty:
            print(f"  [{label}] No 5m data for {t0} – {t1}. Skipping.")
            results[label] = {}
            continue

        n_bars = len(sub["5m"])
        print(f"\n  [{label}]  {t0} – {t1}  ({n_bars:,} 5m bars) ...")

        trade_df, _, equity_df, final_eq = run_backtest_fn(
            sub,
            spread=spread,
            slippage_std=slippage_std,
            risk_pct=risk_pct,
            verbose=False,
            **kwargs,
        )
        m = _metrics(trade_df, final_eq, starting_equity)
        m["start"] = t0
        m["end"]   = t1
        results[label] = m

    # ── Side-by-side comparison table ────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  WALK-FORWARD VALIDATION  —  {pair}")
    print(f"  Train : {train_start} – {train_end}")
    print(f"  Test  : {test_start}  – {test_end}")
    print(SEP)

    fields = [
        ("Trades",           "trades",        "{:>12.0f}", False),
        ("Win Rate (%)",     "win_rate",      "{:>12.1f}", True),
        ("Total Return (%)", "total_return",  "{:>+12.2f}", True),
        ("Max Drawdown (%)","max_drawdown",   "{:>+12.2f}", False),
        ("Sharpe",           "sharpe",        "{:>12.3f}", True),
        ("Profit Factor",    "profit_factor", "{:>12.3f}", True),
    ]

    row_fmt = "  {:<24} {:>12} {:>12} {:>10}"
    print(row_fmt.format("Metric", "TRAIN", "TEST", "Δ (test−train)"))
    print(DIV)

    tr = results.get("TRAIN", {})
    te = results.get("TEST",  {})

    for label_txt, key, fmt, _ in fields:
        tv = tr.get(key, float("nan"))
        xv = te.get(key, float("nan"))
        if _nan(tv) or _nan(xv):
            delta_s = "       n/a"
        else:
            delta_s = f"{xv - tv:>+10.2f}"
        tv_s = fmt.format(tv) if not _nan(tv) else "         n/a"
        xv_s = fmt.format(xv) if not _nan(xv) else "         n/a"
        print(f"  {label_txt:<24} {tv_s} {xv_s} {delta_s}")

    print(DIV)

    # ── Verdict ──────────────────────────────────────────────────────────────
    if tr and te:
        x_ret = te.get("total_return", 0)
        x_shr = te.get("sharpe", 0)
        t_ret = tr.get("total_return", 0)
        decay = (x_ret - t_ret) / abs(t_ret) * 100 if t_ret != 0 else float("nan")

        if x_ret > 0 and x_shr > 1.0:
            verdict   = "PASS — Edge persists strongly out-of-sample"
            edge_type = "STRUCTURAL"
        elif x_ret > 0 and x_shr > 0.5:
            verdict   = "PASS — Edge persists out-of-sample"
            edge_type = "STRUCTURAL"
        elif x_ret > 0:
            verdict   = "MARGINAL — Positive OOS but weak Sharpe"
            edge_type = "POSSIBLY REGIME-DEPENDENT"
        else:
            verdict   = "FAIL — Edge does NOT persist out-of-sample"
            edge_type = "REGIME-DEPENDENT"

        print(f"\n  VERDICT    : {verdict}")
        print(f"  EDGE TYPE  : {edge_type}")
        if not _nan(decay):
            print(f"  Return decay (train → test) : {decay:+.1f}%")

    print(f"{SEP}\n")
    return results


# =============================================================================
# TASK 3 — ROLLING WINDOW ANALYSIS
# =============================================================================

def run_rolling_windows(
    frames: dict,
    pair: str,
    run_backtest_fn,
    spread: float = 0.0,
    slippage_std: float = 0.0,
    window_months: int = 6,
    risk_pct: float = 0.015,
    starting_equity: float = STARTING_EQUITY,
    **kwargs,
) -> pd.DataFrame:
    """
    Evaluate strategy performance in non-overlapping rolling windows.

    Parameters
    ----------
    window_months : length of each window (default 6 months)

    Returns
    -------
    DataFrame with one row per window — return, sharpe, max_drawdown.
    """
    W   = 76
    SEP = "=" * W
    DIV = "  " + "─" * (W - 2)

    df5 = frames["5m"]
    if df5.empty:
        print("  [rolling windows] No 5m data.")
        return pd.DataFrame()

    # ── Build non-overlapping windows ────────────────────────────────────────
    start_dt = df5.index[0].to_period("M").to_timestamp()
    end_dt   = df5.index[-1]

    windows = []
    cur = start_dt
    while cur <= end_dt:
        w_start = cur
        w_end   = cur + pd.DateOffset(months=window_months) - pd.Timedelta(seconds=1)
        if w_start > end_dt:
            break
        w_end = min(w_end, end_dt)
        windows.append((w_start, w_end))
        cur = cur + pd.DateOffset(months=window_months)

    print(f"\n{SEP}")
    print(f"  ROLLING WINDOW ANALYSIS  —  {pair}  "
          f"({window_months}-month windows, {len(windows)} total)")
    print(SEP)

    hdr = (f"  {'Window':<23} {'Return%':>9} {'Sharpe':>8} "
           f"{'MaxDD%':>8} {'Trades':>7} {'WR%':>6}  Status")
    print(hdr)
    print(DIV)

    rows = []
    for w_start, w_end in windows:
        sub = _filter_frames_by_date(
            frames,
            str(w_start.date()),
            str(w_end.date()),
        )
        if sub["5m"].empty or len(sub["5m"]) < 100:
            label = (f"{w_start.strftime('%Y-%m')} – "
                     f"{pd.Timestamp(w_end).strftime('%Y-%m')}")
            print(f"  {label:<23} {'(insufficient data)':>50}")
            continue

        trade_df, _, equity_df, final_eq = run_backtest_fn(
            sub,
            spread=spread,
            slippage_std=slippage_std,
            risk_pct=risk_pct,
            verbose=False,
            **kwargs,
        )
        m = _metrics(trade_df, final_eq, starting_equity)

        label  = (f"{w_start.strftime('%Y-%m')} – "
                  f"{pd.Timestamp(w_end).strftime('%Y-%m')}")
        ret    = m["total_return"]
        shr    = m["sharpe"]
        dd     = m["max_drawdown"]
        n      = m["trades"]
        wr     = m["win_rate"]
        status = ("✓ OK" if ret > 0 and dd > -10
                  else ("⚠ DD>10" if dd <= -10 else "✗ NEG"))

        print(f"  {label:<23} {ret:>+8.2f}%  {shr:>7.3f}  "
              f"{dd:>+7.2f}%  {n:>6}  {wr:>5.1f}%  {status}")

        rows.append({
            "window": label,
            "start":  str(w_start.date()),
            "end":    str(pd.Timestamp(w_end).date()),
            **m,
        })

    if not rows:
        print(f"{SEP}\n")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    n_pos    = (df["total_return"] > 0).sum()
    n_tot    = len(df)
    avg_ret  = df["total_return"].mean()
    avg_shr  = df["sharpe"].mean()
    worst_dd = df["max_drawdown"].min()

    print(DIV)
    print(f"\n  Windows positive  : {n_pos} / {n_tot}  "
          f"({n_pos/n_tot*100:.0f}%)")
    print(f"  Avg return / win  : {avg_ret:+.2f}%")
    print(f"  Avg Sharpe        : {avg_shr:.3f}")
    print(f"  Worst DD window   : {worst_dd:.2f}%")

    consistency = n_pos / n_tot if n_tot > 0 else 0
    if consistency >= 0.75 and avg_shr > 0.5:
        assessment = "CONSISTENT — edge holds across most periods"
    elif consistency >= 0.50:
        assessment = "MIXED — some regime sensitivity detected"
    else:
        assessment = "UNSTABLE — performance concentrated in few windows"

    print(f"\n  ASSESSMENT : {assessment}")
    print(f"{SEP}\n")

    return df


# =============================================================================
# TASK 4 — TRADE DISTRIBUTION STABILITY (MFE/MAE by year)
# =============================================================================

def run_mfe_mae_by_year(
    trade_df: pd.DataFrame,
    pair: str,
    strategy_name: str = "trailing",
) -> pd.DataFrame:
    """
    Break down MFE/MAE by year to detect regime vs structural edge.

    Requires: compute_mfe_mae() already called (mfe_R, mae_R columns present).

    Stable MFE/MAE across years → structural edge.
    High variance → regime-dependent edge.
    """
    W   = 76
    SEP = "=" * W
    DIV = "  " + "─" * (W - 2)

    if trade_df.empty or "mfe_R" not in trade_df.columns:
        print(f"  [MFE/MAE by year] mfe_R column missing — "
              f"call compute_mfe_mae() first.")
        return pd.DataFrame()

    df = trade_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["year"] = df["entry_time"].dt.year

    print(f"\n{SEP}")
    print(f"  TRADE DISTRIBUTION STABILITY  —  {pair}  [{strategy_name}]")
    print(SEP)

    hdr = (f"  {'Year':<6} {'N':>5}  {'MFE(R)':>8} {'MAE(R)':>8}  "
           f"{'Ratio':>6}  {'%≥1R':>6} {'%≥1.5R':>7} {'%≥2R':>6}")
    print(hdr)
    print(DIV)

    rows = []
    for year, grp in df.groupby("year"):
        mfe = grp["mfe_R"].dropna()
        mae = (grp["mae_R"].dropna()
               if "mae_R" in grp.columns else pd.Series(dtype=float))
        n = len(grp)

        avg_mfe = float(mfe.mean()) if len(mfe) else float("nan")
        avg_mae = float(mae.mean()) if len(mae) else float("nan")
        ratio   = (avg_mfe / avg_mae
                   if (not _nan(avg_mae) and avg_mae > 0)
                   else float("nan"))
        p1r   = float((mfe >= 1.0).mean() * 100) if len(mfe) else float("nan")
        p15r  = float((mfe >= 1.5).mean() * 100) if len(mfe) else float("nan")
        p2r   = float((mfe >= 2.0).mean() * 100) if len(mfe) else float("nan")

        def _s(v, fmt):
            return fmt.format(v) if not _nan(v) else "     n/a"

        print(f"  {year:<6} {n:>5}  "
              f"{_s(avg_mfe, '{:>8.3f}')}  "
              f"{_s(avg_mae, '{:>8.3f}')}  "
              f"{_s(ratio,   '{:>6.2f}')}  "
              f"{_s(p1r,   '{:>5.1f}%')}  "
              f"{_s(p15r,  '{:>6.1f}%')}  "
              f"{_s(p2r,   '{:>5.1f}%')}")

        rows.append({
            "year": int(year), "trades": n,
            "avg_mfe_R":     round(avg_mfe, 3) if not _nan(avg_mfe) else float("nan"),
            "avg_mae_R":     round(avg_mae, 3) if not _nan(avg_mae) else float("nan"),
            "mfe_mae_ratio": round(ratio, 2)   if not _nan(ratio)   else float("nan"),
            "pct_1R":        round(p1r, 1)     if not _nan(p1r)     else float("nan"),
            "pct_1_5R":      round(p15r, 1)    if not _nan(p15r)    else float("nan"),
            "pct_2R":        round(p2r, 1)     if not _nan(p2r)     else float("nan"),
        })

    if len(rows) >= 2:
        df_yr     = pd.DataFrame(rows)
        mfe_vals  = df_yr["avg_mfe_R"].dropna()
        mae_vals  = df_yr["avg_mae_R"].dropna()
        mfe_mean  = float(mfe_vals.mean()) if len(mfe_vals) else float("nan")
        mfe_std   = float(mfe_vals.std())  if len(mfe_vals) > 1 else 0.0
        mae_mean  = float(mae_vals.mean()) if len(mae_vals) else float("nan")
        cv_mfe    = mfe_std / mfe_mean if (not _nan(mfe_mean) and mfe_mean > 0) else float("nan")

        print(DIV)
        print(f"\n  Overall avg MFE : {mfe_mean:.3f}R  "
              f"(σ={mfe_std:.3f}R"
              + (f", CV={cv_mfe:.2f}" if not _nan(cv_mfe) else "")
              + ")")
        print(f"  Overall avg MAE : {mae_mean:.3f}R")

        if not _nan(cv_mfe):
            if cv_mfe < 0.20:
                verdict = "STRUCTURAL — MFE highly stable across years (CV < 0.20)"
            elif cv_mfe < 0.40:
                verdict = "MODERATE — some year-to-year MFE variation (CV < 0.40)"
            else:
                verdict = "REGIME-DEPENDENT — MFE varies significantly by year"
        else:
            verdict = "INSUFFICIENT data for stability assessment"

        print(f"  ASSESSMENT  : {verdict}")
        print(f"{SEP}\n")
        return df_yr

    print(f"{SEP}\n")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# TASK 5 — PARAMETER ROBUSTNESS TEST
# =============================================================================

def run_param_robustness(
    frames: dict,
    pair: str,
    run_backtest_fn,
    spread: float = 0.0,
    slippage_std: float = 0.0,
    risk_pct: float = 0.015,
    tp1_variants: list = None,
    tp2_variants: list = None,
    starting_equity: float = STARTING_EQUITY,
    _precomputed_swings: tuple = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Test small variations of TP1 / TP2 R-levels.

    TP1: 0.70, 0.75 (base), 0.80
    TP2: 1.40, 1.50 (base), 1.60

    Stable performance across variants → robust edge.
    Collapse at slight deviations → overfit.
    """
    if tp1_variants is None:
        tp1_variants = [0.70, 0.75, 0.80]
    if tp2_variants is None:
        tp2_variants = [1.40, 1.50, 1.60]

    W   = 80
    SEP = "=" * W
    DIV = "  " + "─" * (W - 2)

    print(f"\n{SEP}")
    print(f"  PARAMETER ROBUSTNESS TEST  —  {pair}")
    print(f"  TP1 variants : {tp1_variants}   TP2 variants : {tp2_variants}")
    print(SEP)

    hdr = (f"  {'TP1':>5} {'TP2':>5}  "
           f"{'Return%':>9} {'MaxDD%':>8} {'Sharpe':>8} "
           f"{'PF':>7} {'Trades':>7} {'WR%':>6}")
    print(hdr)
    print(DIV)

    _swing_kwarg = {"_precomputed_swings": _precomputed_swings} if _precomputed_swings is not None else {}

    # Separate baseline (0.75/1.50) from non-baseline combinations.
    # Baseline is run in the main process (already available from caller or run now).
    # Non-baseline cells run in parallel worker processes.
    all_combos = [(tp1, tp2) for tp1 in tp1_variants for tp2 in tp2_variants]
    base_combos = [(tp1, tp2) for tp1, tp2 in all_combos
                   if abs(tp1 - 0.75) < 0.001 and abs(tp2 - 1.50) < 0.001]
    non_base    = [(tp1, tp2) for tp1, tp2 in all_combos
                   if not (abs(tp1 - 0.75) < 0.001 and abs(tp2 - 1.50) < 0.001)]

    def _run_one(tp1_r, tp2_r):
        tdf, _, _, feq = run_backtest_fn(
            frames,
            spread=spread, slippage_std=slippage_std,
            risk_pct=risk_pct, tp1_r=tp1_r, tp2_r=tp2_r,
            verbose=False, **_swing_kwarg, **kwargs,
        )
        return tp1_r, tp2_r, _metrics(tdf, feq, starting_equity)

    # Run non-baseline cells in parallel (cap at CPU count, max 4 to avoid RAM pressure)
    n_workers = min(len(non_base), max(1, multiprocessing.cpu_count() - 1), 4)
    results_map: dict = {}  # (tp1, tp2) -> metrics

    if n_workers > 1 and len(non_base) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one, tp1, tp2): (tp1, tp2)
                       for tp1, tp2 in non_base}
            for fut in as_completed(futures):
                tp1_r, tp2_r, m = fut.result()
                results_map[(tp1_r, tp2_r)] = m
    else:
        for tp1, tp2 in non_base:
            tp1_r, tp2_r, m = _run_one(tp1, tp2)
            results_map[(tp1_r, tp2_r)] = m

    # Run baseline cell (always sequential in main process)
    for tp1, tp2 in base_combos:
        tp1_r, tp2_r, m = _run_one(tp1, tp2)
        results_map[(tp1_r, tp2_r)] = m

    # Print results in sorted order; accumulate rows list
    rows = []
    for tp1_r in tp1_variants:
        for tp2_r in tp2_variants:
            m = results_map[(tp1_r, tp2_r)]
            is_base = (abs(tp1_r - 0.75) < 0.001 and abs(tp2_r - 1.50) < 0.001)
            tag = " ◀ base" if is_base else ""

            print(f"  {tp1_r:>5.2f} {tp2_r:>5.2f}  "
                  f"{m['total_return']:>+8.2f}%  "
                  f"{m['max_drawdown']:>+7.2f}%  "
                  f"{m['sharpe']:>8.3f}  "
                  f"{m['profit_factor']:>7.3f}  "
                  f"{m['trades']:>7}  "
                  f"{m['win_rate']:>5.1f}%"
                  f"{tag}")

            rows.append({
                "tp1_r": tp1_r, "tp2_r": tp2_r,
                "is_baseline": is_base,
                **m,
            })

    if rows:
        df = pd.DataFrame(rows)
        ret_std  = float(df["total_return"].std())
        ret_mean = float(df["total_return"].mean())
        shr_std  = float(df["sharpe"].std())
        cv = ret_std / abs(ret_mean) if ret_mean != 0 else float("inf")

        # Bounds
        r_min = float(df["total_return"].min())
        r_max = float(df["total_return"].max())
        profitable_frac = (df["total_return"] > 0).mean()

        print(DIV)
        print(f"\n  Return range   : [{r_min:+.2f}%, {r_max:+.2f}%]")
        print(f"  Return σ       : {ret_std:.2f}%    CV = {cv:.2f}")
        print(f"  Sharpe σ       : {shr_std:.3f}")
        print(f"  Profitable configs : {int(profitable_frac * len(df))} / {len(df)}")

        if cv < 0.20 and shr_std < 0.30 and profitable_frac >= 0.80:
            verdict = "ROBUST — performance stable across TP variants"
        elif cv < 0.40 and profitable_frac >= 0.60:
            verdict = "MODERATE — acceptable sensitivity to TP placement"
        else:
            verdict = "SENSITIVE — results depend heavily on TP levels (risk of overfit)"

        print(f"  ASSESSMENT     : {verdict}")
        print(f"{SEP}\n")
        return df

    print(f"{SEP}\n")
    return pd.DataFrame()


# =============================================================================
# TASK 7 — ROLLING PERFORMANCE MONITORING
# =============================================================================

def compute_rolling_performance(
    trade_df: pd.DataFrame,
    pair: str,
    strategy_name: str = "trailing",
    window: int = 20,
    sharpe_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute rolling performance metrics across the trade log.

    Outputs:
    - Rolling Sharpe (last N trades)
    - Rolling PnL sum
    - Rolling win rate
    - Losing streak detection
    - Optional risk reduction advisory (print only — never forced)

    Parameters
    ----------
    window           : number of trades for rolling window (default 20)
    sharpe_threshold : warn if rolling Sharpe drops below this value
    """
    W   = 66
    SEP = "=" * W
    DIV = "  " + "─" * (W - 2)

    if trade_df.empty:
        print(f"  [performance monitoring] No trades.")
        return pd.DataFrame()

    pnl_col = "total_pnl" if "total_pnl" in trade_df.columns else "pnl"
    df = trade_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)

    pnl_pct_arr = (df["pnl_pct"].values / 100
                   if "pnl_pct" in df.columns
                   else df[pnl_col].values / df["equity_after"].values)

    rolling_pnl    = []
    rolling_sharpe = []
    rolling_wr     = []

    for i in range(len(df)):
        lo  = max(0, i - window + 1)
        seg = pnl_pct_arr[lo: i + 1]
        pnl_seg = df[pnl_col].iloc[lo: i + 1]

        rolling_pnl.append(float(pnl_seg.sum()))
        rolling_wr.append(float((pnl_seg > 0).mean() * 100))

        if len(seg) >= 3 and seg.std() > 0:
            rolling_sharpe.append(float(seg.mean() / seg.std() * np.sqrt(252)))
        else:
            rolling_sharpe.append(float("nan"))

    df["rolling_pnl"]    = rolling_pnl
    df["rolling_sharpe"] = rolling_sharpe
    df["rolling_wr"]     = rolling_wr

    last_sharpe = rolling_sharpe[-1] if rolling_sharpe else float("nan")
    last_pnl    = rolling_pnl[-1]    if rolling_pnl    else 0.0
    last_wr     = rolling_wr[-1]     if rolling_wr     else 0.0

    # ── Print header ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  PERFORMANCE MONITORING  —  {pair}  [{strategy_name}]")
    print(f"  Rolling window : last {window} trades  |  "
          f"Sharpe threshold : {sharpe_threshold}")
    print(SEP)

    L = 30
    if not _nan(last_sharpe):
        print(f"  {'Rolling Sharpe (last {w})':<{L}}".replace("{w}", str(window))
              + f" {last_sharpe:>8.3f}")
    else:
        print(f"  {'Rolling Sharpe (last {w})':<{L}}".replace("{w}", str(window))
              + "      n/a")

    print(f"  {'Rolling PnL (last {w})':<{L}}".replace("{w}", str(window))
          + f" ${last_pnl:>10,.2f}")
    print(f"  {'Rolling Win Rate':<{L}} {last_wr:>7.1f}%")

    # ── Sharpe advisory ──────────────────────────────────────────────────────
    if not _nan(last_sharpe) and last_sharpe < sharpe_threshold:
        print(f"\n  ⚠  Rolling Sharpe ({last_sharpe:.3f}) is BELOW threshold "
              f"({sharpe_threshold:.2f})")
        print(f"     → MONITOR: Consider reducing risk_pct until Sharpe recovers")
        print(f"     → This is advisory only — no automatic change made")
    elif not _nan(last_sharpe):
        print(f"\n  ✓  Rolling Sharpe above threshold — performance stable")

    # ── Losing streak analysis ────────────────────────────────────────────────
    consec = 0
    max_consec = 0
    for v in df[pnl_col].values:
        consec = consec + 1 if v < 0 else 0
        max_consec = max(max_consec, consec)

    recent_streak = 0
    for v in reversed(df[pnl_col].values[-window:]):
        if v < 0:
            recent_streak += 1
        else:
            break

    print(f"\n  Max consecutive losses (full history) : {max_consec}")
    print(f"  Current losing streak (recent {window})  : {recent_streak}")

    if recent_streak >= 5:
        print(f"  ⚠  Active losing streak of {recent_streak} trades — "
              f"review current regime conditions")

    print(DIV)

    # ── Last 10 trades at a glance ────────────────────────────────────────────
    show_n = min(10, len(df))
    print(f"\n  Last {show_n} trades:")
    print(f"  {'#':>4}  {'Entry Time':<20} {'Dir':<5} "
          f"{'PnL ($)':>10} {'Roll.PnL':>10} {'Roll.Sharpe':>12}")
    print(f"  {'-'*65}")
    for _, row in df.tail(show_n).iterrows():
        rs_s  = (f"{row['rolling_sharpe']:>12.3f}"
                 if not _nan(row["rolling_sharpe"]) else "         n/a")
        print(f"  {int(row.name)+1:>4}  "
              f"{str(row['entry_time'])[:19]:<20}  "
              f"{row['direction']:<5} "
              f"${row[pnl_col]:>9,.2f} "
              f"${row['rolling_pnl']:>9,.2f} "
              f"{rs_s}")

    print(f"{SEP}\n")

    out_cols = [c for c in
                ["entry_time", "direction", pnl_col,
                 "rolling_pnl", "rolling_sharpe", "rolling_wr"]
                if c in df.columns]
    return df[out_cols]


# =============================================================================
# TASK 8 — ENHANCED FINAL DECISION PANEL
# =============================================================================

def print_final_decision(
    trade_df: pd.DataFrame,
    final_equity: float,
    pair: str,
    strategy_name: str,
    risk_pct: float,
    best_risk_pct: float = None,
    walk_forward_result: dict = None,
    rolling_windows_df: pd.DataFrame = None,
    mfe_mae_df: pd.DataFrame = None,
    starting_equity: float = STARTING_EQUITY,
) -> None:
    """
    Print the deployment decision panel.

    STATUS: ACCEPTABLE / REVIEW / REJECT
    EDGE TYPE: STRUCTURAL / REGIME-DEPENDENT / UNKNOWN
    DD HEADROOM, BEST RISK LEVEL, NEXT STEP
    """
    W   = 68
    SEP = "=" * W
    DIV = "  " + "─" * (W - 2)

    m        = _metrics(trade_df, final_equity, starting_equity)
    max_dd   = m["max_drawdown"]
    sharpe   = m["sharpe"]
    tot_ret  = m["total_return"]
    pf       = m["profit_factor"]
    n_trades = m["trades"]

    # ── STATUS ───────────────────────────────────────────────────────────────
    if n_trades == 0:
        status = "REJECT"
    elif max_dd >= -5.0 and sharpe > 1.5 and tot_ret > 10:
        status = "ACCEPTABLE"
    elif max_dd >= -10.0 and sharpe > 0.5 and tot_ret > 0:
        status = "ACCEPTABLE"
    elif max_dd >= -15.0 and tot_ret > 0:
        status = "REVIEW"
    elif max_dd >= -20.0:
        status = "REVIEW"
    else:
        status = "REJECT"

    # ── EDGE TYPE ────────────────────────────────────────────────────────────
    edge_type = "UNKNOWN"
    edge_notes = []

    if walk_forward_result:
        te = walk_forward_result.get("TEST", {})
        x_ret = te.get("total_return", float("nan"))
        x_shr = te.get("sharpe", float("nan"))
        if not _nan(x_ret) and not _nan(x_shr):
            if x_ret > 0 and x_shr > 1.0:
                edge_type = "STRUCTURAL"
                edge_notes.append("walk-forward OOS positive + Sharpe > 1")
            elif x_ret > 0 and x_shr > 0.5:
                edge_type = "STRUCTURAL"
                edge_notes.append("walk-forward OOS positive")
            elif x_ret > 0:
                edge_type = "POSSIBLY REGIME-DEPENDENT"
                edge_notes.append("OOS positive but weak Sharpe")
            else:
                edge_type = "REGIME-DEPENDENT"
                edge_notes.append("OOS negative — edge did not transfer")

    if edge_type == "UNKNOWN" and rolling_windows_df is not None and not rolling_windows_df.empty:
        pct_pos = float((rolling_windows_df["total_return"] > 0).mean())
        if pct_pos >= 0.75:
            edge_type = "STRUCTURAL"
            edge_notes.append(f"{pct_pos*100:.0f}% of rolling windows profitable")
        elif pct_pos >= 0.50:
            edge_type = "POSSIBLY REGIME-DEPENDENT"
            edge_notes.append(f"only {pct_pos*100:.0f}% of rolling windows profitable")
        else:
            edge_type = "REGIME-DEPENDENT"
            edge_notes.append(f"only {pct_pos*100:.0f}% of rolling windows profitable")

    if edge_type == "UNKNOWN" and mfe_mae_df is not None and not mfe_mae_df.empty:
        if "mfe_mae_ratio" in mfe_mae_df.columns:
            cv = float(mfe_mae_df["avg_mfe_R"].std() / mfe_mae_df["avg_mfe_R"].mean()
                       if mfe_mae_df["avg_mfe_R"].mean() > 0 else float("inf"))
            if cv < 0.25:
                edge_type = "STRUCTURAL"
                edge_notes.append(f"MFE stable across years (CV={cv:.2f})")
            else:
                edge_type = "REGIME-DEPENDENT"
                edge_notes.append(f"MFE varies by year (CV={cv:.2f})")

    # ── DD HEADROOM ──────────────────────────────────────────────────────────
    dd_headroom = 10.0 + max_dd   # max_dd is negative, e.g. -6.8 → 3.2 remaining

    # ── BEST RISK ────────────────────────────────────────────────────────────
    best_r_str = (f"{best_risk_pct:.2%}" if best_risk_pct
                  else f"{risk_pct:.2%}  (current — run risk sweep to optimise)")

    # ── NEXT STEP ────────────────────────────────────────────────────────────
    if status == "REJECT":
        next_step = "Do NOT deploy — re-examine entry logic and edge diagnostics"
    elif status == "REVIEW":
        next_step = "Reduce risk_pct or tighten exit model, then re-validate"
    elif dd_headroom > 4.0 and edge_type == "STRUCTURAL":
        next_step = "Run risk sweep — DD headroom available to increase risk_pct"
    elif dd_headroom > 2.0:
        next_step = "Maintain current parameters — DD headroom is tight"
    else:
        next_step = "Near DD limit — do not increase risk without OOS confirmation"

    # ── Print ────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  FINAL DECISION  —  {pair}  [{strategy_name}]")
    print(SEP)

    L = 28

    # Block 1: Status + edge
    status_flag = {"ACCEPTABLE": "✓", "REVIEW": "⚠", "REJECT": "✗"}.get(status, "?")
    print(f"  {'STATUS':<{L}} {status_flag}  {status}")
    print(f"  {'EDGE TYPE':<{L}} {edge_type}")
    if edge_notes:
        for note in edge_notes:
            print(f"  {'  └─':<{L}} {note}")

    print(DIV)

    # Block 2: Performance snapshot
    print(f"  {'Total Return':<{L}} {tot_ret:>+.2f}%")
    print(f"  {'Max Drawdown':<{L}} {max_dd:.2f}%")
    print(f"  {'Sharpe (ann.)':<{L}} {sharpe:.3f}")
    print(f"  {'Profit Factor':<{L}} {pf:.3f}")
    print(f"  {'Trades':<{L}} {n_trades}")

    print(DIV)

    # Block 3: Risk utilisation
    dd_bar = "▓" * min(20, max(0, int(-max_dd * 2))) + "░" * max(0, 20 - min(20, int(-max_dd * 2)))
    head_bar = "▓" * min(20, max(0, int(dd_headroom * 2))) + "░" * max(0, 20 - min(20, int(dd_headroom * 2)))
    print(f"  {'DD Used (vs 10% limit)':<{L}} {-max_dd:.2f}% used  "
          f"| {dd_headroom:.2f}% remaining")
    print(f"  {'Current Risk':<{L}} {risk_pct:.2%}")
    print(f"  {'Best Risk Level (sweep)':<{L}} {best_r_str}")

    print(DIV)

    # Block 4: Decision
    print(f"  {'NEXT STEP':<{L}} {next_step}")
    print(f"{SEP}\n")


# =============================================================================
# FULL ROBUSTNESS SUITE RUNNER
# =============================================================================

def run_full_robustness_suite(
    frames: dict,
    pair: str,
    run_backtest_fn,
    trade_df: pd.DataFrame,
    final_equity: float,
    spread: float = 0.0,
    slippage_std: float = 0.0,
    risk_pct: float = 0.015,
    best_risk_pct: float = None,
    starting_equity: float = STARTING_EQUITY,
    skip_param_robustness: bool = False,
    _precomputed_swings: tuple = None,
    **kwargs,
) -> dict:
    """
    Run the complete robustness suite in one call.

    Executes (in order):
      1. MFE/MAE by year (uses existing trade_df — fast)
      2. Walk-forward validation
      3. Rolling window analysis
      4. Parameter robustness test  (optional — slow)
      5. Performance monitoring
      6. Final decision panel

    Returns
    -------
    dict with keys: walk_forward, rolling_windows, mfe_mae_df, param_df, perf_df
    """
    print("\n" + "█" * 72)
    print(f"  FULL ROBUSTNESS SUITE  —  {pair}")
    print("█" * 72)

    results = {}

    # 1. MFE/MAE by year
    if "mfe_R" in trade_df.columns:
        mfe_df = run_mfe_mae_by_year(trade_df, pair)
        results["mfe_mae_df"] = mfe_df
    else:
        results["mfe_mae_df"] = pd.DataFrame()

    # 2. Walk-forward
    wf = run_walk_forward(
        frames, pair, run_backtest_fn,
        spread=spread, slippage_std=slippage_std,
        risk_pct=risk_pct, starting_equity=starting_equity,
        **kwargs,
    )
    results["walk_forward"] = wf

    # 3. Rolling windows
    rw = run_rolling_windows(
        frames, pair, run_backtest_fn,
        spread=spread, slippage_std=slippage_std,
        risk_pct=risk_pct, starting_equity=starting_equity,
        **kwargs,
    )
    results["rolling_windows"] = rw

    # 4. Parameter robustness (optional — runs 9 backtests)
    if not skip_param_robustness:
        pd_res = run_param_robustness(
            frames, pair, run_backtest_fn,
            spread=spread, slippage_std=slippage_std,
            risk_pct=risk_pct, starting_equity=starting_equity,
            _precomputed_swings=_precomputed_swings,
            **kwargs,
        )
        results["param_df"] = pd_res
    else:
        print("  [param robustness] Skipped (skip_param_robustness=True)")
        results["param_df"] = pd.DataFrame()

    # 5. Performance monitoring
    perf_df = compute_rolling_performance(
        trade_df, pair, window=20, sharpe_threshold=0.5,
    )
    results["perf_df"] = perf_df

    # 6. Final decision
    print_final_decision(
        trade_df, final_equity, pair, "trailing",
        risk_pct=risk_pct,
        best_risk_pct=best_risk_pct,
        walk_forward_result=wf,
        rolling_windows_df=rw,
        mfe_mae_df=results["mfe_mae_df"],
        starting_equity=starting_equity,
    )

    return results


# =============================================================================
# PHASE 1 — FINAL VALIDATION SUITE  (Tasks 1–5)
#
# PERFORMANCE NOTE
# ----------------
# run_phase1_validation() runs the backtest engine ONCE (full date range,
# baseline TP1=0.75 / TP2=1.50 / risk=1.50%).  All subsequent analysis
# derives results from that single run:
#
#   Walk-forward  : slice baseline trade_df by date → renormalize equity
#   Risk confirm  : scale pnl_pct by (target_risk / base_risk) → exact
#   Param robust  : 8 extra runs (TP variants change trade exits, unavoidable)
#                   baseline (0.75 / 1.50) cell is reused — no double-run
#
# Total engine calls: 1 baseline + 8 param variants = 9  (was 14)
# =============================================================================

def _extended_metrics(
    trade_df: pd.DataFrame,
    final_equity: float,
    starting_equity: float = STARTING_EQUITY,
) -> dict:
    """
    Superset of _metrics(): adds wins, losses, total_pnl, MFE/MAE R,
    TP1/TP2 hit rates.
    """
    base = _metrics(trade_df, final_equity, starting_equity)
    if trade_df.empty:
        base.update({
            "wins": 0, "losses": 0, "total_pnl": 0.0,
            "avg_mfe_R": float("nan"), "avg_mae_R": float("nan"),
            "tp1_hit_rate": float("nan"), "tp2_hit_rate": float("nan"),
        })
        return base

    pnl_col = "total_pnl" if "total_pnl" in trade_df.columns else "pnl"
    n       = len(trade_df)
    wins    = int((trade_df[pnl_col] > 0).sum())
    total_pnl = float(trade_df[pnl_col].sum())

    avg_mfe_R = (
        float(trade_df["mfe_R"].dropna().mean())
        if "mfe_R" in trade_df.columns and trade_df["mfe_R"].notna().any()
        else float("nan")
    )
    avg_mae_R = (
        float(trade_df["mae_R"].dropna().mean())
        if "mae_R" in trade_df.columns and trade_df["mae_R"].notna().any()
        else float("nan")
    )
    tp1_rate = (
        float(trade_df["tp1_hit"].mean() * 100)
        if "tp1_hit" in trade_df.columns else float("nan")
    )
    tp2_rate = (
        float(trade_df["tp2_hit"].mean() * 100)
        if "tp2_hit" in trade_df.columns else float("nan")
    )

    base.update({
        "wins":         wins,
        "losses":       n - wins,
        "total_pnl":    round(total_pnl, 2),
        "avg_mfe_R":    round(avg_mfe_R, 3) if not _nan(avg_mfe_R) else float("nan"),
        "avg_mae_R":    round(avg_mae_R, 3) if not _nan(avg_mae_R) else float("nan"),
        "tp1_hit_rate": round(tp1_rate, 1)  if not _nan(tp1_rate)  else float("nan"),
        "tp2_hit_rate": round(tp2_rate, 1)  if not _nan(tp2_rate)  else float("nan"),
    })
    return base


def _normalize_equity_from_pnl_pct(
    trade_df: pd.DataFrame,
    starting_equity: float = STARTING_EQUITY,
) -> tuple:
    """
    Recompute equity_after and total_pnl for a trade_df slice,
    starting from a fresh equity level.

    WHY THIS IS EXACT
    -----------------
    pnl_pct = risk_pct × R_multiple   (R_multiple = price_move / stop_dist)

    R_multiple is set by entry/exit prices and is independent of equity level.
    Therefore pnl_pct per trade is the same whether the strategy ran from
    $100k or from any other equity.  We just need to replay the pnl_pct
    sequence with a new starting equity to get the correct dollar amounts
    and equity path.

    Returns
    -------
    (trade_df_renormalized, final_equity)
    """
    if trade_df.empty:
        return trade_df.copy(), starting_equity

    df      = trade_df.copy().reset_index(drop=True)
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl"
    equity  = starting_equity
    new_pnl, new_eq = [], []

    for _, row in df.iterrows():
        pct   = float(row.get("pnl_pct", 0.0)) / 100.0
        pnl_d = equity * pct
        equity += pnl_d
        new_pnl.append(round(pnl_d, 2))
        new_eq.append(round(equity, 2))

    df[pnl_col]      = new_pnl
    df["equity_after"] = new_eq
    return df, round(equity, 2)


def _slice_trade_df(
    trade_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Filter trade_df to trades whose entry_time falls in [start_date, end_date]."""
    if trade_df.empty:
        return trade_df.copy()
    df  = trade_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    s   = pd.Timestamp(start_date)
    e   = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
    return df[(df["entry_time"] >= s) & (df["entry_time"] <= e)].reset_index(drop=True)


# -----------------------------------------------------------------------------
# TASK 1 — WALK-FORWARD VALIDATION (Phase 1)
# -----------------------------------------------------------------------------

def run_phase1_walk_forward(
    frames: dict,
    pair: str,
    run_backtest_fn,
    spread: float       = 0.0,
    slippage_std: float = 0.0,
    train_start: str    = "2011-01-01",
    train_end: str      = "2020-12-31",
    test_start: str     = "2021-01-01",
    test_end: str       = "2026-12-31",
    risk_pct: float     = 0.015,
    starting_equity: float = STARTING_EQUITY,
    precomputed_trade_df: pd.DataFrame = None,
    **kwargs,
) -> dict:
    """
    Phase 1 / Task 1: walk-forward with extended metrics.

    FAST PATH  (precomputed_trade_df is provided)
    ----------------------------------------------
    Slices the already-computed full-range trade_df into TRAIN/TEST date
    windows and renormalizes equity from scratch for each slice.
    No extra backtest runs are needed.

    SLOW PATH  (precomputed_trade_df is None)
    -----------------------------------------
    Runs the backtest engine separately for each sub-period.
    Use only when you don't have a full-range run available.

    Returns dict with keys "TRAIN" and "TEST" containing extended metrics.
    Prints a side-by-side table and WALK-FORWARD RESULT: PASS / REVIEW / FAIL.
    """
    try:
        from analytics import compute_mfe_mae as _cmm
    except ImportError:
        _cmm = None

    W   = 76
    SEP = "=" * W
    DIV = "  " + "─" * (W - 2)

    results = {}

    for label, t0, t1 in [
        ("TRAIN", train_start, train_end),
        ("TEST",  test_start,  test_end),
    ]:
        if precomputed_trade_df is not None:
            # ── Fast path: slice + renormalize ────────────────────────────────
            slice_df = _slice_trade_df(precomputed_trade_df, t0, t1)
            trade_df, final_eq = _normalize_equity_from_pnl_pct(
                slice_df, starting_equity)
            print(f"  [{label}]  {t0} – {t1}  "
                  f"({len(trade_df)} trades from baseline slice)", flush=True)
        else:
            # ── Slow path: run backtest for sub-period ────────────────────────
            sub = _filter_frames_by_date(frames, t0, t1)
            if sub["5m"].empty:
                print(f"  [{label}] No 5m data for {t0} – {t1}. Skipping.")
                results[label] = {}
                continue
            print(f"  [{label}]  {t0} – {t1}  ({len(sub['5m']):,} 5m bars) ...",
                  flush=True)
            trade_df, _, _eq_df, final_eq = run_backtest_fn(
                sub, spread=spread, slippage_std=slippage_std,
                risk_pct=risk_pct, verbose=False, **kwargs,
            )
            if _cmm is not None and not trade_df.empty:
                trade_df = _cmm(trade_df, sub["5m"])

        m = _extended_metrics(trade_df, final_eq, starting_equity)
        m["start"] = t0
        m["end"]   = t1
        results[label] = m

    # ── Side-by-side table ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  WALK-FORWARD VALIDATION  —  {pair}")
    print(f"  TRAIN : {train_start} – {train_end}  (in-sample)")
    print(f"  TEST  : {test_start}  – {test_end}  (out-of-sample)")
    print(SEP)

    tr = results.get("TRAIN", {})
    te = results.get("TEST",  {})

    def _fmt(d, key, fmt):
        v = d.get(key, float("nan"))
        if _nan(v):
            return "           n/a"
        return fmt.format(v)

    row_fmt = "  {:<26} {:>15} {:>15}"
    print(row_fmt.format("Metric", "TRAIN", "TEST"))
    print(DIV)

    fields = [
        ("Trades",           "trades",        "{:>15,.0f}"),
        ("Wins",             "wins",          "{:>15,.0f}"),
        ("Losses",           "losses",        "{:>15,.0f}"),
        ("Win Rate (%)",     "win_rate",      "{:>15.1f}"),
        ("Total PnL ($)",    "total_pnl",     "{:>15,.0f}"),
        ("Total Return (%)", "total_return",  "{:>+15.2f}"),
        ("Final Equity ($)", "final_equity",  "{:>15,.0f}"),
        ("Max Drawdown (%)", "max_drawdown",  "{:>+15.2f}"),
        ("Sharpe (ann.)",    "sharpe",        "{:>15.3f}"),
        ("Profit Factor",    "profit_factor", "{:>15.3f}"),
        ("Avg MFE (R)",      "avg_mfe_R",     "{:>15.3f}"),
        ("Avg MAE (R)",      "avg_mae_R",     "{:>15.3f}"),
        ("TP1 Hit Rate (%)", "tp1_hit_rate",  "{:>15.1f}"),
        ("TP2 Hit Rate (%)", "tp2_hit_rate",  "{:>15.1f}"),
    ]

    for label_txt, key, fmt in fields:
        print(row_fmt.format(label_txt, _fmt(tr, key, fmt), _fmt(te, key, fmt)))

    print(DIV)

    # ── Verdict ──────────────────────────────────────────────────────────────
    verdict = "FAIL"
    if tr and te:
        x_ret = te.get("total_return", float("nan"))
        x_shr = te.get("sharpe",       float("nan"))
        x_dd  = te.get("max_drawdown", float("nan"))
        t_ret = tr.get("total_return", float("nan"))

        ret_decay = (
            (x_ret - t_ret) / abs(t_ret) * 100
            if not (_nan(x_ret) or _nan(t_ret) or t_ret == 0)
            else float("nan")
        )

        if (not _nan(x_ret) and x_ret > 0 and
                not _nan(x_dd) and x_dd >= -12.0 and
                not _nan(ret_decay) and abs(ret_decay) < 50):
            verdict = "PASS"
        elif not _nan(x_ret) and x_ret > 0:
            verdict = "REVIEW"
        else:
            verdict = "FAIL"

        print(f"\n  WALK-FORWARD RESULT: {verdict}")
        if verdict == "PASS":
            print("  → Edge persists out-of-sample with acceptable decay.")
        elif verdict == "REVIEW":
            print("  → Profitable OOS but noticeably weaker — review before deploying.")
        else:
            print("  → Edge does NOT persist out-of-sample. Not deployment-ready.")
    else:
        print(f"\n  WALK-FORWARD RESULT: {verdict}")
        print("  → Insufficient data to assess.")

    print(f"{SEP}\n")
    return results


# -----------------------------------------------------------------------------
# TASK 3 — RISK CONFIRMATION (Phase 1)
# -----------------------------------------------------------------------------

def run_risk_confirmation(
    frames: dict,
    pair: str,
    run_backtest_fn,
    spread: float       = 0.0,
    slippage_std: float = 0.0,
    risk_levels: list   = None,
    dd_limit: float     = 10.0,
    starting_equity: float = STARTING_EQUITY,
    baseline_trade_df: pd.DataFrame = None,
    base_risk: float    = 0.015,
    **kwargs,
) -> dict:
    """
    Phase 1 / Task 3: compare 1.25%, 1.50%, 1.75% risk under DD limit.

    FAST PATH  (baseline_trade_df is provided)
    ------------------------------------------
    Derives all risk levels by scaling the baseline pnl_pct sequence.

    WHY THIS IS EXACT
    -----------------
    pnl_pct = risk_pct × R_multiple  (R_multiple = price_move / stop_dist)
    R_multiple is determined by entry/exit prices which are IDENTICAL across
    all risk levels (risk_pct only affects position size, not signal/exit logic).
    Therefore pnl_pct(risk_B) = pnl_pct(risk_A) × (risk_B / risk_A), exactly.

    SLOW PATH  (baseline_trade_df is None)
    ---------------------------------------
    Runs the backtest engine separately for each risk level.

    Returns dict with keys: rows_df, best_risk, keep_current, recommendation.
    """
    if risk_levels is None:
        risk_levels = [0.0125, 0.0150, 0.0175]

    W   = 72
    SEP = "=" * W
    DIV = "  " + "─" * (W - 2)

    print(f"\n{SEP}")
    print(f"  RISK CONFIRMATION  —  {pair}")
    print(f"  DD limit : ≤ {dd_limit:.0f}%   "
          f"Levels : " + ", ".join(f"{r:.2%}" for r in risk_levels))
    if baseline_trade_df is not None:
        print(f"  Source   : derived from baseline run (no extra backtests)")
    print(SEP)

    hdr = (f"  {'Risk %':<10} {'Return%':>10} {'MaxDD%':>9} "
           f"{'Sharpe':>8} {'PF':>8}  DD ok?")
    print(hdr)
    print(DIV)

    rows = []
    for rp in risk_levels:
        if baseline_trade_df is not None:
            # ── Fast path: scale pnl_pct, renormalize equity ──────────────────
            scale    = rp / base_risk
            scaled   = baseline_trade_df.copy()
            scaled["pnl_pct"] = scaled["pnl_pct"] * scale
            trade_df, final_eq = _normalize_equity_from_pnl_pct(
                scaled, starting_equity)
        else:
            # ── Slow path: full backtest run ──────────────────────────────────
            trade_df, _, _eq_df, final_eq = run_backtest_fn(
                frames, spread=spread, slippage_std=slippage_std,
                risk_pct=rp, verbose=False, **kwargs,
            )

        m       = _metrics(trade_df, final_eq, starting_equity)
        dd_ok   = m["max_drawdown"] >= -dd_limit
        is_base = abs(rp - base_risk) < 0.0001
        tag     = "  ◀ current" if is_base else ""

        print(f"  {rp:.2%}      "
              f"{m['total_return']:>+9.2f}%  "
              f"{m['max_drawdown']:>+8.2f}%  "
              f"{m['sharpe']:>8.3f}  "
              f"{m['profit_factor']:>7.3f}  "
              f"  {'✓' if dd_ok else '✗'}{tag}")

        rows.append({"risk_pct": rp, "is_current": is_base, **m, "dd_ok": dd_ok})

    print(DIV)

    df           = pd.DataFrame(rows) if rows else pd.DataFrame()
    best_risk    = None
    keep_current = False
    rec          = "No data"

    if not df.empty:
        eligible = df[df["dd_ok"]]
        if not eligible.empty:
            best_row     = eligible.loc[eligible["total_return"].idxmax()]
            best_risk    = float(best_row["risk_pct"])
            keep_current = abs(best_risk - base_risk) < 0.0001
            if keep_current:
                rec = f"KEEP 1.50% — best return under DD ≤ {dd_limit:.0f}%"
            elif best_risk < base_risk:
                rec = f"REDUCE to {best_risk:.2%} — 1.50% exceeds DD limit"
            else:
                rec = f"{best_risk:.2%} highest; 1.50% also within DD limit"
        else:
            rec = f"ALL levels breach DD ≤ {dd_limit:.0f}% — review strategy"

    print(f"\n  Best risk under DD ≤ {dd_limit:.0f}% : "
          + (f"{best_risk:.2%}" if best_risk else "none within limit"))
    print(f"  RISK DECISION             : {rec}")
    print(f"{SEP}\n")

    return {
        "rows_df":        df,
        "best_risk":      best_risk,
        "keep_current":   keep_current,
        "recommendation": rec,
    }


# -----------------------------------------------------------------------------
# TASK 5 — FINAL VALIDATION DECISION (Phase 1)
# -----------------------------------------------------------------------------

def print_phase1_decision(
    walk_result: dict,
    param_result_df: pd.DataFrame,
    risk_result: dict,
    pair: str,
    current_risk_pct: float = 0.015,
) -> None:
    """
    Phase 1 / Task 5: deployment decision block.

    Prints:
      VALIDATION STATUS   : READY / NEEDS REVIEW / NOT READY
      EDGE CLASSIFICATION : STRUCTURAL / REGIME-AMPLIFIED / TOO FRAGILE
      RISK DECISION       : KEEP 1.50% / REDUCE / RECHECK
      NEXT STEP           : paper trade / refine exits / improve filters / more OOS testing
    """
    W   = 72
    SEP = "=" * W
    DIV = "  " + "─" * (W - 2)

    # ── Walk-forward signals ──────────────────────────────────────────────────
    tr    = walk_result.get("TRAIN", {})
    te    = walk_result.get("TEST",  {})
    x_ret = te.get("total_return", float("nan"))
    x_shr = te.get("sharpe",       float("nan"))
    x_dd  = te.get("max_drawdown", float("nan"))
    t_ret = tr.get("total_return", float("nan"))
    ret_decay = (
        (x_ret - t_ret) / abs(t_ret) * 100
        if not (_nan(x_ret) or _nan(t_ret) or t_ret == 0)
        else float("nan")
    )

    # ── Param stability ───────────────────────────────────────────────────────
    param_stable = True
    if not param_result_df.empty and "total_return" in param_result_df.columns:
        mean_ret  = float(param_result_df["total_return"].mean())
        cv        = (float(param_result_df["total_return"].std()) / abs(mean_ret)
                     if mean_ret != 0 else float("inf"))
        prof_frac = float((param_result_df["total_return"] > 0).mean())
        param_stable = (cv < 0.40 and prof_frac >= 0.60)

    # ── Risk signals ──────────────────────────────────────────────────────────
    best_risk    = risk_result.get("best_risk", None)
    keep_current = risk_result.get("keep_current", False)

    # ── VALIDATION STATUS ─────────────────────────────────────────────────────
    if (not _nan(x_ret) and x_ret > 0 and
            not _nan(x_dd) and x_dd >= -12.0 and
            not _nan(x_shr) and x_shr > 0.5 and
            param_stable):
        status = "READY"
    elif not _nan(x_ret) and x_ret > 0:
        status = "NEEDS REVIEW"
    else:
        status = "NOT READY"

    # ── EDGE CLASSIFICATION ────────────────────────────────────────────────────
    if (not _nan(x_ret) and x_ret > 0 and
            not _nan(x_shr) and x_shr > 1.0 and
            not _nan(ret_decay) and abs(ret_decay) < 40):
        edge = "STRUCTURAL"
    elif not _nan(x_ret) and x_ret > 0 and not _nan(x_shr) and x_shr > 0.3:
        edge = "REGIME-AMPLIFIED"
    else:
        edge = "TOO FRAGILE"

    # ── RISK DECISION ─────────────────────────────────────────────────────────
    if keep_current:
        risk_dec = "KEEP 1.50%"
    elif best_risk is not None and best_risk < current_risk_pct:
        risk_dec = f"REDUCE to {best_risk:.2%}"
    elif best_risk is not None:
        risk_dec = "KEEP 1.50%"
    else:
        risk_dec = "RECHECK"

    # ── NEXT STEP ─────────────────────────────────────────────────────────────
    if status == "READY":
        next_step = "paper trade"
    elif status == "NEEDS REVIEW":
        if edge == "REGIME-AMPLIFIED":
            next_step = "improve filters"
        elif not param_stable:
            next_step = "refine exits"
        else:
            next_step = "more out-of-sample testing"
    else:
        next_step = "improve filters"

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  FINAL VALIDATION DECISION  —  {pair}")
    print(SEP)

    L   = 30
    sfx = {"READY": "✓", "NEEDS REVIEW": "⚠", "NOT READY": "✗"}.get(status, "?")
    print(f"\n  {'VALIDATION STATUS':<{L}} {sfx}  {status}")
    print(f"  {'EDGE CLASSIFICATION':<{L}} {edge}")
    print(f"  {'RISK DECISION':<{L}} {risk_dec}")
    print(f"  {'NEXT STEP':<{L}} {next_step}")

    print(DIV)
    print(f"\n  Supporting evidence:")
    if not _nan(t_ret): print(f"  • In-sample return   : {t_ret:+.2f}%")
    if not _nan(x_ret): print(f"  • OOS return         : {x_ret:+.2f}%")
    if not _nan(x_shr): print(f"  • OOS Sharpe         : {x_shr:.3f}")
    if not _nan(x_dd):  print(f"  • OOS max drawdown   : {x_dd:.2f}%")
    if not _nan(ret_decay):
        print(f"  • Return decay       : {ret_decay:+.1f}%  (train → test)")
    if not param_result_df.empty:
        print(f"  • Param stability    : {'stable' if param_stable else 'sensitive'}")
    if best_risk is not None:
        print(f"  • Best risk (≤ 10%DD): {best_risk:.2%}")

    print(f"{SEP}\n")


# -----------------------------------------------------------------------------
# PHASE 1 COORDINATOR  (optimized: 1 baseline run + 8 param variants = 9 total)
# -----------------------------------------------------------------------------

def run_phase1_validation(
    frames: dict,
    pair: str,
    run_backtest_fn,
    spread: float       = 0.0,
    slippage_std: float = 0.0,
    risk_pct: float     = 0.015,
    dd_limit: float     = 10.0,
    starting_equity: float = STARTING_EQUITY,
    **kwargs,
) -> dict:
    """
    Phase 1 — final validation coordinator.

    BACKTEST ENGINE CALLS: 1 baseline + 8 param TP variants = 9 total.
    Walk-forward and risk confirmation are derived from the baseline — no
    extra engine calls required.

    Executes in sequence:
      0. Baseline run (full range, TP1=0.75 / TP2=1.50 / risk=1.50%)
      1. Walk-forward  (TRAIN 2011–2020 / TEST 2021–2026) — sliced from baseline
      2. Param robustness (TP1 × TP2 grid; baseline cell reused)
      3. Risk confirmation (1.25%, 1.50%, 1.75%) — derived from baseline
      4. Final validation decision

    Returns
    -------
    dict: walk_forward, param_df, risk_result
    """
    try:
        from analytics import compute_mfe_mae as _cmm
    except ImportError:
        _cmm = None

    W = 72
    print("\n" + "█" * W)
    print(f"  PHASE 1 — FINAL VALIDATION  |  {pair}")
    print(f"  Risk: {risk_pct:.2%}   DD limit: ≤ {dd_limit:.0f}%   "
          f"Spread: {spread:.5f}   Slip: {slippage_std:.5f}")
    print(f"  Engine calls: 1 baseline + 8 TP variants = 9 total")
    print("█" * W)

    # ── BASELINE RUN (full range, one time only) ───────────────────────────────
    print(f"\n  Running baseline backtest (full range) ...")
    base_trade_df, _, _base_eq_df, base_final_eq = run_backtest_fn(
        frames,
        spread=spread, slippage_std=slippage_std,
        risk_pct=risk_pct, verbose=True,
        **kwargs,
    )
    if _cmm is not None and not base_trade_df.empty:
        base_trade_df = _cmm(base_trade_df, frames["5m"])

    results = {}

    # ── 1. WALK-FORWARD VALIDATION ────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  1. WALK-FORWARD VALIDATION")
    print(f"  (sliced from baseline — no extra engine calls)")
    print(f"{'─'*W}")
    walk = run_phase1_walk_forward(
        frames, pair, run_backtest_fn,
        spread=spread, slippage_std=slippage_std,
        risk_pct=risk_pct, starting_equity=starting_equity,
        precomputed_trade_df=base_trade_df,
        **kwargs,
    )
    results["walk_forward"] = walk

    # ── 2. PARAMETER ROBUSTNESS CHECK ─────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  2. PARAMETER ROBUSTNESS CHECK")
    print(f"  (8 new runs; baseline (0.75 / 1.50) cell reused)")
    print(f"{'─'*W}")

    tp1_variants = [0.70, 0.75, 0.80]
    tp2_variants = [1.40, 1.50, 1.60]

    W2  = 80
    SEP = "=" * W2
    DIV = "  " + "─" * (W2 - 2)

    print(f"\n{SEP}")
    print(f"  PARAMETER ROBUSTNESS TEST  —  {pair}")
    print(f"  TP1 variants : {tp1_variants}   TP2 variants : {tp2_variants}")
    print(SEP)

    hdr = (f"  {'TP1':>5} {'TP2':>5}  "
           f"{'Return%':>9} {'MaxDD%':>8} {'Sharpe':>8} "
           f"{'PF':>7} {'Trades':>7} {'WR%':>6}")
    print(hdr)
    print(DIV)

    param_rows = []
    for tp1_r in tp1_variants:
        for tp2_r in tp2_variants:
            is_base = (abs(tp1_r - 0.75) < 0.001 and abs(tp2_r - 1.50) < 0.001)
            tag     = " ◀ base" if is_base else ""

            if is_base:
                # Reuse the already-computed baseline
                trade_df_p = base_trade_df
                final_eq_p = base_final_eq
            else:
                trade_df_p, _, _eq_p, final_eq_p = run_backtest_fn(
                    frames,
                    spread=spread, slippage_std=slippage_std,
                    risk_pct=risk_pct, tp1_r=tp1_r, tp2_r=tp2_r,
                    verbose=False, **kwargs,
                )

            m = _metrics(trade_df_p, final_eq_p, starting_equity)
            print(f"  {tp1_r:>5.2f} {tp2_r:>5.2f}  "
                  f"{m['total_return']:>+8.2f}%  "
                  f"{m['max_drawdown']:>+7.2f}%  "
                  f"{m['sharpe']:>8.3f}  "
                  f"{m['profit_factor']:>7.3f}  "
                  f"{m['trades']:>7}  "
                  f"{m['win_rate']:>5.1f}%"
                  f"{tag}")
            param_rows.append({
                "tp1_r": tp1_r, "tp2_r": tp2_r,
                "is_baseline": is_base, **m,
            })

    param_df = pd.DataFrame(param_rows) if param_rows else pd.DataFrame()

    if not param_df.empty:
        ret_std  = float(param_df["total_return"].std())
        ret_mean = float(param_df["total_return"].mean())
        shr_std  = float(param_df["sharpe"].std())
        cv       = ret_std / abs(ret_mean) if ret_mean != 0 else float("inf")
        r_min    = float(param_df["total_return"].min())
        r_max    = float(param_df["total_return"].max())
        prof_frac = float((param_df["total_return"] > 0).mean())

        print(DIV)
        print(f"\n  Return range   : [{r_min:+.2f}%, {r_max:+.2f}%]")
        print(f"  Return σ       : {ret_std:.2f}%    CV = {cv:.2f}")
        print(f"  Sharpe σ       : {shr_std:.3f}")
        print(f"  Profitable configs : {int(prof_frac * len(param_df))} / {len(param_df)}")

        if cv < 0.20 and shr_std < 0.30 and prof_frac >= 0.80:
            param_verdict = "STABLE — performance stable across TP variants"
        elif cv < 0.40 and prof_frac >= 0.60:
            param_verdict = "STABLE — acceptable sensitivity to TP placement"
        else:
            param_verdict = "SENSITIVE — results depend heavily on TP levels"

        print(f"  ROBUSTNESS RESULT: {param_verdict}")
        print(f"{SEP}\n")

    results["param_df"] = param_df

    # ── 3. RISK CONFIRMATION ──────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  3. RISK CONFIRMATION")
    print(f"  (derived from baseline — no extra engine calls)")
    print(f"{'─'*W}")
    risk_result = run_risk_confirmation(
        frames, pair, run_backtest_fn,
        spread=spread, slippage_std=slippage_std,
        risk_levels=[0.0125, 0.0150, 0.0175],
        dd_limit=dd_limit, starting_equity=starting_equity,
        baseline_trade_df=base_trade_df,
        base_risk=risk_pct,
        **kwargs,
    )
    results["risk_result"] = risk_result

    # ── 4. FINAL VALIDATION DECISION ──────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  4. FINAL VALIDATION DECISION")
    print(f"{'─'*W}")
    print_phase1_decision(
        walk, param_df, risk_result,
        pair=pair, current_risk_pct=risk_pct,
    )

    return results
