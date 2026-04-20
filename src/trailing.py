# =============================================================================
# trailing.py  —  Multi-TP Trailing Stop Backtest Strategy  (v2 — realistic)
# =============================================================================
# Strategy: Multi-timeframe liquidity sweep + 15m directional change + FVG + 5m BOS
#
# EXIT MODEL  (exit_mode="dual_tp")
#   TP1 @ +0.75R → close 60%; stop moves to breakeven
#   TP2 @ +1.50R → close remaining 40%
#   Before TP1 : original SL beats TP1 on same bar
#   After  TP1 : breakeven stop beats TP2 on same bar
#
# OPTIONAL TEST MODES
#   exit_mode="fixed_sl"   — same TPs; stop never moves (comparison baseline)
#   exit_mode="single_tp"  — close 100% at 1.2R, original SL throughout
#
# EXECUTION MODEL (v4 — refined retracement entry)
#   Spread + slippage applied to all fills
#   SL/TP fills always at worst realistic price (never perfect)
#   Session filter: London 07-12 UTC, New York 13-17 UTC
#   Retracement entry: wait for 55% pullback to swing midpoint after signal
#   Minimum retracement: price must retrace ≥ 0.33R before fill
#   Structure-based stop: swing low/high from last N bars; ATR as fallback
#   ATR regime filter: skips dead-market sessions (ATR/price < 0.03%)
#   Cost gate: stop distance must be ≥ 4× spread before entry is accepted
#
# STRESS TEST
#   run_stress_test() → Scenario A (ideal), B (realistic), C (worst-case)
# =============================================================================

import os
import warnings

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from data_loader import load_local_data, prompt_data_inputs
from sim_costs import (
    get_default_costs, apply_entry_fill, apply_exit_fill,
    apply_sl_fill, apply_tp_fill, ALL_SCENARIOS,
    SCENARIO_IDEAL, SCENARIO_REALISTIC, SCENARIO_WORST_CASE,
)
from analytics import (
    compute_mfe_mae,
    build_yearly_summary,
    build_pair_summary,
    build_continuation_stats,
    print_analytics,
    print_edge_diagnostics,
    print_yearly_summary,
    print_continuation_summary,
    print_report,
    export_all,
)
from robustness import (
    run_walk_forward,
    run_rolling_windows,
    run_mfe_mae_by_year,
    run_param_robustness,
    compute_rolling_performance,
    print_final_decision,
    run_full_robustness_suite,
    run_phase1_validation,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
STARTING_EQUITY  = 100_000.0
MAX_RISK_PCT     = 0.015
ATR_PERIOD       = 14
SWING_WINDOW     = 5
LIQ_TOUCH_PCT    = 0.001
PROGRESS_EVERY   = 50_000
STRUCT_LOOKBACK  = 20    # bars to look back for structure stop

# --- Robustness filters ---
STOP_MULTIPLIER      = 1.08   # widen computed stop by this factor
MIN_ATR_THRESHOLD    = 0.0    # skip signal if ATR < this value (0 = disabled); tune per pair
MIN_ATR_RATIO        = 0.0003 # skip if ATR/price < 0.03% — filters genuinely dead markets
                               # Conservative enough to pass most normal sessions.
                               # Raise to 0.0005 to filter more aggressively. Set 0.0 to disable.
MAX_SPREAD_FILTER    = None   # max allowed spread (price units); None = no filter
MIN_STOP_SPREAD_MULT = 4.0    # stop distance must be >= this many × spread before entry
                               # Modest raise from 3× — rejects only the tightest setups.

# --- BOS quality filters ---
# Start disabled. Enable and tune one at a time once trade count is healthy.
# Stacking multiple filters simultaneously caused over-filtering (3 trades / 4 years).
# Lesson: each filter should be judged individually against the trade count impact.
MIN_BOS_BODY_PCT  = 0.0   # BOS bar body/range >= this fraction (0.0 = disabled)
                           # When enabling: start at 0.30, observe trade count, step up slowly.
BOS_MARGIN_MULT   = 0.0   # BOS close must clear prior swing by >= this × ATR (0.0 = disabled)
                           # When enabling: start at 0.05, observe trade count.

# --- Exit mode ---
# "dual_tp"  : TP1@0.75R(60%) + TP2@1.50R(40%); stop → breakeven after TP1
# "fixed_sl" : same TPs; stop never moves (comparison baseline)
# "single_tp": 100% close at 1.2R, original SL throughout
EXIT_MODE = "dual_tp"

# --- Default TP levels ---
TP1_R = 0.75   # partial close (60%) at this R multiple
TP2_R = 1.50   # full close (40%) at this R multiple

# --- Alternate TP profile (higher targets) ---
# Only ~5.6% of EURUSD worst-case trades reached 2.0R — test this AFTER MFE improves.
# Switch by passing tp1_r=TP1_R_ALT, tp2_r=TP2_R_ALT to run_backtest().
TP1_R_ALT = 1.00   # hold for stronger expansion before first partial
TP2_R_ALT = 2.00   # runner target once entry quality improves

# --- Retracement entry ---
# Modest deepening from 0.50 → 0.55: improves entry R/R without killing fill rate.
# The jump to 0.65 + MIN_RETRACE_R 0.40 was too aggressive in combination.
RETRACE_LEVEL   = 0.55   # entry at this fraction of (signal_close → liq_level) swing
MIN_RETRACE_R   = 0.33   # minimum pullback in R-multiples required before fill
RETRACE_TIMEOUT = 14     # cancel pending signal after this many 5m bars (70 min)

# --- Dev mode ---
# Set DEV_MODE=True to slice data to a short window for rapid iteration.
# Does NOT change strategy logic — only the date window fed to run_backtest().
# Usage: pass dev=True to run_backtest() or set DEV_MODE=True here globally.
DEV_MODE       = False         # set True for fast iteration
DEV_START_DATE = "2023-01-01"  # slice start (inclusive)
DEV_END_DATE   = "2024-06-30"  # slice end   (inclusive)

# --- Output flags ---
# Set False globally to suppress prints/plots during robustness sweeps.
VERBOSE_DEFAULT = True   # default for run_backtest(verbose=...) parameter
ENABLE_PLOTS    = True   # set False to skip all plt.show() calls system-wide


# =============================================================================
# SESSION FILTER  (timestamps assumed UTC)
# =============================================================================
def is_session_active(ts: pd.Timestamp) -> bool:
    """Return True during London (07-12 UTC) or New York (13-17 UTC) sessions."""
    h = ts.hour
    return (7 <= h < 12) or (13 <= h < 18)


# =============================================================================
# SWING HIGH / LOW DETECTION
# =============================================================================
def detect_swings(df: pd.DataFrame, window: int = SWING_WINDOW) -> pd.DataFrame:
    highs  = df["High"].values
    lows   = df["Low"].values
    opens  = df["Open"].values
    closes = df["Close"].values
    n      = len(df)

    swing_highs = np.full(n, np.nan)
    swing_lows  = np.full(n, np.nan)

    for i in range(window, n - window):
        lo = i - window
        hi = i + window + 1
        cluster_bull = np.any(closes[lo:hi] > opens[lo:hi])
        cluster_bear = np.any(closes[lo:hi] < opens[lo:hi])
        if not (cluster_bull and cluster_bear):
            continue
        if highs[i] == np.max(highs[lo:hi]):
            swing_highs[i] = highs[i]
        if lows[i] == np.min(lows[lo:hi]):
            swing_lows[i] = lows[i]

    out = df.copy()
    out["SwingHigh"] = swing_highs
    out["SwingLow"]  = swing_lows
    return out


# =============================================================================
# UNTOUCHED LIQUIDITY
# =============================================================================
def get_untouched_liquidity(df: pd.DataFrame, as_of_idx: int) -> dict:
    highs_col = df["SwingHigh"].values
    lows_col  = df["SwingLow"].values
    price_hi  = df["High"].values
    price_lo  = df["Low"].values
    result    = {"high": None, "low": None}

    for i in range(as_of_idx - 1, -1, -1):
        if np.isnan(highs_col[i]):
            continue
        if not np.any(price_hi[i + 1: as_of_idx + 1] >= highs_col[i]):
            result["high"] = (highs_col[i], i)
            break

    for i in range(as_of_idx - 1, -1, -1):
        if np.isnan(lows_col[i]):
            continue
        if not np.any(price_lo[i + 1: as_of_idx + 1] <= lows_col[i]):
            result["low"] = (lows_col[i], i)
            break

    return result


def price_near_level(price: float, level: float, pct: float = LIQ_TOUCH_PCT) -> bool:
    return abs(price - level) / level <= pct


# =============================================================================
# 15-MINUTE DIRECTIONAL CHANGE
# =============================================================================
def check_15m_directional_change(df15: pd.DataFrame, as_of_idx: int, direction: str) -> bool:
    lo      = max(0, as_of_idx - 20)
    segment = df15.iloc[lo: as_of_idx + 1]
    if len(segment) < 3:
        return False
    highs = segment["High"].values
    lows  = segment["Low"].values
    if direction == "bear":
        for j in range(1, len(highs) - 1):
            if highs[j] < highs[j - 1]:
                for k in range(j + 1, len(lows)):
                    if lows[k] < lows[j]:
                        return True
    elif direction == "bull":
        for j in range(1, len(lows) - 1):
            if lows[j] > lows[j - 1]:
                for k in range(j + 1, len(highs)):
                    if highs[k] > highs[j]:
                        return True
    return False


# =============================================================================
# FAIR VALUE GAP
# =============================================================================
def find_fvg(df: pd.DataFrame, as_of_idx: int, direction: str) -> bool:
    if as_of_idx < 2:
        return False
    c1 = df.iloc[as_of_idx - 2]
    c3 = df.iloc[as_of_idx]
    if direction == "bull":
        return float(c1["High"]) < float(c3["Low"])
    if direction == "bear":
        return float(c1["Low"]) > float(c3["High"])
    return False


def check_fvg_any_tf(frames: dict, timestamp: pd.Timestamp, direction: str) -> bool:
    for tf in ["1h", "30m", "15m"]:
        df  = frames[tf]
        idx = df.index.searchsorted(timestamp, side="right") - 1
        if idx >= 2 and find_fvg(df, idx, direction):
            return True
    return False


# =============================================================================
# 5-MINUTE BREAK OF STRUCTURE
# =============================================================================
def check_5m_bos(df5: pd.DataFrame, as_of_idx: int, direction: str) -> bool:
    lo = max(0, as_of_idx - 10)
    if as_of_idx < 1:
        return False
    current_close = float(df5.iloc[as_of_idx]["Close"])
    if direction == "bull":
        prior_highs = df5.iloc[lo: as_of_idx]["High"]
        return (not prior_highs.empty) and (current_close > float(prior_highs.max()))
    if direction == "bear":
        prior_lows = df5.iloc[lo: as_of_idx]["Low"]
        return (not prior_lows.empty) and (current_close < float(prior_lows.min()))
    return False


# =============================================================================
# ATR STOP DISTANCE
# =============================================================================
def compute_atr_stop(df5: pd.DataFrame, as_of_idx: int) -> float:
    lo      = max(0, as_of_idx - ATR_PERIOD - 1)
    segment = df5.iloc[lo: as_of_idx + 1]
    high    = segment["High"]
    low     = segment["Low"]
    close   = segment["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean().iloc[-1]
    if pd.isna(atr) or atr == 0:
        atr = float(close.iloc[-1]) * 0.001
    return float(atr)


# =============================================================================
# STRUCTURE-BASED STOP  (primary)  with ATR fallback
# =============================================================================
def compute_structure_stop(df5: pd.DataFrame, as_of_idx: int, direction: str,
                            atr_dist: float, lookback: int = STRUCT_LOOKBACK) -> float:
    """
    Place stop below the lowest low (bull) or above the highest high (bear)
    in the last `lookback` 5m bars.  Falls back to ATR if structure stop
    is tighter than 50% of ATR (stop would be too close to current price).
    """
    lo      = max(0, as_of_idx - lookback)
    segment = df5.iloc[lo: as_of_idx]
    entry_p = float(df5.iloc[as_of_idx]["Close"])

    if direction == "bull":
        struct_level = float(segment["Low"].min()) if len(segment) else entry_p - atr_dist
        stop_dist    = entry_p - struct_level
        if stop_dist < atr_dist * 0.5 or stop_dist <= 0:
            return entry_p - atr_dist    # ATR fallback
        return struct_level
    else:
        struct_level = float(segment["High"].max()) if len(segment) else entry_p + atr_dist
        stop_dist    = struct_level - entry_p
        if stop_dist < atr_dist * 0.5 or stop_dist <= 0:
            return entry_p + atr_dist    # ATR fallback
        return struct_level


# =============================================================================
# POSITION SIZING
# =============================================================================
def compute_position_size(equity: float, stop_distance: float,
                          risk_pct: float = MAX_RISK_PCT) -> float:
    if stop_distance <= 0:
        return 0.0
    return round((equity * risk_pct) / stop_distance, 2)


# =============================================================================
# TRADE ROW BUILDER
# =============================================================================
def _build_trade_row(trade: dict) -> dict:
    entry_price     = trade["entry_price"]
    last_exit_price = trade["last_exit_price"]
    direction       = trade["direction"]
    price_change    = ((last_exit_price - entry_price) if direction == "bull"
                       else (entry_price - last_exit_price))
    R = trade["R"]
    return {
        "trade_id":        trade["trade_id"],
        "timeframe":       trade["timeframe"],
        "entry_time":      trade["entry_time"],
        "exit_time":       trade["last_exit_time"],
        "direction":       direction,
        "entry_price":     round(entry_price, 5),
        "exit_price":      round(last_exit_price, 5),
        "orig_stop_price": round(trade["orig_stop_price"], 5),
        "orig_units":      trade["orig_units"],
        "R_price":         round(R, 5),
        "tp1_price":       round((entry_price + trade.get("tp1_r", 0.75)*R) if direction == "bull" else (entry_price - trade.get("tp1_r", 0.75)*R), 5),
        "tp2_price":       round((entry_price + trade.get("tp2_r", 1.50)*R) if direction == "bull" else (entry_price - trade.get("tp2_r", 1.50)*R), 5),
        "price_change":    round(price_change, 5),
        "pct_change":      round(price_change / entry_price * 100, 4) if entry_price else np.nan,
        "entry_reason":    trade["entry_reason"],
        "exit_mode":       trade.get("exit_mode", "dual_tp"),
        "tp1_hit":         trade["tp1_hit"],
        "tp2_hit":         trade["tp2_hit"],
        "total_pnl":       trade["total_pnl"],
        "pnl_pct":         round(trade["total_pnl"] / trade["equity_at_entry"] * 100, 4),
        "equity_after":    trade["final_equity"],
        "spread_applied":  trade.get("spread_applied", 0.0),
    }


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def run_backtest(
    frames: dict,
    spread: float = 0.0,
    slippage_std: float = 0.0,
    use_session_filter: bool = True,
    use_pullback_entry: bool = True,
    use_structure_stop: bool = True,
    max_spread_filter: float = None,
    min_atr_threshold: float = MIN_ATR_THRESHOLD,
    min_atr_ratio: float = MIN_ATR_RATIO,
    stop_multiplier: float = STOP_MULTIPLIER,
    min_stop_spread_mult: float = MIN_STOP_SPREAD_MULT,
    min_bos_body_pct: float = MIN_BOS_BODY_PCT,
    bos_margin_mult: float = BOS_MARGIN_MULT,
    exit_mode: str = EXIT_MODE,
    retrace_level: float = RETRACE_LEVEL,
    min_retrace_R: float = MIN_RETRACE_R,
    retrace_timeout: int = RETRACE_TIMEOUT,
    risk_pct: float = MAX_RISK_PCT,
    tp1_r: float = TP1_R,
    tp2_r: float = TP2_R,
    verbose: bool = True,
    dev: bool = None,
    dev_start: str = DEV_START_DATE,
    dev_end: str   = DEV_END_DATE,
    _precomputed_swings: tuple = None,
) -> tuple:
    """
    Staged partial take-profit backtest with retracement entry timing.

    Parameters
    ----------
    spread             : bid-ask spread in price units (applied at entry + exit)
    slippage_std       : std dev of normal slippage distribution
    use_session_filter : restrict entries to London (07-12 UTC) + NY (13-17 UTC)
    use_pullback_entry : True → wait for retracement fill; False → immediate close fill
    use_structure_stop : use swing structure for stop; fall back to ATR
    max_spread_filter  : skip signal if spread > this value (None = disabled)
    min_atr_threshold    : skip signal if ATR < this value (0 = disabled); tune per pair
    min_atr_ratio        : skip if ATR/price < this value; filters low-vol regimes
    stop_multiplier      : scale computed stop distance (< 1.0 tightens, > 1.0 widens)
    min_stop_spread_mult : stop distance must be >= this many × spread before entry
    min_bos_body_pct     : BOS bar body/range must be >= this fraction (0.0 = disabled)
    bos_margin_mult      : BOS close must clear prior swing by >= this × ATR (0.0 = disabled)
    exit_mode            : "dual_tp"  — TP1(60%) + TP2(40%); B/E after TP1
                           "fixed_sl" — same TPs; stop never moves to B/E
                           "single_tp"— 100% close at 1.2R, original SL throughout
    retrace_level        : fraction of (signal_close → liq_level) swing to target
    min_retrace_R        : minimum pullback in R-multiples before entry allowed
    retrace_timeout      : bars to wait for retracement before cancelling signal
    risk_pct             : fraction of equity risked per trade (0.0075 = 0.75%)

    Returns
    -------
    (trade_df, exit_events_df, equity_df, final_equity)
    """
    # Dev mode: slice frames to a short window for rapid iteration
    _use_dev = DEV_MODE if dev is None else dev
    if _use_dev:
        _s = pd.Timestamp(dev_start)
        _e = pd.Timestamp(dev_end) + pd.Timedelta(hours=23, minutes=59)
        frames = {tf: df.loc[(_s <= df.index) & (df.index <= _e)].copy()
                  for tf, df in frames.items()}
        if verbose:
            print(f"  [DEV MODE] Sliced to {dev_start} – {dev_end}")
        _precomputed_swings = None   # re-compute swings for sliced frames

    df5  = frames["5m"]
    df15 = frames["15m"]

    if _precomputed_swings is not None:
        df1h_s, df4h_s = _precomputed_swings
    else:
        df1h_s = detect_swings(frames["1h"], window=SWING_WINDOW)
        df4h_s = detect_swings(frames["4h"], window=max(2, SWING_WINDOW // 2))

    df5_open  = df5["Open"].values
    df5_high  = df5["High"].values
    df5_low   = df5["Low"].values
    df5_close = df5["Close"].values
    df5_index = df5.index

    # ------------------------------------------------------------------
    # PRE-COMPUTATION  (replaces per-bar expensive calls with array lookups)
    # ------------------------------------------------------------------
    # ATR array — one full-series rolling pass instead of pd.concat per bar
    _tr = pd.concat([
        df5["High"] - df5["Low"],
        (df5["High"] - df5["Close"].shift(1)).abs(),
        (df5["Low"]  - df5["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    _atr_raw = _tr.rolling(ATR_PERIOD).mean().values
    _atr_arr = np.where(
        np.isnan(_atr_raw) | (_atr_raw == 0),
        df5_close * 0.001,
        _atr_raw,
    )

    # Structure stop arrays — rolling window over prior bars (shift=1 excludes current bar)
    _struct_low  = df5["Low"].rolling(STRUCT_LOOKBACK).min().shift(1).values
    _struct_high = df5["High"].rolling(STRUCT_LOOKBACK).max().shift(1).values

    # BOS arrays — max high / min low of prior 10 bars (same window as check_5m_bos)
    _bos_bull_swing = df5["High"].rolling(10).max().shift(1).values
    _bos_bear_swing = df5["Low"].rolling(10).min().shift(1).values

    # FVG precomputation for each TF: bool arrays + 5m→tf index map
    def _precompute_fvg(df_tf):
        h = df_tf["High"].values
        l = df_tf["Low"].values
        n = len(df_tf)
        bull_fvg = np.zeros(n, dtype=bool)
        bear_fvg = np.zeros(n, dtype=bool)
        if n >= 3:
            bull_fvg[2:] = h[:-2] < l[2:]
            bear_fvg[2:] = l[:-2] > h[2:]
        return bull_fvg, bear_fvg

    _fvg_data = []   # list of (bull_fvg_arr, bear_fvg_arr, idx_map)
    for _tf in ["1h", "30m", "15m"]:
        _df_tf = frames[_tf]
        _bf, _brf = _precompute_fvg(_df_tf)
        _imap = np.searchsorted(_df_tf.index, df5_index, side="right").astype(np.intp) - 1
        _fvg_data.append((_bf, _brf, _imap))

    # HTF index maps — precompute 5m → 1h / 4h bar indices for all bars at once
    _htf_1h_idx = np.searchsorted(df1h_s.index, df5_index, side="right").astype(np.intp) - 1
    _htf_4h_idx = np.searchsorted(df4h_s.index, df5_index, side="right").astype(np.intp) - 1

    # 15m index map
    _df15_idx = np.searchsorted(df15.index, df5_index, side="right").astype(np.intp) - 1

    # Liquidity caches — keyed by HTF bar index; many 5m bars share the same HTF bar
    _liq_cache_1h: dict = {}
    _liq_cache_4h: dict = {}

    # Inline structure-stop using precomputed rolling arrays.
    # Matches compute_structure_stop() exactly: uses bar close as the distance reference,
    # identical to how the original re-reads df5.iloc[as_of_idx]["Close"] internally.
    def _fast_struct_stop(bar_i: int, direction: str, _unused_entry_p: float, atr_d: float) -> float:
        ref_p = df5_close[bar_i]   # original always uses bar close, not fill price
        if direction == "bull":
            sl = _struct_low[bar_i]
            if np.isnan(sl):
                return ref_p - atr_d
            dist = ref_p - sl
            return float(sl) if (dist >= atr_d * 0.5 and dist > 0) else ref_p - atr_d
        else:
            sl = _struct_high[bar_i]
            if np.isnan(sl):
                return ref_p + atr_d
            dist = sl - ref_p
            return float(sl) if (dist >= atr_d * 0.5 and dist > 0) else ref_p + atr_d

    equity          = STARTING_EQUITY
    equity_curve    = []
    trade_log       = []
    exit_events     = []
    in_trade        = False
    trade           = {}
    pending_dir     = None   # direction pending entry on next bar
    pending_meta    = {}
    n5              = len(df5)

    mode_desc = (
        f"spread={spread:.5f}  slip_std={slippage_std:.5f}  "
        f"session={'ON' if use_session_filter else 'OFF'}  "
        f"pullback={'ON' if use_pullback_entry else 'OFF'}  "
        f"struct_stop={'ON' if use_structure_stop else 'OFF'}  "
        f"max_spread_filter={max_spread_filter}  "
        f"min_atr={min_atr_threshold}  min_atr_ratio={min_atr_ratio}  "
        f"stop_mult={stop_multiplier}  min_stop_sp_mult={min_stop_spread_mult}  "
        f"bos_body>={min_bos_body_pct:.0%}  bos_margin={bos_margin_mult}×ATR  "
        f"exit_mode={exit_mode}  tp1={tp1_r}R  tp2={tp2_r}R  "
        f"retrace={retrace_level:.0%}  min_ret_R={min_retrace_R}  timeout={retrace_timeout}bars"
    )
    if verbose:
        print(f"\n  Running trailing backtest on {n5:,} 5m candles ...")
        print(f"  Exec mode: {mode_desc}")

    for i in range(50, n5):
        if verbose and i % PROGRESS_EVERY == 0 and i > 50:
            pct = i / n5 * 100
            print(f"    {pct:.0f}%  bar {i:,}/{n5:,}  trades so far: {len(trade_log)}")

        current_time  = df5_index[i]
        current_open  = float(df5_open[i])
        current_high  = float(df5_high[i])
        current_low   = float(df5_low[i])
        current_close = float(df5_close[i])

        equity_curve.append({"time": current_time, "equity": equity})

        # ------------------------------------------------------------------
        # PENDING RETRACEMENT ENTRY  (use_pullback_entry=True path)
        # ------------------------------------------------------------------
        if pending_dir is not None and not in_trade:
            direction      = pending_dir
            atr_dist       = pending_meta["atr_dist"]
            liq_str        = pending_meta["liq_str"]
            trade_id       = pending_meta["trade_id"]
            signal_bar     = pending_meta["signal_bar"]
            signal_close   = pending_meta["signal_close"]
            retrace_target = pending_meta["retrace_target"]
            min_retrace    = pending_meta["min_retrace"]

            # ---- Timeout: cancel signal if retracement never came ----
            if i - signal_bar > retrace_timeout:
                pending_dir = None; pending_meta = {}
                # Fall through to signal scan so a new signal can fire

            else:
                # ---- Check if retracement is reached this bar ----
                if direction == "bull":
                    # Bull pullback: price must dip to retrace_target
                    retrace_hit = current_low  <= retrace_target
                    # Minimum retracement: close must have been above signal_close,
                    # and this bar's low must reach signal_close - min_retrace
                    min_ret_hit = current_low  <= signal_close - min_retrace
                else:
                    # Bear pullback: price must rise to retrace_target
                    retrace_hit = current_high >= retrace_target
                    min_ret_hit = current_high >= signal_close + min_retrace

                if retrace_hit and min_ret_hit:
                    # Fill as a conservative limit order at the retracement target.
                    # apply_entry_fill adds spread+slip so the fill is realistic.
                    entry_price = apply_entry_fill(retrace_target, direction, spread, slippage_std)

                    if use_structure_stop:
                        stop_price = _fast_struct_stop(i, direction, entry_price, atr_dist)
                    else:
                        stop_price = (entry_price - atr_dist if direction == "bull"
                                      else entry_price + atr_dist)

                    raw_dist  = abs(entry_price - stop_price)
                    widened   = raw_dist * stop_multiplier
                    stop_price = (entry_price - widened if direction == "bull"
                                  else entry_price + widened)
                    stop_distance = widened

                    if stop_distance <= 0:
                        pending_dir = None; pending_meta = {}
                    else:
                        units    = compute_position_size(equity, stop_distance, risk_pct)
                        R        = stop_distance
                        in_trade = True
                        trade    = {
                            "trade_id":        trade_id,
                            "timeframe":       "5m exec | 15m/1h/4h context",
                            "entry_time":      current_time,
                            "entry_price":     entry_price,
                            "direction":       direction,
                            "orig_units":      units,
                            "rem_units":       units,
                            "stop_price":      stop_price,
                            "orig_stop_price": stop_price,
                            "R":               R,
                            "entry_reason":    f"{direction.upper()} | {liq_str} | retrace fill",
                            "equity_at_entry": equity,
                            "spread_applied":  round(spread + slippage_std, 6),
                            "exit_mode":       exit_mode,
                            "tp1_r":           tp1_r,
                            "tp2_r":           tp2_r,
                            "tp1_hit": False, "tp2_hit": False,
                            "closed": False, "total_pnl": 0.0,
                            "final_equity": None, "last_exit_time": None, "last_exit_price": None,
                        }
                        pending_dir = None; pending_meta = {}
                    # Fall through to in-trade check on same bar
                else:
                    # Still waiting for retracement — do not scan for new signals
                    continue

        # ------------------------------------------------------------------
        # IN-TRADE EXIT LOGIC
        # ------------------------------------------------------------------
        if in_trade:
            direction   = trade["direction"]
            entry_price = trade["entry_price"]
            stop_price  = trade["stop_price"]
            R           = trade["R"]
            orig_units  = trade["orig_units"]
            rem_units   = trade["rem_units"]
            tp1_hit     = trade["tp1_hit"]
            tp2_hit     = trade["tp2_hit"]
            mode        = trade["exit_mode"]
            _tp1_r      = trade.get("tp1_r", 0.75)
            _tp2_r      = trade.get("tp2_r", 1.50)

            def _pnl(units, exec_p):
                return (units * (exec_p - entry_price) if direction == "bull"
                        else units * (entry_price - exec_p))

            def _close_trade(event_label, fill_price, units):
                """Close remaining position and log the trade."""
                pnl_exit = _pnl(units, fill_price)
                equity_ref[0] += pnl_exit
                exit_events.append({"trade_id": trade["trade_id"], "event": event_label,
                                    "time": current_time, "price": round(fill_price, 5),
                                    "units_closed": units, "pnl": round(pnl_exit, 2),
                                    "equity_after": round(equity_ref[0], 2)})
                trade["rem_units"]       = 0.0
                trade["closed"]          = True
                trade["total_pnl"]       = round(trade.get("total_pnl", 0.0) + pnl_exit, 2)
                trade["final_equity"]    = round(equity_ref[0], 2)
                trade["last_exit_time"]  = current_time
                trade["last_exit_price"] = fill_price
                trade_log.append(_build_trade_row(trade))

            # equity_ref lets the nested helper mutate equity
            equity_ref = [equity]

            if direction == "bull":
                stop_hit = current_low <= stop_price
            else:
                stop_hit = current_high >= stop_price

            # ----------------------------------------------------------
            # exit_mode == "single_tp"  — 100% close at 1.2R
            # ----------------------------------------------------------
            if mode == "single_tp":
                tp_p = (entry_price + 1.2 * R) if direction == "bull" else (entry_price - 1.2 * R)
                tp_touch = (current_high >= tp_p) if direction == "bull" else (current_low <= tp_p)

                if stop_hit:
                    fill = apply_sl_fill(stop_price, direction, slippage_std)
                    _close_trade("STOP", fill, rem_units)
                    equity = equity_ref[0]
                    in_trade = False; trade = {}
                    continue

                if tp_touch:
                    fill = apply_tp_fill(tp_p, direction, slippage_std)
                    trade["tp1_hit"] = True
                    _close_trade("TP1", fill, rem_units)
                    equity = equity_ref[0]
                    in_trade = False; trade = {}

                continue

            # ----------------------------------------------------------
            # exit_mode == "dual_tp"
            #   TP1 @ 0.75R → close 60%; stop moves to breakeven
            #   TP2 @ 1.50R → close remaining 40%
            #   Before TP1 : original SL beats TP1 on same bar (STOP)
            #   After  TP1 : breakeven stop beats TP2 on same bar (BE_STOP)
            #
            # exit_mode == "fixed_sl"
            #   Same TPs; stop never moves to B/E (comparison mode)
            # ----------------------------------------------------------
            if direction == "bull":
                tp1_p     = entry_price + _tp1_r * R
                tp2_p     = entry_price + _tp2_r * R
                tp1_touch = (not tp1_hit) and (current_high >= tp1_p)
                tp2_touch = tp1_hit and (not tp2_hit) and (current_high >= tp2_p)
            else:
                tp1_p     = entry_price - _tp1_r * R
                tp2_p     = entry_price - _tp2_r * R
                tp1_touch = (not tp1_hit) and (current_low  <= tp1_p)
                tp2_touch = tp1_hit and (not tp2_hit) and (current_low  <= tp2_p)

            # ---- Before TP1: original stop beats TP1 on same bar ----
            if stop_hit and not tp1_hit:
                fill = apply_sl_fill(stop_price, direction, slippage_std)
                _close_trade("STOP", fill, rem_units)
                equity = equity_ref[0]
                in_trade = False; trade = {}
                continue

            # ---- After TP1: breakeven stop beats TP2 on same bar ----
            if stop_hit and tp1_hit:
                fill = apply_sl_fill(stop_price, direction, slippage_std)
                _close_trade("BE_STOP", fill, rem_units)
                equity = equity_ref[0]
                in_trade = False; trade = {}
                continue

            # ---- TP1: close 60%; move stop to breakeven (dual_tp only) ----
            if tp1_touch:
                fill    = apply_tp_fill(tp1_p, direction, slippage_std)
                u       = round(orig_units * 0.60, 2)
                pnl_tp1 = _pnl(u, fill)
                equity += pnl_tp1
                trade["rem_units"] -= u
                trade["tp1_hit"]    = True
                trade["total_pnl"]  = round(trade.get("total_pnl", 0.0) + pnl_tp1, 2)
                exit_events.append({"trade_id": trade["trade_id"], "event": "TP1",
                                    "time": current_time, "price": round(fill, 5),
                                    "units_closed": u, "pnl": round(pnl_tp1, 2),
                                    "equity_after": round(equity, 2)})
                tp1_hit   = True
                rem_units = trade["rem_units"]
                equity_ref[0] = equity
                if mode == "dual_tp":
                    trade["stop_price"] = entry_price   # move stop to breakeven
                    stop_price          = entry_price
                # Re-evaluate TP2 in case both levels hit on the same bar
                if direction == "bull":
                    tp2_touch = (not tp2_hit) and (current_high >= tp2_p)
                else:
                    tp2_touch = (not tp2_hit) and (current_low  <= tp2_p)

            # ---- TP2: close remaining 40%; trade fully closed ----
            if tp2_touch:
                fill = apply_tp_fill(tp2_p, direction, slippage_std)
                u    = trade["rem_units"]
                trade["tp2_hit"] = True
                _close_trade("TP2", fill, u)
                equity = equity_ref[0]
                in_trade = False; trade = {}

            continue

        # ------------------------------------------------------------------
        # SIGNAL SCAN
        # ------------------------------------------------------------------
        if use_session_filter and not is_session_active(current_time):
            continue

        for direction in ["bull", "bear"]:
            liq_hit, liq_level = False, None

            for _htf_df, _htf_imap, _liq_cache in (
                    (df1h_s, _htf_1h_idx, _liq_cache_1h),
                    (df4h_s, _htf_4h_idx, _liq_cache_4h)):
                idx_htf = int(_htf_imap[i])
                if idx_htf < 1:
                    continue
                if idx_htf not in _liq_cache:
                    _liq_cache[idx_htf] = get_untouched_liquidity(_htf_df, idx_htf)
                liq = _liq_cache[idx_htf]
                if direction == "bull" and liq["low"] is not None:
                    if price_near_level(current_low, liq["low"][0]):
                        liq_hit, liq_level = True, liq["low"][0]; break
                elif direction == "bear" and liq["high"] is not None:
                    if price_near_level(current_high, liq["high"][0]):
                        liq_hit, liq_level = True, liq["high"][0]; break

            if not liq_hit:
                continue

            idx15 = int(_df15_idx[i])
            if idx15 < 0 or not check_15m_directional_change(df15, idx15, direction):
                continue
            # FVG check using precomputed arrays (avoids 3× searchsorted per bar)
            _fvg_hit = False
            for _bf, _brf, _imap in _fvg_data:
                _j = int(_imap[i])
                if _j >= 2:
                    if direction == "bull" and _bf[_j]:
                        _fvg_hit = True; break
                    elif direction == "bear" and _brf[_j]:
                        _fvg_hit = True; break
            if not _fvg_hit:
                continue
            # BOS check using precomputed rolling arrays (avoids iloc per bar)
            if direction == "bull":
                _bos_ok = (not np.isnan(_bos_bull_swing[i])) and (df5_close[i] > _bos_bull_swing[i])
            else:
                _bos_ok = (not np.isnan(_bos_bear_swing[i])) and (df5_close[i] < _bos_bear_swing[i])
            if not _bos_ok:
                continue

            # ---- BOS quality filter 1: bar body must be a meaningful fraction of range ----
            # A doji or spinning-top BOS bar (tiny body) shows no real momentum.
            # If the BOS close is marginal and indecisive, post-entry expansion is unlikely.
            # Controls: MIN_BOS_BODY_PCT (0.0 = disabled). Tune: 0.40–0.65.
            if min_bos_body_pct > 0:
                _bos_range = current_high - current_low
                _bos_body  = abs(current_close - current_open)
                if _bos_range > 0 and _bos_body / _bos_range < min_bos_body_pct:
                    continue

            atr_dist = float(_atr_arr[i])

            # ---- BOS quality filter 2: close must clear the prior swing by a real margin ----
            # Prevents triggering on 1-pip breaches of the swing level that immediately
            # reverse. Margin = BOS_MARGIN_MULT × ATR. At EURUSD ATR ~0.0010 and
            # BOS_MARGIN_MULT=0.08, requires ≈ 0.5 pip clearance — eliminating noise.
            if bos_margin_mult > 0:
                if direction == "bull":
                    _prior_swing = float(_bos_bull_swing[i])
                    _bos_margin  = current_close - _prior_swing
                else:
                    _prior_swing = float(_bos_bear_swing[i])
                    _bos_margin  = _prior_swing - current_close
                if _bos_margin < bos_margin_mult * atr_dist:
                    continue

            # ---- Volatility filter: skip if ATR is below minimum threshold ----
            if min_atr_threshold > 0 and atr_dist < min_atr_threshold:
                continue

            # ---- Regime filter: skip if ATR/price ratio is too low ----
            # ATR/price < threshold indicates a low-volatility, compressed regime
            # where spread costs consume a disproportionate share of potential R.
            # Default: 0.0004 (0.04% of price). Typical range: 0.0003–0.0008.
            if min_atr_ratio > 0 and (atr_dist / current_close) < min_atr_ratio:
                continue

            # ---- Spread filter: skip if current spread exceeds threshold ----
            if max_spread_filter is not None and spread > max_spread_filter:
                continue

            # ---- R-to-cost filter: stop must justify spread cost ----
            # Raised from 3× to min_stop_spread_mult (default 5×).
            # At 2× worst-case spread, a 5× threshold = effectively 10× base spread,
            # ensuring every trade has room to absorb costs before stop is hit.
            if use_structure_stop:
                preview_stop = _fast_struct_stop(i, direction, current_close, atr_dist)
                preview_dist = abs(current_close - preview_stop) * stop_multiplier
            else:
                preview_dist = atr_dist * stop_multiplier
            if spread > 0 and preview_dist < min_stop_spread_mult * spread:
                continue

            liq_str   = f"HTF liq {'low' if direction=='bull' else 'high'} @ {liq_level:.5f}"
            trade_id  = len(trade_log) + 1

            if use_pullback_entry:
                # Retracement entry: wait for price to pull back to retrace_level fraction
                # of the swing from signal_close back toward the liquidity level.
                # Deepened to 65% (from 50%): ensures fill only on meaningful pullbacks,
                # improving entry R/R and reducing MAE from shallow-fill entries.
                # retrace_target = signal_close + retrace_level * (liq_level - signal_close)
                #   bull: liq_level < current_close  → target is below current_close (pullback)
                #   bear: liq_level > current_close  → target is above current_close (pullback)
                retrace_target = current_close + retrace_level * (liq_level - current_close)
                min_retrace    = min_retrace_R * preview_dist   # R proxy from cost filter

                pending_dir  = direction
                pending_meta = {
                    "atr_dist":      atr_dist,
                    "liq_str":       liq_str,
                    "liq_level":     liq_level,
                    "trade_id":      trade_id,
                    "signal_close":  current_close,
                    "signal_bar":    i,
                    "retrace_target": retrace_target,
                    "min_retrace":   min_retrace,
                }
            else:
                # Immediate fill at current close + costs
                entry_price = apply_entry_fill(current_close, direction, spread, slippage_std)
                if use_structure_stop:
                    stop_price = _fast_struct_stop(i, direction, entry_price, atr_dist)
                else:
                    stop_price = (entry_price - atr_dist if direction == "bull"
                                  else entry_price + atr_dist)
                # Apply stop multiplier
                raw_dist = abs(entry_price - stop_price)
                widened  = raw_dist * stop_multiplier
                stop_price = (entry_price - widened if direction == "bull"
                              else entry_price + widened)
                stop_distance = widened
                if stop_distance <= 0:
                    break
                units  = compute_position_size(equity, stop_distance, risk_pct)
                R      = stop_distance
                in_trade = True
                trade = {
                    "trade_id":        trade_id,
                    "timeframe":       "5m exec | 15m/1h/4h context",
                    "entry_time":      current_time,
                    "entry_price":     entry_price,
                    "direction":       direction,
                    "orig_units":      units,
                    "rem_units":       units,
                    "stop_price":      stop_price,
                    "orig_stop_price": stop_price,
                    "R":               R,
                    "entry_reason":    f"{direction.upper()} | {liq_str} | 15m dir | FVG | BOS",
                    "equity_at_entry": equity,
                    "spread_applied":  round(spread + slippage_std, 6),
                    "exit_mode":  exit_mode,
                    "tp1_r":      tp1_r,
                    "tp2_r":      tp2_r,
                    "tp1_hit": False, "tp2_hit": False,
                    "closed": False, "total_pnl": 0.0,
                    "final_equity": None, "last_exit_time": None, "last_exit_price": None,
                }
            break

    # ---- End-of-data: force-close ----
    if in_trade and trade.get("rem_units", 0) > 0:
        exit_price  = float(df5_close[-1])
        fill        = apply_exit_fill(exit_price, trade["direction"], spread, slippage_std)
        rem_units   = trade["rem_units"]
        direction   = trade["direction"]
        entry_price = trade["entry_price"]
        pnl_eod     = (rem_units * (fill - entry_price) if direction == "bull"
                       else rem_units * (entry_price - fill))
        equity     += pnl_eod
        exit_events.append({"trade_id": trade["trade_id"], "event": "EOD",
                             "time": df5_index[-1], "price": round(fill, 5),
                             "units_closed": rem_units, "pnl": round(pnl_eod, 2),
                             "equity_after": round(equity, 2)})
        trade["rem_units"]       = 0.0
        trade["closed"]          = True
        trade["total_pnl"]       = round(trade.get("total_pnl", 0.0) + pnl_eod, 2)
        trade["final_equity"]    = round(equity, 2)
        trade["last_exit_time"]  = df5_index[-1]
        trade["last_exit_price"] = fill
        trade_log.append(_build_trade_row(trade))

    trade_df       = pd.DataFrame(trade_log)
    exit_events_df = pd.DataFrame(exit_events)
    equity_df      = pd.DataFrame(equity_curve)
    if verbose:
        print(f"  Done. Trades: {len(trade_df)}   Final equity: ${equity:,.2f}")
    return trade_df, exit_events_df, equity_df, equity


# =============================================================================
# SIGNAL EXPORT  (Phase 6 — Thinkorswim preparation)
# =============================================================================
def export_signals(trade_df: pd.DataFrame, out_path: str) -> None:
    """
    Export trade signals for paper-trading validation and ThinkScript translation.
    Candle-close confirmation only — no lookahead bias.

    out_path should point to the pair subfolder, e.g.:
        results/realistic/EURUSD/signals.csv
    """
    if trade_df.empty:
        return
    base_cols = ["entry_time", "direction", "entry_price", "orig_stop_price", "R_price"]
    tp_cols   = [c for c in ["tp1_price", "tp2_price"] if c in trade_df.columns]
    sig = trade_df[base_cols + tp_cols].copy()
    sig["exit_time"]    = trade_df["exit_time"]
    sig["total_pnl"]    = trade_df["total_pnl"]
    sig["exit_reason"]  = trade_df.apply(
        lambda r: ("TP2" if r.get("tp2_hit") else
                   "TP1" if r.get("tp1_hit") else "STOP/EOD"), axis=1
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sig.to_csv(out_path, index=False)
    print(f"  Signals: {out_path}")


# =============================================================================
# PERFORMANCE REPORTING
# =============================================================================
def print_performance(trade_df: pd.DataFrame, final_equity: float,
                      pair: str, scenario: str = "") -> None:
    label = f"Trailing — {pair}" + (f" [{scenario}]" if scenario else "")
    sep   = "=" * 60
    print(f"\n{sep}\n  BACKTEST RESULTS  —  {label}\n{sep}")

    if trade_df.empty:
        print(f"  No trades.  Final equity: ${final_equity:,.2f}")
        print(sep); return

    n       = len(trade_df)
    wins    = (trade_df["total_pnl"] > 0).sum()
    wr      = wins / n * 100
    total_ret = (final_equity - STARTING_EQUITY) / STARTING_EQUITY * 100
    total_pnl = trade_df["total_pnl"].sum()

    eq    = np.concatenate([[STARTING_EQUITY], trade_df["equity_after"].values])
    peak  = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak * 100).min())

    tp1_rate = trade_df["tp1_hit"].mean() * 100
    tp2_rate = trade_df["tp2_hit"].mean() * 100 if "tp2_hit" in trade_df.columns else 0.0
    mode_col = trade_df["exit_mode"].iloc[0] if "exit_mode" in trade_df.columns else EXIT_MODE

    print(f"  Trades        : {n}")
    print(f"  Wins / Losses : {wins} / {n - wins}  ({wr:.1f}% win rate)")
    print(f"  TP1 hit rate  : {tp1_rate:.1f}%")
    if mode_col != "single_tp":
        print(f"  TP2 hit rate  : {tp2_rate:.1f}%")
    print(f"  Total PnL     : ${total_pnl:,.2f}")
    print(f"  Total return  : {total_ret:+.2f}%")
    print(f"  Max drawdown  : {max_dd:.2f}%")
    print(f"  Final equity  : ${final_equity:,.2f}")
    print(f"{sep}")
    # Build dynamic TP description from actual levels stored in trade rows
    _t1 = float(trade_df["tp1_price"].iloc[0] - trade_df["entry_price"].iloc[0])
    _R  = float(trade_df["R_price"].iloc[0])
    _tp1_r_actual = round(_t1 / _R, 2) if _R > 0 else TP1_R
    _tp2_r_actual = TP2_R
    if "tp2_price" in trade_df.columns:
        _t2 = float(trade_df["tp2_price"].iloc[0] - trade_df["entry_price"].iloc[0])
        _tp2_r_actual = round(abs(_t2) / _R, 2) if _R > 0 else TP2_R
    _exit_desc = {
        "dual_tp":   f"TP1@{_tp1_r_actual}R(60%) | TP2@{_tp2_r_actual}R(40%) | stop → B/E after TP1",
        "fixed_sl":  f"TP1@{_tp1_r_actual}R(60%) | TP2@{_tp2_r_actual}R(40%) | fixed SL (no B/E)",
        "single_tp": "Single TP@1.2R (100%) | fixed SL",
    }
    print(f"  EXIT: {_exit_desc.get(mode_col, mode_col)}")
    print(f"{sep}")


# =============================================================================
# RISK SWEEP  — find optimal risk_pct under a drawdown constraint
# =============================================================================
def run_risk_sweep(
    frames: dict,
    pair: str,
    spread: float = 0.0,
    slippage_std: float = 0.0,
    risk_levels: list = None,
    dd_limit_pct: float = 10.0,
    exit_mode: str = EXIT_MODE,
) -> pd.DataFrame:
    """
    Run the backtest at multiple risk_pct values and print a comparison table.

    Parameters
    ----------
    frames        : data dict from load_local_data
    pair          : e.g. "EURUSD"
    spread        : spread in price units
    slippage_std  : slippage std dev
    risk_levels   : list of risk fractions to test; default [0.75%..1.75%]
    dd_limit_pct  : max acceptable drawdown in % (positive number); default 10.0
    exit_mode     : forwarded to run_backtest

    Returns
    -------
    DataFrame with one row per risk level
    """
    if risk_levels is None:
        risk_levels = [0.0075, 0.0100, 0.0125, 0.0150, 0.0175, 0.0200]

    W   = 72
    SEP = "=" * W
    print(f"\n{SEP}")
    print(f"  RISK SWEEP  —  {pair}  |  spread={spread:.5f}  slip={slippage_std:.5f}")
    print(f"  DD constraint: max_drawdown ≤ {dd_limit_pct:.1f}%")
    print(SEP)

    rows = []
    for rp in risk_levels:
        print(f"  risk={rp:.2%} ...", end="  ", flush=True)

        trade_df, _, equity_df, final_eq = run_backtest(
            frames,
            spread=spread, slippage_std=slippage_std,
            use_session_filter=True, use_pullback_entry=True,
            use_structure_stop=True, exit_mode=exit_mode,
            risk_pct=rp, verbose=False,
        )

        if trade_df.empty:
            print("no trades — skipped")
            continue

        n   = len(trade_df)
        wins = (trade_df["total_pnl"] > 0).sum()
        wr  = wins / n * 100
        tot_ret = (final_eq - STARTING_EQUITY) / STARTING_EQUITY * 100

        eq_arr = np.concatenate([[STARTING_EQUITY], trade_df["equity_after"].values])
        peak   = np.maximum.accumulate(eq_arr)
        max_dd = float(((eq_arr - peak) / peak * 100).min())

        rets   = trade_df["pnl_pct"].values / 100
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
        gp     = trade_df.loc[trade_df["total_pnl"] > 0, "total_pnl"].sum()
        gl     = abs(trade_df.loc[trade_df["total_pnl"] < 0, "total_pnl"].sum())
        pf     = gp / gl if gl > 0 else np.inf

        rows.append({
            "risk_pct":      rp,
            "trades":        n,
            "win_rate":      wr,
            "total_return":  tot_ret,
            "final_equity":  final_eq,
            "max_drawdown":  max_dd,
            "sharpe":        sharpe,
            "profit_factor": pf,
        })
        flag = "  ✓" if max_dd >= -dd_limit_pct else "  ✗"
        print(f"return={tot_ret:+.1f}%  dd={max_dd:.1f}%  sharpe={sharpe:.2f}{flag}")

    if not rows:
        print("  No results — check data or filters.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  {'risk%':<8} {'trades':>7} {'win%':>7} {'return%':>9} "
          f"{'equity':>12} {'max_dd%':>9} {'sharpe':>8} {'PF':>7}  {'DD ok?':>6}")
    print(f"  {'-'*68}")
    for _, r in df.iterrows():
        ok = "  ✓" if r["max_drawdown"] >= -dd_limit_pct else "  ✗"
        print(f"  {r['risk_pct']:.2%}  "
              f"{int(r['trades']):>7}  "
              f"{r['win_rate']:>6.1f}%  "
              f"{r['total_return']:>+8.2f}%  "
              f"${r['final_equity']:>10,.0f}  "
              f"{r['max_drawdown']:>+8.2f}%  "
              f"{r['sharpe']:>7.3f}  "
              f"{r['profit_factor']:>6.3f}"
              f"{ok}")

    # ── Optimal selection ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    eligible = df[df["max_drawdown"] >= -dd_limit_pct]
    if not eligible.empty:
        best = eligible.loc[eligible["total_return"].idxmax()]
        print(f"  OPTIMAL  (max return subject to DD ≤ {dd_limit_pct:.0f}%)")
        print(f"  risk_pct     = {best['risk_pct']:.2%}")
        print(f"  Total return = {best['total_return']:+.2f}%")
        print(f"  Max drawdown = {best['max_drawdown']:.2f}%")
        print(f"  Sharpe       = {best['sharpe']:.3f}")
        print(f"  Profit factor= {best['profit_factor']:.3f}")
    else:
        best = df.loc[df["total_return"].idxmax()]
        print(f"  WARNING: No risk level satisfies DD ≤ {dd_limit_pct:.0f}%.")
        print(f"  Best available: risk_pct={best['risk_pct']:.2%}  "
              f"return={best['total_return']:+.2f}%  dd={best['max_drawdown']:.2f}%")
        print(f"  Consider reviewing exits or reducing position size before scaling up.")
    print(f"{SEP}\n")

    return df


# =============================================================================
# SINGLE-SCENARIO RUNNER
# =============================================================================
def run_single_scenario(
    frames: dict,
    pair: str,
    start_date,
    end_date,
    chart_tf: str,
    scenario: dict,
    base_spread: float = None,
    base_slip: float = None,
    risk_pct: float = MAX_RISK_PCT,
) -> dict:
    """Run one named scenario (ideal / realistic / worst_case) and return its result dict."""
    if base_spread is None or base_slip is None:
        base_spread, base_slip = get_default_costs(pair)

    label = scenario["label"]
    sp    = base_spread * scenario["spread_mult"]
    sl    = base_slip   * scenario["slip_mult"]

    print(f"\n{'='*64}")
    print(f"  SCENARIO: {label.upper()}  |  {scenario['description']}")
    print(f"  spread={sp:.5f}  slippage_std={sl:.5f}")
    print(f"{'='*64}")

    trade_df, exit_df, equity_df, final_eq = run_backtest(
        frames,
        spread=sp, slippage_std=sl,
        use_session_filter=True,
        use_pullback_entry=True,
        use_structure_stop=True,
        risk_pct=risk_pct,
    )

    if not trade_df.empty:
        trade_df = compute_mfe_mae(trade_df, frames["5m"])

    yearly_df = build_yearly_summary(trade_df, STARTING_EQUITY)
    cont_df   = build_continuation_stats(trade_df, frames["5m"])
    pair_sum  = build_pair_summary(trade_df, equity_df, final_eq,
                                   pair, f"trailing_{label}", start_date, end_date)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "results", label)
    export_all(trade_df, equity_df, yearly_df, cont_df, pair_sum,
               pair, "trailing", out_dir)
    export_signals(trade_df, os.path.join(out_dir, f"signals_{pair}_trailing.csv"))

    print_report(
        trade_df, equity_df, final_eq, yearly_df, cont_df,
        pair, "trailing",
        scenario=label, spread=sp, slippage_std=sl,
        risk_pct=risk_pct, exit_mode=EXIT_MODE,
    )
    print_edge_diagnostics(trade_df, pair, f"trailing_{label}")

    compute_rolling_performance(trade_df, pair, "trailing", window=20)
    print_final_decision(trade_df, final_eq, pair, "trailing", risk_pct=risk_pct)
    plot_results(frames, trade_df, exit_df, equity_df, pair, chart_tf)

    eq_arr = (np.concatenate([[STARTING_EQUITY], trade_df["equity_after"].values])
              if not trade_df.empty else np.array([STARTING_EQUITY]))
    peak   = np.maximum.accumulate(eq_arr)
    max_dd = float(((eq_arr - peak) / peak * 100).min())

    return {
        "trade_df":     trade_df,
        "equity_df":    equity_df,
        "final_equity": final_eq,
        "max_dd":       max_dd,
        "pair_sum":     pair_sum,
    }


# =============================================================================
# STRESS TEST RUNNER  (Phase 4)
# =============================================================================
def run_stress_test(
    frames: dict,
    pair: str,
    start_date,
    end_date,
    chart_tf: str,
    base_spread: float = None,
    base_slip: float = None,
    risk_pct: float = MAX_RISK_PCT,
) -> dict:
    """
    Run 3 scenarios:
      A — Ideal      : spread=0, slippage=0
      B — Realistic  : normal defaults
      C — Worst-case : 2x spread + 2x slippage

    Validation criteria (all required):
      • Profitable in Scenario B
      • Max drawdown < 20% in all scenarios
      • Not catastrophically losing in Scenario C
    """
    if base_spread is None or base_slip is None:
        base_spread, base_slip = get_default_costs(pair)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = {}

    for sc in ALL_SCENARIOS:
        label  = sc["label"]
        sp     = base_spread * sc["spread_mult"]
        sl     = base_slip   * sc["slip_mult"]

        print(f"\n{'='*64}")
        print(f"  SCENARIO: {label.upper()}  |  {sc['description']}")
        print(f"  spread={sp:.5f}  slippage_std={sl:.5f}")
        print(f"{'='*64}")

        trade_df, exit_df, equity_df, final_eq = run_backtest(
            frames,
            spread=sp, slippage_std=sl,
            use_session_filter=True,
            use_pullback_entry=True,
            use_structure_stop=True,
            risk_pct=risk_pct,
        )

        if not trade_df.empty:
            trade_df = compute_mfe_mae(trade_df, frames["5m"])

        yearly_df = build_yearly_summary(trade_df, STARTING_EQUITY)
        cont_df   = build_continuation_stats(trade_df, frames["5m"])
        pair_sum  = build_pair_summary(trade_df, equity_df, final_eq,
                                       pair, f"trailing_{label}", start_date, end_date)

        out_dir = os.path.join(project_root, "results", label)
        export_all(trade_df, equity_df, yearly_df, cont_df, pair_sum,
                   pair, "trailing", out_dir)
        export_signals(trade_df,
                       os.path.join(out_dir, f"signals_{pair}_trailing.csv"))

        print_report(
            trade_df, equity_df, final_eq, yearly_df, cont_df,
            pair, "trailing",
            scenario=label, spread=sp, slippage_std=sl,
            risk_pct=risk_pct, exit_mode=EXIT_MODE,
        )

        # Drawdown check
        eq_arr = (np.concatenate([[STARTING_EQUITY], trade_df["equity_after"].values])
                  if not trade_df.empty else np.array([STARTING_EQUITY]))
        peak   = np.maximum.accumulate(eq_arr)
        max_dd = float(((eq_arr - peak) / peak * 100).min())

        results[label] = {
            "trade_df":     trade_df,
            "equity_df":    equity_df,
            "final_equity": final_eq,
            "max_dd":       max_dd,
            "pair_sum":     pair_sum,
        }

    # ---- Validation summary ----
    print(f"\n{'='*64}")
    print("  STRESS TEST VALIDATION SUMMARY")
    print(f"{'='*64}")
    b = results["realistic"]
    c = results["worst_case"]
    a = results["ideal"]
    for lbl, r in results.items():
        status = "PROFITABLE" if r["final_equity"] > STARTING_EQUITY else "LOSING"
        dd_ok  = r["max_dd"] > -20.0
        print(f"  {lbl:12s}: ${r['final_equity']:>12,.2f}  "
              f"max_dd={r['max_dd']:>6.1f}%  "
              f"{'[DD OK]' if dd_ok else '[DD BREACH]'}  {status}")

    scenario_b_ok = b["final_equity"] > STARTING_EQUITY
    scenario_c_ok = c["final_equity"] > STARTING_EQUITY * 0.80  # not catastrophic
    dd_ok_all     = all(r["max_dd"] > -20.0 for r in results.values())

    print(f"\n  Criteria:")
    print(f"  Profitable in Scenario B : {'PASS' if scenario_b_ok else 'FAIL'}")
    print(f"  Not catastrophic in C    : {'PASS' if scenario_c_ok else 'FAIL'}")
    print(f"  Max DD < 20% all scenarios: {'PASS' if dd_ok_all else 'FAIL'}")

    if scenario_b_ok and scenario_c_ok and dd_ok_all:
        print("\n  [RESULT] Strategy ACCEPTABLE for deployment.")
    elif scenario_b_ok:
        print("\n  [RESULT] Strategy profitable in realistic conditions but review worst-case.")
    else:
        print("\n  [RESULT] Strategy NOT acceptable. Review edge diagnostics above.")
    print(f"{'='*64}")

    plot_scenario_comparison(results, pair, "trailing")
    return results


# =============================================================================
# SCENARIO COMPARISON CHART
# =============================================================================
def plot_scenario_comparison(results: dict, pair: str, strategy: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    colors = {"ideal": "#2ecc71", "realistic": "#4da6ff", "worst_case": "#e74c3c"}

    for label, r in results.items():
        eq_df = r.get("equity_df", pd.DataFrame())
        if eq_df.empty:
            continue
        ax.plot(pd.to_datetime(eq_df["time"]), eq_df["equity"],
                color=colors.get(label, "white"),
                label=f"{label} (${r['final_equity']:,.0f})", linewidth=1.2)

    ax.axhline(STARTING_EQUITY, color="gray", linestyle="--", alpha=0.5, label="Start")
    ax.set_title(f"{pair} — {strategy.title()} Stress Test Comparison",
                 color="white", fontsize=13)
    ax.set_ylabel("Equity ($)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#111111", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    plt.tight_layout()
    if ENABLE_PLOTS:
        plt.show()


# =============================================================================
# CANDLESTICK CHART
# =============================================================================
def make_dark_mpf_style():
    mc = mpf.make_marketcolors(
        up="#2ecc71", down="#e74c3c",
        edge={"up": "#2ecc71", "down": "#e74c3c"},
        wick={"up": "#2ecc71", "down": "#e74c3c"},
        volume={"up": "#2ecc71", "down": "#e74c3c"},
    )
    return mpf.make_mpf_style(
        base_mpf_style="nightclouds", marketcolors=mc,
        facecolor="#000000", edgecolor="white", figcolor="#000000",
        gridcolor="#333333", gridstyle="--",
        rc={
            "axes.labelcolor": "white", "axes.edgecolor": "white",
            "xtick.color": "white", "ytick.color": "white",
            "text.color": "white", "figure.facecolor": "#000000",
            "axes.facecolor": "#000000",
        },
    )


def map_trade_times_to_chart(df_chart, trade_df, col_name):
    mapped = []
    for ts in trade_df[col_name]:
        idx = df_chart.index.searchsorted(ts, side="right") - 1
        mapped.append(df_chart.index[idx] if idx >= 0 else pd.NaT)
    return mapped


def plot_results(frames: dict, trade_df: pd.DataFrame, exit_events_df: pd.DataFrame,
                 equity_df: pd.DataFrame, pair: str, chart_tf: str) -> None:
    if chart_tf not in frames:
        print(f"  chart_tf '{chart_tf}' not in frames — skipping plot.")
        return

    df_chart = frames[chart_tf].copy()
    df_chart = df_chart.iloc[-2000:] if len(df_chart) > 2000 else df_chart
    style    = make_dark_mpf_style()
    addplots = []

    if not trade_df.empty:
        ce = trade_df.copy()
        ce["chart_entry_time"] = map_trade_times_to_chart(df_chart, trade_df, "entry_time")
        ce["chart_exit_time"]  = map_trade_times_to_chart(df_chart, trade_df, "exit_time")

        long_ent  = pd.Series(np.nan, index=df_chart.index)
        short_ent = pd.Series(np.nan, index=df_chart.index)
        long_ex   = pd.Series(np.nan, index=df_chart.index)
        short_ex  = pd.Series(np.nan, index=df_chart.index)

        for _, row in ce.iterrows():
            if pd.notna(row["chart_entry_time"]) and row["chart_entry_time"] in df_chart.index:
                (long_ent if row["direction"] == "bull" else short_ent).loc[row["chart_entry_time"]] = row["entry_price"]
            if pd.notna(row["chart_exit_time"]) and row["chart_exit_time"] in df_chart.index:
                (long_ex if row["direction"] == "bull" else short_ex).loc[row["chart_exit_time"]] = row["exit_price"]

        addplots.extend([
            mpf.make_addplot(long_ent,  type="scatter", marker="^", markersize=80,  color="#2ecc71"),
            mpf.make_addplot(short_ent, type="scatter", marker="v", markersize=80,  color="#e74c3c"),
            mpf.make_addplot(long_ex,   type="scatter", marker="x", markersize=60,  color="#7dffb3"),
            mpf.make_addplot(short_ex,  type="scatter", marker="x", markersize=60,  color="#ff9f9f"),
        ])

    if not equity_df.empty:
        eq = equity_df.copy()
        eq["time"] = pd.to_datetime(eq["time"])
        eq = eq.sort_values("time").drop_duplicates("time", keep="last")
        eq_s    = eq.set_index("time")["equity"]
        merged  = df_chart.index.union(eq_s.index).sort_values()
        eq_s    = eq_s.reindex(merged).ffill()
        eq_line = eq_s.reindex(df_chart.index).ffill().fillna(STARTING_EQUITY)
        addplots.append(mpf.make_addplot(eq_line, panel=1, color="#4da6ff",
                                         width=1.4, ylabel="Equity ($)"))
        panel_ratios = (3, 1)
    else:
        panel_ratios = (1,)

    fig, axes = mpf.plot(
        df_chart, type="candle", style=style,
        title=f"{pair} — Trailing Backtest ({chart_tf})",
        ylabel="Price", volume=False,
        addplot=addplots if addplots else None,
        panel_ratios=panel_ratios, figsize=(18, 9),
        returnfig=True, tight_layout=True,
    )
    for ax in axes:
        ax.set_facecolor("#000000")
        ax.tick_params(colors="white")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    pair, file_path, start_date, end_date, chart_tf = prompt_data_inputs()

    print("\n  Run mode:")
    print("    1 = Ideal only")
    print("    2 = Realistic only")
    print("    3 = Worst-case only")
    print("    4 = Full stress test (all 3 scenarios)")
    mode = input("  Choice [4]: ").strip() or "4"

    frames = load_local_data(file_path, pair, start_date, end_date, chart_tf)
    base_spread, base_slip = get_default_costs(pair)

    # ── TP PROFILE SELECTION ────────────────────────────────────────────────────
    # Default: TP1_R=0.75, TP2_R=1.50  (run_single_scenario uses these via defaults)
    # Alternate profile (higher targets — test after MFE improves):
    #   pass extra_kwargs={"tp1_r": TP1_R_ALT, "tp2_r": TP2_R_ALT} to run_single_scenario
    #   or call run_stress_test(..., tp1_r=TP1_R_ALT, tp2_r=TP2_R_ALT)
    # ───────────────────────────────────────────────────────────────────────────

    # ── MODE 1: Ideal only ────────────────────────────────────────────────────
    if mode == "1":
        run_single_scenario(frames, pair, start_date, end_date, chart_tf,
                            SCENARIO_IDEAL, base_spread, base_slip)

    # ── MODE 2: Realistic only ────────────────────────────────────────────────
    elif mode == "2":
        run_single_scenario(frames, pair, start_date, end_date, chart_tf,
                            SCENARIO_REALISTIC, base_spread, base_slip)

    # ── MODE 3: Worst-case only ───────────────────────────────────────────────
    elif mode == "3":
        run_single_scenario(frames, pair, start_date, end_date, chart_tf,
                            SCENARIO_WORST_CASE, base_spread, base_slip)

    # ── MODE 4: Full stress test ──────────────────────────────────────────────
    elif mode == "4":
        run_stress_test(frames, pair, start_date, end_date, chart_tf)

    else:
        print(f"  Unknown mode '{mode}'. Running stress test by default.")
        run_stress_test(frames, pair, start_date, end_date, chart_tf)
