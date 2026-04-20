# =============================================================================
# data_loader.py
# Shared local-data pipeline for the forex backtesting project.
#
# Replaces Yahoo Finance intraday downloads with local 1-minute CSV/Parquet
# data. All higher timeframes are resampled from 1m so every bar aligns
# perfectly across the strategy timeframes.
#
# PUBLIC API
# ----------
# load_local_data(file_path, pair, start_date, end_date, chart_tf) -> dict
#   Main entry point. Returns a frames dict ready for run_backtest().
#
# build_timeframes(df_1m, chart_tf) -> dict
#   Build 5m/15m/30m/1h/4h (+ optional chart_tf) from a 1m DataFrame.
#
# EXPECTED CSV FORMAT  (column names are case-insensitive)
# ---------------------------------------------------------
# Datetime,Open,High,Low,Close,Volume
# 2020-01-01 00:00:00,1.1210,1.1212,1.1209,1.1211,0
# =============================================================================

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# OHLCV aggregation rule shared by all resamplers
# ---------------------------------------------------------------------------
_OHLCV_AGG = {
    "Open":   "first",
    "High":   "max",
    "Low":    "min",
    "Close":  "last",
    "Volume": "sum",
}

# Resample rule strings for each supported timeframe label
_TF_RULES: dict[str, str] = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
    "1wk": "1W",
    "1mo": "1ME",   # month-end
}


# ===========================================================================
# 1. NORMALISE OHLCV
# ===========================================================================
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names, index type, and data quality.

    Accepts any of these column name styles (case-insensitive, stripped):
        datetime / date / time / timestamp
        open / high / low / close / volume

    Returns a DataFrame with:
        DatetimeIndex (timezone-naive, ascending, no duplicates)
        Columns: Open, High, Low, Close, Volume
    """
    df = df.copy()

    # ---- Flatten MultiIndex if present (yfinance artefact) ----
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ---- Normalise column names to Title case ----
    df.columns = [c.strip() for c in df.columns]
    col_map: dict[str, str] = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("datetime", "date", "time", "timestamp"):
            col_map[col] = "_datetime_"
        elif lc == "open":
            col_map[col] = "Open"
        elif lc == "high":
            col_map[col] = "High"
        elif lc == "low":
            col_map[col] = "Low"
        elif lc == "close":
            col_map[col] = "Close"
        elif lc == "volume":
            col_map[col] = "Volume"
    df = df.rename(columns=col_map)

    # ---- Move datetime column to index if not already ----
    if "_datetime_" in df.columns:
        df = df.set_index("_datetime_")

    # ---- Build proper DatetimeIndex ----
    df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
    # Strip timezone info if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # ---- Add Volume = 0 if missing (forex often has no volume) ----
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # ---- Keep only OHLCV ----
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"data_loader: missing required columns after normalisation: {missing}")

    df = df[required].copy()
    df[required] = df[required].apply(pd.to_numeric, errors="coerce")

    # ---- Clean up ----
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "Datetime"

    return df


# ===========================================================================
# 2. LOAD RAW 1-MINUTE DATA
# ===========================================================================
def load_local_1m(
    file_path: str,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Load 1-minute OHLCV data from a CSV or Parquet file.

    Parameters
    ----------
    file_path  : path to .csv or .parquet file
    start_date : inclusive start  (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
    end_date   : inclusive end    (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)

    Returns
    -------
    Normalised 1m DataFrame with DatetimeIndex.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"data_loader: file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    print(f"  Loading 1m data from: {file_path}")

    if ext == ".parquet":
        raw = pd.read_parquet(file_path)
    elif ext in (".csv", ".txt", ".tsv"):
        # Try to sniff delimiter
        sep = "\t" if ext == ".tsv" else ","
        raw = pd.read_csv(file_path, sep=sep, low_memory=False)
    else:
        raise ValueError(f"data_loader: unsupported file extension '{ext}'. Use .csv or .parquet")

    df = normalize_ohlcv(raw)

    # ---- Date slicing ----
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        # include full end day
        end_ts = pd.Timestamp(end_date)
        if end_ts.hour == 0 and end_ts.minute == 0:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[df.index <= end_ts]

    if df.empty:
        raise ValueError(
            f"data_loader: no 1m rows in [{start_date}, {end_date}]. "
            "Check file content and date range."
        )

    print(f"  1m rows loaded : {len(df):,}")
    print(f"  First bar      : {df.index[0]}")
    print(f"  Last  bar      : {df.index[-1]}")
    return df


# ===========================================================================
# 3. RESAMPLE A SINGLE TIMEFRAME
# ===========================================================================
def resample_timeframe(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample a 1m DataFrame to the given pandas offset rule.

    Parameters
    ----------
    df_1m : 1m normalised DataFrame
    rule  : pandas offset string, e.g. '5min', '1h', '4h'

    Returns
    -------
    Resampled OHLCV DataFrame — incomplete trailing candle dropped via dropna().
    """
    resampled = (
        df_1m
        .resample(rule, label="left", closed="left")
        .agg(_OHLCV_AGG)
        .dropna(subset=["Open", "Close"])
    )
    return resampled


# ===========================================================================
# 4. BUILD ALL TIMEFRAMES FROM 1M
# ===========================================================================
def build_timeframes(df_1m: pd.DataFrame, chart_tf: Optional[str] = None) -> dict:
    """
    Build the full set of strategy timeframes plus an optional chart timeframe.

    Always produces: 5m, 15m, 30m, 1h, 4h
    Optionally adds chart_tf if it is not already in the standard set.

    Returns
    -------
    dict mapping timeframe label -> OHLCV DataFrame
    """
    standard = ["5m", "15m", "30m", "1h", "4h"]
    frames: dict[str, pd.DataFrame] = {}

    for tf in standard:
        rule = _TF_RULES[tf]
        frames[tf] = resample_timeframe(df_1m, rule)
        print(f"  {tf:>4s}: {len(frames[tf]):>8,} candles")

    # ---- Optional chart timeframe ----
    if chart_tf and chart_tf not in frames:
        if chart_tf in _TF_RULES:
            frames[chart_tf] = resample_timeframe(df_1m, _TF_RULES[chart_tf])
            print(f"  {chart_tf:>4s}: {len(frames[chart_tf]):>8,} candles  (chart only)")
        else:
            print(f"  Warning: unknown chart_tf '{chart_tf}' — skipping.")

    return frames


# ===========================================================================
# 5. MAIN ENTRY POINT
# ===========================================================================
def load_local_data(
    file_path:  str,
    pair:       str,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    chart_tf:   Optional[str] = "1h",
) -> dict:
    """
    Full pipeline: load 1m CSV → slice dates → resample all timeframes.

    Parameters
    ----------
    file_path  : path to 1m OHLCV CSV or Parquet
    pair       : display label, e.g. 'EURUSD'
    start_date : 'YYYY-MM-DD'
    end_date   : 'YYYY-MM-DD'
    chart_tf   : timeframe for candlestick chart, e.g. '1h', '4h', '1d'

    Returns
    -------
    dict of DataFrames keyed by timeframe label (5m, 15m, 30m, 1h, 4h, ...)
    """
    print(f"\n{'='*60}")
    print(f"  DATA LOADER  —  {pair}")
    print(f"{'='*60}")

    df_1m = load_local_1m(file_path, start_date, end_date)

    print(f"\n  Building timeframes...")
    frames = build_timeframes(df_1m, chart_tf)

    # Validate strategy-required timeframes have enough bars
    for tf in ["5m", "15m", "1h", "4h"]:
        if len(frames.get(tf, pd.DataFrame())) < 100:
            raise ValueError(
                f"data_loader: timeframe '{tf}' has only "
                f"{len(frames.get(tf, pd.DataFrame()))} bars — "
                "too few for a meaningful backtest. Check date range."
            )

    print(f"{'='*60}\n")
    return frames


# ===========================================================================
# 6. INTERACTIVE PROMPT  (used by strategy scripts when run standalone)
# ===========================================================================
def prompt_data_inputs(default_pair: str = "EURUSD") -> tuple:
    """
    Interactive CLI prompt shared by 2to1.py and trailing.py.

    Returns
    -------
    (pair, file_path, start_date, end_date, chart_tf)
    """
    print("\n" + "=" * 60)
    print("  BACKTEST SETUP")
    print("=" * 60)

    pair = input(f"  Pair / label [{default_pair}]: ").strip().upper() or default_pair

    file_path = ""
    while not file_path:
        file_path = input("  Path to 1m CSV/Parquet file: ").strip()
        # Strip surrounding quotes if user drag-and-dropped
        file_path = file_path.strip('"').strip("'")
        if not os.path.isfile(file_path):
            print(f"  ✗ File not found: {file_path}. Please try again.")
            file_path = ""

    start_date = input("  Start date YYYY-MM-DD [default: start of file]: ").strip() or None
    end_date   = input("  End   date YYYY-MM-DD [default: end of file]:   ").strip() or None

    print("\n  Chart timeframe:")
    print("    1 = 5m   2 = 15m   3 = 30m   4 = 1h   5 = 4h   6 = 1d")
    _aliases = {"1": "5m", "2": "15m", "3": "30m", "4": "1h",
                "5": "4h", "6": "1d", "": "1h"}
    chart_tf = _aliases.get(input("  Choice [4 = 1h]: ").strip(), "1h")

    return pair, file_path, start_date, end_date, chart_tf
