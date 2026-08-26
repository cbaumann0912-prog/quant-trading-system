from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import numpy as np
import pandas as pd

from src.analysis.performance_analyzer import PerformanceAnalyzer

DEV_START = "20110101"
DEV_END = "20231231"

DATA_DIR = REPO_ROOT.parent / "data"
OUT_PATH = REPO_ROOT / "paper" / "tables" / ".day72_data_section_scan.json"

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
]

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

if len(sys.argv) > 1:
    targets = [p.upper() for p in sys.argv[1:]]
else:
    targets = list(PAIRS)

if OUT_PATH.exists():
    results = json.loads(OUT_PATH.read_text())
else:
    results = {}

for pair in targets:
    path = DATA_DIR / f"{pair}.csv"
    frame = pd.read_csv(
        path,
        dtype={"Open": "float64", "High": "float64", "Low": "float64",
               "Close": "float64", "Volume": "float64"},
    )

    stamps = frame["Datetime"].values.astype("U15")
    del frame["Datetime"]
    days = stamps.astype("U8")

    in_dev = (days >= DEV_START) & (days <= DEV_END)
    stamps = stamps[in_dev]
    days = days[in_dev]
    opens = frame["Open"].values[in_dev]
    highs = frame["High"].values[in_dev]
    lows = frame["Low"].values[in_dev]
    closes = frame["Close"].values[in_dev]
    volumes = frame["Volume"].values[in_dev]
    del frame

    order = np.argsort(stamps, kind="stable")
    stamps = stamps[order]
    days = days[order]
    opens = opens[order]
    highs = highs[order]
    lows = lows[order]
    closes = closes[order]
    volumes = volumes[order]

    record = {}
    record["minute_rows"] = int(stamps.size)
    record["first_stamp"] = str(stamps[0])
    record["last_stamp"] = str(stamps[-1])

    duplicated = pd.Index(stamps).duplicated()
    record["duplicate_stamps"] = int(duplicated.sum())
    dup_months = sorted({str(d)[:6] for d in days[duplicated]})
    record["duplicate_months"] = dup_months

    record["missing_close"] = int(np.isnan(closes).sum())
    record["nonpositive_prices"] = int(
        ((opens <= 0) | (highs <= 0) | (lows <= 0) | (closes <= 0)).sum()
    )
    record["ohlc_violations"] = int(
        (
            (highs < lows)
            | (highs < opens)
            | (highs < closes)
            | (lows > opens)
            | (lows > closes)
        ).sum()
    )

    record["volume_max"] = float(np.nanmax(volumes))
    record["volume_nonzero"] = int((volumes != 0).sum())

    flat = int((highs == lows).sum())
    record["flat_bars"] = flat
    record["flat_bars_pct"] = round(100.0 * flat / stamps.size, 4)

    close_series = pd.Series(closes)
    daily_close = close_series.groupby(days).last()
    minute_counts = close_series.groupby(days).size()
    daily_index = pd.to_datetime(daily_close.index, format="%Y%m%d")
    daily_close.index = daily_index
    minute_counts.index = daily_index

    record["daily_obs"] = int(daily_close.size)
    record["daily_first"] = str(daily_index.min().date())
    record["daily_last"] = str(daily_index.max().date())

    gaps = np.diff(daily_index.values).astype("timedelta64[D]").astype(int)
    record["max_daily_gap_days"] = int(gaps.max())
    record["gap_gt_4_days"] = int((gaps > 4).sum())

    weekday = daily_index.dayofweek
    record["median_minutes_by_weekday"] = {
        WEEKDAY_NAMES[d]: float(minute_counts[weekday == d].median())
        for d in range(7)
        if (weekday == d).any()
    }
    record["daily_obs_by_weekday"] = {
        WEEKDAY_NAMES[d]: int((weekday == d).sum())
        for d in range(7)
        if (weekday == d).any()
    }

    log_returns = np.log(daily_close).diff().dropna()
    ret_weekday = log_returns.index.dayofweek
    record["mean_abs_logret_by_weekday"] = {
        WEEKDAY_NAMES[d]: round(float(log_returns[ret_weekday == d].abs().mean()), 8)
        for d in range(7)
        if (ret_weekday == d).any()
    }

    sunday_obs = int((weekday == 6).sum())
    record["sunday_daily_obs"] = sunday_obs
    record["sunday_share_pct"] = round(100.0 * sunday_obs / daily_close.size, 4)

    analyzer_all = PerformanceAnalyzer(log_returns)
    record["ann_factor_all_days"] = round(float(analyzer_all.compute_ann_factor()), 4)
    record["sharpe_all_days"] = round(float(analyzer_all.compute_sharpe()), 6)
    record["total_log_return_all_days"] = round(float(log_returns.sum()), 6)
    record["std_all_days"] = round(float(log_returns.std()), 8)

    weekday_close = daily_close[weekday < 5]
    weekday_returns = np.log(weekday_close).diff().dropna()
    analyzer_weekday = PerformanceAnalyzer(weekday_returns)
    record["ann_factor_weekday_only"] = round(float(analyzer_weekday.compute_ann_factor()), 4)
    record["sharpe_weekday_only"] = round(float(analyzer_weekday.compute_sharpe()), 6)
    record["total_log_return_weekday_only"] = round(float(weekday_returns.sum()), 6)
    record["std_weekday_only"] = round(float(weekday_returns.std()), 8)
    record["weekday_obs"] = int(weekday_returns.size)

    record["sharpe_hardcoded_252"] = round(
        float(log_returns.mean() / log_returns.std() * np.sqrt(252.0)), 6
    )

    results[pair] = record
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"{pair}: {record['minute_rows']} minute rows, {record['daily_obs']} daily closes, "
          f"ann_factor {record['ann_factor_all_days']}")
