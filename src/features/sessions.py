"""
Partition of raw 1-minute FX bars into Asian, London, and New York sessions.

Session boundaries are derived from local exchange opens converted to UTC,
so daylight-saving transitions shift the London and New York boundaries as
they do in reality rather than being pinned to a fixed UTC hour. The raw
files carry a fixed UTC offset (:data:`FILE_UTC_OFFSET_HOURS`), applied
before any localization.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

FILE_UTC_OFFSET_HOURS = 5
FIRST_HOUR_MINUTES = 60

LONDON_ZONE = ZoneInfo("Europe/London")
NY_ZONE = ZoneInfo("America/New_York")
LONDON_LOCAL_OPEN = time(8, 0)
NY_LOCAL_OPEN = time(8, 0)

SESSION_NAMES = ["asian", "london", "ny"]


def _local_open_to_utc(dates: pd.DatetimeIndex, local_open: time, zone: ZoneInfo) -> pd.DatetimeIndex:
    naive_local = pd.DatetimeIndex([pd.Timestamp.combine(d.date(), local_open) for d in dates])
    localized = naive_local.tz_localize(zone, nonexistent="shift_forward", ambiguous="NaT")
    return localized.tz_convert("UTC")


def _bucketed_1min_bars(pair: str, data_dir: str | Path) -> pd.DataFrame:
    """
    Shared internal step for every session-return function in this module:
    reads a pair's raw 1-minute bars, converts to true UTC, and labels every
    bar with which non-overlapping open-to-open session block it falls in
    (Asian [Asian open, London open), London [London open, New York open),
    New York [New York open, next Asian open)) and which UTC calendar date
    that block belongs to. See `load_session_returns` for the full
    rationale (fixed-offset file convention, DST handling, why open-to-open
    rather than each market's own quoted hours).

    Returns
    -------
    pd.DataFrame
        columns : ["Datetime" (tz-aware UTC), "Close", "session", "date"],
        one row per raw 1-minute bar that falls inside some session.

    Raises
    ------
    FileNotFoundError
        If `{pair}.csv` does not exist in `data_dir`.
    """
    path = Path(data_dir) / f"{pair}.csv"
    if not path.exists():
        logger.error("Missing 1-minute data file for %s at %s.", pair, path)
        raise FileNotFoundError(f"No 1-minute data file for {pair} at {path}")

    logger.info("Reading 1-minute bars for %s from %s.", pair, path)
    try:
        raw = pd.read_csv(path, usecols=["Datetime", "Close"])
    except ValueError as exc:
        logger.error("Schema mismatch reading %s: %s", path, exc)
        raise ValueError(
            f"{path} does not contain the required columns "
            f"['Datetime', 'Close']: {exc}"
        ) from exc
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.error("Failed to read %s: %s", path, exc)
        raise OSError(f"Could not read 1-minute bars for {pair} at {path}: {exc}") from exc

    logger.debug("%s: %d raw 1-minute bars read.", pair, len(raw))
    raw["Datetime"] = pd.to_datetime(raw["Datetime"], format="%Y%m%d %H%M%S")
    raw = raw.set_index("Datetime").sort_index()

    utc_index = (raw.index + pd.Timedelta(hours=FILE_UTC_OFFSET_HOURS)).tz_localize("UTC")
    raw = raw.set_axis(utc_index, axis=0)

    all_dates = pd.date_range(
        utc_index.normalize().min(), utc_index.normalize().max() + pd.Timedelta(days=1), freq="D", tz="UTC"
    )
    asian_open = all_dates
    london_open = _local_open_to_utc(all_dates, LONDON_LOCAL_OPEN, LONDON_ZONE)
    ny_open = _local_open_to_utc(all_dates, NY_LOCAL_OPEN, NY_ZONE)

    breakpoints = []
    for i in range(len(all_dates)):
        breakpoints.append((asian_open[i], "asian", all_dates[i]))
        breakpoints.append((london_open[i], "london", all_dates[i]))
        breakpoints.append((ny_open[i], "ny", all_dates[i]))
    breakpoints.sort(key=lambda item: item[0])

    boundary_index = pd.DatetimeIndex([bp[0] for bp in breakpoints])
    boundary_labels = np.array([bp[1] for bp in breakpoints])
    boundary_dates = pd.DatetimeIndex([bp[2] for bp in breakpoints])

    bucket_idx = boundary_index.searchsorted(raw.index, side="right") - 1
    valid = bucket_idx >= 0

    bucketed = pd.DataFrame({
        "Datetime": raw.index[valid],
        "Close": raw["Close"].to_numpy()[valid],
        "session": boundary_labels[bucket_idx[valid]],
        "date": boundary_dates[bucket_idx[valid]],
        "session_open": boundary_index[bucket_idx[valid]],
    })
    return bucketed


def load_session_returns(pair: str, start: str, end: str, data_dir: str | Path) -> pd.DataFrame:
    """
    Splits a pair's raw 1-minute bars into Asian/London/New York session log
    returns, one row per UTC calendar day, using non-overlapping open-to-open
    blocks rather than each market's own quoted close time.

    Parameters
    ----------
    pair : str
        e.g. "EURUSD". Must have a `{pair}.csv` file in `data_dir`.
    start, end : str
        ISO date strings, inclusive, applied to the resulting session-date
        index (not to the raw 1-minute timestamps before session slicing).
    data_dir : str | Path
        Directory containing `{pair}.csv`.

    Returns
    -------
    pd.DataFrame
        index : DatetimeIndex, one row per UTC calendar date with at least
            one session present, ascending.
        columns : ["asian_return", "london_return", "ny_return"], log
            return of session close vs. session open for that block.

    Raises
    ------
    FileNotFoundError
        If `{pair}.csv` does not exist in `data_dir`.
    """
    bucketed = _bucketed_1min_bars(pair, data_dir)

    session_series = {}
    for session_name in SESSION_NAMES:
        sub = bucketed.loc[bucketed["session"] == session_name]
        grouped = sub.groupby("date")["Close"]
        open_price = grouped.first()
        close_price = grouped.last()
        session_series[f"{session_name}_return"] = np.log(close_price / open_price)

    combined = pd.DataFrame(session_series).sort_index()
    combined.index = combined.index.tz_localize(None)
    combined = combined.loc[(combined.index >= pd.Timestamp(start)) & (combined.index <= pd.Timestamp(end))]
    return combined


def load_session_returns_with_first_hour(
    pair: str,
    start: str,
    end: str,
    data_dir: str | Path,
    first_hour_minutes: int = FIRST_HOUR_MINUTES,
) -> pd.DataFrame:
    """
    Same session partition as `load_session_returns`, plus a first-hour
    return for each session (log return of the session's own open vs. its
    close at `first_hour_minutes` minutes after that session's open),
    letting predictor/target legs of a lead-lag test be built from either
    the full session or just the opening slice of it.

    Parameters
    ----------
    pair, start, end, data_dir : see `load_session_returns`.
    first_hour_minutes : int, default 60
        Width of the opening slice, in minutes from that session's own open.

    Returns
    -------
    pd.DataFrame
        index : DatetimeIndex, one row per UTC calendar date.
        columns : ["asian_return", "london_return", "ny_return",
            "asian_first_hour_return", "london_first_hour_return",
            "ny_first_hour_return"].

    Raises
    ------
    FileNotFoundError
        If `{pair}.csv` does not exist in `data_dir`.
    """
    bucketed = _bucketed_1min_bars(pair, data_dir)
    bucketed["minutes_since_session_open"] = (
        bucketed["Datetime"] - bucketed["session_open"]
    ).dt.total_seconds() / 60.0

    session_series = {}
    for session_name in SESSION_NAMES:
        sub = bucketed.loc[bucketed["session"] == session_name]
        grouped = sub.groupby("date")["Close"]
        open_price = grouped.first()
        close_price = grouped.last()
        session_series[f"{session_name}_return"] = np.log(close_price / open_price)

        first_hour_sub = sub.loc[sub["minutes_since_session_open"] < first_hour_minutes]
        grouped_fh = first_hour_sub.groupby("date")["Close"]
        fh_open = grouped_fh.first()
        fh_close = grouped_fh.last()
        session_series[f"{session_name}_first_hour_return"] = np.log(fh_close / fh_open)

    combined = pd.DataFrame(session_series).sort_index()
    combined.index = combined.index.tz_localize(None)
    combined = combined.loc[(combined.index >= pd.Timestamp(start)) & (combined.index <= pd.Timestamp(end))]
    return combined
