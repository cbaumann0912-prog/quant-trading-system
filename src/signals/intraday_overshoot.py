"""
Intraday overshoot signal: the framework's primary validated strategy.

Builds per-session overshoot measures from raw 1-minute bars and gates them
on a walk-forward GARCH conditional volatility estimate. The volatility
estimate is refit strictly on data available at each decision point --
see :func:`walk_forward_conditional_vol` -- because an in-sample GARCH fit
would leak the very volatility clustering the signal conditions on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.garch import fit_garch
from src.features.sessions import FILE_UTC_OFFSET_HOURS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

NY_ZONE = "America/New_York"


def load_ny_minute_bars(pair: str, data_dir: str | Path) -> pd.DataFrame:
    """Read a pair's raw 1-minute bars and label each with its New York date
    and minute-of-day.

    Parameters
    ----------
    pair : str
        Six-character pair code.
    data_dir : str or Path
        Directory holding `{pair}.csv` with Datetime and Close columns.

    Returns
    -------
    pd.DataFrame
        Columns: c (close), d (NY calendar date, tz-naive), m (minutes since
        NY midnight). One row per raw 1-minute bar.

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
    parsed = pd.to_datetime(raw["Datetime"], format="%Y%m%d %H%M%S")
    ny = pd.DatetimeIndex(
        (parsed + pd.Timedelta(hours=FILE_UTC_OFFSET_HOURS)).dt.tz_localize("UTC")
    ).tz_convert(NY_ZONE)

    return pd.DataFrame({
        "c": raw["Close"].to_numpy(),
        "d": ny.normalize().tz_localize(None),
        "m": ny.hour.values * 60 + ny.minute.values,
    })


def walk_forward_conditional_vol(
    daily_ret: pd.Series,
    garch_min_train: int = 500,
) -> pd.Series:
    """GARCH(1,1) conditional volatility path fit walk-forward by year.

    Parameters
    ----------
    daily_ret : pd.Series
        Daily log returns indexed by date.
    garch_min_train : int, optional
        Minimum prior observations required before a year is fit. Years with
        less history are left NaN. Defaults to 500.

    Returns
    -------
    pd.Series
        Conditional volatility indexed by date, NaN dropped.
    """
    conditional_vol = pd.Series(np.nan, index=daily_ret.index)

    for year in sorted(daily_ret.index.year.unique()):
        train = daily_ret[daily_ret.index.year < year]
        if len(train) < garch_min_train:
            continue

        g = fit_garch(train)
        hist = daily_ret[daily_ret.index.year <= year]
        eps = (hist - train.mean()).to_numpy()

        var = np.empty(len(eps))
        var[0] = eps.var()
        for i in range(1, len(eps)):
            var[i] = g["omega"] + g["alpha"] * eps[i - 1] ** 2 + g["beta"] * var[i - 1]

        path = pd.Series(np.sqrt(var), index=hist.index)
        conditional_vol.loc[conditional_vol.index.year == year] = path[
            path.index.year == year
        ]

    return conditional_vol.dropna()


def build_overshoot_sessions(
    pair: str,
    data_dir: str | Path,
    start: str,
    end: str,
    ks: list[float],
    entry_delays: list[int],
    scan_open: int,
    scan_close: int,
    exit_min: int,
    vol_ratio_min_obs: int = 250,
    garch_min_train: int = 500,
) -> pd.DataFrame:
    """Build the per-day session frame for the intraday overshoot strategy.

    Parameters
    ----------
    pair : str
        Six-character pair code.
    data_dir : str or Path
        Directory holding the raw 1-minute CSVs.
    start, end : str
        Inclusive date bounds applied after construction, so the expanding
        vol-ratio warmup can use all available prior history.
    ks : list of float
        Threshold multiples of the session sigma.
    entry_delays : list of int
        Minutes after the crossing bar at which to record an entry price.
    scan_open, scan_close, exit_min : int
        Minutes since NY midnight for the scan start, scan end and exit.
    vol_ratio_min_obs : int, optional
        Minimum observations before the expanding session/daily vol ratio is
        defined. Defaults to 250.
    garch_min_train : int, optional
        Passed to `walk_forward_conditional_vol`. Defaults to 500.

    Returns
    -------
    pd.DataFrame
        Indexed by session date, with columns sigma, open, exit_px and, per k,
        `t_{k}` (minutes into the scan at the crossing), `disp_{k}` (signed log
        displacement at the crossing) and `px_{k}_d{delay}` for each delay.
        Rows where no crossing occurred carry NaN in that k's columns.
    """
    bars = load_ny_minute_bars(pair, data_dir)

    daily_close = bars.groupby("d")["c"].last()
    daily_ret = np.log(daily_close / daily_close.shift(1)).dropna()
    conditional_vol = walk_forward_conditional_vol(daily_ret, garch_min_train)

    scan = bars[(bars["m"] >= scan_open) & (bars["m"] <= exit_min)].sort_values(["d", "m"])
    sess_open = scan.groupby("d")["c"].first()
    scan_close_px = scan[scan["m"] <= scan_close].groupby("d")["c"].last()
    exit_px = scan.groupby("d")["c"].last()
    sess_ret = np.log(scan_close_px / sess_open)

    ratio = (
        sess_ret.expanding(vol_ratio_min_obs).std()
        / daily_ret.reindex(sess_ret.index).expanding(vol_ratio_min_obs).std()
    ).shift(1)
    sigma_sess = conditional_vol.reindex(sess_ret.index).shift(1) * ratio

    out = pd.DataFrame({
        "sigma": sigma_sess.values,
        "open": sess_open.values,
        "exit_px": exit_px.reindex(sess_open.index).values,
    }, index=sess_open.index)
    out.index.name = "date"

    grouped = {
        d: (g["m"].to_numpy(), g["c"].to_numpy())
        for d, g in scan[scan["m"] <= scan_close].groupby("d")
    }

    for k in ks:
        cols = {f"t_{k}": [], f"disp_{k}": []}
        for delay in entry_delays:
            cols[f"px_{k}_d{delay}"] = []

        for date in out.index:
            s = out["sigma"].get(date, np.nan)
            mm, cc = grouped.get(date, (np.array([]), np.array([])))

            hit_i = -1
            if np.isfinite(s) and s > 0 and len(cc):
                hit = np.abs(np.log(cc / cc[0])) > k * s
                if hit.any():
                    hit_i = int(np.argmax(hit))

            if hit_i < 0:
                cols[f"t_{k}"].append(np.nan)
                cols[f"disp_{k}"].append(np.nan)
                for delay in entry_delays:
                    cols[f"px_{k}_d{delay}"].append(np.nan)
                continue

            cols[f"t_{k}"].append(mm[hit_i] - scan_open)
            cols[f"disp_{k}"].append(np.log(cc[hit_i] / cc[0]))
            for delay in entry_delays:
                j = np.searchsorted(mm, mm[hit_i] + delay)
                cols[f"px_{k}_d{delay}"].append(cc[j] if j < len(cc) else np.nan)

        for name, vals in cols.items():
            out[name] = vals

    return out.loc[(out.index >= start) & (out.index <= end)]


def overshoot_trades(
    sessions: dict[str, pd.DataFrame],
    k: float,
    delay: int,
) -> pd.DataFrame:
    """Fade trades from per-pair session frames.

    Parameters
    ----------
    sessions : dict of str to pd.DataFrame
        Per-pair output of `build_overshoot_sessions`.
    k : float
        Threshold multiple to read.
    delay : int
        Entry delay in minutes to read.

    Returns
    -------
    pd.DataFrame
        Columns pair, date, ret. One row per triggered trade.
    """
    frames = []
    for pair, frame in sessions.items():
        px = f"px_{k}_d{delay}"
        x = frame.dropna(subset=[px, "exit_px", f"disp_{k}"])
        frames.append(pd.DataFrame({
            "pair": pair,
            "date": x.index,
            "ret": (-np.sign(x[f"disp_{k}"]) * np.log(x["exit_px"] / x[px])).values,
        }))

    return pd.concat(frames, ignore_index=True)
