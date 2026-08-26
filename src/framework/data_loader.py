"""
Loading and leakage-safe splitting of the raw FX price data.

This module is the framework's only boundary with the on-disk minute bars
and with FRED's public CSV endpoint. Every read is guarded and logged here
rather than in the research modules, so that a failure surfaces as a
diagnosable error naming the pair and path instead of a bare pandas
traceback three call frames deep.

The embargo logic in :meth:`DataLoader.train_test_split` is the leakage
control the rest of the framework depends on; see the class docstring for
the precise definition of an embargo "day".
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_DIR = Path(
    os.environ.get("QUANT_DATA_DIR", Path(__file__).resolve().parents[3] / "data")
)

SUPPORTED_PAIRS = {
    "EURUSD", "GBPUSD", "USDJPY",
    "USDCHF", "AUDUSD", "USDCAD",
    "NZDUSD", "EURGBP", "EURJPY",
    "EURCHF",
}


class DataLoader:
    """
    Loads daily FX close prices for a fixed set of pairs over [start, end]
    and produces a leakage-safe train/test split.

    Parameters
    ----------
    pairs : list[str]
        Must be a subset of {"EURUSD", "GBPUSD", "USDJPY"}. Any other pair
        (e.g. "USDCHF") raises ValueError
    start, end : str
        ISO date strings, e.g. "2015-01-01". Inclusive on both ends.
    embargo_days : int, default 5
        Number of rows (trading days present in the resampled daily index,
        NOT literal calendar days -- weekends are already dropped by
        resample("D").last().dropna(), so a "day" here means a row in the
        index) enforced as a gap between the end of train and the start of
        test.
    data_dir : str | Path
        Directory containing "{pair}.csv" files with columns
        [Datetime, Open, High, Low, Close, Volume], Datetime formatted
        "%Y%m%d %H%M%S".
    """

    def __init__(
        self,
        pairs: list[str],
        start: str,
        end: str,
        embargo_days: int = 5,
        data_dir: str | Path = DEFAULT_DATA_DIR,
    ) -> None:
        invalid = set(pairs) - SUPPORTED_PAIRS
        if invalid:
            raise ValueError(
                f"Unsupported pair(s) {sorted(invalid)}. DataLoader only "
                f"supports {sorted(SUPPORTED_PAIRS)}."
            )
        if not pairs:
            raise ValueError("Must specify at least one pair.")
        if embargo_days < 0:
            raise ValueError("embargo_days must be >= 0.")

        self.pairs = list(pairs)
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.embargo_days = embargo_days
        self.data_dir = Path(data_dir)

        self._data: pd.DataFrame | None = None
        self._returns: dict[bool, pd.DataFrame] = {}

    def load(self) -> pd.DataFrame:
        """
        Reads {pair}.csv for each pair, resamples minute data to daily close
        (last observation per day), aligns all pairs on a shared sorted,
        unique DatetimeIndex, and restricts to [start, end].

        Returns
        -------
        pd.DataFrame
            index   : DatetimeIndex, daily, ascending, unique
            columns : one per pair, values = daily close price
        """
        if self._data is not None:
            logger.debug("load() cache hit: %d rows already resolved.", len(self._data))
            return self._data

        logger.info(
            "Loading %d pair(s) %s from %s over [%s, %s].",
            len(self.pairs), self.pairs, self.data_dir,
            self.start.date(), self.end.date(),
        )
        load_started = time.perf_counter()

        series = {}
        for pair in self.pairs:
            path = self.data_dir / f"{pair}.csv"
            if not path.exists():
                logger.error("Missing data file for %s at %s.", pair, path)
                raise FileNotFoundError(f"No data file for {pair} at {path}")

            try:
                df = pd.read_csv(path, usecols=["Datetime", "Close"])
            except ValueError as exc:
                logger.error("Schema mismatch reading %s: %s", path, exc)
                raise ValueError(
                    f"{path} does not contain the required columns "
                    f"['Datetime', 'Close']. Got a schema error: {exc}"
                ) from exc
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                logger.error("Failed to read %s: %s", path, exc)
                raise OSError(f"Could not read price data for {pair} at {path}: {exc}") from exc

            logger.debug("%s: read %d raw minute rows from %s.", pair, len(df), path.name)

            stamps = df["Datetime"].values.astype("U15")
            order = np.argsort(stamps, kind="stable")
            day_keys = stamps[order].astype("U8")
            closes = df["Close"].values[order]

            daily = pd.Series(closes).groupby(day_keys).last()
            daily.index = pd.to_datetime(daily.index, format="%Y%m%d")
            daily.index.name = "Datetime"
            series[pair] = daily
            logger.debug(
                "%s: resampled to %d daily closes spanning [%s, %s].",
                pair, len(daily), daily.index.min().date(), daily.index.max().date(),
            )

        data = pd.DataFrame(series)
        pre_filter_n = len(data)
        data = data.loc[(data.index >= self.start) & (data.index <= self.end)]
        data = data.sort_index()

        if data.index.duplicated().any():
            logger.error("Duplicate timestamps present after resampling.")
            raise ValueError(
                "Duplicate timestamps after resampling -- investigate source data."
            )
        if data.empty:
            logger.error(
                "Date filter emptied the panel: %d rows before filtering, 0 after.",
                pre_filter_n,
            )
            raise ValueError(
                f"No data in range [{self.start.date()}, {self.end.date()}] "
                f"for pairs {self.pairs}."
            )

        n_missing = int(data.isna().sum().sum())
        if n_missing:
            logger.warning(
                "Aligned panel contains %d missing values across %d pairs; "
                "downstream statistics will see NaNs unless handled.",
                n_missing, data.shape[1],
            )

        logger.info(
            "Loaded panel: %d rows x %d pairs, [%s, %s], in %.2fs (%d rows dropped by date filter).",
            data.shape[0], data.shape[1], data.index.min().date(), data.index.max().date(),
            time.perf_counter() - load_started, pre_filter_n - len(data),
        )

        self._data = data
        return self._data

    def split_train_test(self, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Strictly temporal train/test split with an embargo gap between them.

        The embargo is subtracted from the sample BEFORE applying test_ratio,
        so test_ratio always describes a fraction of the post-embargo usable
        range.

        Returns
        -------
        (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
            Copies, not views -- mutating one must not corrupt self._data.

        Raises
        ------
        ValueError
            If test_ratio is out of (0, 1), or if embargo_days + the
            requested test size leaves no room for a non-empty training set.
        """
        if not (0.0 < test_ratio < 1.0):
            raise ValueError("test_ratio must be in (0, 1).")

        data = self.load()
        n = len(data)

        usable_n = n - self.embargo_days
        if usable_n <= 0:
            raise ValueError(
                f"embargo_days={self.embargo_days} consumes the entire "
                f"sample (n={n}). Reduce embargo or extend the date range."
            )

        test_n = int(round(usable_n * test_ratio))
        if test_n <= 0:
            raise ValueError("test_ratio too small -- resulting test set is empty.")

        train_end_idx = n - self.embargo_days - test_n
        if train_end_idx <= 0:
            raise ValueError(
                "test_ratio + embargo_days leaves no room for a training set."
            )

        train_df = data.iloc[:train_end_idx].copy()
        test_df = data.iloc[train_end_idx + self.embargo_days:].copy()

        train_end_pos = data.index.get_loc(train_df.index[-1])
        test_start_pos = data.index.get_loc(test_df.index[0])
        assert test_start_pos - train_end_pos >= self.embargo_days, (
            "Embargo contract violated -- this should be unreachable."
        )

        return train_df, test_df

    def get_window(
        self,
        train_index: pd.DatetimeIndex,
        test_index: pd.DatetimeIndex,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Given explicit boundary indices, returns (train_data, test_data) as
        actual sliced DataFrame copies.
        """
        data = self.load()
        train_df = data.loc[train_index].copy()
        test_df = data.loc[test_index].copy()
        return train_df, test_df

    def get_returns(self, log: bool = True) -> pd.DataFrame:
        """
        Single-period returns for every pair, same index as load() minus the
        first row (no return defined for it).

        Parameters
        ----------
        log : bool, default True
            If True, log returns: log(P_t / P_t-1). Time-additive across
            periods required for SignalBuilder to aggregate returns over
            arbitrary lookback/holding windows with plain summation, and
            closer to normal for small daily FX moves, which most downstream
            hypothesis tests assume.
            If False, simple returns: P_t / P_t-1 - 1. Simple returns
            are additive across a portfolio of positions at a single point
            in time; log returns are not. Any future cross-pair portfolio
            aggregation must use simple returns.
        """
        if log not in self._returns:
            data = self.load()
            if log:
                r = np.log(data / data.shift(1))
            else:
                r = data.pct_change(fill_method=None)
            self._returns[log] = r.dropna(how="all")
        return self._returns[log]


_RATE_SERIES = {
    "us": "IR3TIB01USM156N",
    "ea": "IR3TIB01EZM156N",
    "uk": "IR3TIB01GBM156N",
    "jp": "IR3TIB01JPM156N",
    "ch": "IR3TIB01CHM156N",
    "ca": "IR3TIB01CAM156N",
    "au": "IR3TIB01AUM156N",
    "nz": "IR3TIB01NZM156N",
}
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

_FRED_MAX_ATTEMPTS = 3
_FRED_BACKOFF_SECONDS = 2.0
_FRED_USER_AGENT = "quant-research-framework/1.0"


def fetch_rate_differentials(data_dir: str | Path) -> None:
    """
    Refreshes the four OECD 3-month interbank rate CSVs
    ({region}_3m_interbank.csv) in `data_dir` from FRED's public CSV
    endpoint (no API key required). Monthly, not seasonally adjusted.
    """
    out_dir = Path(data_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create output directory %s: %s", out_dir, exc)
        raise

    failures: list[str] = []
    for region, series_id in _RATE_SERIES.items():
        url = _FRED_CSV_URL.format(series_id=series_id)
        df = None

        for attempt in range(1, _FRED_MAX_ATTEMPTS + 1):
            try:
                df = pd.read_csv(url, storage_options={"User-Agent": _FRED_USER_AGENT})
                break
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                logger.warning(
                    "FRED fetch failed for %s (%s), attempt %d/%d: %s",
                    region, series_id, attempt, _FRED_MAX_ATTEMPTS, exc,
                )
                if attempt == _FRED_MAX_ATTEMPTS:
                    failures.append(region)
                else:
                    time.sleep(_FRED_BACKOFF_SECONDS * attempt)
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                logger.error(
                    "FRED returned unparseable content for %s (%s): %s",
                    region, series_id, exc,
                )
                failures.append(region)
                break

        if df is None:
            continue

        if df.shape[1] != 2:
            logger.error(
                "Unexpected FRED schema for %s: expected 2 columns, got %d (%s). "
                "Leaving the existing local file untouched.",
                region, df.shape[1], list(df.columns),
            )
            failures.append(region)
            continue

        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        n_missing = int(df["value"].isna().sum())
        if n_missing:
            logger.warning(
                "%s: %d of %d observations are missing after numeric coercion.",
                region, n_missing, len(df),
            )

        target = out_dir / f"{region}_3m_interbank.csv"
        try:
            df.to_csv(target, index=False)
        except OSError as exc:
            logger.error("Could not write %s: %s", target, exc)
            failures.append(region)
            continue

        logger.info(
            "Refreshed %s: %d observations through %s -> %s.",
            region, len(df), df["date"].max().date(), target.name,
        )

    if failures:
        raise RuntimeError(
            f"Rate differential refresh failed for {sorted(failures)}. "
            f"Local files for these regions were left unchanged; the data "
            f"directory is now in a mixed-vintage state. Re-run before "
            f"computing any carry signal."
        )
