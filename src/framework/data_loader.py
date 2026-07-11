from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED_PAIRS = {"EURUSD", "GBPUSD", "USDJPY"}


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
        data_dir: str | Path = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data",
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
            return self._data

        series = {}
        for pair in self.pairs:
            path = self.data_dir / f"{pair}.csv"
            if not path.exists():
                raise FileNotFoundError(f"No data file for {pair} at {path}")

            df = pd.read_csv(path, usecols=["Datetime", "Close"])
            df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
            df = df.set_index("Datetime").sort_index()

            daily = df["Close"].resample("D").last().dropna()
            series[pair] = daily

        data = pd.DataFrame(series)
        data = data.loc[(data.index >= self.start) & (data.index <= self.end)]
        data = data.sort_index()

        if data.index.duplicated().any():
            raise ValueError(
                "Duplicate timestamps after resampling -- investigate source data."
            )
        if data.empty:
            raise ValueError(
                f"No data in range [{self.start.date()}, {self.end.date()}] "
                f"for pairs {self.pairs}."
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
}
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_rate_differentials(data_dir: str | Path) -> None:
    """
    Refreshes the four OECD 3-month interbank rate CSVs
    ({region}_3m_interbank.csv) in `data_dir` from FRED's public CSV
    endpoint (no API key required). Monthly, not seasonally adjusted.
    """
    for region, series_id in _RATE_SERIES.items():
        df = pd.read_csv(_FRED_CSV_URL.format(series_id=series_id))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
        df.to_csv(Path(data_dir) / f"{region}_3m_interbank.csv", index=False)
