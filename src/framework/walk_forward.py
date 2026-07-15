from __future__ import annotations

from typing import Any, Callable, Dict, List

import pandas as pd


class WalkForwardValidator:
    """
    Rolling-window walk-forward split generator with an embargo gap between
    train and test.

    Skeleton scope (Day 45): window generation and embargo only. `signal_fn`
    is stored but not yet invoked -- `run()` returns the train/test slices
    for each window without scoring. Wiring this into SignalBuilder /
    PerformanceAnalyzer for actual OOS scoring is deliberately deferred to a
    later day rather than guessed at here.

    Parameters
    ----------
    signal_fn : Callable
        Reserved for future use (scoring integration). Not called by this
        skeleton. No signature is enforced yet.
    data : pd.DataFrame
        Must have a sorted, monotonic increasing DatetimeIndex. No column
        requirements at this stage since signal_fn isn't invoked.
    n_windows : int
        Number of rolling windows to generate. Must be >= 1.
    train_years : int
        Length of each training window, in calendar years (via
        `pd.DateOffset(years=...)`). Must be >= 1.
    test_months : int
        Length of each test window, in calendar months. Also the amount
        `train_start` advances between consecutive windows (rolling, not
        expanding -- train length stays fixed at `train_years`). Must be >= 1.
    embargo_days : int, default 5
        Calendar-day gap enforced between `train_end` and `test_start`, sized
        to clear any lookback-window feature so no test-set feature reaches
        back into training data. This is a single directional gap (train ->
        test); no purging of trailing training rows is performed by this
        class. Must be >= 0.

    Raises
    ------
    ValueError
        If `data.index` is not a monotonic increasing DatetimeIndex, or if
        `n_windows` / `train_years` / `test_months` < 1, or `embargo_days` < 0.
    """

    def __init__(
        self,
        signal_fn: Callable,
        data: pd.DataFrame,
        n_windows: int,
        train_years: int,
        test_months: int,
        embargo_days: int = 5,
    ) -> None:
        if not isinstance(data.index, pd.DatetimeIndex) or not data.index.is_monotonic_increasing:
            raise ValueError("data.index must be a monotonic increasing DatetimeIndex")
        if n_windows < 1:
            raise ValueError("n_windows must be >= 1")
        if train_years < 1:
            raise ValueError("train_years must be >= 1")
        if test_months < 1:
            raise ValueError("test_months must be >= 1")
        if embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")

        self.signal_fn = signal_fn
        self.data = data
        self.n_windows = n_windows
        self.train_years = train_years
        self.test_months = test_months
        self.embargo_days = embargo_days

    def generate_windows(self) -> List[Dict[str, Any]]:
        """
        Generate rolling-window train/embargo/test boundaries in calendar
        time.

        Window i (0-indexed), rolling (train_start advances, train length
        fixed at train_years):

            train_start = data_start + DateOffset(months=test_months * i)
            train_end   = train_start + DateOffset(years=train_years)
            embargo_end = train_end + Timedelta(days=embargo_days)
            test_start  = embargo_end
            test_end    = test_start + DateOffset(months=test_months)

        Returns
        -------
        list[dict]
            One dict per window with keys:
            train_start, train_end, embargo_end, test_start, test_end
            (all pd.Timestamp).

        Raises
        ------
        ValueError
            If the requested n_windows does not fit inside the available
            data range -- fails loudly rather than silently returning fewer
            windows than asked for.
        """
        data_start = self.data.index[0]
        data_end = self.data.index[-1]

        windows: List[Dict[str, Any]] = []
        for i in range(self.n_windows):
            train_start = data_start + pd.DateOffset(months=self.test_months * i)
            train_end = train_start + pd.DateOffset(years=self.train_years)
            embargo_end = train_end + pd.Timedelta(days=self.embargo_days)
            test_start = embargo_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            if test_end > data_end:
                raise ValueError(
                    f"Requested n_windows={self.n_windows} does not fit: "
                    f"window {i} requires data through {test_end.date()}, "
                    f"but data only extends to {data_end.date()}. "
                    f"Reduce n_windows, train_years, or test_months."
                )

            windows.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "embargo_end": embargo_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )

        return windows

    def _slice(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Half-open slice [start, end) against the actual DatetimeIndex."""
        mask = (self.data.index >= start) & (self.data.index < end)
        return self.data.loc[mask]

    def run(self) -> Dict[str, Any]:
        """
        Generate windows and slice train/test data for each. Scoring
        (signal_fn invocation, IC/Sharpe) is not implemented in this
        skeleton -- see class docstring.

        Returns
        -------
        dict
            {
                "window_results": [
                    {window_idx, train_start, train_end, embargo_end,
                     test_start, test_end, n_train, n_test}, ...
                ],
                "aggregate_stats": {
                    "n_windows", "avg_n_train", "avg_n_test",
                },
            }
        """
        windows = self.generate_windows()
        window_results: List[Dict[str, Any]] = []

        for idx, w in enumerate(windows):
            train_df = self._slice(w["train_start"], w["train_end"])
            test_df = self._slice(w["test_start"], w["test_end"])

            window_results.append(
                {
                    "window_idx": idx,
                    "train_start": w["train_start"],
                    "train_end": w["train_end"],
                    "embargo_end": w["embargo_end"],
                    "test_start": w["test_start"],
                    "test_end": w["test_end"],
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                }
            )

        n_windows = len(window_results)
        aggregate_stats = {
            "n_windows": n_windows,
            "avg_n_train": sum(r["n_train"] for r in window_results) / n_windows,
            "avg_n_test": sum(r["n_test"] for r in window_results) / n_windows,
        }

        return {"window_results": window_results, "aggregate_stats": aggregate_stats}
