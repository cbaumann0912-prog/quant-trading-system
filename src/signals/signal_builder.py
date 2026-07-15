from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.analysis.performance_analyzer import information_coefficient


class SignalBuilder:
    """
    Signal-agnostic wrapper around a user-supplied ``signal_fn``: handles
    forward-return construction, IC measurement, and rolling-IC robustness
    checks so every strategy signal in this framework is scored the same way.

    Parameters
    ----------
    signal_fn : Callable[[pd.DataFrame, int], pd.Series]
        Vectorized, causal transform called as `signal_fn(data, lookback)`.
        See contract above.
    data : pd.DataFrame
        Feature data for one instrument, indexed by datetime ascending.
        Must include `price_col`. Used by `compute_forward_returns` and as
        the default input to `compute()` inside `compute_ic`/`compute_rolling_ic`.
    price_col : str, default "price"
        Column in `data` used to compute forward returns.
    lookback : int
        Forwarded to `signal_fn` on every call, e.g. 78 for the strategy
        spec's momentum lookback. Must be a deliberate choice, not left at
        a default -- it directly controls signal_fn's rolling window.
    holding_period : int
        Number of bars ahead used for forward returns, e.g. 26 to match the
        strategy spec's shared validation horizon (Section 10).
    return_type : str, default "log"
        "log" or "simple". Log returns are time-additive, consistent with
        `DataLoader.get_returns`.
    ic_method : str, default "spearman"
        Passed through to `information_coefficient`.

    Raises
    ------
    ValueError
        If `price_col` is not in `data.columns`, `data.index` is not a
        monotonic increasing DatetimeIndex, `lookback < 1`, `holding_period < 1`,
        or `return_type` is not "log"/"simple".
    """

    def __init__(
        self,
        signal_fn: Callable[[pd.DataFrame, int], pd.Series],
        data: pd.DataFrame,
        price_col: str = "price",
        lookback: int = 1,
        holding_period: int = 1,
        return_type: str = "log",
        ic_method: str = "spearman",
    ) -> None:
        if price_col not in data.columns:
            raise ValueError(f"price_col '{price_col}' not found in data.columns")
        if not isinstance(data.index, pd.DatetimeIndex) or not data.index.is_monotonic_increasing:
            raise ValueError("data.index must be a monotonic increasing DatetimeIndex")
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if holding_period < 1:
            raise ValueError("holding_period must be >= 1")
        if return_type not in ("log", "simple"):
            raise ValueError("return_type must be 'log' or 'simple'")

        self.signal_fn = signal_fn
        self.data = data
        self.price_col = price_col
        self.lookback = lookback
        self.holding_period = holding_period
        self.return_type = return_type
        self.ic_method = ic_method

        self._forward_returns: Optional[pd.Series] = None

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = self.signal_fn(data, self.lookback)

        if not isinstance(signal, pd.Series):
            raise TypeError(
                f"signal_fn must return a pd.Series, got {type(signal).__name__}"
            )
        if not signal.index.isin(data.index).all():
            raise ValueError("signal_fn returned an index not contained in data.index")

        return signal

    def compute_forward_returns(self) -> pd.Series:
        if self._forward_returns is not None:
            return self._forward_returns

        prices = self.data[self.price_col]
        future_prices = prices.shift(-self.holding_period)

        if self.return_type == "log":
            forward_returns = np.log(future_prices / prices)
        else:
            forward_returns = future_prices / prices - 1

        self._forward_returns = forward_returns
        return forward_returns

    def compute_ic(self, forward_returns: pd.Series) -> float:
        signal = self.compute(self.data)

        aligned = pd.concat(
            [signal.rename("signal"), forward_returns.rename("forward_returns")],
            axis=1,
            join="inner",
        ).dropna()

        if len(aligned) < 2:
            return float("nan")

        return information_coefficient(
            aligned["signal"], aligned["forward_returns"], method=self.ic_method
        )

    def compute_rolling_ic(self, forward_returns: pd.Series, window: int) -> pd.Series:
        if window < 2:
            raise ValueError("window must be >= 2")

        signal = self.compute(self.data)

        aligned = pd.concat(
            [signal.rename("signal"), forward_returns.rename("forward_returns")],
            axis=1,
            join="inner",
        ).dropna()

        n = len(aligned)
        ic_values = []
        ic_index = []

        for start in range(0, n - window + 1, window):
            chunk = aligned.iloc[start : start + window]
            if len(chunk) < 2:
                continue
            if chunk["signal"].nunique() < 2 or chunk["forward_returns"].nunique() < 2:
                continue
            ic = information_coefficient(
                chunk["signal"], chunk["forward_returns"], method=self.ic_method
            )
            if np.isnan(ic):
                continue
            ic_values.append(ic)
            ic_index.append(aligned.index[start])

        return pd.Series(ic_values, index=ic_index)

    def validate_no_lookahead(self, cutoff: pd.Timestamp) -> bool:
        truncated_data = self.data.loc[:cutoff]

        full_signal = self.compute(self.data)
        truncated_signal = self.compute(truncated_data)

        full_slice = full_signal.loc[truncated_signal.index]

        both_nan = full_slice.isna() & truncated_signal.isna()
        equal = full_slice == truncated_signal

        return bool((equal | both_nan).all())
