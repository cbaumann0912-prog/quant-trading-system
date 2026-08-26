"""
SignalBuilder: assembly of lagged signals and aligned forward returns.

The alignment contract enforced here is the framework's main defence
against lookahead bias. A signal observed at time t may only be paired with
returns realized strictly after t, and exposure must be lagged relative to
the return it earns. Most backtest overstatement traces to a violation of
exactly this rule, so the lag is applied centrally rather than left to
each individual signal module.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.analysis.performance_analyzer import information_coefficient

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


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
        """
        Evaluates the wrapped signal function and enforces its output contract.

        Parameters
        ----------
        data : pd.DataFrame
            Price panel to evaluate on. Passed explicitly rather than taken
            from `self.data` so that `validate_no_lookahead` can re-run the
            signal on a truncated panel.

        Returns
        -------
        pd.Series
            The raw signal, indexed by a subset of `data.index`.

        Raises
        ------
        TypeError
            If `signal_fn` returns something other than a Series.
        ValueError
            If the returned index is not contained in `data.index`. An index
            containing timestamps absent from the input is the signature of a
            signal that has fabricated or reindexed observations, which would
            silently misalign against forward returns downstream.
        """
        signal = self.signal_fn(data, self.lookback)

        if not isinstance(signal, pd.Series):
            raise TypeError(
                f"signal_fn must return a pd.Series, got {type(signal).__name__}"
            )
        if not signal.index.isin(data.index).all():
            raise ValueError("signal_fn returned an index not contained in data.index")

        return signal

    def compute_forward_returns(self) -> pd.Series:
        """
        Returns the forward return realized over `holding_period` steps.

        Returns
        -------
        pd.Series
            Indexed like `self.data`. The value at time t is the return from
            t to t + holding_period, so the final `holding_period` entries
            are NaN by construction -- their realizations have not occurred.

        Notes
        -----
        This is the alignment contract the whole framework rests on. The
        return at t is deliberately *forward* looking while the signal at t
        is *backward* looking, which is what makes pairing them a valid
        out-of-sample test rather than a contemporaneous correlation.

        The trailing NaNs must not be dropped and backfilled by callers: doing
        so pulls realized returns backwards in time and produces exactly the
        lookahead bias this construction exists to prevent.

        Cached after first call, which is safe only because `self.data` is
        treated as immutable for the lifetime of the builder.
        """
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
        """
        Full-sample information coefficient between signal and forward returns.

        Parameters
        ----------
        forward_returns : pd.Series
            Typically the output of `compute_forward_returns`.

        Returns
        -------
        float
            Rank correlation (Spearman by default), or NaN if fewer than two
            aligned non-null observations survive.

        Notes
        -----
        Spearman is the default because it is invariant to monotone
        transformation and far less sensitive to the outliers that dominate
        FX return tails, where a single large move can drive a Pearson IC on
        its own.

        This is an in-sample descriptive statistic. Overlapping forward
        windows make consecutive observations dependent, so the naive
        standard error implied by the sample size is too small and this IC
        must not be converted to a p-value without accounting for that
        overlap.
        """
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
        """
        Information coefficient computed on consecutive non-overlapping blocks.

        Parameters
        ----------
        forward_returns : pd.Series
            Forward returns aligned to the signal.
        window : int
            Block length in observations. Must be >= 2.

        Returns
        -------
        pd.Series
            One IC per block, indexed by the timestamp that opens the block.
            Blocks are skipped when either side is constant, since rank
            correlation is undefined there.

        Raises
        ------
        ValueError
            If `window < 2`.

        Notes
        -----
        Blocks step by `window`, not by 1, so they do not overlap. This is
        the deliberate difference from a conventional rolling statistic: the
        resulting ICs are approximately independent, which is what makes
        their dispersion a usable estimate of IC stability and their mean
        divided by their standard deviation interpretable as an information
        ratio. A stride-1 rolling IC would produce a smooth, highly
        autocorrelated series whose apparent stability is an artifact of the
        overlap.

        The cost is resolution: with n observations only n // window blocks
        exist, so short samples yield few points and a correspondingly noisy
        dispersion estimate.
        """
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
            chunk = aligned.iloc[start: start + window]
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
        """
        Checks that the signal at times <= `cutoff` does not change when data
        after `cutoff` is withheld.

        Parameters
        ----------
        cutoff : pd.Timestamp
            Boundary at which the panel is truncated.

        Returns
        -------
        bool
            True if the signal computed on the truncated panel matches the
            corresponding slice of the signal computed on the full panel,
            treating NaN-in-both as agreement.

        Notes
        -----
        This is a falsification test, and its logic is worth stating
        precisely: if a signal value at time t <= cutoff changes depending on
        whether future data was present, the signal used the future. That is
        direct evidence of lookahead.

        The converse does not hold. Passing at one cutoff does not prove the
        signal is leakage-free -- a leak that only manifests at other cutoffs,
        or one that enters through a parameter fitted on the full sample
        before the builder was ever constructed, will pass this check.
        Treat a pass as failure to detect leakage, not as a certificate of
        its absence, and run it at several cutoffs.
        """
        truncated_data = self.data.loc[:cutoff]

        full_signal = self.compute(self.data)
        truncated_signal = self.compute(truncated_data)

        full_slice = full_signal.loc[truncated_signal.index]

        both_nan = full_slice.isna() & truncated_signal.isna()
        equal = full_slice == truncated_signal

        return bool((equal | both_nan).all())
