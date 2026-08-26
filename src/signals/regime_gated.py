"""
Factory for regime-gated wrappers around a base signal.

Gating multiplies a base signal by a regime condition. This adds at least
one threshold parameter, so a gated signal that outperforms its ungated
parent has been given extra fitting freedom and must clear a
correspondingly higher bar before the improvement is credited to the
regime hypothesis rather than to the added flexibility.
"""
from typing import Callable

import numpy as np
import pandas as pd

from src.signals.mean_reversion import _ladder_step, price_zscore_signal
from src.signals.momentum import momentum_signal

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def make_regime_gated_signal_fn(
    regime: pd.Series,
    reversion_lookback: int,
    entry_thresholds: tuple[float, float, float] = (2.0, 2.5, 3.0),
    exit_z: float = 0.5,
    time_stop: int = 26,
) -> Callable[[pd.DataFrame, int], pd.Series]:
    """
    Full-spec `signal_fn(data, lookback) -> pd.Series` for `SignalBuilder`,
    combining `momentum_signal` (turbulent) with the stateful 3-rung
    mean-reversion ladder (calm)

    Parameters
    ----------
    regime : pd.Series
        Output of `src.features.regime_classifier.classify_regime`.
        Reindexed onto `data.index`; any bar without a matching regime label
        is treated like deadzone (hold-through, no new entries triggered by
        regime alone).
    reversion_lookback : int
        Rolling window for the calm leg's z-score (spec: 26 trading days).
    entry_thresholds : tuple[float, float, float], default (2.0, 2.5, 3.0)
        Ladder rung thresholds on |price_z| (spec parameter table).
    exit_z : float, default 0.5
        Ladder reversion-to-band exit threshold.
    time_stop : int, default 26
        Ladder max holding period in bars from the rung-1 entry.

    Returns
    -------
    Callable[[pd.DataFrame, int], pd.Series]
        A `signal_fn` usable directly as `SignalBuilder(signal_fn=...)`. The
        `lookback` argument passed in by `SignalBuilder` is used for the
        momentum leg (spec: 78 trading days); the ladder uses
        `reversion_lookback` instead, bound into this closure.
    """

    def signal_fn(data: pd.DataFrame, lookback: int) -> pd.Series:
        """
        Regime-gated exposure: momentum in turbulent regimes, a mean-reversion
        ladder in calm ones, hold-through otherwise.

        Parameters
        ----------
        data : pd.DataFrame
            Price panel supplied by `SignalBuilder`.
        lookback : int
            Momentum leg lookback, passed through by `SignalBuilder`. The
            reversion leg uses `reversion_lookback` from the enclosing
            closure instead.

        Returns
        -------
        pd.Series
            Exposure per bar, NaN where no position is held.

        Notes
        -----
        Bars whose regime label is missing are treated as deadzone: an
        existing position is held but no new entry is triggered. Treating a
        missing label as "no regime" rather than defaulting to one of the two
        legs keeps an absent classification from silently becoming a trading
        decision.

        The loop is sequential rather than vectorized because ladder state
        (rung count, direction, entry index) depends on its own history, and
        the time stop is measured from the rung-1 entry. Rewriting this as a
        vectorized expression is the most likely way to introduce lookahead
        into this module.
        """
        momentum = momentum_signal(data, lookback)
        z = price_zscore_signal(data, reversion_lookback)
        aligned_regime = regime.reindex(data.index)

        n = len(data)
        exposure = np.full(n, np.nan)

        active_type = None
        rungs, direction, entry_idx = 0, 0, None

        for i in range(n):
            r = aligned_regime.iloc[i]

            if r == "turbulent":
                if active_type == "calm":
                    rungs, direction, entry_idx = 0, 0, None
                active_type = "turbulent"
                exposure[i] = momentum.iloc[i]
                continue

            if r == "calm":
                active_type = "calm"
                exposure[i], rungs, direction, entry_idx = _ladder_step(
                    z.iloc[i], rungs, direction, entry_idx, i, entry_thresholds, exit_z, time_stop
                )
                continue

            if active_type == "turbulent":
                exposure[i] = momentum.iloc[i]
            elif active_type == "calm":
                exposure[i], rungs, direction, entry_idx = _ladder_step(
                    z.iloc[i], rungs, direction, entry_idx, i, entry_thresholds, exit_z, time_stop
                )

        return pd.Series(exposure, index=data.index)

    return signal_fn
