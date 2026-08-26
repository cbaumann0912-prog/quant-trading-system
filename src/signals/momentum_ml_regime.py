"""
Momentum outcome labelling and non-overlapping subsampling.

`non_overlapping_subsample` exists to address a specific inference problem:
momentum labels built from overlapping forward windows are serially
dependent, so the effective sample size is far smaller than the row count.
Tests run on the overlapping panel report standard errors that are too
small and p-values that are too optimistic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_LOOKBACK = 26
DEFAULT_HOLDING = 5


def momentum_signal_outcome(
    daily_prices: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
    holding: int = DEFAULT_HOLDING,
) -> pd.DataFrame:
    """Pair each day's trailing return with the forward return that follows it.

    Both legs are built from the cumulative log-return index so that the
    trailing and forward windows share no observations:

        trailing_return_t = cumsum_t - cumsum_{t-lookback}
        forward_return_t  = cumsum_{t+holding} - cumsum_t

    The windows meet at ``t`` but do not overlap: ``forward_start`` is the day
    after ``trailing_end``. An earlier construction of strategy 2 overlapped
    them and produced a spurious IC near +0.30, so the returned frame carries
    the window boundaries explicitly for auditing.

    Parameters
    ----------
    daily_prices : pd.Series
        Daily closes indexed by date, ascending, no gaps beyond weekends.
    lookback : int
        Trailing window length in observations.
    holding : int
        Forward window length in observations.

    Returns
    -------
    pd.DataFrame
        One row per scored day with columns ``date``, ``trailing_return``,
        ``forward_return``, ``trailing_start``, ``trailing_end``,
        ``forward_start``, ``forward_end``. Length is
        ``len(daily_prices) - 1 - lookback - holding``.

    Notes
    -----
    Consecutive rows overlap by ``lookback - 1`` days on the trailing leg and
    ``holding - 1`` on the forward leg. Any test treating rows as independent
    will overstate significance; see
    ``research/strategies/s02_momentum_ml_regime/`` for a case where the naive
    p-value understates the permutation p-value by 37x.
    """
    log_returns = np.log(daily_prices / daily_prices.shift(1)).dropna()
    cumsum = log_returns.cumsum()
    n = len(cumsum)

    records = []
    for i in range(lookback, n - holding):
        trailing_return = cumsum.iloc[i] - cumsum.iloc[i - lookback]
        forward_return = cumsum.iloc[i + holding] - cumsum.iloc[i]
        records.append({
            "date": cumsum.index[i],
            "trailing_return": trailing_return,
            "forward_return": forward_return,
            "trailing_start": cumsum.index[i - lookback],
            "trailing_end": cumsum.index[i],
            "forward_start": cumsum.index[i + 1],
            "forward_end": cumsum.index[i + holding],
        })

    return pd.DataFrame(records)


def non_overlapping_subsample(
    outcomes: pd.DataFrame,
    holding: int = DEFAULT_HOLDING,
) -> pd.DataFrame:
    """Take every ``holding``-th row so no two rows share a forward window.

    This is the control arm of strategy 2's Test A: with the overlap removed by
    construction, a naive Spearman p-value and a permutation p-value agree.
    """
    return outcomes.iloc[::holding]
