from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MONTH_END_DAYS = 2


def month_to_date_return(daily_log_return: pd.Series) -> pd.Series:
    """Cumulative log return within each calendar month, through ``t-1``.

    The ``shift(1)`` is what keeps the signal causal: the reading available on
    day ``t`` uses returns through the previous close only, so it can be acted
    on at ``t`` without lookahead.

    Known quirk, preserved from the original construction: the shift is applied
    *before* the groupby, so the first observation of each month carries the
    previous month's final return rather than starting at zero. The signal
    stays causal, but on day one of a month it is not a month-to-date reading.
    This affects only non-month-end rows, since ``month_end`` flags the last
    days of a month, so it does not touch the cell H1 was fitted on. Documented
    rather than corrected, because changing it would move the recorded figures
    in ``month_end_fx_flow_h1_result.md``.
    """
    period = daily_log_return.index.to_period("M")
    return daily_log_return.shift(1).groupby(period).cumsum()


def hedging_need_signal(daily_log_return: pd.Series) -> pd.Series:
    """Signed hedging-demand proxy: ``-sign(month-to-date return)``.

    A price-insensitive hedger who has watched a position appreciate must sell
    into the month-end fix to restore the hedge ratio, which predicts reversal
    of the month's move.

    This is a **proxy and a known misspecification**. Melvin & Prins identify
    hedging direction from the foreign *equity* return: a rising European
    equity book leaves a US investor's EUR hedge too small regardless of where
    EUR/USD went. Substituting the month-to-date FX return does not measure
    hedging need. Strategy 5's H1 was significant with the sign inverted, and
    the spec flagged this substitution in advance as the most likely route to a
    false negative. See
    ``research/strategies/s05_month_end_fx_flow/month_end_fx_flow_h1_result.md``.

    Returns
    -------
    pd.Series
        Values in ``{-1.0, 0.0, +1.0}``. Zeros mark months with no accumulated
        move and are dropped before fitting.
    """
    return -np.sign(month_to_date_return(daily_log_return))


def month_end_flag(
    index: pd.DatetimeIndex,
    month_end_days: int = DEFAULT_MONTH_END_DAYS,
) -> pd.Series:
    """Indicator for the last ``month_end_days`` trading days of each month.

    Counted from the end of each month's observed rows rather than from the
    calendar, so a month whose final days are missing still flags its own last
    observations.
    """
    period = index.to_period("M")
    frame = pd.Series(0, index=index)
    return (frame.groupby(period).cumcount(ascending=False) < month_end_days).astype(float)


def build_interaction_panel(
    staged_pairs: dict[str, pd.DataFrame],
    pairs: list[str],
    fix_col: str,
    control_col: str = "control_return",
    month_end_days: int = DEFAULT_MONTH_END_DAYS,
) -> pd.DataFrame:
    """Stack pairs into the long panel the three-way interaction is fitted on.

    Each pair contributes two rows per date: one for the fix window
    (``fix = 1``) and one for the control window (``fix = 0``). The control
    window is what lets the test distinguish a month-end fix effect from a
    month-end effect that happens to show up at any time of day.

    Parameters
    ----------
    staged_pairs : dict[str, pd.DataFrame]
        Per-pair frames indexed by London date, carrying window returns and
        ``daily_log_return``.
    pairs : list[str]
        Pairs to include, in order.
    fix_col : str
        Which fix-window column to use as the treatment return.
    control_col : str
        Column supplying the ``fix = 0`` rows.
    month_end_days : int
        Days at month end treated as the flow window.

    Returns
    -------
    pd.DataFrame
        Columns ``date``, ``y``, ``signal``, ``month_end``, ``fix``, sorted by
        date, with null and zero-signal rows removed.

    Notes
    -----
    Month-end flows hit every pair on the same date, so ten pairs on one date
    are nowhere near ten independent observations. Analytic OLS errors on this
    panel understate the true standard error by roughly 1.6x; use a block
    bootstrap over date blocks instead.
    """
    frames = []
    for pair in pairs:
        df = staged_pairs[pair]
        signal = hedging_need_signal(df["daily_log_return"])
        month_end = month_end_flag(df.index, month_end_days=month_end_days)

        for col, is_fix in [(fix_col, 1.0), (control_col, 0.0)]:
            frames.append(pd.DataFrame({
                "date": df.index,
                "y": df[col].to_numpy(),
                "signal": signal.to_numpy(),
                "month_end": month_end.to_numpy(),
                "fix": is_fix,
            }))

    built = pd.concat(frames, ignore_index=True).dropna(
        subset=["y", "signal", "month_end"]
    )
    built = built.loc[built["signal"] != 0.0]
    return built.sort_values("date").reset_index(drop=True)
