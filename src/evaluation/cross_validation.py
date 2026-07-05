import numpy as np
import pandas as pd
from typing import Iterator, Tuple


def purged_cross_validation(
    t1: pd.Series,
    n_splits: int,
    embargo_pct: float = 0.0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate purged and embargoed train/test index splits for time-series
    cross-validation, per Lopez de Prado (AML, Ch. 7).

    Parameters
    ----------
    t1 : pd.Series
        Index is the observation start time; values are the corresponding
        label end time.Must be sorted ascending by start time, with positional 
        order matching chronological order.
    n_splits : int
        Number of contiguous folds to partition observations into.
    embargo_pct : float, default 0.0
        Fraction of total observations to additionally exclude from
        training immediately after each test fold, on top of overlap
        purging.

    Yields
    ------
    train_indices : np.ndarray
        Positional indices into t1 for the purged, embargoed training set
        for this fold.
    test_indices : np.ndarray
        Positional indices into t1 for this fold's contiguous test set.

    Raises
    ------
    ValueError
        If t1 is not a pd.Series, n_splits < 2, or embargo_pct not in [0, 1).
    """
    if not isinstance(t1, pd.Series):
        raise ValueError("t1 must be a pd.Series")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if not (0.0 <= embargo_pct < 1.0):
        raise ValueError("embargo_pct must be in [0, 1)")

    n = len(t1)
    start_times = np.asarray(t1.index)
    end_times = np.asarray(t1.values)
    indices = np.arange(n)
    embargo = int(n * embargo_pct)

    fold_positions = np.array_split(indices, n_splits)

    for test_indices in fold_positions:
        test_start = start_times[test_indices[0]]
        test_end = end_times[test_indices].max()

        overlap = (
            ((start_times >= test_start) & (start_times <= test_end))
            | ((end_times >= test_start) & (end_times <= test_end))
            | ((start_times <= test_start) & (end_times >= test_end))
        )

        train_mask = ~overlap
        train_mask[test_indices] = False

        if embargo > 0:
            after_test = indices[indices > test_indices[-1]]
            kept_after_test = after_test[train_mask[after_test]]
            if kept_after_test.size > 0:
                first_kept = kept_after_test[0]
                embargo_zone = np.arange(first_kept, min(first_kept + embargo, n))
                train_mask[embargo_zone] = False

        train_indices = indices[train_mask]
        yield train_indices, test_indices