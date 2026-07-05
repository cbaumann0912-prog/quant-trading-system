import numpy as np
import pandas as pd
import pytest

from src.evaluation.cross_validation import purged_cross_validation


def _make_t1(n_days: int, holding_period: int) -> pd.Series:
    start = np.arange(1, n_days + 1)
    end = start + holding_period
    return pd.Series(end, index=start)


def _windows_overlap(s_i: int, e_i: int, s_j: int, e_j: int) -> bool:
    cond1 = s_j <= s_i <= e_j
    cond2 = s_j <= e_i <= e_j
    cond3 = s_i <= s_j <= e_j <= e_i
    return cond1 or cond2 or cond3


def test_no_leakage_between_train_test():
    t1 = _make_t1(n_days=100, holding_period=5)
    starts = t1.index.values
    ends = t1.values

    splits = list(purged_cross_validation(t1, n_splits=5, embargo_pct=0.0))
    
    assert len(splits) == 5

    for train_indices, test_indices in splits:
        for i in train_indices:
            for j in test_indices:
                assert not _windows_overlap(
                    starts[i], ends[i], starts[j], ends[j]
                ), f"leakage: train day {starts[i]} overlaps test day {starts[j]}"

    _, fold3_test = splits[2]
    fold3_train, _ = splits[2]

    assert list(fold3_test) == list(range(40, 60))

    purged_days = set(range(36, 41)) | set(range(61, 66))
    remaining_days = set(range(1, 101)) - set(range(41, 61)) - purged_days
    train_days = set(starts[fold3_train])

    assert train_days == remaining_days


def test_embargo_gap_respected():
    n_days = 100
    holding_period = 5
    embargo_pct = 0.05
    t1 = _make_t1(n_days=n_days, holding_period=holding_period)
    starts = t1.index.values
    splits_no_embargo = list(
        purged_cross_validation(t1, n_splits=5, embargo_pct=0.0)
    )
    splits_embargo = list(
        purged_cross_validation(t1, n_splits=5, embargo_pct=embargo_pct)
    )
    expected_embargo = int(n_days * embargo_pct)

    assert expected_embargo > 0

    for (train_no_emb, test_indices), (train_emb, _) in zip(
        splits_no_embargo, splits_embargo
    ):
        after_test = [i for i in train_no_emb if i > test_indices[-1]]
        after_test_embargo = [i for i in train_emb if i > test_indices[-1]]
        if not after_test:
            continue
        first_resume_no_embargo = min(after_test)
        first_resume_with_embargo = (
            min(after_test_embargo) if after_test_embargo else None
        )
        if first_resume_with_embargo is None:
            continue
        gap = first_resume_with_embargo - first_resume_no_embargo
    
        assert gap >= expected_embargo


def test_n_splits_correct():
    t1 = _make_t1(n_days=100, holding_period=5)

    for n_splits in (2, 4, 5, 10):
        splits = list(purged_cross_validation(t1, n_splits=n_splits, embargo_pct=0.0))
        
        assert len(splits) == n_splits

        all_test_indices = np.concatenate([test for _, test in splits])

        assert len(all_test_indices) == len(np.unique(all_test_indices))
        assert set(all_test_indices) == set(range(100))

        for test_indices in [test for _, test in splits]:
            assert np.all(np.diff(test_indices) == 1)


def test_invalid_inputs_raise():
    t1 = _make_t1(n_days=10, holding_period=2)

    with pytest.raises(ValueError):
        list(purged_cross_validation([1, 2, 3], n_splits=3))

    with pytest.raises(ValueError):
        list(purged_cross_validation(t1, n_splits=1))

    with pytest.raises(ValueError):
        list(purged_cross_validation(t1, n_splits=3, embargo_pct=1.0))

    with pytest.raises(ValueError):
        list(purged_cross_validation(t1, n_splits=3, embargo_pct=-0.1))