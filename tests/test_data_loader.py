import pandas as pd
import pytest

from src.framework.data_loader import DataLoader, SUPPORTED_PAIRS, DEFAULT_DATA_DIR

DATA_DIR = DEFAULT_DATA_DIR

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
START = "2011-01-01"
END = "2016-01-01"
EMBARGO_DAYS = 5

@pytest.fixture(scope="module")
def loader():
    return DataLoader(
        pairs=PAIRS,
        start=START,
        end=END,
        embargo_days=EMBARGO_DAYS,
        data_dir=DATA_DIR,
    )


def test_load_returns_dataframe(loader):
    data = loader.load()

    assert isinstance(data, pd.DataFrame)
    assert set(data.columns) == set(PAIRS)
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.is_monotonic_increasing
    assert data.index.is_unique
    assert not data.empty


def test_train_test_split_respects_embargo(loader):
    train_df, test_df = loader.split_train_test(test_ratio=0.2)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(train_df) > 0
    assert len(test_df) > 0

    full_index = loader.load().index
    train_end_pos = full_index.get_loc(train_df.index[-1])
    test_start_pos = full_index.get_loc(test_df.index[0])

    assert test_start_pos - train_end_pos >= EMBARGO_DAYS
    assert train_df.index.max() < test_df.index.min()


def test_train_test_split_ratio_applies_to_post_embargo_range(loader):
    test_ratio = 0.25
    train_df, test_df = loader.split_train_test(test_ratio=test_ratio)

    n = len(loader.load())
    usable_n = n - EMBARGO_DAYS
    expected_test_n = int(round(usable_n * test_ratio))

    assert len(test_df) == expected_test_n


def test_log_returns_shape(loader):
    data = loader.load()
    returns = loader.get_returns(log=True)

    assert isinstance(returns, pd.DataFrame)
    assert set(returns.columns) == set(PAIRS)
    assert len(returns) == len(data) - 1
    assert returns.index.max() <= data.index.max()
    assert returns.index.min() > data.index.min()


def test_get_returns_simple_vs_log_differ(loader):
    log_returns = loader.get_returns(log=True)
    simple_returns = loader.get_returns(log=False)

    assert not log_returns.equals(simple_returns)


def test_unsupported_pair_raises():
    with pytest.raises(ValueError):
        DataLoader(pairs=["USDSEK"], start=START, end=END, data_dir=DATA_DIR)


def test_empty_pairs_raises():
    with pytest.raises(ValueError):
        DataLoader(pairs=[], start=START, end=END, data_dir=DATA_DIR)


def test_supported_pairs_frozen_to_universe():
    assert SUPPORTED_PAIRS == {
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
        "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
    }


def test_original_three_pairs_still_supported():
    assert {"EURUSD", "GBPUSD", "USDJPY"}.issubset(SUPPORTED_PAIRS)


def _write_synthetic_pair_csv(tmp_path, pair: str, n_days: int = 20) -> None:
    """One row per calendar day (2024-01-01 onward), which is sufficient
    since DataLoader.load() resamples to daily last-close anyway."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({
        "Datetime": [d.strftime("%Y%m%d") + " 120000" for d in dates],
        "Open": 1.10,
        "High": 1.11,
        "Low": 1.09,
        "Close": [1.10 + 0.001 * i for i in range(n_days)],
        "Volume": 0,
    })
    df.to_csv(tmp_path / f"{pair}.csv", index=False)


def test_negative_embargo_days_raises(tmp_path):
    _write_synthetic_pair_csv(tmp_path, "EURUSD")
    with pytest.raises(ValueError):
        DataLoader(
            pairs=["EURUSD"], start="2024-01-01", end="2024-01-20",
            embargo_days=-1, data_dir=tmp_path,
        )


def test_missing_data_file_raises_file_not_found(tmp_path):
    loader = DataLoader(pairs=["EURUSD"], start="2024-01-01", end="2024-01-20", data_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_empty_date_range_raises_value_error(tmp_path):
    _write_synthetic_pair_csv(tmp_path, "EURUSD")
    loader = DataLoader(
        pairs=["EURUSD"], start="2030-01-01", end="2030-01-31", data_dir=tmp_path,
    )
    with pytest.raises(ValueError):
        loader.load()


def test_split_train_test_ratio_out_of_bounds_raises(tmp_path):
    _write_synthetic_pair_csv(tmp_path, "EURUSD", n_days=30)
    loader = DataLoader(pairs=["EURUSD"], start="2024-01-01", end="2024-01-30", data_dir=tmp_path)

    with pytest.raises(ValueError):
        loader.split_train_test(test_ratio=0.0)
    with pytest.raises(ValueError):
        loader.split_train_test(test_ratio=1.0)


def test_split_train_test_embargo_consumes_whole_sample_raises(tmp_path):
    _write_synthetic_pair_csv(tmp_path, "EURUSD", n_days=10)
    loader = DataLoader(
        pairs=["EURUSD"], start="2024-01-01", end="2024-01-10",
        embargo_days=50, data_dir=tmp_path,
    )
    with pytest.raises(ValueError):
        loader.split_train_test(test_ratio=0.5)


def test_split_train_test_too_small_ratio_raises(tmp_path):
    _write_synthetic_pair_csv(tmp_path, "EURUSD", n_days=10)
    loader = DataLoader(
        pairs=["EURUSD"], start="2024-01-01", end="2024-01-10",
        embargo_days=1, data_dir=tmp_path,
    )
    with pytest.raises(ValueError):
        loader.split_train_test(test_ratio=0.001)


def test_split_train_test_leaves_no_room_for_training_raises(tmp_path):
    _write_synthetic_pair_csv(tmp_path, "EURUSD", n_days=10)
    loader = DataLoader(
        pairs=["EURUSD"], start="2024-01-01", end="2024-01-10",
        embargo_days=3, data_dir=tmp_path,
    )
    with pytest.raises(ValueError):
        loader.split_train_test(test_ratio=0.95)


def test_get_window_returns_requested_slices(tmp_path):
    _write_synthetic_pair_csv(tmp_path, "EURUSD", n_days=20)
    loader = DataLoader(pairs=["EURUSD"], start="2024-01-01", end="2024-01-20", data_dir=tmp_path)
    full_index = loader.load().index

    train_index = full_index[:10]
    test_index = full_index[10:]
    train_df, test_df = loader.get_window(train_index, test_index)

    assert list(train_df.index) == list(train_index)
    assert list(test_df.index) == list(test_index)
    train_df.iloc[0, 0] = -999.0
    assert loader.load().iloc[0, 0] != -999.0
