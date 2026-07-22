import numpy as np
import pandas as pd
import pytest

from src.features.sessions import (
    FILE_UTC_OFFSET_HOURS,
    load_session_returns,
    load_session_returns_with_first_hour,
)


def _write_synthetic_pair_csv(tmp_path, pair, utc_start, utc_end):
    utc_index = pd.date_range(utc_start, utc_end, freq="1min", tz="UTC")
    close = np.arange(1, len(utc_index) + 1, dtype=float)
    file_native_index = utc_index.tz_localize(None) - pd.Timedelta(hours=FILE_UTC_OFFSET_HOURS)
    df = pd.DataFrame({
        "Datetime": file_native_index.strftime("%Y%m%d %H%M%S"),
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": 0,
    })
    path = tmp_path / f"{pair}.csv"
    df.to_csv(path, index=False)
    return utc_index, close


def test_returns_dataframe_with_expected_columns(tmp_path):
    _write_synthetic_pair_csv(tmp_path, "TESTPAIR", "2023-01-08 00:00", "2023-01-14 23:59")
    result = load_session_returns("TESTPAIR", "2023-01-01", "2023-01-31", tmp_path)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"asian_return", "london_return", "ny_return"}
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.is_monotonic_increasing
    assert not result.empty


def test_asian_session_is_midnight_to_london_open(tmp_path):
    utc_index, close = _write_synthetic_pair_csv(tmp_path, "TESTPAIR", "2023-01-08 00:00", "2023-01-14 23:59")
    result = load_session_returns("TESTPAIR", "2023-01-01", "2023-01-31", tmp_path)

    monday = pd.Timestamp("2023-01-09")
    open_val = close[utc_index.get_loc(pd.Timestamp("2023-01-09 00:00", tz="UTC"))]
    close_val = close[utc_index.get_loc(pd.Timestamp("2023-01-09 07:59", tz="UTC"))]  # GMT: London opens 08:00 UTC

    assert result.loc[monday, "asian_return"] == pytest.approx(np.log(close_val / open_val))


def test_sessions_partition_the_day_with_no_overlap_or_gap(tmp_path):
    utc_index, close = _write_synthetic_pair_csv(tmp_path, "TESTPAIR", "2023-01-08 00:00", "2023-01-14 23:59")
    result = load_session_returns("TESTPAIR", "2023-01-01", "2023-01-31", tmp_path)

    monday = pd.Timestamp("2023-01-09")

    asian_open = close[utc_index.get_loc(pd.Timestamp("2023-01-09 00:00", tz="UTC"))]
    asian_close = close[utc_index.get_loc(pd.Timestamp("2023-01-09 07:59", tz="UTC"))]
    london_open = close[utc_index.get_loc(pd.Timestamp("2023-01-09 08:00", tz="UTC"))]
    london_close = close[utc_index.get_loc(pd.Timestamp("2023-01-09 12:59", tz="UTC"))]
    ny_open = close[utc_index.get_loc(pd.Timestamp("2023-01-09 13:00", tz="UTC"))]
    ny_close = close[utc_index.get_loc(pd.Timestamp("2023-01-09 23:59", tz="UTC"))]

    assert asian_close + 1 == london_open
    assert london_close + 1 == ny_open

    expected_asian = np.log(asian_close / asian_open)
    expected_london = np.log(london_close / london_open)
    expected_ny = np.log(ny_close / ny_open)

    assert result.loc[monday, "asian_return"] == pytest.approx(expected_asian)
    assert result.loc[monday, "london_return"] == pytest.approx(expected_london)
    assert result.loc[monday, "ny_return"] == pytest.approx(expected_ny)


def test_ny_open_shifts_with_us_dst_transition(tmp_path):
    utc_index, close = _write_synthetic_pair_csv(tmp_path, "TESTPAIR", "2023-03-06 00:00", "2023-03-20 23:59")
    result = load_session_returns("TESTPAIR", "2023-03-01", "2023-03-31", tmp_path)

    pre_dst_monday = pd.Timestamp("2023-03-06")
    post_dst_monday = pd.Timestamp("2023-03-13")

    pre_ny_open = close[utc_index.get_loc(pd.Timestamp("2023-03-06 13:00", tz="UTC"))]
    pre_ny_close = close[utc_index.get_loc(pd.Timestamp("2023-03-06 23:59", tz="UTC"))]
    post_ny_open = close[utc_index.get_loc(pd.Timestamp("2023-03-13 12:00", tz="UTC"))]
    post_ny_close = close[utc_index.get_loc(pd.Timestamp("2023-03-13 23:59", tz="UTC"))]

    assert result.loc[pre_dst_monday, "ny_return"] == pytest.approx(np.log(pre_ny_close / pre_ny_open))
    assert result.loc[post_dst_monday, "ny_return"] == pytest.approx(np.log(post_ny_close / post_ny_open))


def test_london_ny_dst_mismatch_window_still_partitions_cleanly(tmp_path):
    utc_index, close = _write_synthetic_pair_csv(tmp_path, "TESTPAIR", "2023-03-13 00:00", "2023-03-19 23:59")
    result = load_session_returns("TESTPAIR", "2023-03-01", "2023-03-31", tmp_path)

    day = pd.Timestamp("2023-03-13")
    london_open_utc = pd.Timestamp("2023-03-13 08:00", tz="UTC")
    ny_open_utc = pd.Timestamp("2023-03-13 12:00", tz="UTC")

    assert (ny_open_utc - london_open_utc) == pd.Timedelta(hours=4)
    assert not result.loc[day, ["asian_return", "london_return", "ny_return"]].isna().any()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_returns("NOPE", "2023-01-01", "2023-01-31", tmp_path)


def test_date_range_filters_result(tmp_path):
    _write_synthetic_pair_csv(tmp_path, "TESTPAIR", "2023-01-01 00:00", "2023-01-31 23:59")
    result = load_session_returns("TESTPAIR", "2023-01-10", "2023-01-20", tmp_path)

    assert result.index.min() >= pd.Timestamp("2023-01-10")
    assert result.index.max() <= pd.Timestamp("2023-01-20")


def test_first_hour_columns_present_and_full_session_matches_load_session_returns(tmp_path):
    _write_synthetic_pair_csv(tmp_path, "TESTPAIR", "2023-01-08 00:00", "2023-01-14 23:59")
    full = load_session_returns("TESTPAIR", "2023-01-01", "2023-01-31", tmp_path)
    extended = load_session_returns_with_first_hour("TESTPAIR", "2023-01-01", "2023-01-31", tmp_path)

    assert set(extended.columns) == {
        "asian_return", "london_return", "ny_return",
        "asian_first_hour_return", "london_first_hour_return", "ny_first_hour_return",
    }
    for col in ["asian_return", "london_return", "ny_return"]:
        pd.testing.assert_series_equal(extended[col], full[col])


def test_first_hour_return_matches_first_60_minutes(tmp_path):
    utc_index, close = _write_synthetic_pair_csv(tmp_path, "TESTPAIR", "2023-01-08 00:00", "2023-01-14 23:59")
    result = load_session_returns_with_first_hour("TESTPAIR", "2023-01-01", "2023-01-31", tmp_path)

    monday = pd.Timestamp("2023-01-09")
    asian_open = close[utc_index.get_loc(pd.Timestamp("2023-01-09 00:00", tz="UTC"))]
    asian_first_hour_close = close[utc_index.get_loc(pd.Timestamp("2023-01-09 00:59", tz="UTC"))]

    expected = np.log(asian_first_hour_close / asian_open)
    assert result.loc[monday, "asian_first_hour_return"] == pytest.approx(expected)


def test_first_hour_window_is_narrower_than_full_session(tmp_path):
    _write_synthetic_pair_csv(tmp_path, "TESTPAIR", "2023-01-08 00:00", "2023-01-14 23:59")
    result = load_session_returns_with_first_hour("TESTPAIR", "2023-01-01", "2023-01-31", tmp_path)

    monday = pd.Timestamp("2023-01-09")
    assert result.loc[monday, "ny_first_hour_return"] != result.loc[monday, "ny_return"]
