import numpy as np
import pandas as pd
import pytest

from src.signals.intraday_overshoot import (
    build_overshoot_sessions,
    load_ny_minute_bars,
    overshoot_trades,
    walk_forward_conditional_vol,
)

SCAN_OPEN, SCAN_CLOSE, EXIT_MIN = 9 * 60, 12 * 60, 13 * 60


def _write_bars(tmp_path, pair, n_days=700, step=10, seed=28):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2011-01-03", periods=n_days)
    minutes = np.arange(0, 24 * 60, step)

    rows, price = [], 1.2000
    for d in dates:
        price *= float(np.exp(rng.standard_normal() * 0.004))
        for m in minutes:
            tick = price * float(np.exp(rng.standard_normal() * 0.0001))
            rows.append((f"{d:%Y%m%d} {m // 60:02d}{m % 60:02d}00", round(tick, 6)))

    path = tmp_path / f"{pair}.csv"
    pd.DataFrame(rows, columns=["Datetime", "Close"]).to_csv(path, index=False)
    return path


def _build(tmp_path, pair="EURUSD", ks=(2.0,), delays=(0, 5)):
    return build_overshoot_sessions(
        pair=pair, data_dir=tmp_path, start="2011-01-01", end="2014-12-31",
        ks=list(ks), entry_delays=list(delays),
        scan_open=SCAN_OPEN, scan_close=SCAN_CLOSE, exit_min=EXIT_MIN,
        vol_ratio_min_obs=100, garch_min_train=250,
    )


def _sessions(tmp_path, pair="EURUSD", ks=(2.0,), delays=(0, 5), **kw):
    _write_bars(tmp_path, pair, **kw)
    return _build(tmp_path, pair, ks, delays)


def _returns(n=900, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.standard_normal(n) * 0.006, index=pd.bdate_range("2011-01-03", periods=n)
    )


def _frame(disp, entry, exit_px, dates):
    return pd.DataFrame(
        {"disp_2.0": disp, "px_2.0_d5": entry, "exit_px": exit_px},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def test_load_bars_returns_expected_columns(tmp_path):
    _write_bars(tmp_path, "EURUSD", n_days=5)
    bars = load_ny_minute_bars("EURUSD", tmp_path)

    assert list(bars.columns) == ["c", "d", "m"]
    assert bars["m"].between(0, 24 * 60 - 1).all()


def test_load_bars_applies_file_offset_and_dst_aware_ny_conversion(tmp_path):
    path = _write_bars(tmp_path, "EURUSD", n_days=400, step=60)
    raw = pd.read_csv(path)
    bars = load_ny_minute_bars("EURUSD", tmp_path)

    winter_row = raw.index[raw["Datetime"] == "20110105 090000"][0]
    summer_row = raw.index[raw["Datetime"] == "20110706 090000"][0]

    assert bars.loc[winter_row, "m"] == 9 * 60
    assert bars.loc[summer_row, "m"] == 10 * 60


def test_load_bars_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ny_minute_bars("NOPAIR", tmp_path)


def test_conditional_vol_skips_years_without_enough_history():
    ret = _returns()
    vol = walk_forward_conditional_vol(ret, garch_min_train=500)

    assert vol.index.min().year > ret.index.min().year
    assert (vol > 0).all()


def test_conditional_vol_is_deterministic():
    ret = _returns()

    assert walk_forward_conditional_vol(ret).equals(walk_forward_conditional_vol(ret))


def test_sessions_have_expected_columns(tmp_path):
    out = _sessions(tmp_path, ks=(1.5, 2.0), delays=(0, 5))

    for col in ["sigma", "open", "exit_px",
                "t_1.5", "disp_1.5", "px_1.5_d0", "px_1.5_d5",
                "t_2.0", "disp_2.0", "px_2.0_d0", "px_2.0_d5"]:
        assert col in out.columns


def test_sessions_respect_date_bounds(tmp_path):
    out = _sessions(tmp_path)

    assert out.index.min() >= pd.Timestamp("2011-01-01")
    assert out.index.max() <= pd.Timestamp("2014-12-31")
    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates


def test_crossing_columns_are_all_nan_together(tmp_path):
    out = _sessions(tmp_path)

    assert out["t_2.0"].isna().equals(out["disp_2.0"].isna())


def test_no_crossing_while_sigma_is_undefined(tmp_path):
    out = _sessions(tmp_path)
    warmup = out[out["sigma"].isna()]

    assert len(warmup) > 0
    assert warmup["disp_2.0"].isna().all()
    assert warmup["t_2.0"].isna().all()


def test_higher_threshold_triggers_less_often(tmp_path):
    out = _sessions(tmp_path, ks=(1.5, 2.0, 2.5))

    n15 = out["disp_1.5"].notna().sum()
    n20 = out["disp_2.0"].notna().sum()
    n25 = out["disp_2.5"].notna().sum()

    assert n15 >= n20 >= n25


def test_displacement_exceeds_threshold_at_crossing(tmp_path):
    out = _sessions(tmp_path).dropna(subset=["disp_2.0", "sigma"])

    assert len(out) > 0
    assert (out["disp_2.0"].abs() > 2.0 * out["sigma"]).all()


def test_trigger_time_falls_inside_scan_window(tmp_path):
    out = _sessions(tmp_path).dropna(subset=["t_2.0"])

    assert (out["t_2.0"] >= 0).all()
    assert (out["t_2.0"] <= SCAN_CLOSE - SCAN_OPEN).all()


def test_delayed_entry_price_differs_from_crossing_bar(tmp_path):
    out = _sessions(tmp_path, delays=(0, 5)).dropna(subset=["px_2.0_d0", "px_2.0_d5"])

    assert len(out) > 0
    assert not np.allclose(out["px_2.0_d0"], out["px_2.0_d5"])


def test_sigma_uses_no_same_day_information(tmp_path):
    pair = "EURUSD"
    _write_bars(tmp_path, pair)
    base = _build(tmp_path, pair, ks=(2.0,), delays=(0,))

    target = base.dropna(subset=["sigma"]).index[-1]
    raw = pd.read_csv(tmp_path / f"{pair}.csv")
    stamp = raw["Datetime"].str.slice(0, 8)
    raw.loc[stamp == f"{target:%Y%m%d}", "Close"] *= 1.05
    raw.to_csv(tmp_path / f"{pair}.csv", index=False)

    edited = _build(tmp_path, pair, ks=(2.0,), delays=(0,))

    assert edited.loc[target, "sigma"] == pytest.approx(base.loc[target, "sigma"])


def test_rebuild_is_reproducible(tmp_path):
    _write_bars(tmp_path, "EURUSD")

    pd.testing.assert_frame_equal(
        _build(tmp_path, "EURUSD", ks=(2.0,), delays=(5,)),
        _build(tmp_path, "EURUSD", ks=(2.0,), delays=(5,)),
    )


def test_trades_fade_direction_is_opposite_displacement():
    dates = pd.bdate_range("2011-01-03", periods=2)
    sessions = {"EURUSD": _frame([0.01, -0.01], [1.21, 1.19], [1.20, 1.20], dates)}

    t = overshoot_trades(sessions, 2.0, 5)

    assert (t["ret"] > 0).all()


def test_trades_lose_when_move_continues():
    dates = pd.bdate_range("2011-01-03", periods=2)
    sessions = {"EURUSD": _frame([0.01, -0.01], [1.21, 1.19], [1.22, 1.18], dates)}

    t = overshoot_trades(sessions, 2.0, 5)

    assert (t["ret"] < 0).all()


def test_trades_drop_non_triggering_days():
    dates = pd.bdate_range("2011-01-03", periods=3)
    sessions = {
        "EURUSD": _frame([0.01, np.nan, 0.02], [1.21, np.nan, 1.22],
                         [1.20, 1.20, 1.21], dates)
    }

    t = overshoot_trades(sessions, 2.0, 5)

    assert len(t) == 2
    assert pd.Timestamp("2011-01-04") not in set(t["date"])


def test_trades_pool_across_pairs():
    dates = pd.bdate_range("2011-01-03", periods=2)
    sessions = {
        "EURUSD": _frame([0.01, 0.01], [1.21, 1.21], [1.20, 1.20], dates),
        "GBPUSD": _frame([0.01, np.nan], [1.51, np.nan], [1.50, 1.50], dates),
    }

    t = overshoot_trades(sessions, 2.0, 5)

    assert len(t) == 3
    assert set(t["pair"]) == {"EURUSD", "GBPUSD"}


def test_trade_return_matches_hand_calculation():
    dates = pd.bdate_range("2011-01-03", periods=1)
    sessions = {"EURUSD": _frame([0.01], [1.21], [1.20], dates)}

    t = overshoot_trades(sessions, 2.0, 5)

    assert t["ret"].iloc[0] == pytest.approx(-np.log(1.20 / 1.21))
