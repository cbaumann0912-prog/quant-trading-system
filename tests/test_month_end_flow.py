import numpy as np
import pandas as pd

from src.signals.month_end_flow import (
    build_interaction_panel,
    hedging_need_signal,
    month_end_flag,
    month_to_date_return,
)


def _daily(values, start="2020-01-01"):
    index = pd.date_range(start=start, periods=len(values), freq="1D")
    return pd.Series(np.asarray(values, dtype=float), index=index)


def test_month_to_date_excludes_the_current_day():
    r = _daily([0.01, 0.02, 0.03, 0.04])
    mtd = month_to_date_return(r)

    assert np.isnan(mtd.iloc[0])
    assert np.isclose(mtd.iloc[1], 0.01)
    assert np.isclose(mtd.iloc[2], 0.03)
    assert np.isclose(mtd.iloc[3], 0.06)


def test_month_to_date_carries_the_prior_month_final_return_into_day_one():
    index = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02"])
    r = pd.Series([0.01, 0.02, 0.03, 0.04], index=index)
    mtd = month_to_date_return(r)

    assert np.isclose(mtd.iloc[2], 0.02)
    assert np.isclose(mtd.iloc[3], 0.05)


def test_month_to_date_accumulates_within_a_month():
    index = pd.to_datetime(["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05"])
    r = pd.Series([0.01, 0.02, 0.03, 0.04], index=index)
    mtd = month_to_date_return(r)
    assert np.isnan(mtd.iloc[0])
    assert np.isclose(mtd.iloc[1], 0.01)
    assert np.isclose(mtd.iloc[2], 0.03)
    assert np.isclose(mtd.iloc[3], 0.06)


def test_signal_is_the_negative_sign_of_month_to_date():
    r = _daily([0.01, 0.02, -0.05, 0.01])
    sig = hedging_need_signal(r)
    assert sig.iloc[1] == -1.0
    assert sig.iloc[2] == -1.0
    assert sig.iloc[3] == 1.0


def test_signal_is_causal():
    r = _daily([0.01, 0.02, 0.03, 0.04])
    original = hedging_need_signal(r)

    perturbed = r.copy()
    perturbed.iloc[3] = -99.0
    assert original.iloc[3] == hedging_need_signal(perturbed).iloc[3]


def test_month_end_flag_marks_the_last_n_observations_of_each_month():
    index = pd.to_datetime([
        "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31",
        "2020-02-03", "2020-02-04", "2020-02-05",
    ])
    flag = month_end_flag(index, month_end_days=2)
    assert list(flag.to_numpy()) == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]


def test_month_end_flag_counts_from_observed_rows_not_the_calendar():
    index = pd.to_datetime(["2020-01-05", "2020-01-06"])
    flag = month_end_flag(index, month_end_days=2)
    assert list(flag.to_numpy()) == [1.0, 1.0]


def _staged(pairs, n=90, seed=11):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    return {p: pd.DataFrame({
        "fix_return": rng.normal(0, 1e-3, n),
        "control_return": rng.normal(0, 1e-3, n),
        "daily_log_return": rng.normal(0, 5e-3, n),
    }, index=index) for p in pairs}


def test_panel_emits_a_fix_row_and_a_control_row_per_observation():
    pairs = ["A", "B"]
    panel = build_interaction_panel(_staged(pairs), pairs, "fix_return")
    counts = panel.groupby("fix").size()
    assert counts[0.0] == counts[1.0]


def test_panel_drops_zero_signal_rows():
    pairs = ["A"]
    staged = _staged(pairs)
    staged["A"]["daily_log_return"] = 0.0
    panel = build_interaction_panel(staged, pairs, "fix_return")
    assert panel.empty


def test_panel_columns_and_sort_order():
    pairs = ["A", "B"]
    panel = build_interaction_panel(_staged(pairs), pairs, "fix_return")
    assert list(panel.columns) == ["date", "y", "signal", "month_end", "fix"]
    assert panel["date"].is_monotonic_increasing
    assert set(np.unique(panel["signal"])) <= {-1.0, 1.0}


def test_panel_fix_rows_carry_the_requested_column():
    pairs = ["A"]
    staged = _staged(pairs)
    panel = build_interaction_panel(staged, pairs, "fix_return")
    fix_rows = panel.loc[panel["fix"] == 1.0].set_index("date")["y"]
    source = staged["A"]["fix_return"]
    assert np.allclose(fix_rows.to_numpy(), source.loc[fix_rows.index].to_numpy())
