import numpy as np
import pandas as pd
import pytest

from src.signals.regime_refit import compute_composite_regime_score_walkforward
from src.features.regime_classifier import classify_regime


def _series(values, start="2020-01-01"):
    index = pd.date_range(start=start, periods=len(values), freq="1D")
    return pd.Series(values, index=index)


def _two_fold_windows(n_train=100, n_test=50, gap=0, start="2020-01-01"):
    base = pd.Timestamp(start)
    day = pd.Timedelta(days=1)

    w0_train_start = base
    w0_train_end = w0_train_start + n_train * day
    w0_test_start = w0_train_end + gap * day
    w0_test_end = w0_test_start + n_test * day

    w1_train_start = w0_test_end
    w1_train_end = w1_train_start + n_train * day
    w1_test_start = w1_train_end + gap * day
    w1_test_end = w1_test_start + n_test * day

    return [
        {
            "train_start": w0_train_start,
            "train_end": w0_train_end,
            "test_start": w0_test_start,
            "test_end": w0_test_end,
        },
        {
            "train_start": w1_train_start,
            "train_end": w1_train_end,
            "test_start": w1_test_start,
            "test_end": w1_test_end,
        },
    ], w1_test_end


def _make_features(n, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    vol = _series(rng.normal(0.01, 0.002, n), start=start)
    rate_diff = _series(rng.normal(0, 1, n), start=start)
    return vol, rate_diff


def test_diagnostics_unaffected_by_test_window_values():
    windows, total_end = _two_fold_windows()
    n_total = (total_end - pd.Timestamp("2020-01-01")).days
    vol, rate_diff = _make_features(n_total)

    _, diag_a = compute_composite_regime_score_walkforward(vol, rate_diff, windows)

    w0 = windows[0]
    test_mask = (vol.index >= w0["test_start"]) & (vol.index < w0["test_end"])
    vol_corrupted = vol.copy()
    vol_corrupted[test_mask] = vol_corrupted[test_mask] * 1000 + 5

    _, diag_b = compute_composite_regime_score_walkforward(vol_corrupted, rate_diff, windows)

    pd.testing.assert_frame_equal(
        diag_a[["mean_vol", "std_vol", "pc1_vol", "pc1_rate_diff"]],
        diag_b[["mean_vol", "std_vol", "pc1_vol", "pc1_rate_diff"]],
    )


def test_composite_test_uses_frozen_train_params_not_test_stats():
    windows, total_end = _two_fold_windows()
    n_total = (total_end - pd.Timestamp("2020-01-01")).days
    vol, rate_diff = _make_features(n_total)

    composite_z, diag = compute_composite_regime_score_walkforward(vol, rate_diff, windows)

    combined = pd.concat([vol.rename("vol"), rate_diff.rename("rate_diff")], axis=1, sort=False).dropna()
    w0 = windows[0]
    row = diag.iloc[0]
    test = combined.loc[(combined.index >= w0["test_start"]) & (combined.index < w0["test_end"])]

    mean = pd.Series({"vol": row["mean_vol"], "rate_diff": row["mean_rate_diff"]})
    std = pd.Series({"vol": row["std_vol"], "rate_diff": row["std_rate_diff"]})
    pc1 = np.array([row["pc1_vol"], row["pc1_rate_diff"]])

    z_test_manual = (test - mean) / std
    composite_manual = z_test_manual.to_numpy() @ pc1
    composite_manual_z = (composite_manual - row["composite_train_mean"]) / row["composite_train_std"]

    expected = pd.Series(composite_manual_z, index=test.index)
    actual = composite_z.loc[test.index]

    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-10)


def test_covers_both_test_windows_sorted():
    windows, _ = _two_fold_windows()
    n_total = (windows[1]["test_end"] - pd.Timestamp("2020-01-01")).days
    vol, rate_diff = _make_features(n_total)

    composite_z, _ = compute_composite_regime_score_walkforward(vol, rate_diff, windows)

    assert composite_z.index.is_monotonic_increasing

    w0, w1 = windows
    assert composite_z.index.min() >= w0["test_start"]
    assert composite_z.index.max() < w1["test_end"]


def test_overlapping_test_windows_raise():
    windows, total_end = _two_fold_windows()
    n_total = (total_end - pd.Timestamp("2020-01-01")).days
    vol, rate_diff = _make_features(n_total)

    windows[1]["test_start"] = windows[0]["test_start"]
    windows[1]["test_end"] = windows[0]["test_end"]
    windows[1]["train_end"] = windows[1]["train_start"] + pd.Timedelta(days=50)

    with pytest.raises(ValueError, match="overlaps"):
        compute_composite_regime_score_walkforward(vol, rate_diff, windows)


def test_empty_windows_raises():
    vol, rate_diff = _make_features(10)
    with pytest.raises(ValueError, match="non-empty"):
        compute_composite_regime_score_walkforward(vol, rate_diff, [])


def test_insufficient_train_observations_raises():
    vol, rate_diff = _make_features(20)
    windows = [
        {
            "train_start": pd.Timestamp("2020-01-01"),
            "train_end": pd.Timestamp("2020-01-02"),
            "test_start": pd.Timestamp("2020-01-05"),
            "test_end": pd.Timestamp("2020-01-15"),
        }
    ]
    with pytest.raises(ValueError, match="training observations"):
        compute_composite_regime_score_walkforward(vol, rate_diff, windows)


def test_empty_test_slice_raises():
    vol, rate_diff = _make_features(30)
    windows = [
        {
            "train_start": pd.Timestamp("2020-01-01"),
            "train_end": pd.Timestamp("2020-01-20"),
            "test_start": pd.Timestamp("2025-01-01"),
            "test_end": pd.Timestamp("2025-01-10"),
        }
    ]
    with pytest.raises(ValueError, match="test slice"):
        compute_composite_regime_score_walkforward(vol, rate_diff, windows)


def test_zero_train_std_raises():
    n = 30
    vol = _series(np.full(n, 0.01))
    rate_diff = _series(np.random.default_rng(1).normal(0, 1, n))
    windows = [
        {
            "train_start": pd.Timestamp("2020-01-01"),
            "train_end": pd.Timestamp("2020-01-20"),
            "test_start": pd.Timestamp("2020-01-20"),
            "test_end": pd.Timestamp("2020-01-30"),
        }
    ]
    with pytest.raises(ValueError, match="training-window std"):
        compute_composite_regime_score_walkforward(vol, rate_diff, windows)


def test_pc1_vol_loading_always_nonnegative():
    windows, total_end = _two_fold_windows()
    n_total = (total_end - pd.Timestamp("2020-01-01")).days
    vol, rate_diff = _make_features(n_total, seed=7)
    _, diag = compute_composite_regime_score_walkforward(vol, rate_diff, windows)

    assert (diag["pc1_vol"] >= 0).all()


def test_pc1_vol_pinned_to_inverse_sqrt2():
    windows, total_end = _two_fold_windows()
    n_total = (total_end - pd.Timestamp("2020-01-01")).days
    vol, rate_diff = _make_features(n_total, seed=13)
    _, diag = compute_composite_regime_score_walkforward(vol, rate_diff, windows)

    np.testing.assert_allclose(diag["pc1_vol"], 1 / np.sqrt(2), rtol=1e-10)


def test_explained_variance_ratio_matches_rho_identity():
    windows, total_end = _two_fold_windows()
    n_total = (total_end - pd.Timestamp("2020-01-01")).days
    vol, rate_diff = _make_features(n_total, seed=17)
    _, diag = compute_composite_regime_score_walkforward(vol, rate_diff, windows)

    expected = (1 + diag["rho"].abs()) / 2
    np.testing.assert_allclose(diag["explained_variance_ratio"], expected, rtol=1e-10)


def test_rho_matches_pc1_rate_diff_sign():
    windows, total_end = _two_fold_windows()
    n_total = (total_end - pd.Timestamp("2020-01-01")).days
    vol, rate_diff = _make_features(n_total, seed=11)
    _, diag = compute_composite_regime_score_walkforward(vol, rate_diff, windows)

    assert np.array_equal(np.sign(diag["rho"]), np.sign(diag["pc1_rate_diff"]))


def test_feeds_directly_into_classify_regime():
    windows, total_end = _two_fold_windows()
    n_total = (total_end - pd.Timestamp("2020-01-01")).days
    vol, rate_diff = _make_features(n_total)
    composite_z, _ = compute_composite_regime_score_walkforward(vol, rate_diff, windows)
    regime = classify_regime(composite_z)

    assert set(regime.unique()) <= {"turbulent", "calm", "deadzone"}
    assert (regime.index == composite_z.index).all()