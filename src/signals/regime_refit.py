from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.pca import pca


def compute_composite_regime_score_walkforward(
    vol: pd.Series,
    rate_diff: pd.Series,
    windows: list[dict],
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Per-walk-forward-window refit of the 2-feature PCA regime composite.

    Parameters
    ----------
    vol : pd.Series
        Rolling realized volatility feature, DatetimeIndex.
    rate_diff : pd.Series
        Rate differential feature (already publication-lag-shifted and
        forward-filled), DatetimeIndex.
    windows : list[dict]
        Walk-forward window boundaries, one dict per fold, each with keys
        "train_start", "train_end", "test_start", "test_end" (pd.Timestamp).

    Returns
    -------
    composite_z : pd.Series
        Out-of-sample composite z-score, index = concatenation of each
        window's test-period dates (in window order, sorted ascending),
        name "composite_z".
    diagnostics : pd.DataFrame
        One row per window: window_idx, train_start, train_end,
        test_start, test_end, n_train, n_test, mean_vol, mean_rate_diff,
        std_vol, std_rate_diff, pc1_vol, pc1_rate_diff,
        explained_variance_ratio, rho, composite_train_mean,
        composite_train_std.

    Raises
    ------
    ValueError
        If `windows` is empty; if a window's train slice has fewer than
        2 overlapping non-NaN (vol, rate_diff) observations; if a
        window's train-slice std is zero or numerically indistinguishable
        from zero (atol=1e-10) for either feature (z-score undefined); if
        a window's train-slice composite std is zero or numerically
        indistinguishable from zero; if a window's test slice is empty;
        or if two windows' test periods overlap (ambiguous which fold's
        frozen parameters should apply to the overlapping dates).
    """
    if not windows:
        raise ValueError("windows must be a non-empty list of window boundary dicts.")

    combined = pd.concat([vol.rename("vol"), rate_diff.rename("rate_diff")], axis=1, sort=False).dropna()

    composite_chunks: list[pd.Series] = []
    diagnostics_rows: list[dict] = []
    seen_test_dates: set = set()

    for window_idx, w in enumerate(windows):
        train_mask = (combined.index >= w["train_start"]) & (combined.index < w["train_end"])
        test_mask = (combined.index >= w["test_start"]) & (combined.index < w["test_end"])

        train = combined.loc[train_mask]
        test = combined.loc[test_mask]

        if len(train) < 2:
            raise ValueError(
                f"Window {window_idx}: need at least 2 overlapping non-NaN "
                f"(vol, rate_diff) training observations, got {len(train)} "
                f"in [{w['train_start']}, {w['train_end']})."
            )
        if len(test) == 0:
            raise ValueError(
                f"Window {window_idx}: test slice [{w['test_start']}, "
                f"{w['test_end']}) has 0 overlapping non-NaN observations."
            )

        overlap = seen_test_dates.intersection(test.index)
        if overlap:
            raise ValueError(
                f"Window {window_idx}: test period overlaps a prior window's "
                f"test period (e.g. {sorted(overlap)[0]}) -- ambiguous which "
                f"fold's frozen parameters apply to the overlapping dates."
            )
        seen_test_dates.update(test.index)

        mean = train.mean()
        std = train.std()
        near_zero = np.isclose(std, 0.0, atol=1e-10)
        if near_zero.any():
            zero_cols = list(std.index[near_zero])
            raise ValueError(
                f"Window {window_idx}: zero (or numerically-zero) "
                f"training-window std for {zero_cols} -- z-score undefined."
            )

        z_train = (train - mean) / std

        components, explained_variance, _projected = pca(z_train.to_numpy(), n_components=1)
        pc1 = components[:, 0]
        if pc1[0] < 0:
            pc1 = -pc1

        rho = float(np.corrcoef(train["vol"], train["rate_diff"])[0, 1])

        composite_train = z_train.to_numpy() @ pc1
        composite_train_mean = composite_train.mean()
        composite_train_std = composite_train.std(ddof=1)
        if np.isclose(composite_train_std, 0.0, atol=1e-10):
            raise ValueError(
                f"Window {window_idx}: zero (or numerically-zero) "
                f"training-window composite std -- z-score undefined."
            )

        z_test = (test - mean) / std
        composite_test = z_test.to_numpy() @ pc1
        composite_test_z = (composite_test - composite_train_mean) / composite_train_std

        composite_chunks.append(
            pd.Series(composite_test_z, index=test.index, name="composite_z")
        )

        diagnostics_rows.append(
            {
                "window_idx": window_idx,
                "train_start": w["train_start"],
                "train_end": w["train_end"],
                "test_start": w["test_start"],
                "test_end": w["test_end"],
                "n_train": len(train),
                "n_test": len(test),
                "mean_vol": mean["vol"],
                "mean_rate_diff": mean["rate_diff"],
                "std_vol": std["vol"],
                "std_rate_diff": std["rate_diff"],
                "pc1_vol": pc1[0],
                "pc1_rate_diff": pc1[1],
                "explained_variance_ratio": float(explained_variance[0]),
                "rho": rho,
                "composite_train_mean": composite_train_mean,
                "composite_train_std": composite_train_std,
            }
        )

    composite_z = pd.concat(composite_chunks).sort_index()
    composite_z.name = "composite_z"
    diagnostics = pd.DataFrame(diagnostics_rows)

    return composite_z, diagnostics
