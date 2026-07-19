import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.framework.data_loader import DataLoader
from src.framework.walk_forward import WalkForwardValidator
from src.signals.momentum import momentum_signal
from src.signals.mean_reversion import price_zscore_signal
from src.features.regime_classifier import compute_composite_regime_score, classify_regime

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
START = "2011-01-01"
DEV_END = "2023-12-31"

MOMENTUM_LOOKBACK = 78
REVERSION_LOOKBACK = 26
TRADING_DAYS_PER_YEAR = 312
REVERSION_ENTRY_Z = 2.0

TRAIN_YEARS = 5
TEST_MONTHS = 12
EMBARGO_DAYS = 5
N_WINDOWS = 7

PUBLICATION_LAG_MONTHS = 2
_RATE_FILES = {"EURUSD": ("ea", "us"), "GBPUSD": ("uk", "us"), "USDJPY": ("us", "jp")}


def load_rate_diff(pair: str) -> pd.Series:
    a, b = _RATE_FILES[pair]
    a_series = pd.read_csv(DATA_DIR / f"{a}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
    b_series = pd.read_csv(DATA_DIR / f"{b}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
    diff_monthly = (a_series - b_series).dropna()
    return diff_monthly.shift(PUBLICATION_LAG_MONTHS)


def spearman_ic(signal: pd.Series, forward_returns: pd.Series) -> float:
    aligned = pd.concat([signal.rename("s"), forward_returns.rename("f")], axis=1, join="inner").dropna()
    if len(aligned) < 2 or aligned["s"].nunique() < 2 or aligned["f"].nunique() < 2:
        return float("nan")
    return aligned["s"].corr(aligned["f"], method="spearman")


def window_sharpe(exposure: pd.Series, daily_log_return: pd.Series) -> float:
    """
    Simplified stand-in Sharpe: pnl_t = exposure_{t-1} * daily_log_return_t
    (exposure decided at the prior close, applied to the next day's realized
    return -- causal). NOT vol-targeted per Section 7, no transaction costs.
    A real Sharpe is put off for a later date's proper build.
    """
    pnl = exposure.shift(1) * daily_log_return
    pnl = pnl.dropna()
    if len(pnl) < 2 or pnl.std() == 0:
        return float("nan")
    return (pnl.mean() / pnl.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_distribution_summary(values: list[float]) -> dict:
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "frac_positive": float("nan")}
    return {
        "n": len(arr),
        "mean": arr.mean(),
        "std": arr.std(ddof=1) if len(arr) > 1 else float("nan"),
        "min": arr.min(),
        "max": arr.max(),
        "frac_positive": float((arr > 0).mean()),
    }


for pair in PAIRS:
    loader = DataLoader(pairs=[pair], start=START, end=DEV_END, data_dir=str(DATA_DIR))
    prices = loader.load()[pair]
    data = prices.to_frame(name="price")

    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.rolling(MOMENTUM_LOOKBACK).std()

    rate_diff_monthly = load_rate_diff(pair)
    rate_diff_daily = rate_diff_monthly.reindex(
        pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    ).ffill()

    composite_z = compute_composite_regime_score(vol, rate_diff_daily)
    regime = classify_regime(composite_z)

    momentum = momentum_signal(data, MOMENTUM_LOOKBACK)
    price_z = price_zscore_signal(data, REVERSION_LOOKBACK)
    reversion_exposure = pd.Series(0.0, index=price_z.index)
    reversion_exposure[price_z > REVERSION_ENTRY_Z] = -1.0
    reversion_exposure[price_z < -REVERSION_ENTRY_Z] = 1.0
    reversion_exposure = reversion_exposure.where(price_z.notna())

    forward_returns_26d = np.log(prices.shift(-26) / prices)

    validator = WalkForwardValidator(
        signal_fn=momentum_signal,
        data=data,
        n_windows=N_WINDOWS,
        train_years=TRAIN_YEARS,
        test_months=TEST_MONTHS,
        embargo_days=EMBARGO_DAYS,
    )
    windows = validator.generate_windows()

    momentum_ic_by_window, momentum_sharpe_by_window = [], []
    reversion_ic_by_window, reversion_sharpe_by_window = [], []

    print(f"--- {pair} ---")
    print(f"dev data: n_obs={len(data)}  range={data.index.min().date()}..{data.index.max().date()}  "
          f"(lockbox 2024-2026 excluded at load time, never touched)")

    for w in windows:
        test_mask = (data.index >= w["test_start"]) & (data.index < w["test_end"])
        test_idx = data.index[test_mask]

        m_ic = spearman_ic(momentum.loc[test_idx], forward_returns_26d.loc[test_idx])
        r_ic = spearman_ic(price_z.loc[test_idx], forward_returns_26d.loc[test_idx])
        m_sharpe = window_sharpe(momentum.loc[test_idx], log_returns.loc[test_idx])
        r_sharpe = window_sharpe(reversion_exposure.loc[test_idx], log_returns.loc[test_idx])

        momentum_ic_by_window.append(m_ic)
        reversion_ic_by_window.append(r_ic)
        momentum_sharpe_by_window.append(m_sharpe)
        reversion_sharpe_by_window.append(r_sharpe)

        print(
            f"  window test=[{w['test_start'].date()}, {w['test_end'].date()}) "
            f"n_test={test_mask.sum()}  momentum(IC={m_ic:.4f}, Sharpe={m_sharpe:.3f})  "
            f"reversion(IC={r_ic:.4f}, Sharpe={r_sharpe:.3f})"
        )

    m_sharpe_dist = sharpe_distribution_summary(momentum_sharpe_by_window)
    r_sharpe_dist = sharpe_distribution_summary(reversion_sharpe_by_window)
    m_ic_valid = [v for v in momentum_ic_by_window if not np.isnan(v)]
    r_ic_valid = [v for v in reversion_ic_by_window if not np.isnan(v)]

    print(
        f"  momentum   OOS IC across windows: n={len(m_ic_valid)}  "
        f"mean={np.mean(m_ic_valid) if m_ic_valid else float('nan'):.4f}"
    )
    print(
        f"  reversion  OOS IC across windows: n={len(r_ic_valid)}  "
        f"mean={np.mean(r_ic_valid) if r_ic_valid else float('nan'):.4f}"
    )
    print(f"  momentum   Sharpe distribution: {m_sharpe_dist}")
    print(f"  reversion  Sharpe distribution: {r_sharpe_dist}")
    print()

print("Reminder: preliminary diagnostic only, NOT the official validation run. Full-sample-fit")
print("regime classifier + embargo-only (unpurged) windows -- see module docstring.")
