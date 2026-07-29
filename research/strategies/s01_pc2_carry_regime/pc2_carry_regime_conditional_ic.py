import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from src.signals.pc2_carry import fit_pc2_loadings, pc2_scores, pc2_factor_returns
from src.stats.regression import interaction_regression
from src.evaluation.significance import permutation_test

DEV_END = "2023-12-31"

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT.parent / "data"

FILES = {
    "EURUSD": "EURUSD.csv",
    "GBPUSD": "GBPUSD.csv",
    "USDJPY": "USDJPY.csv",
}

SPLIT_DATE = "2021-01-01"
N_COMPONENTS = 3
PAIRS_ORDER = ["EURUSD", "GBPUSD", "USDJPY"]

VOL_ESTIMATION_WINDOW = 26
REGIME_THRESHOLD_WINDOW = 156

returns = {}
for pair_name, filename in FILES.items():
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime").sort_index().loc[:DEV_END]
    daily_close = df["Close"].resample("D").last().dropna()
    returns[pair_name] = np.log(daily_close / daily_close.shift(1)).dropna()

returns_df = pd.DataFrame(returns)[PAIRS_ORDER].dropna()

train_returns = returns_df[returns_df.index < SPLIT_DATE]
test_returns = returns_df[returns_df.index >= SPLIT_DATE]

pc2_loadings, pc2_loadings_by_pair, train_mean = fit_pc2_loadings(
    train_returns, PAIRS_ORDER, n_components=N_COMPONENTS
)
pc2_scores_test = pc2_scores(test_returns, pc2_loadings, train_mean)

r_pc2_test = pc2_factor_returns(test_returns, pc2_loadings_by_pair)

signal = pc2_scores_test.iloc[:-1]
forward_returns = r_pc2_test.iloc[1:]
forward_returns.index = signal.index

aligned = pd.concat([signal.rename("signal"), forward_returns.rename("forward_returns")], axis=1).dropna()
pc2_signal_series = aligned["signal"]
forward_return_series = aligned["forward_returns"]

underlying_return_series = r_pc2_test

rolling_vol = underlying_return_series.rolling(VOL_ESTIMATION_WINDOW).std()

interaction_result = interaction_regression(
    y=forward_return_series,
    x1=pc2_signal_series,
    x2=rolling_vol,
)

rolling_vol_threshold = rolling_vol.rolling(REGIME_THRESHOLD_WINDOW).median()
high_vol_regime_full = rolling_vol > rolling_vol_threshold.shift(1)
valid_regime_mask_full = high_vol_regime_full.notna()

high_vol_regime = high_vol_regime_full.reindex(pc2_signal_series.index).fillna(False)
valid_regime_mask = valid_regime_mask_full.reindex(pc2_signal_series.index).fillna(False)

high_vol_pc2 = pc2_signal_series[valid_regime_mask & high_vol_regime]
high_vol_returns = forward_return_series[valid_regime_mask & high_vol_regime]

low_vol_pc2 = pc2_signal_series[valid_regime_mask & ~high_vol_regime]
low_vol_returns = forward_return_series[valid_regime_mask & ~high_vol_regime]

high_vol_ic_result = permutation_test(high_vol_pc2, high_vol_returns)
low_vol_ic_result = permutation_test(low_vol_pc2, low_vol_returns)

rolling_vol_aligned = rolling_vol.reindex(pc2_signal_series.index)
median_split_threshold = rolling_vol.median()

median_split_high_mask = rolling_vol_aligned > median_split_threshold
median_split_low_mask = rolling_vol_aligned <= median_split_threshold

median_split_high_pc2 = pc2_signal_series[median_split_high_mask]
median_split_high_returns = forward_return_series[median_split_high_mask]
median_split_low_pc2 = pc2_signal_series[median_split_low_mask]
median_split_low_returns = forward_return_series[median_split_low_mask]

median_split_high_result = permutation_test(median_split_high_pc2, median_split_high_returns)
median_split_low_result = permutation_test(median_split_low_pc2, median_split_low_returns)

print("PC2 loadings (train period, sign-normalized to USD/JPY positive):")
for pair, w in pc2_loadings_by_pair.items():
    print(f"  {pair}: {w:.4f}")
print()

print("=== Primary test: interaction regression ===")
print("Coefficients:", interaction_result["coefficients"])
print("Std errors:", interaction_result["std_errors"])
print("t-stats:", interaction_result["t_stats"])
print("p-values:", interaction_result["p_values"])
print("R-squared:", interaction_result["r_squared"])
print("Condition number:", interaction_result["condition_number"])
print("N obs:", interaction_result["n_obs"])

print()
print("=== Robustness check 1: rolling 156-day threshold, split-sample conditional IC ===")
print(f"High-vol regime: IC={high_vol_ic_result['observed_ic']:.4f}, "
      f"p_value={high_vol_ic_result['p_value']:.4f}, n={len(high_vol_pc2)}")
print(f"Low-vol regime: IC={low_vol_ic_result['observed_ic']:.4f}, "
      f"p_value={low_vol_ic_result['p_value']:.4f}, n={len(low_vol_pc2)}")

print()
print("=== Robustness check 2: full-sample median split, conditional IC ===")
print(f"High-vol (median split): IC={median_split_high_result['observed_ic']:.4f}, "
      f"p_value={median_split_high_result['p_value']:.4f}, n={len(median_split_high_pc2)}")
print(f"Low-vol (median split): IC={median_split_low_result['observed_ic']:.4f}, "
      f"p_value={median_split_low_result['p_value']:.4f}, n={len(median_split_low_pc2)}")