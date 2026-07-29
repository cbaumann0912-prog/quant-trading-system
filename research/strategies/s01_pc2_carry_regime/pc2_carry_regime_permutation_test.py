import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.signals.pc2_carry import fit_pc2_loadings, pc2_scores, pc2_factor_returns
from src.evaluation.significance import (
    permutation_test,
    bonferroni_correction,
    benjamini_hochberg_correction,
)

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
SEED = 42
PAIRS_ORDER = ["EURUSD", "GBPUSD", "USDJPY"]

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
signal = aligned["signal"]
forward_returns = aligned["forward_returns"]

pooled_result = permutation_test(signal, forward_returns, alternative="two-sided", seed=SEED)

pos_mask = signal > 0
neg_mask = signal < 0
n_pos = int(pos_mask.sum())
n_neg = int(neg_mask.sum())

pos_result = permutation_test(signal[pos_mask], forward_returns[pos_mask], alternative="greater", seed=SEED)
neg_result = permutation_test(signal[neg_mask], forward_returns[neg_mask], alternative="greater", seed=SEED)

test_labels = ["pooled (two-sided)", "positive subset (greater)", "negative subset (greater)"]
raw_p_values = [pooled_result["p_value"], pos_result["p_value"], neg_result["p_value"]]

ALPHA = 0.05
bonferroni_reject = bonferroni_correction(raw_p_values, ALPHA)
bh_reject = benjamini_hochberg_correction(raw_p_values, ALPHA)

rho, spearman_p = spearmanr(signal.to_numpy(), forward_returns.to_numpy())
n = len(signal)
t_stat = rho * np.sqrt((n - 2) / (1 - rho**2))

print("PC2 loadings (train period, sign-normalized to USD/JPY positive):")
for pair, w in pc2_loadings_by_pair.items():
    print(f"  {pair}: {w:.4f}")
print()

print(f"n_obs (pooled, post-align): {n}")
print()

print("Pooled permutation test (two-sided):")
print(f"  observed_ic: {pooled_result['observed_ic']:.4f}")
print(f"  p_value:     {pooled_result['p_value']:.4f}")
print()

print("Split-sample permutation tests (exploratory, reduced power):")
print(f"  positive-signal subset: n={n_pos}  observed_ic={pos_result['observed_ic']:.4f}  p_value={pos_result['p_value']:.4f}  (alternative=greater)")
print(f"  negative-signal subset: n={n_neg}  observed_ic={neg_result['observed_ic']:.4f}  p_value={neg_result['p_value']:.4f}  (alternative=greater)")
print()

print(f"Multiple testing correction across all 3 tests run today (alpha={ALPHA}):")
for label, p, bf, bh in zip(test_labels, raw_p_values, bonferroni_reject, bh_reject):
    print(f"  {label:30s} p={p:.4f}  bonferroni_reject={bf}  bh_reject={bh}")
print()

print("Spearman correlation t-test (Day 8 equivalent, pooled):")
print(f"  rho:      {rho:.4f}")
print(f"  t_stat:   {t_stat:.4f}")
print(f"  p_value:  {spearman_p:.6f}")