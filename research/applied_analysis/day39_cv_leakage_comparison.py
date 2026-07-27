import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

from src.evaluation.cross_validation import purged_cross_validation
from src.evaluation.significance import paired_sign_permutation_test

DEV_END = "2023-12-31"

DATA_DIR = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"
FILENAME = "EURUSD.csv"

MOMENTUM_LOOKBACK = 20
HOLD_PERIOD = 5
N_SPLITS_LIST = [5, 15]
EMBARGO_PCT = 0.01
RANDOM_STATE = 42
N_NEIGHBORS = 5

path = f"{DATA_DIR}\\{FILENAME}"
df = pd.read_csv(path)
df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
df = df.set_index("Datetime").sort_index().loc[:DEV_END]

daily_close = df["Close"].resample("D").last().dropna()
log_returns = np.log(daily_close / daily_close.shift(1)).dropna()

momentum_signal = np.log(daily_close / daily_close.shift(MOMENTUM_LOOKBACK)).dropna()
forward_return = np.log(daily_close.shift(-HOLD_PERIOD) / daily_close).dropna()

data = pd.concat(
    {"signal": momentum_signal, "forward_return": forward_return},
    axis=1,
    join="inner",
).dropna()

signal = data["signal"].to_numpy().reshape(-1, 1)
target = data["forward_return"].to_numpy()
dates = data.index
n_obs = len(data)

full_index = daily_close.index
start_positions = full_index.get_indexer(dates)
through_date_positions = start_positions + HOLD_PERIOD
through_dates = full_index[through_date_positions]

t1 = pd.Series(through_dates, index=dates)

for n_splits in N_SPLITS_LIST:

    ics_unshuffled = []
    kfold_unshuffled = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, test_idx in kfold_unshuffled.split(signal):
        model = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
        model.fit(signal[train_idx], target[train_idx])
        predictions = model.predict(signal[test_idx])
        ic, _ = spearmanr(predictions, target[test_idx])
        ics_unshuffled.append(ic)

    ics_shuffled = []
    kfold_shuffled = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in kfold_shuffled.split(signal):
        model = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
        model.fit(signal[train_idx], target[train_idx])
        predictions = model.predict(signal[test_idx])
        ic, _ = spearmanr(predictions, target[test_idx])
        ics_shuffled.append(ic)

    ics_purged = []
    for train_idx, test_idx in purged_cross_validation(t1, n_splits=n_splits, embargo_pct=EMBARGO_PCT):
        model = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
        model.fit(signal[train_idx], target[train_idx])
        predictions = model.predict(signal[test_idx])
        ic, _ = spearmanr(predictions, target[test_idx])
        ics_purged.append(ic)

    results = pd.DataFrame({
        "fold": np.arange(1, n_splits + 1),
        "unsh": ics_unshuffled,
        "shuf": ics_shuffled,
        "purg": ics_purged,
    })
    results["diff"] = results["unsh"] - results["purg"]

    print(f"n_splits = {n_splits}")
    print("\nPer-fold IC (KNN, k={}):".format(N_NEIGHBORS))
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    summary = pd.DataFrame({
        "method": ["unshuffled", "shuffled", "purged"],
        "mean_ic": [
            np.mean(ics_unshuffled),
            np.mean(ics_shuffled),
            np.mean(ics_purged),
        ],
        "std_ic": [
            np.std(ics_unshuffled),
            np.std(ics_shuffled),
            np.std(ics_purged),
        ],
    })

    inflation_unshuffled_all = summary.loc[0, "mean_ic"] - summary.loc[2, "mean_ic"]
    inflation_shuffled_all = summary.loc[1, "mean_ic"] - summary.loc[2, "mean_ic"]

    diffs = results["diff"].to_numpy()
    n_positive = int(np.sum(diffs > 0))
    n_negative = int(np.sum(diffs < 0))
    n_zero = int(np.sum(diffs == 0))

    max_gap_idx = int(np.argmax(np.abs(diffs)))
    max_gap_fold = int(results.loc[max_gap_idx, "fold"])
    inflation_excl_max = float(np.mean(np.delete(diffs, max_gap_idx)))

    perm_result = paired_sign_permutation_test(diffs, n_permutations=10000, seed=42, alternative="two-sided")
    p_value = perm_result["p_value"]

    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nunsh-purg inflation, all folds: {inflation_unshuffled_all:.4f}")
    print(f"shuf-purg inflation, all folds: {inflation_shuffled_all:.4f}")
    print(f"sign count (unsh-purg): +{n_positive} -{n_negative} 0={n_zero}")
    print(f"largest-gap fold: {max_gap_fold} (|diff|={abs(diffs[max_gap_idx]):.4f})")
    print(f"inflation excl. fold {max_gap_fold}: {inflation_excl_max:.4f}")
    print(f"permutation p-value (unsh-purg != 0): {p_value:.4f}")
    print(f"n={n_obs} splits={n_splits} embargo={EMBARGO_PCT} lookback={MOMENTUM_LOOKBACK} hold={HOLD_PERIOD} k={N_NEIGHBORS}")