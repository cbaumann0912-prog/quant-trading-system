from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from src.analysis.portfolio import markowitz_weights
from src.analysis.portfolio_stats import compute_covariance_matrix

DEV_END = "2023-12-31"

DATA_DIR = REPO_ROOT.parent / "data"

FILES = {
    "EURUSD": "EURUSD.csv",
    "GBPUSD": "GBPUSD.csv",
    "USDJPY": "USDJPY.csv",
}

pairs = {}

for pair_name, filename in FILES.items():
    path = f"{DATA_DIR}\\{filename}"
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime").sort_index().loc[:DEV_END]

    daily_close = df["Close"].resample("D").last().dropna()
    log_returns = np.log(daily_close / daily_close.shift(1)).dropna()

    pairs[pair_name] = log_returns

returns = pd.concat(pairs, axis=1, join="inner")
returns.columns = list(FILES.keys())

p_bar = returns.mean().to_numpy()
n_assets = returns.shape[1]
ones = np.ones(n_assets)

sigma = compute_covariance_matrix(returns)
sigma_inv = np.linalg.inv(sigma)

x_minvar = sigma_inv @ ones / (ones @ sigma_inv @ ones)
minvar_return = float(p_bar @ x_minvar)

upper_bound = 1.5 * p_bar.max()

n_points = 50
target_grid = np.linspace(minvar_return, upper_bound, n_points)

ann_factor = len(returns) / ((returns.index.max() - returns.index.min()).days / 365.25)

results = []
for target in target_grid:
    try:
        res = markowitz_weights(returns, target_return=target, ann_factor=ann_factor)
        results.append({
            "target_return": target,
            "weights": res["weights"],
        })
    except np.linalg.LinAlgError:
        print(f"Skipped target_return={target:.6f}: singular system")
    except RuntimeError as exc:
        print(f"Skipped target_return={target:.6f}: {exc}")

frontier = pd.DataFrame(results)

n_rows_wanted = 10
row_indices = np.linspace(0, len(frontier) - 1, n_rows_wanted).round().astype(int)
table_rows = frontier.iloc[row_indices].reset_index(drop=True)

weights_table = pd.DataFrame(
    table_rows["weights"].tolist(),
    columns=FILES.keys(),
)
weights_table.insert(0, "target_return", table_rows["target_return"].values)

print("\nWeights table (10 points across the swept range):")
print(weights_table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

first_row = weights_table.iloc[0]
last_row = weights_table.iloc[-1]
delta_target = last_row["target_return"] - first_row["target_return"]

print("\nWeight sensitivity (Δweight / Δtarget_return) between consecutive points:")

slope_rows = []
for i in range(len(weights_table) - 1):
    row = weights_table.iloc[i]
    next_row = weights_table.iloc[i + 1]
    delta_target = next_row["target_return"] - row["target_return"]

    slope_row = {
        "target_return_from": row["target_return"],
        "target_return_to": next_row["target_return"],
    }
    for asset in FILES.keys():
        delta_weight = next_row[asset] - row[asset]
        slope_row[asset] = delta_weight / delta_target
    slope_rows.append(slope_row)

slope_table = pd.DataFrame(slope_rows)
print(slope_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))