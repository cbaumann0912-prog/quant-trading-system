"""
Day 12 Research: Multiple testing correction — methodology demonstration.
Tests null of zero mean log return per pair (not strategy edge).
"""

import sys
import numpy as np
import pandas as pd
sys.path.append(".")

from src.evaluation.multiple_testing import bonferroni_correction, benjamini_hochberg
from src.stats.hypothesis_tests import t_test_mean

DATA_DIR = r"C:\Users\clayb\onedrive\desktop\career\02_quant_projects\data"
PAIRS = {
    "EUR/USD": "EURUSD.csv",
    "GBP/USD": "GBPUSD.csv",
    "USD/JPY": "USDJPY.csv",
}
alpha = 0.05

p_values = []
labels = []

for pair, filename in PAIRS.items():
    df = pd.read_csv(f"{DATA_DIR}\\{filename}", header=0)
    df.columns = ["datetime", "open", "high", "low", "close", "volume"]
    df["datetime"] = pd.to_datetime(df["datetime"], format='%Y%m%d %H%M%S')

    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna().values

    result = t_test_mean(log_returns, 0, 1-alpha)
    p = result["p_value"]
    labels.append(pair)
    p_values.append(float(p))

bonferroni_results = bonferroni_correction(p_values, alpha)
bh_results = benjamini_hochberg(p_values, alpha)

print("Null: zero mean log return per pair (methodology demonstration)")
print(f"alpha = {alpha}, m = {len(p_values)}")
print(f"Bonferroni threshold: {alpha / len(p_values):.5f}\n")

print(f"{'Pair':<12} {'p-value':>10} {'Bonferroni':>12} {'BH':>6}")
print("-" * 44)
for label, p, bf, bh in zip(labels, p_values, bonferroni_results, bh_results):
    print(f"{label:<12} {p:>10.5f} {str(bf):>12} {str(bh):>6}")