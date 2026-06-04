import sys
sys.path.insert(0, r'C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\summer2026')

import pandas as pd
import numpy as np
from scipy import stats
from src.evaluation.multiple_testing import bonferroni_correction, benjamini_hochberg
from src.stats.hypothesis_tests import t_test_mean

alpha = 0.05

p_values = []
labels = []

PAIRS = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/JPY": "USDJPY",
}

ARCHIVE_DIR = r"C:\Users\clayb\onedrive\desktop\career\02_quant_projects\summer2026\archive\OG_results\realistic"

for pair, folder in PAIRS.items():
    df = pd.read_csv(f"{ARCHIVE_DIR}\\{folder}\\daily_returns.csv", parse_dates=["date"])
    df = df[df["daily_return_pct"] != 0].dropna()
    returns = df["daily_return_pct"].values

    result = t_test_mean(returns, 0, 1 - alpha)
    labels.append(pair)
    p_values.append(float(result["p_value"]))

bonferroni_results = bonferroni_correction(p_values, alpha)
bh_results = benjamini_hochberg(p_values, alpha)

print("Null: zero mean log return per pair (methodology demonstration)")
print(f"alpha = {alpha}, m = {len(p_values)}")
print(f"Bonferroni threshold: {alpha / len(p_values):.5f}\n")

print(f"{'Pair':<12} {'p-value':>10} {'Bonferroni':>12} {'BH':>6}")
print("-" * 44)
for label, p, bf, bh in zip(labels, p_values, bonferroni_results, bh_results):
    print(f"{label:<12} {p:>10.5f} {str(bf):>12} {str(bh):>6}")