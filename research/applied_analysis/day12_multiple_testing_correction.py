import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
from scipy import stats
from src.evaluation.significance import bonferroni_correction, benjamini_hochberg_correction
from src.stats.hypothesis_tests import t_test_mean

alpha = 0.05

p_values = []
labels = []

PAIRS = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/JPY": "USDJPY",
}

ARCHIVE_DIR = Path(
    os.environ.get("ORIGINAL_STRATEGY_DIR", REPO_ROOT.parent / "_original_strategy")
) / "OG_results" / "realistic"

if not ARCHIVE_DIR.is_dir():
    raise FileNotFoundError(
        f"Original-strategy results not found at {ARCHIVE_DIR}. "
        "Set ORIGINAL_STRATEGY_DIR to the directory containing OG_results/."
    )

for pair, folder in PAIRS.items():
    df = pd.read_csv(ARCHIVE_DIR / folder / "daily_returns.csv", parse_dates=["date"])
    df = df[df["daily_return_pct"] != 0].dropna()
    returns = df["daily_return_pct"].values

    result = t_test_mean(returns, 0, 1 - alpha)
    labels.append(pair)
    p_values.append(float(result["p_value"]))

bonferroni_results = bonferroni_correction(p_values, alpha)
bh_results = benjamini_hochberg_correction(p_values, alpha)

print("Null: zero mean log return per pair (methodology demonstration)")
print(f"alpha = {alpha}, m = {len(p_values)}")
print(f"Bonferroni threshold: {alpha / len(p_values):.5f}\n")

print(f"{'Pair':<12} {'p-value':>10} {'Bonferroni':>12} {'BH':>6}")
print("-" * 44)
for label, p, bf, bh in zip(labels, p_values, bonferroni_results, bh_results):
    print(f"{label:<12} {p:>10.5f} {str(bf):>12} {str(bh):>6}")