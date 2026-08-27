from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from src.signals.cointegration import johansen_test

DEV_END = "2023-12-31"

DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = {
    "EURUSD": DATA_DIR / "EURUSD.csv",
    "GBPUSD": DATA_DIR / "GBPUSD.csv",
    "USDJPY": DATA_DIR / "USDJPY.csv",
}

def load_daily_close(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime").sort_index().loc[:DEV_END]
    return df["Close"].resample("D").last().dropna()


series = {name: load_daily_close(path) for name, path in PAIRS.items()}
combined = pd.DataFrame(series).dropna()

print(f"Combined sample: {combined.shape[0]} days, {combined.shape[1]} series")
print(f"Date range: {combined.index.min()} to {combined.index.max()}\n")

result = johansen_test(combined)

print("Trace Test")
print("Stat:           ", result["trace_stat"])
print("Crit (90/95/99):\n", result["trace_crit_vals"])
print()

print("Max Eigenvalue Test")
print("Stat:           ", result["max_eig_stat"])
print("Crit (90/95/99):\n", result["max_eig_crit_vals"])
print()

print("Eigenvalues")
print(result["eigenvalues"])
print()

print("Eigenvectors (columns, ordered by descending eigenvalue)")
print("Series order:", result["series_names"])
print(result["eigenvectors"])
print()

print("Inferred Rank (95% confidence)")
print("rank_trace:  ", result["rank_trace"])
print("rank_max_eig:", result["rank_max_eig"])

if result["rank_trace"] != result["rank_max_eig"]:
    print("\n⚠ Trace and max-eigenvalue tests DISAGREE on rank — address explicitly in audit.")

if result["rank_trace"] >= 1:
    dominant_vec = result["eigenvectors"][:, 0]
    print("\nDominant cointegrating vector (largest eigenvalue):")
    for name, weight in zip(result["series_names"], dominant_vec):
        print(f"  {name}: {weight:.4f}")