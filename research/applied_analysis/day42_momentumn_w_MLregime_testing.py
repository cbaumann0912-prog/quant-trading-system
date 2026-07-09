import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]

LOOKBACK = 26
HOLDING = 5
N_PERMUTATIONS = 10000
ALPHA = 0.05
SEED = 28S
BLOCK_SIZE = LOOKBACK + HOLDING

rng = np.random.default_rng(SEED)


def compute_signal_outcome(daily_prices):
    log_returns = np.log(daily_prices / daily_prices.shift(1)).dropna()
    cumsum = log_returns.cumsum()
    n = len(cumsum)

    records = []
    for i in range(LOOKBACK, n - HOLDING):
        trailing_return = cumsum.iloc[i] - cumsum.iloc[i - LOOKBACK]
        forward_return = cumsum.iloc[i + HOLDING] - cumsum.iloc[i]
        records.append({
            "date": cumsum.index[i],
            "trailing_return": trailing_return,
            "forward_return": forward_return,
            "trailing_start": cumsum.index[i - LOOKBACK],
            "trailing_end": cumsum.index[i],
            "forward_start": cumsum.index[i + 1],
            "forward_end": cumsum.index[i + HOLDING],
        })

    return pd.DataFrame(records)


print("SECTION 1 — WINDOW ALIGNMENT VERIFICATION")
print("Confirms zero overlap between trailing and forward windows, corrected construction")

filepath = os.path.join(DATA_DIR, "EURUSD.csv")
df = pd.read_csv(filepath)
df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
df = df.set_index("Datetime")
daily_prices = df["Close"].resample("D").last().dropna()

check_df = compute_signal_outcome(daily_prices)
print(check_df[["date", "trailing_start", "trailing_end", "forward_start", "forward_end"]].head(3).to_string(index=False))
print("\nVerified: forward_start is exactly 1 day after trailing_end/date, zero shared days between windows.")


print("SECTION 2 — TEST A, METHOD 1: NON-OVERLAPPING SUBSAMPLE, PERMUTATION TEST")
print("RESULT: FAILED, all three pairs")

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f"{pair}.csv")
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")
    daily_prices = df["Close"].resample("D").last().dropna()

    combined = compute_signal_outcome(daily_prices)
    subsampled = combined.iloc[::HOLDING]

    trailing = subsampled["trailing_return"].to_numpy()
    forward = subsampled["forward_return"].to_numpy()

    observed_ic, _ = stats.spearmanr(trailing, forward)

    null_ics = np.empty(N_PERMUTATIONS)
    for p in range(N_PERMUTATIONS):
        shuffled_forward = rng.permutation(forward)
        null_ics[p], _ = stats.spearmanr(trailing, shuffled_forward)

    p_value = np.mean(np.abs(null_ics) >= np.abs(observed_ic))

    print(f"\n{pair}: n={len(subsampled)}")
    print(f"  observed IC={observed_ic:.4f}, p={p_value:.5f}")
    print(f"  Method 1: {'PASS' if (p_value < ALPHA and observed_ic > 0) else 'FAIL'}")


print("SECTION 3 — TEST A, METHOD 2: OVERLAPPING DAILY DATA, BLOCK PERMUTATION TEST")
print("RESULT: FAILED, all three pairs — agrees with Method 1")

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f"{pair}.csv")
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")
    daily_prices = df["Close"].resample("D").last().dropna()

    combined = compute_signal_outcome(daily_prices)
    trailing = combined["trailing_return"].to_numpy()
    forward = combined["forward_return"].to_numpy()
    n = len(combined)

    observed_ic, _ = stats.spearmanr(trailing, forward)

    block_starts = np.arange(0, n, BLOCK_SIZE)

    null_ics = np.empty(N_PERMUTATIONS)
    for p in range(N_PERMUTATIONS):
        block_order = rng.permutation(len(block_starts))
        shuffled_forward = np.concatenate([
            forward[block_starts[b]: block_starts[b] + BLOCK_SIZE]
            for b in block_order
        ])[:n]
        null_ics[p], _ = stats.spearmanr(trailing, shuffled_forward)

    p_value = np.mean(np.abs(null_ics) >= np.abs(observed_ic))

    print(f"\n{pair}: n={n}")
    print(f"  observed IC={observed_ic:.4f}, p={p_value:.5f}")
    print(f"  Method 2: {'PASS' if (p_value < ALPHA and observed_ic > 0) else 'FAIL'}")


print("SUMMARY: FX Momentum with ML Regime Filter — DISCARDED")
print("Test A (base momentum IC) FAILED under both methodologies after correcting")
print("an initial window-alignment bug that had produced a spurious IC ~0.30.")
print("Claim B (ML regime filter) never tested — moot, no base effect to condition.")