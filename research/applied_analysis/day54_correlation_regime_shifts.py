import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.framework.data_loader import DataLoader, SUPPORTED_PAIRS
from src.stats.correlation import detect_correlation_regime_shifts

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = sorted(SUPPORTED_PAIRS)
START = "2011-01-01"
END = "2023-12-31"

WINDOW = 60
THRESHOLD = 3.0

loader = DataLoader(pairs=PAIRS, start=START, end=END, data_dir=str(DATA_DIR))
returns = loader.get_returns(log=True).dropna(how="any")

combos = list(combinations(PAIRS, 2))
results = {
    (a, b): detect_correlation_regime_shifts(returns[a], returns[b], window=WINDOW, threshold=THRESHOLD)
    for a, b in combos
}
flag_counts = {pair: flags.sum() for pair, flags in results.items()}
total_flags = sum(flag_counts.values())

combined = pd.DataFrame(results)
cluster_count = combined.sum(axis=1)
cluster_count = cluster_count[cluster_count > 0].sort_values(ascending=False)

print(f"universe: {len(PAIRS)} pairs, {len(combos)} combinations, {START} to {END}")
print(f"returns: {returns.shape[0]} obs x {returns.shape[1]} pairs, "
      f"{returns.index.min().date()} to {returns.index.max().date()}")
print()

print(f"total flags: {total_flags} ({total_flags / len(combos):.2f} / combo)")
print()

print("top 15 clustering dates (n of 45 combos flagged):")
print(cluster_count.head(15).to_string())
print()

print("cluster-size distribution:")
print(cluster_count.value_counts().sort_index(ascending=False).to_string())
print()

print("combos flagged, top 5 dates:")
for date in cluster_count.head(5).index:
    row = combined.loc[date]
    flagged = [f"{a}/{b}" for (a, b) in row[row].index]
    print(f"  {date.date()} ({len(flagged)}): {flagged}")


def paired_block_bootstrap(s1: np.ndarray, s2: np.ndarray, block_size: int,
                            n_samples: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Joint block bootstrap: same block-start indices applied to both series."""
    n = len(s1)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    out = []
    for _ in range(n_samples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        r1 = np.concatenate([s1[s:s + block_size] for s in starts])[:n]
        r2 = np.concatenate([s2[s:s + block_size] for s in starts])[:n]
        out.append((r1, r2))
    return out


def false_alarm_rate(s1: np.ndarray, s2: np.ndarray, window: int, threshold: float,
                      block_size: int, n_samples: int, seed: int) -> tuple[int, int]:
    """(total false flags, total evaluation points) across all resamples."""
    resamples = paired_block_bootstrap(s1, s2, block_size, n_samples, seed=seed)
    total_flags = 0
    total_eval_points = 0
    for r1, r2 in resamples:
        idx = pd.RangeIndex(len(r1))
        f = detect_correlation_regime_shifts(
            pd.Series(r1, index=idx), pd.Series(r2, index=idx),
            window=window, threshold=threshold,
        )
        total_flags += int(f.sum())
        n_valid = len(r1) - window + 1
        total_eval_points += int(np.ceil(n_valid / window))
    return total_flags, total_eval_points


QUIET_START, QUIET_END = "2014-08-29", "2018-04-26"
BLOCK_SIZE = 21
N_BOOTSTRAP = 100
BOOTSTRAP_SEED = 28

quiet_segment = returns.loc[QUIET_START:QUIET_END, ["EURUSD", "GBPUSD"]]

false_flags, eval_points = false_alarm_rate(
    quiet_segment["EURUSD"].to_numpy(), quiet_segment["GBPUSD"].to_numpy(),
    window=WINDOW, threshold=THRESHOLD,
    block_size=BLOCK_SIZE, n_samples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
)
p_hat = false_flags / eval_points

print()
print(f"bootstrap false-alarm rate (EUR/USD-GBP/USD, {QUIET_START} to {QUIET_END}, "
      f"block_size={BLOCK_SIZE}, n={N_BOOTSTRAP}): {false_flags}/{eval_points} = {p_hat:.4f}")

print()
print("P(>= k of 45 combos flag same date | chance, iid approx, rate=p_hat):")
for k in [8, 9, 10, 11, 12, 13]:
    prob = 1 - binom.cdf(k - 1, len(combos), p_hat)
    print(f"  k={k}: {prob:.2e}")

WINDOW_FAST = 30
fast_false_flags, fast_eval_points = false_alarm_rate(
    quiet_segment["EURUSD"].to_numpy(), quiet_segment["GBPUSD"].to_numpy(),
    window=WINDOW_FAST, threshold=THRESHOLD,
    block_size=BLOCK_SIZE, n_samples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
)
p_hat_fast = fast_false_flags / fast_eval_points

print()
print(f"window={WINDOW_FAST} false-alarm rate: {fast_false_flags}/{fast_eval_points} = {p_hat_fast:.4f} "
      f"(window={WINDOW}: {p_hat:.4f})")

if p_hat_fast <= 2 * p_hat:
    results_fast = {
        (a, b): detect_correlation_regime_shifts(returns[a], returns[b], window=WINDOW_FAST, threshold=THRESHOLD)
        for a, b in combos
    }
    total_fast = sum(f.sum() for f in results_fast.values())
    combined_fast = pd.DataFrame(results_fast)
    cluster_fast = combined_fast.sum(axis=1)
    cluster_fast = cluster_fast[cluster_fast > 0].sort_values(ascending=False)

    print()
    print(f"window={WINDOW_FAST} total flags: {total_fast} ({total_fast / len(combos):.2f} / combo) "
          f"vs window={WINDOW}: {total_flags} ({total_flags / len(combos):.2f} / combo)")
    print(f"window={WINDOW_FAST} top 10 clustering dates:")
    print(cluster_fast.head(10).to_string())
else:
    print("false-alarm rate degrades too much at this window; skipping real-data run.")

identified_events = {
    "2013-02-13": "Abenomics yen depreciation (Abe elected Dec 26 2012, USD/JPY 80->90 by late Jan 2013)",
    "2022-02-28": "Russia's invasion of Ukraine (Feb 24 2022)",
    "2012-05-08": "Greek election (May 6 2012), Grexit fears",
}
print()
print("event lookup:")
for date, driver in identified_events.items():
    print(f"  {date}: {driver}")
