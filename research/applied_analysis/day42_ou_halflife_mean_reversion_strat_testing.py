import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from scipy import stats
from src.data.stationarity import adf_test, kpss_test
from src.signals.cointegration import ou_half_life
from src.evaluation.bootstrap import block_bootstrap

DATA_DIR = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]

MA_WINDOW = 100
VOL_WINDOW = 100
ENTRY_THRESHOLD = 1.0
POOL_SPLIT = 1.5
REVERSION_X = 1.0
CAP_MULTIPLIER = 3
FORWARD_HORIZON = 5
N_BOOTSTRAP = 1000
BLOCK_SIZE = 20
N_PERMUTATIONS = 10000
ALPHA = 0.05
SEED = 42

rng = np.random.default_rng(SEED)


print("=" * 70)
print("SECTION 1 — MA WINDOW SELECTION VIA HALF-LIFE PLATEAU SEARCH")
print("DISREGARDED: no plateau found, half-life scales mechanically with window")
print("=" * 70)

DAYS_PER_MONTH = 26
MONTHS = list(range(3, 25, 3))
GRID_WINDOWS = [m * DAYS_PER_MONTH for m in MONTHS]

grid_results = []

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f"{pair}.csv")
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")
    prices = df["Close"]

    for months, window in zip(MONTHS, GRID_WINDOWS):
        ma = prices.rolling(window=window).mean()
        deviation = (prices - ma).dropna()
        ou_result = ou_half_life(deviation)
        theta = ou_result["theta"]
        half_life = np.log(2) / theta
        grid_results.append({
            "pair": pair,
            "months": months,
            "ma_window_days": window,
            "half_life": round(half_life, 5),
            "half_life_to_window_ratio": round(half_life / window, 5),
        })

grid_df = pd.DataFrame(grid_results)
pd.set_option("display.float_format", lambda x: f"{x:.5f}")
for pair in PAIRS:
    print(f"\n{pair}")
    pair_table = grid_df[grid_df["pair"] == pair].drop(columns="pair")
    print(pair_table.to_string(index=False))


print("\n" + "=" * 70)
print("SECTION 2 — Z-SCORE CONSTRUCTION, THRESHOLD SEPARATION, HALF-LIFE")
print("ADOPTED: informed threshold cutoff (1.5) and censoring cap (3x half-life)")
print("=" * 70)

Z_THRESHOLDS = np.round(np.arange(1.0, 3.1, 0.1), 1)

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f"{pair}.csv")
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")

    daily_prices = df["Close"].resample("D").last().dropna()
    ma = daily_prices.rolling(window=MA_WINDOW).mean()
    deviation = (daily_prices - ma).dropna()
    rolling_vol = deviation.rolling(window=VOL_WINDOW).std()
    z_score = (deviation / rolling_vol).dropna()

    print(f"\n{pair}  (n_obs={len(z_score)})")
    print(z_score.describe())

    ou_result = ou_half_life(z_score)
    theta = ou_result["theta"]
    half_life = np.log(2) / theta
    cap_3x = CAP_MULTIPLIER * half_life

    print(f"z-score OU fit: theta={theta:.5f}, half_life={half_life:.2f} days, 3x cap={cap_3x:.1f} days")

    print("Threshold separation (|z| >= x):")
    for threshold in Z_THRESHOLDS:
        count_above = (z_score.abs() >= threshold).sum()
        pct_above = 100 * count_above / len(z_score)
        print(f"  |z| >= {threshold:.1f}: {count_above:6d} obs ({pct_above:5.2f}%)")


print("\n" + "=" * 70)
print("SECTION 3 — PEAK-CONFIRMATION BUFFER (X) DIAGNOSTIC")
print("DISREGARDED: monotonic decline, distance-to-target confound, no plateau")
print("ADOPTED BY CONVENTION INSTEAD: X = 1.0 (matches entry threshold)")
print("=" * 70)

X_CANDIDATES = np.round(np.arange(0.3, 1.01, 0.1), 1)
MAX_LOOKFORWARD_DIAGNOSTIC = 100

combined_reverted = {x: 0 for x in X_CANDIDATES}
combined_retouched = {x: 0 for x in X_CANDIDATES}
combined_unresolved = {x: 0 for x in X_CANDIDATES}

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f"{pair}.csv")
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")

    daily_prices = df["Close"].resample("D").last().dropna()
    ma = daily_prices.rolling(window=MA_WINDOW).mean()
    deviation = (daily_prices - ma).dropna()
    rolling_vol = deviation.rolling(window=VOL_WINDOW).std()
    z_score = (deviation / rolling_vol).dropna()

    z = z_score.values
    n = len(z)

    reverted_counts = {x: 0 for x in X_CANDIDATES}
    retouched_counts = {x: 0 for x in X_CANDIDATES}
    unresolved_counts = {x: 0 for x in X_CANDIDATES}

    i = 0
    while i < n:
        if abs(z[i]) >= ENTRY_THRESHOLD:
            sign = 1 if z[i] > 0 else -1
            running_peak = z[i] * sign
            peak_events = [(i, running_peak)]

            j = i + 1
            while j < n and np.sign(z[j]) == sign:
                mag = z[j] * sign
                if mag > running_peak:
                    running_peak = mag
                    peak_events.append((j, running_peak))
                j += 1
            excursion_end = j - 1

            for (peak_idx, peak_val) in peak_events:
                for x in X_CANDIDATES:
                    target = peak_val - x
                    resolved = False
                    scan_end = min(peak_idx + 1 + MAX_LOOKFORWARD_DIAGNOSTIC, n)
                    for k in range(peak_idx + 1, scan_end):
                        if np.sign(z[k]) != sign:
                            break
                        mag_k = z[k] * sign
                        if mag_k <= target:
                            reverted_counts[x] += 1
                            resolved = True
                            break
                        if mag_k > peak_val:
                            retouched_counts[x] += 1
                            resolved = True
                            break
                    if not resolved:
                        unresolved_counts[x] += 1

            i = excursion_end + 1
        else:
            i += 1

    print(f"\n{pair}")
    for x in X_CANDIDATES:
        total = reverted_counts[x] + retouched_counts[x] + unresolved_counts[x]
        pct_reverted = 100 * reverted_counts[x] / total if total else float("nan")
        print(f"  X={x}: reverted_first={reverted_counts[x]:4d} ({pct_reverted:5.1f}%)  "
              f"retouched_first={retouched_counts[x]:4d}  unresolved={unresolved_counts[x]:4d}  total={total}")
        combined_reverted[x] += reverted_counts[x]
        combined_retouched[x] += retouched_counts[x]
        combined_unresolved[x] += unresolved_counts[x]

print("\nCOMBINED ACROSS ALL THREE PAIRS")
for x in X_CANDIDATES:
    total = combined_reverted[x] + combined_retouched[x] + combined_unresolved[x]
    pct_reverted = 100 * combined_reverted[x] / total if total else float("nan")
    print(f"  X={x}: reverted_first={combined_reverted[x]:4d} ({pct_reverted:5.1f}%)  "
          f"retouched_first={combined_retouched[x]:4d}  unresolved={combined_unresolved[x]:4d}  total={total}")


print("\n" + "=" * 70)
print("SECTION 4 — TEST 1: BASE MEAN-REVERSION EXISTENCE (ADF, KPSS, OU THETA CI)")
print("RESULT: PASSED on ADF/KPSS agreement; bootstrap CI on theta unreliable, disregarded")
print("=" * 70)

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f"{pair}.csv")
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")

    daily_prices = df["Close"].resample("D").last().dropna()
    ma = daily_prices.rolling(window=MA_WINDOW).mean()
    deviation = (daily_prices - ma).dropna()
    rolling_vol = deviation.rolling(window=VOL_WINDOW).std()
    z_score = (deviation / rolling_vol).dropna()

    print(f"\n{pair}  (n_obs={len(z_score)})")

    adf_result = adf_test(z_score, regression="c")
    print(f"ADF: stat={adf_result['adf_stat']:.4f}, p={adf_result['adf_p']:.5f}, "
          f"reject_unit_root={adf_result['reject_null']}")

    kpss_result = kpss_test(z_score, regression="c")
    print(f"KPSS: stat={kpss_result['kpss_stat']:.4f}, p={kpss_result['kpss_p']:.5f}, "
          f"reject_stationarity={kpss_result['reject_null']}")

    stationarity_verdict = adf_result["reject_null"] and not kpss_result["reject_null"]
    print(f"Combined stationarity verdict: {stationarity_verdict}")

    point_estimate = ou_half_life(z_score)
    theta_point = point_estimate["theta"]
    print(f"OU theta point estimate: {theta_point:.5f}")

    z_array = z_score.to_numpy()

    def theta_statistic(sample_array):
        sample_series = pd.Series(sample_array)
        result = ou_half_life(sample_series)
        return result["theta"]

    theta_boot_samples = block_bootstrap(
        series=z_array,
        block_size=BLOCK_SIZE,
        n_samples=N_BOOTSTRAP,
        statistic_fn=theta_statistic,
        seed=SEED,
    )

    ci_lower = np.percentile(theta_boot_samples, 100 * ALPHA / 2)
    ci_upper = np.percentile(theta_boot_samples, 100 * (1 - ALPHA / 2))
    print(f"Block bootstrap 95% CI on theta (UNRELIABLE, block-boundary artifact): "
          f"[{ci_lower:.5f}, {ci_upper:.5f}]")

    test1_pass = stationarity_verdict
    print(f"Test 1 (base mean-reversion existence): {'PASS' if test1_pass else 'FAIL'}")


print("\n" + "=" * 70)
print("SECTION 5 — TEST 2: NONLINEARITY VIA REVERSION TIME (PERMUTATION TEST)")
print("RESULT: FAILED, all three pairs")
print("=" * 70)

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f"{pair}.csv")
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")

    daily_prices = df["Close"].resample("D").last().dropna()
    ma = daily_prices.rolling(window=MA_WINDOW).mean()
    deviation = (daily_prices - ma).dropna()
    rolling_vol = deviation.rolling(window=VOL_WINDOW).std()
    z_score = (deviation / rolling_vol).dropna()

    ou_result = ou_half_life(z_score)
    half_life = np.log(2) / ou_result["theta"]
    censoring_cap = CAP_MULTIPLIER * half_life

    z = z_score.values
    n = len(z)

    excursions = []
    i = 0
    while i < n:
        if abs(z[i]) >= ENTRY_THRESHOLD:
            sign = 1 if z[i] > 0 else -1
            running_peak = z[i] * sign
            peak_idx = i

            j = i + 1
            while j < n and np.sign(z[j]) == sign:
                mag = z[j] * sign
                if mag > running_peak:
                    running_peak = mag
                    peak_idx = j
                j += 1
            excursion_end_idx = j - 1

            target = running_peak - REVERSION_X
            reversion_time = None
            scan_end = min(peak_idx + 1 + int(np.ceil(censoring_cap)), n)
            for k in range(peak_idx + 1, scan_end):
                if np.sign(z[k]) != sign:
                    reversion_time = k - peak_idx
                    break
                mag_k = z[k] * sign
                if mag_k <= target:
                    reversion_time = k - peak_idx
                    break

            if reversion_time is not None and reversion_time <= censoring_cap:
                excursions.append({"peak": running_peak, "reversion_time": reversion_time, "censored": False})
            else:
                excursions.append({"peak": running_peak, "reversion_time": censoring_cap, "censored": True})

            i = excursion_end_idx + 1
        else:
            i += 1

    excursions_df = pd.DataFrame(excursions)
    large_pool = excursions_df[excursions_df["peak"] >= POOL_SPLIT]["reversion_time"].values
    small_pool = excursions_df[excursions_df["peak"] < POOL_SPLIT]["reversion_time"].values

    observed_diff = np.mean(small_pool) - np.mean(large_pool)
    pooled = np.concatenate([small_pool, large_pool])
    n_small = len(small_pool)

    null_diffs = np.empty(N_PERMUTATIONS)
    for p in range(N_PERMUTATIONS):
        shuffled = rng.permutation(pooled)
        null_diffs[p] = np.mean(shuffled[:n_small]) - np.mean(shuffled[n_small:])

    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    print(f"\n{pair}: half_life={half_life:.2f}d, cap={censoring_cap:.2f}d, n_excursions={len(excursions_df)}")
    print(f"  small pool n={len(small_pool)}, mean={np.mean(small_pool):.2f}d")
    print(f"  large pool n={len(large_pool)}, mean={np.mean(large_pool):.2f}d")
    print(f"  observed_diff(small-large)={observed_diff:.3f}d, p={p_value:.5f}")
    print(f"  Test 2: {'PASS' if (p_value < ALPHA and observed_diff > 0) else 'FAIL'}")


print("\n" + "=" * 70)
print("SECTION 6 — TEST 2b: NONLINEARITY VIA 5-DAY FORWARD RETURN (PERMUTATION TEST)")
print("RESULT: FAILED, all three pairs (2 of 3 wrong-signed)")
print("=" * 70)

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f"{pair}.csv")
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")

    daily_prices = df["Close"].resample("D").last().dropna()
    ma = daily_prices.rolling(window=MA_WINDOW).mean()
    deviation = (daily_prices - ma).dropna()
    rolling_vol = deviation.rolling(window=VOL_WINDOW).std()
    z_score = (deviation / rolling_vol).dropna()

    aligned_prices = daily_prices.loc[z_score.index]
    z = z_score.values
    prices = aligned_prices.values
    n = len(z)

    excursion_records = []
    i = 0
    while i < n:
        if abs(z[i]) >= ENTRY_THRESHOLD:
            sign = 1 if z[i] > 0 else -1
            running_peak = z[i] * sign
            peak_idx = i

            j = i + 1
            while j < n and np.sign(z[j]) == sign:
                mag = z[j] * sign
                if mag > running_peak:
                    running_peak = mag
                    peak_idx = j
                j += 1
            excursion_end_idx = j - 1

            forward_idx = peak_idx + FORWARD_HORIZON
            if forward_idx < n:
                price_at_peak = prices[peak_idx]
                price_forward = prices[forward_idx]
                raw_return = (price_forward - price_at_peak) / price_at_peak
                signed_return = -sign * raw_return
                excursion_records.append({"peak": running_peak, "signed_forward_return": signed_return})

            i = excursion_end_idx + 1
        else:
            i += 1

    excursions_df = pd.DataFrame(excursion_records)
    large_pool = excursions_df[excursions_df["peak"] >= POOL_SPLIT]["signed_forward_return"].values
    small_pool = excursions_df[excursions_df["peak"] < POOL_SPLIT]["signed_forward_return"].values

    observed_diff = np.mean(large_pool) - np.mean(small_pool)
    pooled = np.concatenate([small_pool, large_pool])
    n_small = len(small_pool)

    null_diffs = np.empty(N_PERMUTATIONS)
    for p in range(N_PERMUTATIONS):
        shuffled = rng.permutation(pooled)
        null_diffs[p] = np.mean(shuffled[n_small:]) - np.mean(shuffled[:n_small])

    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    print(f"\n{pair}: n={len(excursions_df)}")
    print(f"  small pool n={len(small_pool)}, mean_return={np.mean(small_pool)*100:.4f}%")
    print(f"  large pool n={len(large_pool)}, mean_return={np.mean(large_pool)*100:.4f}%")
    print(f"  observed_diff(large-small)={observed_diff*100:.4f}%, p={p_value:.5f}")
    print(f"  Test 2b: {'PASS' if (p_value < ALPHA and observed_diff > 0) else 'FAIL'}")


print("\n" + "=" * 70)
print("SECTION 7 — TEST 2c: NONLINEARITY VIA IC (SPEARMAN, PERMUTATION TEST)")
print("RESULT: FAILED, all three pairs, closest to zero of any variant")
print("=" * 70)

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f"{pair}.csv")
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")

    daily_prices = df["Close"].resample("D").last().dropna()
    ma = daily_prices.rolling(window=MA_WINDOW).mean()
    deviation = (daily_prices - ma).dropna()
    rolling_vol = deviation.rolling(window=VOL_WINDOW).std()
    z_score = (deviation / rolling_vol).dropna()

    aligned_prices = daily_prices.loc[z_score.index]
    z = z_score.values
    prices = aligned_prices.values
    n = len(z)

    excursion_records = []
    i = 0
    while i < n:
        if abs(z[i]) >= ENTRY_THRESHOLD:
            sign = 1 if z[i] > 0 else -1
            running_peak = z[i] * sign
            peak_idx = i

            j = i + 1
            while j < n and np.sign(z[j]) == sign:
                mag = z[j] * sign
                if mag > running_peak:
                    running_peak = mag
                    peak_idx = j
                j += 1
            excursion_end_idx = j - 1

            forward_idx = peak_idx + FORWARD_HORIZON
            if forward_idx < n:
                price_at_peak = prices[peak_idx]
                price_forward = prices[forward_idx]
                raw_return = (price_forward - price_at_peak) / price_at_peak
                signed_return = -sign * raw_return
                excursion_records.append({"peak": running_peak, "signed_forward_return": signed_return})

            i = excursion_end_idx + 1
        else:
            i += 1

    excursions_df = pd.DataFrame(excursion_records)
    peaks = excursions_df["peak"].to_numpy()
    returns = excursions_df["signed_forward_return"].to_numpy()

    observed_ic, _ = stats.spearmanr(peaks, returns)

    null_ics = np.empty(N_PERMUTATIONS)
    for p in range(N_PERMUTATIONS):
        shuffled_returns = rng.permutation(returns)
        null_ics[p], _ = stats.spearmanr(peaks, shuffled_returns)

    p_value = np.mean(np.abs(null_ics) >= np.abs(observed_ic))

    print(f"\n{pair}: n={len(excursions_df)}")
    print(f"  observed IC={observed_ic:.4f}, p={p_value:.5f}")
    print(f"  Test 2c: {'PASS' if (p_value < ALPHA and observed_ic > 0) else 'FAIL'}")


print("\n" + "=" * 70)
print("SUMMARY: OU Half-Life Mean-Reversion (Z-Score Threshold) — DISCARDED")
print("Test 1 PASSED. Tests 2, 2b, 2c all FAILED. Strategy candidate closed.")
print("=" * 70)