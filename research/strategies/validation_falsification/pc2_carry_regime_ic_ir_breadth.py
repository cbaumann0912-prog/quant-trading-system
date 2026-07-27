import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from src.features.pca import pca
from src.analysis.performance_analyzer import information_coefficient, information_ratio
from src.evaluation.bootstrap import block_bootstrap

DATA_DIR = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"

FILES = {
    "EURUSD": "EURUSD.csv",
    "GBPUSD": "GBPUSD.csv",
    "USDJPY": "USDJPY.csv",
}

SPLIT_DATE = "2021-01-01"
N_COMPONENTS = 3
PAIRS_ORDER = ["EURUSD", "GBPUSD", "USDJPY"]
MAX_LAG = 30
RHO_THRESHOLD = 0.1
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
CI_ALPHA = 0.05

returns = {}
for pair_name, filename in FILES.items():
    path = f"{DATA_DIR}\\{filename}"
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")
    daily_close = df["Close"].resample("D").last().dropna()
    returns[pair_name] = np.log(daily_close / daily_close.shift(1)).dropna()

returns_df = pd.DataFrame(returns)[PAIRS_ORDER].dropna()

train_returns = returns_df[returns_df.index < SPLIT_DATE]
test_returns = returns_df[returns_df.index >= SPLIT_DATE]

components, explained_variance, projected = pca(train_returns.to_numpy(), n_components=N_COMPONENTS)
pc2_loadings = components[:, 1]
pc2_loadings_by_pair = dict(zip(PAIRS_ORDER, pc2_loadings))
train_mean = train_returns.to_numpy().mean(axis=0)

if pc2_loadings_by_pair["USDJPY"] < 0:
    pc2_loadings = -pc2_loadings
    pc2_loadings_by_pair = dict(zip(PAIRS_ORDER, pc2_loadings))

centered_test_returns = test_returns.to_numpy() - train_mean
pc2_scores_test = pd.Series(centered_test_returns @ pc2_loadings, index=test_returns.index)

r_pc2_test = (
    pc2_loadings_by_pair["EURUSD"] * test_returns["EURUSD"]
    + pc2_loadings_by_pair["GBPUSD"] * test_returns["GBPUSD"]
    + pc2_loadings_by_pair["USDJPY"] * test_returns["USDJPY"]
)

signal = pc2_scores_test.iloc[:-1]
forward_returns = r_pc2_test.iloc[1:]
forward_returns.index = signal.index

aligned = pd.concat([signal.rename("signal"), forward_returns.rename("forward_returns")], axis=1).dropna()
signal = aligned["signal"]
forward_returns = aligned["forward_returns"]

n = len(signal)

pos_mask = signal > 0
neg_mask = signal < 0

regimes = {
    "pooled": (signal, forward_returns),
    "positive": (signal[pos_mask], forward_returns[pos_mask]),
    "negative": (signal[neg_mask], forward_returns[neg_mask]),
}

WINDOW_GRID = [20, 40, 60, 90, 120]
MAX_LAG_REGIME = 10

results = {}

for regime_name, (regime_signal, regime_forward_returns) in regimes.items():
    regime_n = len(regime_signal)
    regime_ic = information_coefficient(regime_signal, regime_forward_returns)

    robustness_rows = []
    for window in WINDOW_GRID:
        if window >= regime_n:
            continue
        rolling_ic_values = []
        rolling_ic_dates = []
        for start in range(0, regime_n - window, window):
            window_signal = regime_signal.iloc[start:start + window]
            window_forward_returns = regime_forward_returns.iloc[start:start + window]
            window_ic = information_coefficient(window_signal, window_forward_returns)
            rolling_ic_values.append(window_ic)
            rolling_ic_dates.append(regime_signal.index[start])

        rolling_ic_series = pd.Series(rolling_ic_values, index=rolling_ic_dates)
        ir_empirical_window = information_ratio(rolling_ic_series, method="empirical")

        robustness_rows.append({
            "window": window,
            "n_windows": len(rolling_ic_series),
            "mean_ic": rolling_ic_series.mean(),
            "std_ic": rolling_ic_series.std(),
            "ir_empirical": ir_empirical_window,
        })

    max_lag_regime = min(MAX_LAG_REGIME, regime_n - 1)
    lag_1_rho = regime_signal.autocorr(lag=1) if regime_n > 1 else float("nan")

    decorrelation_lag = None
    for lag in range(1, max_lag_regime + 1):
        lag_rho = regime_signal.autocorr(lag=lag)
        if abs(lag_rho) < RHO_THRESHOLD:
            decorrelation_lag = lag
            break

    if decorrelation_lag is not None:
        rho_estimate = np.mean([regime_signal.autocorr(lag=k) for k in range(1, decorrelation_lag + 1)])
    else:
        rho_estimate = lag_1_rho

    br_raw = regime_n
    br_eff = regime_n / (1 + (regime_n - 1) * rho_estimate) if not np.isnan(rho_estimate) else float("nan")

    ir_fundamental_law_raw = information_ratio(regime_ic, method="fundamental_law", breadth=br_raw)
    ir_fundamental_law_eff = (
        information_ratio(regime_ic, method="fundamental_law", breadth=br_eff)
        if not np.isnan(br_eff) else float("nan")
    )

    bootstrap_block_size = max(2, decorrelation_lag if decorrelation_lag is not None else 2)
    position_indices = np.arange(regime_n)

    ic_statistic_fn = lambda resampled_positions, s=regime_signal, f=regime_forward_returns: (
        information_coefficient(
            pd.Series(s.to_numpy()[resampled_positions.astype(int)]),
            pd.Series(f.to_numpy()[resampled_positions.astype(int)]),
        )
    )

    bootstrap_ics = block_bootstrap(
        position_indices,
        block_size=bootstrap_block_size,
        n_samples=N_BOOTSTRAP,
        statistic_fn=ic_statistic_fn,
        seed=BOOTSTRAP_SEED,
    )

    ic_ci_lower = np.percentile(bootstrap_ics, 100 * CI_ALPHA / 2)
    ic_ci_upper = np.percentile(bootstrap_ics, 100 * (1 - CI_ALPHA / 2))
    ci_excludes_zero = (ic_ci_lower > 0) or (ic_ci_upper < 0)

    n_runs = float("nan")
    avg_run_length = float("nan")
    rho_runs = float("nan")
    br_eff_runs = float("nan")
    ir_fundamental_law_run_adjusted = float("nan")
    if regime_name in ("positive", "negative"):
        mask_bool = (pos_mask if regime_name == "positive" else neg_mask).to_numpy()
        runs = []
        run_start = None
        for i, val in enumerate(mask_bool):
            if val and run_start is None:
                run_start = i
            elif not val and run_start is not None:
                runs.append((run_start, i - 1))
                run_start = None
        if run_start is not None:
            runs.append((run_start, len(mask_bool) - 1))

        run_lengths = [end - start + 1 for start, end in runs]
        n_runs = len(runs)
        avg_run_length = np.mean(run_lengths) if n_runs > 0 else float("nan")

        if n_runs > 2:
            run_mean_returns = [
                forward_returns.iloc[start:end + 1].mean() for start, end in runs
            ]
            run_returns_series = pd.Series(run_mean_returns)
            rho_runs = run_returns_series.autocorr(lag=1)
            if np.isnan(rho_runs):
                rho_runs = 0.0
            rho_runs_for_breadth = max(rho_runs, 0.0)
            br_eff_runs = n_runs / (1 + (n_runs - 1) * rho_runs_for_breadth)
            ir_fundamental_law_run_adjusted = information_ratio(
                regime_ic, method="fundamental_law", breadth=br_eff_runs
            )

    results[regime_name] = {
        "n": regime_n,
        "ic": regime_ic,
        "robustness_rows": robustness_rows,
        "lag_1_rho": lag_1_rho,
        "decorrelation_lag": decorrelation_lag,
        "rho_estimate": rho_estimate,
        "br_raw": br_raw,
        "br_eff": br_eff,
        "ir_fundamental_law_raw": ir_fundamental_law_raw,
        "ir_fundamental_law_eff": ir_fundamental_law_eff,
        "ic_ci_lower": ic_ci_lower,
        "ic_ci_upper": ic_ci_upper,
        "ci_excludes_zero": ci_excludes_zero,
        "bootstrap_block_size": bootstrap_block_size,
        "n_runs": n_runs,
        "avg_run_length": avg_run_length,
        "rho_runs": rho_runs,
        "br_eff_runs": br_eff_runs,
        "ir_fundamental_law_run_adjusted": ir_fundamental_law_run_adjusted,
    }

print("PC2 loadings (train period, sign-normalized to USD/JPY positive):")
for pair, w in pc2_loadings_by_pair.items():
    print(f"  {pair}: {w:.4f}")
print()

for regime_name in ["pooled", "positive", "negative"]:
    r = results[regime_name]
    print(f"=== {regime_name.upper()} ===")
    print(f"n_obs: {r['n']}")
    print(f"IC (Spearman): {r['ic']:.4f}")
    print()

    print("Rolling IC / empirical IR robustness check across window lengths:")
    print(f"{'window':>8} {'n_windows':>10} {'mean_ic':>10} {'std_ic':>10} {'ir_empirical':>13}")
    for row in r["robustness_rows"]:
        print(f"{row['window']:>8} {row['n_windows']:>10} {row['mean_ic']:>10.4f} {row['std_ic']:>10.4f} {row['ir_empirical']:>13.4f}")
    print()

    print(f"Lag-1 autocorrelation: {r['lag_1_rho']:.4f}")
    if r["decorrelation_lag"] is not None:
        print(f"First lag where |autocorr| < {RHO_THRESHOLD}: lag {r['decorrelation_lag']}")
    print(f"rho estimate used for BR_eff: {r['rho_estimate']:.4f}")
    print(f"BR_raw: {r['br_raw']}")
    print(f"BR_eff: {r['br_eff']:.2f}")
    print(f"IR (fundamental law, BR_raw): {r['ir_fundamental_law_raw']:.4f}")
    print(f"IR (fundamental law, BR_eff): {r['ir_fundamental_law_eff']:.4f}")
    print()

    print(f"Block bootstrap IC 95% CI (block_size={r['bootstrap_block_size']}, n_bootstrap={N_BOOTSTRAP}):")
    print(f"  [{r['ic_ci_lower']:.4f}, {r['ic_ci_upper']:.4f}]  excludes zero: {r['ci_excludes_zero']}")
    print()

    if regime_name in ("positive", "negative"):
        print(f"Contiguous-run stats (descriptive):")
        print(f"  n_runs (contiguous same-sign spans): {r['n_runs']}")
        print(f"  avg_run_length (days): {r['avg_run_length']:.2f}")
        print(f"Run-level correlation-adjusted breadth:")
        print(f"  rho_runs (lag-1 autocorr of per-run mean forward return): {r['rho_runs']:.4f}")
        print(f"  br_eff_runs: {r['br_eff_runs']:.2f}")
        print(f"  IR (fundamental law, breadth=br_eff_runs): {r['ir_fundamental_law_run_adjusted']:.4f}")
        print()