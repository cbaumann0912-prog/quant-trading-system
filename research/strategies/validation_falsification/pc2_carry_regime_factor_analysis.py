import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.framework.data_loader import DataLoader
from src.features.pca import pca

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
START = "2011-01-01"
END = "2023-12-31"

FUNDING_REGIONS = ["jp", "ch"]
TARGET_REGIONS = ["au", "ca", "nz"]
PUBLICATION_LAG_MONTHS = 2

SNB_WINDOW_START = "2015-01-05"
SNB_WINDOW_END = "2015-01-26"
COVID_WINDOW_START = "2020-02-10"
COVID_WINDOW_END = "2020-04-03"

LEAD_LAG_HORIZONS_DAYS = [21, 63, 126]

print("PC2 carry-factor analysis -- 7 USD-quoted pairs, descriptive only")
print("No forward-return test; does not consume a trial (n_trials stays at 4 per framework_map.md)")
print()

loader = DataLoader(pairs=PAIRS, start=START, end=END, data_dir=str(DATA_DIR))
returns = loader.get_returns(log=True).dropna()

n_pairs = len(PAIRS)
jpy_idx = PAIRS.index("USDJPY")

print(f"n_observations: {len(returns)}")
print(f"pairs (loading order): {PAIRS}")
print()

v, explained_variance, Z = pca(returns.to_numpy(), n_components=n_pairs)
cumulative = np.cumsum(explained_variance)

print(f"explained_variance: {explained_variance}")
print(f"cumulative_explained_variance: {cumulative}")
print()

for k in range(n_pairs):
    loadings_str = ", ".join(f"{pair}={v[i, k]:.4f}" for i, pair in enumerate(PAIRS))
    print(f"PC{k + 1} loadings: {loadings_str}")
print()

pc_corr = np.corrcoef(Z.T)
print("PC score correlation matrix (should be ~identity):")
print(pc_corr)
print()

for k in range(n_pairs):
    pc_scores = Z[:, k]
    print(
        f"PC{k + 1}: mean={pc_scores.mean():.6f}  std={pc_scores.std(ddof=1):.6f}  "
        f"skew={skew(pc_scores):.4f}  excess_kurtosis={kurtosis(pc_scores, fisher=True):.4f}"
    )
print()

pc2_loadings = v[:, 1].copy()
if pc2_loadings[jpy_idx] < 0:
    pc2_loadings = -pc2_loadings
    pc2_score = -Z[:, 1]
else:
    pc2_score = Z[:, 1]
pc2_series = pd.Series(pc2_score, index=returns.index, name="pc2_score")

start_year = returns.index.year.min()
end_year = returns.index.year.max()

print("Per-calendar-year eigenstructure stability (development window only, lockbox intact)")
for year in range(start_year, end_year + 1):
    subset = returns.loc[f"{year}-01-01":f"{year}-12-31"].to_numpy()
    if len(subset) < 50:
        print(f"{year}: skipped, n={len(subset)} insufficient")
        continue
    _, ev_year, _ = pca(subset, n_components=n_pairs)
    ev_str = "  ".join(f"PC{k + 1}={ev_year[k]:.4f}" for k in range(n_pairs))
    print(f"{year}: n={len(subset)}  {ev_str}")
print()

print("Per-calendar-year PC2 loading stability (sign-normalized so USDJPY loading > 0)")
header = "Year  " + "  ".join(f"{pair:>8}" for pair in PAIRS) + "  var_explained"
print(header)
for year in range(start_year, end_year + 1):
    subset = returns.loc[f"{year}-01-01":f"{year}-12-31"].to_numpy()
    if len(subset) < 50:
        continue
    v_year, ev_year, _ = pca(subset, n_components=n_pairs)
    pc2_year = v_year[:, 1].copy()
    if pc2_year[jpy_idx] < 0:
        pc2_year = -pc2_year
    row = "  ".join(f"{pc2_year[i]:>8.3f}" for i in range(n_pairs))
    print(f"{year}  {row}  {ev_year[1]:.4f}")
print()

print(f"PC2 variance explained (full sample): {explained_variance[1]:.4f}")
print("PC2 loadings (sign-normalized so USDJPY > 0): " + ", ".join(f"{pair}={pc2_loadings[i]:.4f}" for i, pair in enumerate(PAIRS)))
print()

funding_rates = [pd.read_csv(DATA_DIR / f"{r}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"] for r in FUNDING_REGIONS]
target_rates = [pd.read_csv(DATA_DIR / f"{r}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"] for r in TARGET_REGIONS]

funding_rate = pd.concat(funding_rates, axis=1, sort=True).mean(axis=1, skipna=False).dropna()
target_rate = pd.concat(target_rates, axis=1, sort=True).mean(axis=1, skipna=False).dropna()

spread_monthly = (funding_rate - target_rate).dropna()
spread_monthly_lagged = spread_monthly.shift(PUBLICATION_LAG_MONTHS)

spread_daily = spread_monthly_lagged.reindex(
    pd.date_range(returns.index.min(), returns.index.max(), freq="D")
).ffill()

aligned = pd.concat([pc2_series, spread_daily.rename("spread")], axis=1, join="inner").dropna()
print(f"n_observations (PC2 score aligned with rate spread): {len(aligned)}")

level_pearson, level_pearson_p = pearsonr(aligned["pc2_score"], aligned["spread"])
level_spearman, level_spearman_p = spearmanr(aligned["pc2_score"], aligned["spread"])
print("PC2 score vs. funding-minus-target rate spread, LEVEL, contemporaneous")
print(f"Pearson r: {level_pearson:.4f}  p={level_pearson_p:.4f}")
print(f"Spearman rho: {level_spearman:.4f}  p={level_spearman_p:.4f}")
print()

spread_monthly_change = spread_monthly_lagged.diff().dropna()
spread_change_daily = spread_monthly_change.reindex(
    pd.date_range(returns.index.min(), returns.index.max(), freq="D")
).ffill()
aligned_change = pd.concat([pc2_series, spread_change_daily.rename("spread_change")], axis=1, join="inner").dropna()
change_pearson, change_pearson_p = pearsonr(aligned_change["pc2_score"], aligned_change["spread_change"])
change_spearman, change_spearman_p = spearmanr(aligned_change["pc2_score"], aligned_change["spread_change"])
print("PC2 score vs. month-over-month CHANGE in rate spread, contemporaneous")
print(f"Pearson r: {change_pearson:.4f}  p={change_pearson_p:.4f}")
print(f"Spearman rho: {change_spearman:.4f}  p={change_spearman_p:.4f}")
print()

print("Lead-lag check: does the rate spread level predict subsequent PC2 drift, rather than move with it same-day?")
print("Still descriptive -- checking whether the spread has any forward relationship with PC2 at all, not a pre-registered predictive test, does not consume a trial")
for horizon in LEAD_LAG_HORIZONS_DAYS:
    forward_pc2_drift = pc2_series.rolling(horizon).mean().shift(-horizon)
    lead_lag = pd.concat(
        [spread_daily.rename("spread"), forward_pc2_drift.rename("forward_pc2_drift")], axis=1, join="inner"
    ).dropna()
    lp, lpp = pearsonr(lead_lag["spread"], lead_lag["forward_pc2_drift"])
    ls, lsp = spearmanr(lead_lag["spread"], lead_lag["forward_pc2_drift"])
    print(f"Horizon={horizon} trading days  n={len(lead_lag)}  Pearson r={lp:.4f} p={lpp:.4f}  Spearman rho={ls:.4f} p={lsp:.4f}")
print()

print("Overlap-corrected lead-lag check")
print()

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42

for horizon in LEAD_LAG_HORIZONS_DAYS:
    forward_pc2_drift = pc2_series.rolling(horizon).mean().shift(-horizon)
    lead_lag = pd.concat(
        [spread_daily.rename("spread"), forward_pc2_drift.rename("forward_pc2_drift")], axis=1, join="inner"
    ).dropna()

    non_overlapping = lead_lag.iloc[::horizon]
    non_overlapping_r, non_overlapping_p = pearsonr(non_overlapping["spread"], non_overlapping["forward_pc2_drift"])
    print(f"Horizon={horizon}d  non-overlapping subsample n={len(non_overlapping)}  Pearson r={non_overlapping_r:.4f} p={non_overlapping_p:.4f}")

    spread_arr = lead_lag["spread"].to_numpy()
    fwd_arr = lead_lag["forward_pc2_drift"].to_numpy()
    n_obs = len(lead_lag)
    n_blocks = int(np.ceil(n_obs / horizon))
    max_start = n_obs - horizon
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    boot_r = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + horizon) for s in starts])[:n_obs]
        boot_r[b] = pearsonr(spread_arr[idx], fwd_arr[idx])[0]

    boot_lower = np.percentile(boot_r, 2.5)
    boot_upper = np.percentile(boot_r, 97.5)
    includes_zero = bool(boot_lower <= 0 <= boot_upper)
    print(
        f"Horizon={horizon}d  block bootstrap (block_size={horizon}, n_samples={N_BOOTSTRAP})  "
        f"95% CI on r: [{boot_lower:.4f}, {boot_upper:.4f}]  includes zero: {includes_zero}"
    )
    print()

print(f"SNB floor removal window ({SNB_WINDOW_START} to {SNB_WINDOW_END}), daily PC2 score:")
snb_window = pc2_series.loc[SNB_WINDOW_START:SNB_WINDOW_END]
for date, value in snb_window.items():
    print(f"{date.date()}  pc2_score={value:.4f}")
print(f"SNB window mean PC2 score: {snb_window.mean():.4f}  min: {snb_window.min():.4f}")
print()

print(f"COVID crash window ({COVID_WINDOW_START} to {COVID_WINDOW_END}), daily PC2 score:")
covid_window = pc2_series.loc[COVID_WINDOW_START:COVID_WINDOW_END]
for date, value in covid_window.items():
    print(f"{date.date()}  pc2_score={value:.4f}")
print(f"COVID window mean PC2 score: {covid_window.mean():.4f}  min: {covid_window.min():.4f}")
print()

full_sample_mean = pc2_series.mean()
full_sample_std = pc2_series.std()
print(f"Full-sample PC2 score mean: {full_sample_mean:.6f}  std: {full_sample_std:.6f}")
print(f"SNB window z-score of window mean: {(snb_window.mean() - full_sample_mean) / full_sample_std:.4f}")
print(f"COVID window z-score of window mean: {(covid_window.mean() - full_sample_mean) / full_sample_std:.4f}")
