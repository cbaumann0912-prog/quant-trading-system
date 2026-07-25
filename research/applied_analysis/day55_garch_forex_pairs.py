import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from src.framework.data_loader import DataLoader
from src.framework.walk_forward import WalkForwardValidator
from src.features.garch import fit_garch

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
]
REALIZED_VOL_WINDOW = 20
loader = DataLoader(pairs=PAIRS, start="2011-01-03", end="2023-12-31")
returns_df = loader.get_returns(log=True)

results = {}
realized_vol = {}
for pair in PAIRS:
    r = returns_df[pair].dropna()
    results[pair] = fit_garch(r)
    realized_vol[pair] = r.rolling(REALIZED_VOL_WINDOW).std()

print(f"n_obs per pair (daily log returns): {[len(returns_df[p].dropna()) for p in PAIRS]}")
print()
print(f"{'Pair':<8}{'omega':>12}{'alpha':>10}{'beta':>10}{'persistence':>13}{'long_run_vol':>14}")
for pair in PAIRS:
    res = results[pair]
    print(
        f"{pair:<8}{res['omega']:>12.3e}{res['alpha']:>10.4f}{res['beta']:>10.4f}"
        f"{res['persistence']:>13.4f}{res['long_run_vol']:>14.5f}"
    )

print()
print("=== Persistence ranking (most -> least persistent) ===")
ranked = sorted(PAIRS, key=lambda p: results[p]["persistence"], reverse=True)
for pair in ranked:
    print(f"  {pair}: persistence={results[pair]['persistence']:.4f}")

print()
print(f"=== GARCH conditional vol vs {REALIZED_VOL_WINDOW}-day rolling realized vol ===")
for pair in PAIRS:
    cond = results[pair]["conditional_vol"]
    rv = realized_vol[pair].reindex(cond.index)
    aligned = pd.concat([cond.rename("garch"), rv.rename("realized")], axis=1).dropna()
    corr = aligned["garch"].corr(aligned["realized"])
    mean_garch = aligned["garch"].mean()
    mean_realized = aligned["realized"].mean()
    print(
        f"  {pair}: corr(garch, {REALIZED_VOL_WINDOW}d realized)={corr:.4f}, "
        f"mean garch vol={mean_garch:.5f}, mean realized vol={mean_realized:.5f}"
    )

YEARS = range(2011, 2024)
yearly_persistence = {pair: {} for pair in PAIRS}
yearly_nobs = {pair: {} for pair in PAIRS}
for pair in PAIRS:
    r = returns_df[pair].dropna()
    for year in YEARS:
        yr_r = r[r.index.year == year]
        yearly_nobs[pair][year] = len(yr_r)
        if len(yr_r) < 50:
            yearly_persistence[pair][year] = float("nan")
            continue
        yearly_persistence[pair][year] = fit_garch(yr_r)["persistence"]

print()
print("=== Per-year GARCH(1,1) persistence, by pair (one fit per pair per calendar year) ===")
header = "Pair    " + "".join(f"{y:>7}" for y in YEARS)
print(header)
for pair in PAIRS:
    row = f"{pair:<8}" + "".join(
        f"{yearly_persistence[pair][y]:>7.3f}" if yearly_persistence[pair][y] == yearly_persistence[pair][y] else f"{'n/a':>7}"
        for y in YEARS
    )
    print(row)

print()
print("=== Per-year persistence summary stats (min, max, mean, std across years) ===")
for pair in PAIRS:
    vals = [v for v in yearly_persistence[pair].values() if v == v]
    series = pd.Series(vals)
    print(
        f"  {pair}: min={series.min():.4f} max={series.max():.4f} "
        f"mean={series.mean():.4f} std={series.std():.4f}"
    )
    
wfv = WalkForwardValidator(
    signal_fn=None, data=returns_df, n_windows=7,
    train_years=5, test_months=12, embargo_days=5,
)
windows = wfv.generate_windows()
window_labels = [f"{w['train_start'].year}-{w['train_end'].year}" for w in windows]

window_persistence = {pair: [] for pair in PAIRS}
window_nobs = []
for w in windows:
    mask = (returns_df.index >= w["train_start"]) & (returns_df.index < w["train_end"])
    sub = returns_df.loc[mask]
    window_nobs.append(len(sub))
    for pair in PAIRS:
        window_persistence[pair].append(fit_garch(sub[pair].dropna())["persistence"])

print()
print(f"=== Rolling 5-year window GARCH(1,1) persistence, by pair (n_obs/window: {window_nobs}) ===")
header = "Pair    " + "".join(f"{l:>10}" for l in window_labels)
print(header)
for pair in PAIRS:
    row = f"{pair:<8}" + "".join(f"{v:>10.4f}" for v in window_persistence[pair])
    print(row)

print()
print("=== Rolling-window persistence summary stats (min, max, mean, std across 7 windows) ===")
for pair in PAIRS:
    series = pd.Series(window_persistence[pair])
    print(
        f"  {pair}: min={series.min():.4f} max={series.max():.4f} "
        f"mean={series.mean():.4f} std={series.std():.4f}"
    )
