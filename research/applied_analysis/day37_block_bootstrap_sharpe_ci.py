import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.evaluation.bootstrap import bootstrap_confidence_interval, block_bootstrap
from src.analysis.performance_analyzer import PerformanceAnalyzer

DATA_DIR = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"

FILES = {
    "EURUSD": "EURUSD.csv",
    "GBPUSD": "GBPUSD.csv",
    "USDJPY": "USDJPY.csv",
}

N_BOOTSTRAP = 1000
CONFIDENCE = 0.95
RISK_FREE_RATE = 0.0
PRIMARY_BLOCK_SIZE = 21
LB_LAGS = [5, 10, 20, 40, 60]
BLOCK_SIZE_SWEEP = [5, 10, 21, 40, 60, 100, 150, 250]

pairs = {}
for pair_name, filename in FILES.items():
    path = f"{DATA_DIR}\\{filename}"
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")
    daily_close = df["Close"].resample("D").last().dropna()
    pairs[pair_name] = np.log(daily_close / daily_close.shift(1)).dropna()

results = {}

for pair_name, returns in pairs.items():
    analyzer = PerformanceAnalyzer(returns, risk_free_rate=RISK_FREE_RATE)
    ann_factor = analyzer.compute_ann_factor()
    point_sharpe = analyzer.compute_sharpe()
    returns_arr = returns.to_numpy()
    n_obs = len(returns)

    def sharpe_fn(r, af=ann_factor, rf=RISK_FREE_RATE):
        std = r.std()
        if std == 0:
            return float("nan")
        return (r.mean() - rf) / std * np.sqrt(af)

    def std_fn(r):
        return r.std()

    iid_sharpe_lower, iid_sharpe_upper = bootstrap_confidence_interval(
        returns_arr, statistic_fn=sharpe_fn, n_bootstrap=N_BOOTSTRAP, confidence=CONFIDENCE
    )
    iid_sharpe_width = iid_sharpe_upper - iid_sharpe_lower

    block_sharpe_stats = block_bootstrap(
        returns_arr, block_size=PRIMARY_BLOCK_SIZE, n_samples=N_BOOTSTRAP,
        statistic_fn=sharpe_fn, seed=42
    )
    block_sharpe_lower = np.percentile(block_sharpe_stats, ((1 - CONFIDENCE) / 2) * 100)
    block_sharpe_upper = np.percentile(block_sharpe_stats, (1 - (1 - CONFIDENCE) / 2) * 100)
    block_sharpe_width = block_sharpe_upper - block_sharpe_lower

    iid_std_lower, iid_std_upper = bootstrap_confidence_interval(
        returns_arr, statistic_fn=std_fn, n_bootstrap=N_BOOTSTRAP, confidence=CONFIDENCE
    )
    iid_std_width = iid_std_upper - iid_std_lower

    block_std_stats = block_bootstrap(
        returns_arr, block_size=PRIMARY_BLOCK_SIZE, n_samples=N_BOOTSTRAP,
        statistic_fn=std_fn, seed=42
    )
    block_std_lower = np.percentile(block_std_stats, ((1 - CONFIDENCE) / 2) * 100)
    block_std_upper = np.percentile(block_std_stats, (1 - (1 - CONFIDENCE) / 2) * 100)
    block_std_width = block_std_upper - block_std_lower

    abs_returns = np.abs(returns_arr)
    lb = acorr_ljungbox(abs_returns, lags=LB_LAGS, return_df=True)
    lb_results = list(zip(LB_LAGS, lb["lb_stat"].tolist(), lb["lb_pvalue"].tolist()))

    sweep_results = []
    for bs in BLOCK_SIZE_SWEEP:
        stats_arr = block_bootstrap(
            returns_arr, block_size=bs, n_samples=N_BOOTSTRAP,
            statistic_fn=sharpe_fn, seed=42
        )
        lower = np.percentile(stats_arr, ((1 - CONFIDENCE) / 2) * 100)
        upper = np.percentile(stats_arr, (1 - (1 - CONFIDENCE) / 2) * 100)
        sweep_results.append((bs, upper - lower))

    results[pair_name] = {
        "n_obs": n_obs,
        "ann_factor": ann_factor,
        "point_sharpe": point_sharpe,
        "iid_sharpe_ci": (iid_sharpe_lower, iid_sharpe_upper),
        "iid_sharpe_width": iid_sharpe_width,
        "block_sharpe_ci": (block_sharpe_lower, block_sharpe_upper),
        "block_sharpe_width": block_sharpe_width,
        "sharpe_pct_wider": (block_sharpe_width / iid_sharpe_width - 1) * 100,
        "iid_std_ci": (iid_std_lower, iid_std_upper),
        "iid_std_width": iid_std_width,
        "block_std_ci": (block_std_lower, block_std_upper),
        "block_std_width": block_std_width,
        "std_pct_wider": (block_std_width / iid_std_width - 1) * 100,
        "lb_abs_returns": lb_results,
        "block_size_sweep": sweep_results,
    }

for pair_name, r in results.items():
    print(pair_name)
    print(f"n_obs: {r['n_obs']}")
    print(f"ann_factor: {r['ann_factor']:.2f}")
    print(f"point Sharpe: {r['point_sharpe']:.4f}")
    print()
    print("Sharpe CI comparison (block_size={}):".format(PRIMARY_BLOCK_SIZE))
    print(f"  iid:   ({r['iid_sharpe_ci'][0]:.4f}, {r['iid_sharpe_ci'][1]:.4f})  width={r['iid_sharpe_width']:.4f}")
    print(f"  block: ({r['block_sharpe_ci'][0]:.4f}, {r['block_sharpe_ci'][1]:.4f})  width={r['block_sharpe_width']:.4f}")
    print(f"  block vs iid: {r['sharpe_pct_wider']:.1f}%")
    print()
    print("Std CI comparison (isolation test, block_size={}):".format(PRIMARY_BLOCK_SIZE))
    print(f"  iid:   ({r['iid_std_ci'][0]:.6f}, {r['iid_std_ci'][1]:.6f})  width={r['iid_std_width']:.6f}")
    print(f"  block: ({r['block_std_ci'][0]:.6f}, {r['block_std_ci'][1]:.6f})  width={r['block_std_width']:.6f}")
    print(f"  block vs iid: {r['std_pct_wider']:.1f}%")
    print()
    print("Ljung-Box on |r_t|:")
    for lag, stat, p in r["lb_abs_returns"]:
        print(f"  lag={lag:3d}  stat={stat:10.3f}  p={p:.6f}  reject_white_noise={p < 0.05}")
    print()
    print("Block size sweep (Sharpe CI width, single seed=42):")
    for bs, width in r["block_size_sweep"]:
        print(f"  block_size={bs:4d}  width={width:.4f}")
    print()