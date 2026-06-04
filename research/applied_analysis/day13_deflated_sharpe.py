import sys
sys.path.insert(0, r'C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\summer2026')

import pandas as pd
import numpy as np
from scipy import stats

from src.analysis.performance_analyzer import PerformanceAnalyzer

ARCHIVE_DIR = r"C:\Users\clayb\onedrive\desktop\career\02_quant_projects\summer2026\archive\OG_results\realistic"

PAIRS = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/JPY": "USDJPY",
}

print(f"{'Pair':<12} {'Sharpe':>8} {'n_obs':>6} {'Skew':>7} {'Kurt':>7} {'DSR(N=2)':>10} {'DSR(N=10)':>10} {'DSR(N=30)':>10} {'N<0.95':>8} {'N<0.5':>6}")
print("-" * 105)

for pair, folder in PAIRS.items():
    df = pd.read_csv(f"{ARCHIVE_DIR}\\{folder}\\daily_returns.csv", parse_dates=["date"])
    df = df.dropna()
    returns = df["daily_return_pct"].values / 100
    active_returns = returns[returns != 0]

    n_obs = len(active_returns)
    years = (df["date"].max() - df["date"].min()).days / 365.25
    ann_factor = int(round(n_obs / years))

    df_fit, mu_fit, sigma_fit = stats.t.fit(active_returns)
    skewness = stats.skew(active_returns)
    kurtosis = 6 / (df_fit - 4) if df_fit > 4 else stats.kurtosis(active_returns)

    analyzer = PerformanceAnalyzer(returns=pd.Series(active_returns), trades=None, ann_factor=ann_factor)
    SR = analyzer.compute_sharpe()

    dsr_at = {}
    for n in [2, 10, 30]:
        dsr_at[n] = analyzer.deflated_sharpe_ratio(
            observed_sharpe=SR, n_trials=n,
            n_obs=n_obs, skewness=skewness, kurtosis=kurtosis
        )

    trial_range = range(2, 10001)
    dsr_series = [
        analyzer.deflated_sharpe_ratio(SR, n, n_obs, skewness, kurtosis)
        for n in trial_range
    ]

    threshold_95 = next((n for n, d in zip(trial_range, dsr_series) if d < 0.95), None)
    threshold_50 = next((n for n, d in zip(trial_range, dsr_series) if d < 0.50), None)

    print(f"{pair:<12} {SR:>8.3f} {n_obs:>6} {skewness:>7.3f} {kurtosis:>7.3f} "
          f"{dsr_at[2]:>10.4f} {dsr_at[10]:>10.4f} {dsr_at[30]:>10.4f} "
          f"{str(threshold_95) if threshold_95 else '>10000':>8} "
          f"{str(threshold_50) if threshold_50 else '>10000':>6}")