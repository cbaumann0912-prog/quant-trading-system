import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from src.analysis.portfolio import minimum_variance_portfolio, risk_parity_weights
from scipy.stats import binomtest, ttest_rel, wilcoxon

DATA_DIR = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"

FILES = {
    "EURUSD": "EURUSD.csv",
    "GBPUSD": "GBPUSD.csv",
    "USDJPY": "USDJPY.csv",
}

ANN_FACTOR = 252

pairs = {}
for pair_name, filename in FILES.items():
    path = f"{DATA_DIR}\\{filename}"
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")
    daily_close = df["Close"].resample("D").last().dropna()
    pairs[pair_name] = np.log(daily_close / daily_close.shift(1)).dropna()

returns = pd.concat(pairs, axis=1, join="inner")
returns.columns = list(FILES.keys())

years = sorted(returns.index.year.unique())
estimation_years = years[:-1]

rows = []

for est_year in estimation_years:
    train = returns[returns.index.year == est_year]

    if len(train) < 63:
        continue

    mv  = minimum_variance_portfolio(train)
    erc = risk_parity_weights(train)

    oos_year = est_year + 1
    oos = returns[returns.index.year == oos_year]

    if len(oos) == 0:
        continue

    oos["_q"] = oos.index.quarter
    for q in sorted(oos["_q"].unique()):
        quarter = oos[oos["_q"] == q].drop(columns="_q")

        if len(quarter) < 5:
            continue

        mv_port  = quarter.to_numpy() @ mv["weights"]
        erc_port = quarter.to_numpy() @ erc["weights"]

        def stats(port):
            vol    = port.std()  * np.sqrt(ANN_FACTOR)
            sharpe = (port.mean() / port.std()) * np.sqrt(ANN_FACTOR)
            ret    = port.mean() * ANN_FACTOR
            return vol, sharpe, ret

        mv_vol,  mv_sharpe,  mv_ret  = stats(mv_port)
        erc_vol, erc_sharpe, erc_ret = stats(erc_port)

        rows.append({
            "Est Year":    est_year,
            "OOS Quarter": f"{oos_year} Q{q}",
            "n_train":     len(train),
            "n_oos":       len(quarter),
            "MV_EUR":      round(mv["weights"][0],  4),
            "MV_GBP":      round(mv["weights"][1],  4),
            "MV_JPY":      round(mv["weights"][2],  4),
            "ERC_EUR":     round(erc["weights"][0], 4),
            "ERC_GBP":     round(erc["weights"][1], 4),
            "ERC_JPY":     round(erc["weights"][2], 4),
            "MV_Vol":      round(mv_vol,    4),
            "ERC_Vol":     round(erc_vol,   4),
            "MV_Sharpe":   round(mv_sharpe, 4),
            "ERC_Sharpe":  round(erc_sharpe,4),
            "MV_Ret":      round(mv_ret,    4),
            "ERC_Ret":     round(erc_ret,   4),
            "Lower_Vol":   "MV"  if mv_vol    < erc_vol    else "ERC",
            "Hi_Sharpe":   "MV"  if mv_sharpe > erc_sharpe else "ERC"
        })

results = pd.DataFrame(rows)

print("ANNUAL ESTIMATION — QUARTERLY OOS PERFORMANCE")
print(results[[
    "OOS Quarter", "n_train",
    "MV_EUR", "MV_GBP", "MV_JPY",
    "ERC_EUR", "ERC_GBP", "ERC_JPY",
    "MV_Vol", "ERC_Vol", "MV_Sharpe", "ERC_Sharpe",
    "Lower_Vol", "Hi_Sharpe"
]].to_string(index=False))

total_q       = len(results)
mv_vol_wins   = (results["Lower_Vol"] == "MV").sum()
erc_vol_wins  = (results["Lower_Vol"] == "ERC").sum()
mv_sr_wins    = (results["Hi_Sharpe"] == "MV").sum()
erc_sr_wins   = (results["Hi_Sharpe"] == "ERC").sum()


print("SCOREBOARD")
print(f"Total OOS quarters : {total_q}")
print(f"Lower Vol  — MV  : {mv_vol_wins:>3} ({100*mv_vol_wins/total_q:.1f}%)")
print(f"Lower Vol  — ERC : {erc_vol_wins:>3} ({100*erc_vol_wins/total_q:.1f}%)")
print(f"Higher Sharpe — MV  : {mv_sr_wins:>3} ({100*mv_sr_wins/total_q:.1f}%)")
print(f"Higher Sharpe — ERC : {erc_sr_wins:>3} ({100*erc_sr_wins/total_q:.1f}%)")
print()

print("Average OOS metrics across all quarters:")
print(f"  MV  — Vol: {results['MV_Vol'].mean():.4f}  Sharpe: {results['MV_Sharpe'].mean():.4f}  Ret: {results['MV_Ret'].mean():.4f}")
print(f"  ERC — Vol: {results['ERC_Vol'].mean():.4f}  Sharpe: {results['ERC_Sharpe'].mean():.4f}  Ret: {results['ERC_Ret'].mean():.4f}")
print()

sharpe_diff = results["MV_Sharpe"] - results["ERC_Sharpe"]
vol_diff    = results["MV_Vol"]    - results["ERC_Vol"]
 
mv_sharpe_wins = (sharpe_diff > 0).sum()
mv_vol_wins_n  = (vol_diff    < 0).sum()
 
binom_sharpe = binomtest(mv_sharpe_wins, total_q, p=0.5)
binom_vol    = binomtest(mv_vol_wins_n,  total_q, p=0.5)
 
ttest_sharpe = ttest_rel(results["MV_Sharpe"], results["ERC_Sharpe"])
ttest_vol    = ttest_rel(results["MV_Vol"],    results["ERC_Vol"])
 
wilcox_sharpe = wilcoxon(sharpe_diff)
wilcox_vol    = wilcoxon(vol_diff)
 
print("STATISTICAL SIGNIFICANCE TESTS")
print(f"\n{'Test':<30} {'Sharpe p-val':>14} {'Vol p-val':>14}")
print(f"{'Binomial (win rate vs 50/50)':<30} {binom_sharpe.pvalue:>14.4f} {binom_vol.pvalue:>14.4f}")
print(f"{'Paired t-test':<30} {ttest_sharpe.pvalue:>14.4f} {ttest_vol.pvalue:>14.4f}")
print(f"{'Wilcoxon signed-rank':<30} {wilcox_sharpe.pvalue:>14.4f} {wilcox_vol.pvalue:>14.4f}")
 
print(f"\nMean Sharpe diff (MV - ERC) : {sharpe_diff.mean():+.4f}  (std: {sharpe_diff.std():.4f})")
print(f"Mean Vol diff    (MV - ERC) : {vol_diff.mean():+.4f}  (std: {vol_diff.std():.4f})")
print(f"\nConclusion — Sharpe: {'MV significantly better' if binom_sharpe.pvalue < 0.05 and ttest_sharpe.pvalue < 0.05 else 'no significant difference'}")
print(f"Conclusion — Vol:    {'ERC significantly better' if binom_vol.pvalue < 0.05 and ttest_vol.pvalue < 0.05 else 'no significant difference'}")