import sys
sys.path.insert(0, r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\summer2026")

import os
os.makedirs(r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\summer2026\research\audit", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.signals.cointegration import engle_granger_test, cointegration_spread
from src.stats.regression import fit_ols

DATA_PATH = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"
ROLLING_WINDOW = 252
COMBINATIONS = [
    ("EURUSD", "GBPUSD"),
    ("EURUSD", "USDJPY"),
    ("GBPUSD", "USDJPY"),
]

def load_prices(pair: str) -> pd.Series:
    path = rf"{DATA_PATH}\{pair}.csv"
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime").sort_index()
    daily = df["Close"].resample("D").last().dropna()
    return daily

eurusd = load_prices("EURUSD")
gbpusd = load_prices("GBPUSD")
usdjpy = load_prices("USDJPY")

prices = pd.DataFrame({
    "EURUSD": eurusd,
    "GBPUSD": gbpusd,
    "USDJPY": usdjpy,
}).dropna()

print(f"Price levels loaded: {len(prices)} daily observations")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}\n")

print("Full-Sample Cointegration Results")

full_sample_results = []

for y_name, x_name in COMBINATIONS:
    result = engle_granger_test(prices[y_name], prices[x_name])
    full_sample_results.append({
        "pair"            : f"{y_name} ~ {x_name}",
        "hedge_ratio"     : round(result["hedge_ratio"], 4),
        "alpha"           : round(result["alpha"], 4),
        "adf_stat"        : round(result["adf_stat"], 4),
        "adf_p"           : round(result["adf_p"], 4),
        "is_cointegrated" : result["is_cointegrated"],
    })

results_df = pd.DataFrame(full_sample_results)
print(results_df.to_string(index=False))

print("Rolling Hedge Ratio Stability")

rolling_betas = {}

for y_name, x_name in COMBINATIONS:
    betas = []
    dates = []

    for end in range(ROLLING_WINDOW, len(prices)):
        window = prices.iloc[end - ROLLING_WINDOW:end]
        y_w = window[y_name].values
        x_w = window[x_name].values
        # OLS inline — no ADF, just the beta
        A = x_w.reshape(-1, 1)  # fit_ols adds intercept internally
        b = y_w
        ols = fit_ols(A, b)
        betas.append(ols["coefficients"][1])
        dates.append(prices.index[end])

    pair_label = f"{y_name}~{x_name}"
    rolling_betas[pair_label] = pd.Series(betas, index=dates)

rolling_df = pd.DataFrame(rolling_betas)
print(rolling_df.describe().round(4))

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
fig.suptitle("Day 24 — Cointegration Spreads (Full Sample)", fontsize=13)

for i, row in enumerate(full_sample_results):
    y_name, x_name = COMBINATIONS[i]
    spread = cointegration_spread(
        prices[y_name],
        prices[x_name],
        alpha=row["alpha"],
        hedge_ratio=row["hedge_ratio"],
    )
    axes[i].plot(spread.index, spread.values, linewidth=0.8, color="steelblue")
    axes[i].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[i].set_title(
        f"{row['pair']} | β={row['hedge_ratio']} | ADF p={row['adf_p']} | "
        f"Cointegrated: {row['is_cointegrated']}"
    )
    axes[i].set_ylabel("Spread")

plt.tight_layout()
plt.savefig(r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\summer2026\research\audit\day24_spreads.png", dpi=150)
plt.show()
print("\nSpread plot saved to research/audit/day24_spreads.png")

fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
fig2.suptitle("Day 24 — Rolling Hedge Ratio Stability (252-day window)", fontsize=13)

for i, col in enumerate(rolling_df.columns):
    axes2[i].plot(rolling_df.index, rolling_df[col].values, linewidth=0.8, color="darkorange")
    axes2[i].axhline(rolling_df[col].mean(), color="black", linewidth=0.8, linestyle="--", label="mean")
    axes2[i].set_title(f"{col} rolling β")
    axes2[i].set_ylabel("β")
    axes2[i].legend()

plt.tight_layout()
plt.savefig(r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\summer2026\research\audit\day24_rolling_betas.png", dpi=150)
plt.show()
print("Rolling beta plot saved to research/audit/day24_rolling_betas.png")

print("\nDone. Fill results into research/audit/day24_cointegration_analysis.md")