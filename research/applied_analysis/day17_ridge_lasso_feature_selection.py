import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from src.stats.regression import ridge_fit, fit_ols

DATA_PATH = Path(r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data\EURUSD.csv")
OUTPUT_PATH = Path("research/notes/day17_ridge_lasso_feature_selection.md")
LAMBDA_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
N_LAGS = 5

df_1m = pd.read_csv(r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data\EURUSD.csv", parse_dates=["Datetime"], index_col="Datetime")
daily = df_1m["Close"].resample("1D").last().dropna()
returns = daily.pct_change().dropna()

lag_df = pd.DataFrame(index=returns.index)
for i in range(1, N_LAGS + 1):
    lag_df[f"lag{i}"] = returns.shift(i)
lag_df["target"] = returns
lag_df.dropna(inplace=True)

X = lag_df[[f"lag{i}" for i in range(1, N_LAGS + 1)]].values
y = lag_df["target"].values

ols_result = fit_ols(X, y, add_intercept=False)

ridge_results = [ridge_fit(X, y, lambda_=lam) for lam in LAMBDA_GRID]

rows = [{"lambda": "OLS", **{f"lag{i+1}": ols_result["coefficients"][i] for i in range(N_LAGS)}}]
for lam, res in zip(LAMBDA_GRID, ridge_results):
    rows.append({"lambda": lam, **{f"lag{i+1}": res["coefficients"][i] for i in range(N_LAGS)}})
coef_df = pd.DataFrame(rows).set_index("lambda")

ols_coefs = ols_result["coefficients"]
threshold = None
for lam, res in zip(LAMBDA_GRID, ridge_results):
    if np.any(np.abs(res["coefficients"] - ols_coefs) > 0.01):
        threshold = lam
        break

print(coef_df.to_string())
print(f"\nShrinkage becomes significant at lambda = {threshold}")

from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(returns, lags=[5, 10, 20], return_df=True)
print("\nLjung-Box Test on Raw EURUSD Returns:")
print(lb.to_string())
