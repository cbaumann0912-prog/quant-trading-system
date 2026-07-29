import numpy as np
import pandas as pd
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.stats.regression import fit_ols, residual_diagnostics

DEV_END = "2023-12-31"

DATA_DIR = REPO_ROOT.parent / "data"
EURUSD_PATH = DATA_DIR / "EURUSD.csv"
GBPUSD_PATH = DATA_DIR / "GBPUSD.csv"

def load_pair(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime").sort_index().loc[:DEV_END]
    return df

eur = load_pair(EURUSD_PATH)
gbp = load_pair(GBPUSD_PATH)

eur_daily = eur["Close"].resample("D").last().dropna()
gbp_daily = gbp["Close"].resample("D").last().dropna()

common_idx = eur_daily.index.intersection(gbp_daily.index)
eur_daily = eur_daily.loc[common_idx]
gbp_daily = gbp_daily.loc[common_idx]

eur_ret = np.log(eur_daily / eur_daily.shift(1)).dropna()
gbp_ret = np.log(gbp_daily / gbp_daily.shift(1)).dropna()

common_idx_ret = eur_ret.index.intersection(gbp_ret.index)
eur_ret = eur_ret.loc[common_idx_ret]
gbp_ret = gbp_ret.loc[common_idx_ret]

A = gbp_ret.values.reshape(-1, 1)
b = eur_ret.values

result = fit_ols(A, b, add_intercept=True)

diag = residual_diagnostics(b, b - result['residuals'], lags=20)

for k, v in diag.items():
    print(f"{k}: {v:.6f}")