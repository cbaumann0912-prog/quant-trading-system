import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from src.features.pca import eigendecomposition
from src.analysis.portfolio_stats import compute_covariance_matrix

DATA_DIR = Path(r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data")
EURUSD_PATH = DATA_DIR / "EURUSD.csv"
GBPUSD_PATH = DATA_DIR / "GBPUSD.csv"
USDJPY_PATH = DATA_DIR / "USDJPY.csv"

def load_pair(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime").sort_index()
    return df


eurusd = load_pair(EURUSD_PATH)
gbpusd = load_pair(GBPUSD_PATH)
usdjpy = load_pair(USDJPY_PATH)

eurusd_daily = eurusd["Close"].resample("D").last().dropna()
gbpusd_daily = gbpusd["Close"].resample("D").last().dropna()
usdjpy_daily = usdjpy["Close"].resample("D").last().dropna()

eurusd_ret = np.log(eurusd_daily / eurusd_daily.shift(1)).dropna()
gbpusd_ret = np.log(gbpusd_daily / gbpusd_daily.shift(1)).dropna()
usdjpy_ret = np.log(usdjpy_daily / usdjpy_daily.shift(1)).dropna()

returns = pd.concat([eurusd_ret, gbpusd_ret, usdjpy_ret], axis=1).dropna()
returns.columns = ["EURUSD", "GBPUSD", "USDJPY"]

cov_matrix = compute_covariance_matrix(returns)

lambdas, v = eigendecomposition(cov_matrix)

variance_explained = lambdas / lambdas.sum()
cumulative = np.cumsum(variance_explained)

print("Eigenvalues:", lambdas)
print(f"\nVariance explained: {variance_explained}")
print(f"Cumulative variance explained: {cumulative}")
print(f"\nEigenvectors:\n{v}")
