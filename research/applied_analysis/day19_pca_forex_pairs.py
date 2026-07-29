import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from src.features.pca import pca
from src.analysis.portfolio_stats import compute_covariance_matrix
from scipy.stats import kurtosis

DEV_END = "2023-12-31"

DATA_DIR = REPO_ROOT.parent / "data"
EURUSD_PATH = DATA_DIR / "EURUSD.csv"
GBPUSD_PATH = DATA_DIR / "GBPUSD.csv"
USDJPY_PATH = DATA_DIR / "USDJPY.csv"

def load_pair(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime").sort_index().loc[:DEV_END]
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

v, explained_variance, Z = pca(returns.values, n_components=3)

cumulative = np.cumsum(explained_variance)

print(f"\nVariance explained: {explained_variance}")
print(f"Cumulative variance explained: {cumulative}")
print(f"\nEigenvectors:\n{v}")
print(f"\nProjections:\n{Z}")

print(f"\nPC score correlation matrix:\n{np.corrcoef(Z.T)}")

total_var = np.var(returns.values, axis=0, ddof=1).sum()
eigenvalues = explained_variance * total_var
for k in range(3):
    print(f"PC{k+1}: eigenvalue={eigenvalues[k]:.6e}, Var(Z_{k+1})={np.var(Z[:,k], ddof=1):.6e}")

for k in range(3):
    print(f"PC{k+1} excess kurtosis: {kurtosis(Z[:,k], fisher=True):.4f}")

n = len(returns)
n_splits = 15
split_size = n // n_splits

print("Split-sample eigenstructure stability (yearly)\n")

start_year = returns.index.year.min()
end_year = returns.index.year.max()

for year in range(start_year, end_year + 1):
    subset = returns.loc[f"{year}-01-01":f"{year}-12-31"].values
    
    if len(subset) < 50:
        continue
    
    _, ev, _ = pca(subset, n_components=3)
    
    print(f"{year}: PC1={ev[0]:.4f}  PC2={ev[1]:.4f}  PC3={ev[2]:.4f}  (n={len(subset)})")

    from scipy.stats import skew, kurtosis

print(f"{'Stat':<12} {'PC1':>12} {'PC2':>12} {'PC3':>12}")
print("-" * 50)
stats = {
    'Mean':   Z.mean(axis=0),
    'Std':    Z.std(axis=0, ddof=1),
    'Skew':   [skew(Z[:,k]) for k in range(3)],
    'Kurt':   [kurtosis(Z[:,k], fisher=True) for k in range(3)],
    'Min':    Z.min(axis=0),
    'Max':    Z.max(axis=0),
}
for name, vals in stats.items():
    print(f"{name:<12} {vals[0]:>12.6f} {vals[1]:>12.6f} {vals[2]:>12.6f}")

pc2 = Z[:, 1]
pc2_z = (pc2 - pc2.mean()) / pc2.std(ddof=1)

for threshold in [1.5, 2.0, 2.5, 3.0]:
    n_above = (pc2_z > threshold).sum()
    n_below = (pc2_z < -threshold).sum()
    total = n_above + n_below
    per_year = total / 15
    print(f"|z| > {threshold}: {total} events ({per_year:.1f}/year)  "
          f"[+{n_above} / -{n_below}]")
    
for k in range(3):
    ac = np.corrcoef(Z[:-1, k], Z[1:, k])[0, 1]
    print(f"PC{k+1} lag-1 autocorrelation: {ac:.4f}")