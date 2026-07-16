import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.stats.regression import fit_ols

DATA_DIR = Path(r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data")
EURUSD_PATH = DATA_DIR / "EURUSD.csv"
GBPUSD_PATH = DATA_DIR / "GBPUSD.csv"

def load_pair(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime").sort_index()
    return df

eur = load_pair(EURUSD_PATH)
gbp = load_pair(GBPUSD_PATH)

print(f"  EUR/USD: {len(eur):,} rows  |  {eur.index[0].date()} → {eur.index[-1].date()}")
print(f"  GBP/USD: {len(gbp):,} rows  |  {gbp.index[0].date()} → {gbp.index[-1].date()}")

eur_daily = eur["Close"].resample("D").last().dropna()
gbp_daily = gbp["Close"].resample("D").last().dropna()

common_idx = eur_daily.index.intersection(gbp_daily.index)
eur_daily = eur_daily.loc[common_idx]
gbp_daily = gbp_daily.loc[common_idx]

print(f"\nDaily close aligned: {len(common_idx)} trading days")
print(f"  Date range: {common_idx[0].date()} → {common_idx[-1].date()}")

eur_ret = np.log(eur_daily / eur_daily.shift(1)).dropna()
gbp_ret = np.log(gbp_daily / gbp_daily.shift(1)).dropna()

common_idx_ret = eur_ret.index.intersection(gbp_ret.index)
eur_ret = eur_ret.loc[common_idx_ret]
gbp_ret = gbp_ret.loc[common_idx_ret]

print(f"\nLog return observations: {len(eur_ret)}")

X = gbp_ret.values.reshape(-1, 1)
y = eur_ret.values

result = fit_ols(X, y, add_intercept=True)

beta_0   = result['coefficients'][0]
beta_1   = result['coefficients'][1]
r2       = result['r_squared']
residuals = result['residuals']
se       = result['std_errors']

t_alpha = beta_0 / se[0]
t_beta  = beta_1 / se[1]

resid_mean   = residuals.mean()
resid_std    = residuals.std()
resid_skew   = pd.Series(residuals).skew()
resid_kurt   = pd.Series(residuals).kurt() 

lag1_autocorr = pd.Series(residuals).autocorr(lag=1)

Xfull = np.column_stack([np.ones(len(y)), X])
xtx_residual_check = np.abs(Xfull.T @ residuals).max()


print("OLS RESULTS — EUR/USD ~ alpha + beta * GBP/USD (log returns)")
print(f"  alpha (intercept)  : {beta_0:.8f}  |  SE: {se[0]:.8f}  |  t: {t_alpha:.4f}")
print(f"  beta  (hedge ratio): {beta_1:.8f}  |  SE: {se[1]:.8f}  |  t: {t_beta:.4f}")
print(f"  R-squared          : {r2:.6f}")
print(f"  Observations       : {len(y)}")
print("RESIDUAL DIAGNOSTICS")
print(f"  Mean               : {resid_mean:.2e}   (should be ~0 when intercept included)")
print(f"  Std dev            : {resid_std:.6f}")
print(f"  Skewness           : {resid_skew:.4f}")
print(f"  Excess kurtosis    : {resid_kurt:.4f}")
print(f"  Lag-1 autocorr     : {lag1_autocorr:.6f}  (>0.1 suggests serial correlation)")
print(f"  max|X'e|           : {xtx_residual_check:.2e}  (should be ~machine epsilon)")
print("INTERPRETATION")
print(f"  Hedge ratio beta = {beta_1:.4f}")
print(f"  For every 1 unit long EUR/USD, short {beta_1:.4f} units GBP/USD")
print(f"  R-squared = {r2:.4f}: GBP/USD returns explain {r2*100:.1f}% of EUR/USD return variance")
print(f"\n  NOTE: Stationarity of residuals not formally tested here.")
print(f"  ADF test on residuals put off for a later date (stationarity module).")
print(f"  If residuals are non-stationary, this spread is not a valid mean-reversion signal.")