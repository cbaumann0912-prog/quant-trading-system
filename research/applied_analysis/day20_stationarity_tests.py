import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from src.data.stationarity import check_stationarity
from src.stats.regression import fit_ols
from src.evaluation.significance import benjamini_hochberg_correction

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

common_idx = (eurusd_daily.index.intersection(gbpusd_daily.index).intersection(usdjpy_daily.index))

eurusd_daily = eurusd_daily.loc[common_idx]
gbpusd_daily = gbpusd_daily.loc[common_idx]
usdjpy_daily = usdjpy_daily.loc[common_idx]

log_eur = pd.Series(np.log(eurusd_daily), index=eurusd_daily.index)
log_gbp = pd.Series(np.log(gbpusd_daily), index=gbpusd_daily.index)
log_jpy = pd.Series(np.log(usdjpy_daily), index=usdjpy_daily.index)

print("LAYER 1 — Individual Log Price Level Stationarity")
print("Expected: all three I(1)")
print(f"{'Pair':<12} {'ADF Stat':>10} {'ADF p':>8} {'KPSS Stat':>10} {'KPSS p':>8}  Verdict")

for name, series in [("EUR/USD", log_eur), ("GBP/USD", log_gbp), ("USD/JPY", log_jpy)]:
    result = check_stationarity(series)
    verdict = "I(0) stationary" if result["is_stationary"] else "I(1) unit root"
    print(
        f"{name:<12}"
        f"{result['adf_stat']:>10.4f}"
        f"{result['adf_p']:>8.4f}"
        f"{result['kpss_stat']:>10.4f}"
        f"{result['kpss_p']:>8.4f}  "
        f"{verdict}"
    )

print("LAYER 2 & 3 — Engle-Granger Step 1 (OLS) + Step 2 (Residual Stationarity)")
print(f"{'Spread':<25} {'Beta':>8} {'ADF Stat':>10} {'ADF p':>8} {'KPSS Stat':>10} {'KPSS p':>8}  Verdict")

y = log_eur.values
X = log_gbp.values.reshape(-1, 1)
ols_eur_gbp   = fit_ols(X, y, True)
beta_eur_gbp  = ols_eur_gbp["coefficients"][1]
resid_eur_gbp = ols_eur_gbp["residuals"]
stat_eur_gbp  = check_stationarity(pd.Series(resid_eur_gbp))
verdict_eur_gbp = "I(0) cointegrated" if stat_eur_gbp["is_stationary"] else "I(1) no cointegration"

print(f"{'EUR/USD ~ GBP/USD':<25} {beta_eur_gbp:>8.4f} {stat_eur_gbp['adf_stat']:>10.4f} {stat_eur_gbp['adf_p']:>8.4f} {stat_eur_gbp['kpss_stat']:>10.4f} {stat_eur_gbp['kpss_p']:>8.4f}  {verdict_eur_gbp}")

y = log_eur.values
X = log_jpy.values.reshape(-1, 1)
ols_eur_jpy  = fit_ols(X, y, True)
beta_eur_jpy = ols_eur_jpy["coefficients"][1]
resid_eur_jpy = ols_eur_jpy["residuals"]                  
stat_eur_jpy = check_stationarity(pd.Series(resid_eur_jpy)) 
verdict_eur_jpy = "I(0) cointegrated" if stat_eur_jpy["is_stationary"] else "I(1) no cointegration"

print(f"{'EUR/USD ~ USD/JPY':<25} {beta_eur_jpy:>8.4f} {stat_eur_jpy['adf_stat']:>10.4f} {stat_eur_jpy['adf_p']:>8.4f} {stat_eur_jpy['kpss_stat']:>10.4f} {stat_eur_jpy['kpss_p']:>8.4f}  {verdict_eur_jpy}")

y = log_gbp.values
X = log_jpy.values.reshape(-1, 1)
ols_gbp_jpy  = fit_ols(X, y, True)                      
beta_gbp_jpy = ols_gbp_jpy["coefficients"][1]
resid_gbp_jpy = ols_gbp_jpy["residuals"]
stat_gbp_jpy = check_stationarity(pd.Series(resid_gbp_jpy))    
verdict_gbp_jpy = "I(0) cointegrated" if stat_gbp_jpy["is_stationary"] else "I(1) no cointegration"

print(f"{'GBP/USD ~ USD/JPY':<25} {beta_gbp_jpy:>8.4f} {stat_gbp_jpy['adf_stat']:>10.4f} {stat_gbp_jpy['adf_p']:>8.4f} {stat_gbp_jpy['kpss_stat']:>10.4f} {stat_gbp_jpy['kpss_p']:>8.4f}  {verdict_gbp_jpy}")

print("MULTIPLE TESTING CORRECTION — Benjamini-Hochberg on ADF p-values")
print(f"{'Spread':<25} {'Raw ADF p':>10} {'BH Reject':>12}")

adf_pvalues = [stat_eur_gbp["adf_p"], stat_eur_jpy["adf_p"], stat_gbp_jpy["adf_p"]]
pair_labels = ["EUR/USD ~ GBP/USD", "EUR/USD ~ USD/JPY", "GBP/USD ~ USD/JPY"]

bh_rejections = benjamini_hochberg_correction(adf_pvalues, 0.05)

for label, raw_p, reject in zip(pair_labels, adf_pvalues, bh_rejections):
    print(f"{label:<25} {raw_p:>10.4f} {str(reject):>12}")

print("RECOMMENDATIONS")

for label, stat in [
    ("EUR/USD ~ GBP/USD", stat_eur_gbp),
    ("EUR/USD ~ USD/JPY", stat_eur_jpy),
    ("GBP/USD ~ USD/JPY", stat_gbp_jpy),
]:
    print(f"\n{label}")
    print(f"  {stat['recommendation']}")