import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from src.data.stationarity import plot_acf_pacf, ljung_box_test

DEV_END = "2023-12-31"

DATA_DIR = Path(r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data")
EURUSD_PATH = DATA_DIR / "EURUSD.csv"
GBPUSD_PATH = DATA_DIR / "GBPUSD.csv"
USDJPY_PATH = DATA_DIR / "USDJPY.csv"

MAX_LAG_PLOT = 40
MAX_LAG_LB   = 20


def load_pair(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime").sort_index().loc[:DEV_END]
    daily = df["Close"].resample("D").last().dropna()
    log_returns = np.log(daily / daily.shift(1)).dropna()
    return log_returns


eurusd_returns = load_pair(EURUSD_PATH)
gbpusd_returns = load_pair(GBPUSD_PATH)
usdjpy_returns = load_pair(USDJPY_PATH)

pairs = {
    "EUR/USD": eurusd_returns,
    "GBP/USD": gbpusd_returns,
    "USD/JPY": usdjpy_returns,
}

PLOT_DIR = Path("research/applied_analysis/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

for name, returns in pairs.items():
    print(f"  {name}")
    print(f"  Obs: {len(returns):,}")
    print(f"  Date range: {returns.index[0]} → {returns.index[-1]}")

    plot_acf_pacf(
        series=returns.values,
        lags=MAX_LAG_PLOT,
        title=name,
        save_path=PLOT_DIR / f"day22_{name.replace('/', '')}_acf_pacf.png",
    )

    lb = ljung_box_test(series=returns.values, lags=MAX_LAG_LB)

    print(f"\n  Ljung-Box results (lags 1–{MAX_LAG_LB}):")
    print(lb.to_string())

    lag5  = lb.iloc[4]
    lag20 = lb.iloc[19]
    print(f"\n  Lag 5  → Q = {lag5['lb_stat']:.4f},  p = {lag5['lb_pvalue']:.4f}")
    print(f"  Lag 20 → Q = {lag20['lb_stat']:.4f}, p = {lag20['lb_pvalue']:.4f}")

    reject = (lb["lb_pvalue"] < 0.05).sum()
    print(f"  Lags with p < 0.05: {reject} of {MAX_LAG_LB}")