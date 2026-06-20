import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from src.signals.cointegration import engle_granger_test, ou_half_life

DATA_DIR = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"
BAR_DURATION_DAYS = 1  # set to whatever resampling you used in Day 24/25 (daily = 1)
TRANSACTION_COST_BPS = 2.0  # round-trip cost in bps, adjust to your actual cost model

eurusd = pd.read_csv(f"{DATA_DIR}\\EURUSD.csv")
eurusd["Datetime"] = pd.to_datetime(eurusd["Datetime"], format="%Y%m%d %H%M%S")
eurusd = eurusd.set_index("Datetime").sort_index()
eurusd_daily = eurusd["Close"].resample("D").last().dropna()

gbpusd = pd.read_csv(f"{DATA_DIR}\\GBPUSD.csv")
gbpusd["Datetime"] = pd.to_datetime(gbpusd["Datetime"], format="%Y%m%d %H%M%S")
gbpusd = gbpusd.set_index("Datetime").sort_index()
gbpusd_daily = gbpusd["Close"].resample("D").last().dropna()

usdjpy = pd.read_csv(f"{DATA_DIR}\\USDJPY.csv")
usdjpy["Datetime"] = pd.to_datetime(usdjpy["Datetime"], format="%Y%m%d %H%M%S")
usdjpy = usdjpy.set_index("Datetime").sort_index()
usdjpy_daily = usdjpy["Close"].resample("D").last().dropna()

aligned = pd.DataFrame({
    "EURUSD": eurusd_daily,
    "GBPUSD": gbpusd_daily,
    "USDJPY": usdjpy_daily,
}).dropna()

pairs_to_test = [
    ("EURUSD", "GBPUSD"),
    ("EURUSD", "USDJPY"),
    ("GBPUSD", "USDJPY"),
]

results = []

for y_name, x_name in pairs_to_test:
    y = aligned[y_name]
    x = aligned[x_name]

    eg_result = engle_granger_test(y, x)
    hl_result = ou_half_life(eg_result["residuals"])

    results.append({
        "pair": f"{y_name}_{x_name}",
        "adf_p": eg_result["adf_p"],
        "is_cointegrated": eg_result["is_cointegrated"],
        "hedge_ratio": eg_result["hedge_ratio"],
        "theta": hl_result["theta"],
        "mu": hl_result["mu"],
        "sigma": hl_result["sigma"],
        "half_life_bars": hl_result["half_life"],
    })

results_table = pd.DataFrame(results)
results_table["half_life_days"] = results_table["half_life_bars"] * BAR_DURATION_DAYS

results_table["cost_flag"] = results_table["half_life_days"].apply(
    lambda hl: "REVIEW" if hl > 30 or np.isinf(hl) else "OK"
)

results_table["entry_z"] = 2.0
results_table["exit_z"] = 0.5

print(results_table.to_string(index=False))