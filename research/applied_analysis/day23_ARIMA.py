import sys
sys.path.insert(0, r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\summer2026")

import pandas as pd
import numpy as np
from src.data.time_series import fit_arima, select_arima_order
from src.data.stationarity import ljung_box_test

DEV_END = "2023-12-31"



DATA_DIR  = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"
PAIRS     = {"EUR/USD": "EURUSD.csv", "GBP/USD": "GBPUSD.csv", "USD/JPY": "USDJPY.csv"}
MAX_P     = 3
MAX_Q     = 3
LB_LAG    = 10
OUTPUT    = r"research/notes/day23_arima_forex_pairs.md"


def load_daily_returns(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime").sort_index().loc[:DEV_END]
    daily = df["Close"].resample("D").last().dropna()
    log_returns = np.log(daily / daily.shift(1)).dropna()
    return log_returns

rows = []
test = None

for pair, filename in PAIRS.items():
    path = f"{DATA_DIR}\\{filename}"
    returns = load_daily_returns(path)

    order = select_arima_order(returns, max_p=MAX_P, max_q=MAX_Q, d=0, criterion="aic")
    fit_result = fit_arima(returns, order=order)
    lb_result = ljung_box_test(fit_result["residuals"], lags=LB_LAG)

    aic = fit_result["aic"]
    lb_pval = lb_result["lb_pvalue"].iloc[-1]

    if lb_pval > 0.05:
        test = True
    else:
        test = False

    rows.append({
        "pair":    pair,
        "order":   order,
        "aic":     aic,
        "lb_pval": lb_pval,
        "pass":    test,
    })


print(f"\n{'Pair':<10} {'Order':<12} {'AIC':>10} {'LB p (10)':>12} {'Residuals OK':>14}")
for r in rows:
    print(f"{r['pair']:<10} {str(r['order']):<12} {r['aic']:>10.2f} "
          f"{r['lb_pval']:>12.4f} {str(r['pass']):>14}")
    
gbp_returns = load_daily_returns(f"{DATA_DIR}\\GBPUSD.csv")
fit = fit_arima(gbp_returns, order=(1, 0, 0))
print(fit["params"])
