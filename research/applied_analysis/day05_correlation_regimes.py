from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import pandas as pd
import numpy as np
from scipy import stats
import sys
sys.path.append(str(REPO_ROOT))

from src.stats.correlation import rolling_correlation

DEV_END = "2023-12-31"

EURUSD = pd.read_csv(
    REPO_ROOT.parent / "data" / "EURUSD.csv",
    parse_dates=['Datetime'],
    date_format='%Y%m%d %H%M%S'
)
GBPUSD = pd.read_csv(
    REPO_ROOT.parent / "data" / "GBPUSD.csv",
    parse_dates=['Datetime'],
    date_format='%Y%m%d %H%M%S'
)
USDJPY = pd.read_csv(
    REPO_ROOT.parent / "data" / "USDJPY.csv",
    parse_dates=['Datetime'],
    date_format='%Y%m%d %H%M%S'
)

EURUSD = EURUSD.set_index('Datetime').sort_index().loc[:DEV_END]
dailyEURUSD = EURUSD['Close'].resample('D').last()
GBPUSD = GBPUSD.set_index('Datetime').sort_index().loc[:DEV_END]
dailyGBPUSD = GBPUSD['Close'].resample('D').last()
USDJPY = USDJPY.set_index('Datetime').sort_index().loc[:DEV_END]
dailyUSDJPY = USDJPY['Close'].resample('D').last()

returnsEURUSD = np.log(dailyEURUSD / dailyEURUSD.shift(1))
returnsEURUSD = returnsEURUSD.dropna()
returnsGBPUSD = np.log(dailyGBPUSD / dailyGBPUSD.shift(1))
returnsGBPUSD = returnsGBPUSD.dropna()
returnsUSDJPY = np.log(dailyUSDJPY / dailyUSDJPY.shift(1))
returnsUSDJPY = returnsUSDJPY.dropna()

combined = pd.DataFrame({
    "EURUSD": returnsEURUSD,
    "GBPUSD": returnsGBPUSD,
    "USDJPY": returnsUSDJPY
})
combined = combined.dropna()

corrEU_GU = rolling_correlation(combined["EURUSD"], combined["GBPUSD"], 30)
corrGU_UJ = rolling_correlation(combined["GBPUSD"], combined["USDJPY"], 30)
corrEU_UJ = rolling_correlation(combined["EURUSD"], combined["USDJPY"], 30)

print (corrEU_GU.mean())
print (corrGU_UJ.mean())
print (corrEU_UJ.mean())

count = 0
regime_breaks = []
in_break = None

for date, value in corrEU_GU.items():
    if value < 0.4:
        count += 1
        if count == 14:
            start = date
            in_break = True
    elif value >= 0.4 and in_break:
        count = 0
        in_break = False
        end = date
        regime_breaks.append((start,end))
    elif value >= 0.4:
        count = 0

print(f"Regime breaks EURUSD/GBPUSD: {regime_breaks}")
avg_duration = np.mean([end - start for start, end in regime_breaks])
print(avg_duration)

count = 0
regime_breaks = []
in_break = None

for date, value in corrGU_UJ.items():
    if value > -0.1:
        count += 1
        if count == 14:
            start = date
            in_break = True
    elif value <= -0.1 and in_break:
        count = 0
        in_break = False
        end = date
        regime_breaks.append((start,end))
    elif value <= -0.1:
        count = 0

print(f"Regime breaks GBPUSD/USDJPY: {regime_breaks}")
avg_duration = np.mean([end - start for start, end in regime_breaks])
print(avg_duration)

count = 0
regime_breaks = []
in_break = None

for date, value in corrEU_UJ.items():
    if value > -0.2:
        count += 1
        if count == 14:
            start = date
            in_break = True
    elif value <= -0.2 and in_break:
        count = 0
        in_break = False
        end = date
        regime_breaks.append((start,end))
    elif value <= -0.2:
        count = 0

print(f"Regime breaks EURUSD/USDJPY: {regime_breaks}")
avg_duration = np.mean([end - start for start, end in regime_breaks])
print(avg_duration)