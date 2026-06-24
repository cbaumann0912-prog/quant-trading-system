import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from src.analysis.performance_analyzer import PerformanceAnalyzer
 
DATA_DIR = r"C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\data"
 
FILES = {
    "EURUSD": "EURUSD.csv",
    "GBPUSD": "GBPUSD.csv",
    "USDJPY": "USDJPY.csv",
}
 
pairs = {}
 
for pair_name, filename in FILES.items():
    path = f"{DATA_DIR}\\{filename}"
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S")
    df = df.set_index("Datetime")
 
    daily_close = df["Close"].resample("D").last().dropna()
    log_returns = np.log(daily_close / daily_close.shift(1)).dropna()
 
    pairs[pair_name] = log_returns
 
results = {}
 
for pair_name, returns in pairs.items():
    analyzer = PerformanceAnalyzer(returns=returns, trades=None)
    report = analyzer.run_report()
    results[pair_name] = report
    print(f"--- {pair_name} ---")
    print(report)
    print()