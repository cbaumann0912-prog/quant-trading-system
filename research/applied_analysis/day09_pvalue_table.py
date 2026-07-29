from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
from src.stats.hypothesis_tests import compute_effect_size_cohens_d, t_test_mean

df = pd.read_csv(
    REPO_ROOT.parent / "_original_strategy" / "OG_results" / "realistic" / "daily_balance.csv",
    parse_dates=['date']
)
df = df.set_index('date')
returns = np.log(df['account_balance'] / df['account_balance'].shift(1))
returns = returns[returns!=0].dropna()

table = []
result = t_test_mean(returns, null_mean=0.0, confidence=0.95)

table.append(result["t_stat"])
table.append(result["p_value"])

sp500 = yf.download('^GSPC', start='2011-01-01', end='2026-03-15')
sp500_returns = np.log(sp500['Close']['^GSPC'] / sp500['Close']['^GSPC'].shift(1)).dropna()

table.append(compute_effect_size_cohens_d(returns,sp500_returns))

if result["p_value"] < 0.05:
    table.append("True")
else:
    table.append("False")

print(table)