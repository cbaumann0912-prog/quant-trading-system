import sys
sys.path.insert(0, r'C:\Users\clayb\onedrive\desktop\career\02_quant_projects\summer2026')

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
from src.stats.hypothesis_tests import compute_effect_size_cohens_d, t_test_mean

df = pd.read_csv(
    r'C:\Users\clayb\onedrive\desktop\career\02_quant_projects\summer2026\results\OG_results\realistic\daily_balance.csv',
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