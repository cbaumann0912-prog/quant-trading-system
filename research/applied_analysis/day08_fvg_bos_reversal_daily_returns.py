import pandas as pd
import numpy as np
from scipy import stats

import sys
import os
sys.path.append(r'C:\Users\clayb\OneDrive\Desktop\Career\02_quant_projects\summer2026')
from src.stats.hypothesis_tests import t_test_mean

df = pd.read_csv(
    r'C:\Users\clayb\onedrive\desktop\career\02_quant_projects\summer2026\results\OG_results\realistic\daily_balance.csv',
    parse_dates=['date']
)
df = df.set_index('date')
returns = np.log(df['account_balance'] / df['account_balance'].shift(1))
returns = returns[returns!=0].dropna()

result = t_test_mean(returns, null_mean=0.0, confidence=0.95)

print(f"n observations : {len(returns)}")
print(f"mean return    : {returns.mean():.6f}")
print(f"std deviation  : {returns.std():.6f}")
print(f"std error      : {returns.std() / np.sqrt(len(returns)):.6f}")
print(f"min return     : {returns.min():.6f}")
print(f"max return     : {returns.max():.6f}")
print(f"t-stat         : {result['t_stat']:.4f}")
print(f"p-value        : {result['p_value']:.4f}")
print(f"reject null    : {result['reject_null']}")
print(f"CI             : ({result['confidence_interval'][0]:.6f}, {result['confidence_interval'][1]:.6f})")
