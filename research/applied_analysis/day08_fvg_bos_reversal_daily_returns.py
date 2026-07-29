from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import pandas as pd
import numpy as np
from scipy import stats

import sys
import os
sys.path.append(str(REPO_ROOT))
from src.stats.hypothesis_tests import t_test_mean

df = pd.read_csv(
    REPO_ROOT.parent / "_original_strategy" / "OG_results" / "realistic" / "daily_balance.csv",
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
