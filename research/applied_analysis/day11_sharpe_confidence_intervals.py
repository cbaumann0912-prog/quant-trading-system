from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
from scipy import stats
from src.evaluation.bootstrap import bootstrap_confidence_interval

df = pd.read_csv(
    REPO_ROOT.parent / "_original_strategy" / "OG_results" / "realistic" / "daily_balance.csv",
    parse_dates=['date']
)
df = df.set_index('date')
returns = np.log(df['account_balance'] / df['account_balance'].shift(1))
returns = returns[returns!=0].dropna()

risk_free_rate = 0 

years = (returns.index.max() - returns.index.min()).days / 365.25
ann_factor = len(returns) / years
print(round(ann_factor, 2))

sharpe_point = ((returns.mean() - risk_free_rate) / returns.std()) * np.sqrt(ann_factor)
print(round(sharpe_point, 4))

def compute_sharpe(sample):
    return ((sample.mean() - risk_free_rate) / sample.std()) * np.sqrt(ann_factor)

result = bootstrap_confidence_interval(returns, compute_sharpe, 10000, 0.95)
print(result)
print (result[1]-result[0])