from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import pandas as pd
import numpy as np
from scipy import stats

DEV_END = "2023-12-31"

df = pd.read_csv(
    REPO_ROOT.parent / "data" / "USDJPY.csv",
    parse_dates=['Datetime'],
    date_format='%Y%m%d %H%M%S'
)
df = df.set_index('Datetime').sort_index().loc[:DEV_END]
daily = df['Close'].resample('D').last()
returns = np.log(daily / daily.shift(1))
returns = returns.dropna()
kurtosis = stats.kurtosis(returns)
df_fit, mu_fit, sigma_fit = stats.t.fit(returns)

print(returns.head())
print("kurtosis = ", kurtosis)
print(f"degrees of freedom: {df_fit:.2f}")