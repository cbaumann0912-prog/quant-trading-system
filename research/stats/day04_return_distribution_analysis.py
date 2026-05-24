import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv(
    r'C:\Users\clayb\onedrive\desktop\career\02_quant_projects\data\USDJPY.csv',
    parse_dates=['Datetime'],
    date_format='%Y%m%d %H%M%S'
)
df = df.set_index('Datetime')
daily = df['Close'].resample('D').last()
returns = np.log(daily / daily.shift(1))
returns = returns.dropna()
kurtosis = stats.kurtosis(returns)
df_fit, mu_fit, sigma_fit = stats.t.fit(returns)

print(returns.head())
print("kurtosis = ", kurtosis)
print(f"degrees of freedom: {df_fit:.2f}")