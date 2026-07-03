import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.analysis.portfolio import efficient_frontier, minimum_variance_portfolio, leverage_bounded_return_range
from src.analysis.portfolio_stats import compute_portfolio_return, compute_portfolio_variance

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
    pairs[pair_name] = np.log(daily_close / daily_close.shift(1)).dropna()

returns = pd.concat(pairs, axis=1, join="inner")
returns.columns = list(FILES.keys())

LEVERAGE_CAPS = np.array([50.0, 50.0, 50.0])

ann_factor = len(returns) / ((returns.index.max() - returns.index.min()).days / 365.25)

mv = minimum_variance_portfolio(returns)
frontier = efficient_frontier(returns, ann_factor=ann_factor, n_points=50)

p_bar = returns.mean().to_numpy()
n_assets = returns.shape[1]
x_equal = np.full(n_assets, 1.0 / n_assets)
equal_return = compute_portfolio_return(x_equal, p_bar)
equal_variance = compute_portfolio_variance(x_equal, returns.cov().to_numpy())
equal_volatility = np.sqrt(equal_variance)

max_sharpe_idx = np.argmax(frontier["sharpes"])
max_sharpe_return = frontier["returns"][max_sharpe_idx]
max_sharpe_vol = frontier["volatilities"][max_sharpe_idx]
max_sharpe_sharpe = frontier["sharpes"][max_sharpe_idx]
max_sharpe_weights = frontier["weights"][max_sharpe_idx]

print(f"Global min-variance: return={mv['return']:.6f}, variance={mv['variance']:.8f}, weights={mv['weights']}")
print(f"\nMax-Sharpe (grid): return={max_sharpe_return:.6f}, "
      f"vol={max_sharpe_vol:.6f}, sharpe={max_sharpe_sharpe:.4f}, weights={max_sharpe_weights}")
print(f"\nEqual-weight: return={equal_return:.6f}, variance={equal_variance:.8f}, volatility={equal_volatility:.6f}")
print(f"\nEqual-weight vs. global min-variance return: "
      f"{'BELOW min-variance return' if equal_return < mv['return'] else 'within frontier range'}")
print(f"Equal-weight vs. global min-variance variance: "
      f"{'HIGHER' if equal_variance > mv['variance'] else 'lower'}")

lev_r_min, lev_r_max = leverage_bounded_return_range(returns, LEVERAGE_CAPS)
max_sharpe_within_cap = np.all(np.abs(max_sharpe_weights) <= LEVERAGE_CAPS)

print(f"\nLeverage check (50:1 per-pair):")
print(f"  feasible r range under leverage cap: [{lev_r_min:.6f}, {lev_r_max:.6f}]")
print(f"  max-Sharpe weights within cap: {max_sharpe_within_cap} "
      f"(largest |weight| = {np.max(np.abs(max_sharpe_weights)):.4f} vs. cap of {LEVERAGE_CAPS[0]:.0f})")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(frontier["volatilities"], frontier["returns"],
        label="Efficient Frontier", color="steelblue", linewidth=2)
ax.scatter(equal_volatility, equal_return, color="darkorange", marker="o", s=80, zorder=5,
           label="Equal-Weight (1/3, 1/3, 1/3)")
ax.scatter(max_sharpe_vol, max_sharpe_return, color="crimson", marker="*", s=200, zorder=5,
           label="Max-Sharpe (grid)")
ax.scatter(np.sqrt(mv["variance"]), mv["return"], color="black", marker="x", s=80, zorder=5,
           label="Global Min-Variance")
ax.set_xlabel("Volatility (std. dev.)")
ax.set_ylabel("Expected Return")
ax.set_title("Efficient Frontier — EUR/USD, GBP/USD, USD/JPY")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("research/audit_images/day33_efficient_frontier_forex.png", dpi=150)
plt.show()