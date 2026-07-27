import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.framework.data_loader import DataLoader

REGIME_WINDOW = 78
PRICE_Z_WINDOW = 26

DATA_DIR = Path(__file__).resolve().parents[3].parent / "data"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
REGIME_THRESHOLDS = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
PRICE_Z_THRESHOLDS = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
FORWARD_HORIZONS = {"1wk": 5, "2wk": 10, "1mo": PRICE_Z_WINDOW}
PUBLICATION_LAG_MONTHS = 2 

us = pd.read_csv(DATA_DIR / "us_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"].ffill()
ea = pd.read_csv(DATA_DIR / "ea_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
uk = pd.read_csv(DATA_DIR / "uk_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
jp = pd.read_csv(DATA_DIR / "jp_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]

rate_diffs_monthly = pd.DataFrame({
    "EURUSD": ea - us,
    "GBPUSD": uk - us,
    "USDJPY": us - jp,
}).dropna()
rate_diffs_monthly_lagged = rate_diffs_monthly.shift(PUBLICATION_LAG_MONTHS)

for pair in PAIRS:
    loader = DataLoader(pairs=[pair], start="2011-01-01", end="2026-05-01",
                         embargo_days=5, data_dir=str(DATA_DIR))
    prices = loader.load()[pair]

    log_ret = np.log(prices / prices.shift(1)).dropna()
    rolling_vol = log_ret.rolling(REGIME_WINDOW).std().dropna()

    rate_diff_daily = rate_diffs_monthly_lagged[pair].reindex(
        pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    ).ffill()

    reg_df = pd.concat([rolling_vol.rename("vol"), rate_diff_daily.rename("rate_diff")], axis=1).dropna()
    z = (reg_df - reg_df.mean()) / reg_df.std()

    cov = np.cov(z["vol"], z["rate_diff"])
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, np.argmax(eigvals)]
    if pc1[0] < 0:
        pc1 = -pc1
    explained_var_ratio = max(eigvals) / eigvals.sum()

    composite = z["vol"] * pc1[0] + z["rate_diff"] * pc1[1]
    composite_z = (composite - composite.mean()) / composite.std()

    print(f"=== {pair} === PC1 loadings (vol, rate_diff): {pc1.round(3)}, "
          f"explained variance: {explained_var_ratio:.3f}")

    print(f"--- {pair}: regime composite threshold ---")
    print("threshold | % obs with |composite z| exceeding it | n obs")
    for t in REGIME_THRESHOLDS:
        pct = 100 * (composite_z.abs() > t).mean()
        n = int((composite_z.abs() > t).sum())
        print(f"  {t:.2f}    | {pct:6.2f}%  | {n}")
    deadzone_pct = 100 * ((composite_z.abs() >= 1.0) & (composite_z.abs() <= 1.5)).mean()
    print(f"deadzone (1.0-1.5, no trade): {deadzone_pct:.2f}% of observations")
    print()


    price_mean = prices.rolling(PRICE_Z_WINDOW).mean()
    price_std = prices.rolling(PRICE_Z_WINDOW).std()
    price_z = ((prices - price_mean) / price_std).dropna()

    fwd_rets = {}
    for name, h in FORWARD_HORIZONS.items():
        fwd_rets[name] = (prices.shift(-h) / prices - 1).rename(f"fwd_{name}")

    combined = pd.concat(
        [composite_z.rename("regime_z"), price_z.rename("price_z")] + list(fwd_rets.values()), axis=1
    ).dropna()
    calm = combined[combined["regime_z"].abs() < 1.0]

    print(f"--- {pair}: mean-reversion trigger, conditional forward return (calm n={len(calm)}) ---")
    header = "thresh | " + " | ".join(f"{h} sign-adj mean fwd ret (n)" for h in FORWARD_HORIZONS)
    print(header)
    for t in PRICE_Z_THRESHOLDS:
        extreme = calm[calm["price_z"].abs() > t]
        sign = -np.sign(extreme["price_z"])
        row = [f"{t:.2f}  "]
        for name in FORWARD_HORIZONS:
            reversion_ret = (sign * extreme[f"fwd_{name}"]).mean() if len(extreme) else float("nan")
            row.append(f"{reversion_ret*100:+.3f}% (n={len(extreme)})")
        print(" | ".join(row))
    print()
