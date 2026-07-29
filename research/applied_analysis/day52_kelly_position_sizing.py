import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.framework.data_loader import DataLoader
from src.features.regime_classifier import classify_regime
from src.signals.regime_refit import compute_composite_regime_score_walkforward
from src.signals.momentum import momentum_signal
from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.analysis.portfolio import kelly_fraction, fractional_kelly
from research.strategies.s04_vol_regime_breakout.vol_regime_signal_report_pipeline import load_rate_diff, fit_windows, regime_gated_pnl

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
START = "2011-01-01"
END = "2023-12-31"

MOMENTUM_LOOKBACK = 78
REGIME_WINDOW = 78
TURBULENT_THRESHOLD = 1.5
CALM_THRESHOLD = 1.0

TRAIN_YEARS = 5
TEST_MONTHS = 12
EMBARGO_DAYS = 5

KELLY_FRACTION = 0.25
CAPITAL = 100_000

print("Day 52 Kelly sizing mechanics demo")
print()

momentum_pnl_chunks = []

for pair in PAIRS:
    loader = DataLoader(pairs=[pair], start=START, end=END, data_dir=str(DATA_DIR))
    prices = loader.load()[pair]
    data = prices.to_frame(name="price")

    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.rolling(REGIME_WINDOW).std()

    rate_diff_monthly = load_rate_diff(pair)
    rate_diff = rate_diff_monthly.reindex(
        pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    ).ffill()

    windows = fit_windows(prices)

    composite_z, _diag = compute_composite_regime_score_walkforward(vol, rate_diff, windows)
    regime = classify_regime(composite_z, turbulent_threshold=TURBULENT_THRESHOLD, calm_threshold=CALM_THRESHOLD)
    turbulent_dummy = (regime == "turbulent").astype(float)

    momentum = momentum_signal(data, MOMENTUM_LOOKBACK)

    momentum_pnl_chunks.append(regime_gated_pnl(momentum, log_returns, turbulent_dummy))

momentum_pnl = pd.concat(momentum_pnl_chunks, axis=1).mean(axis=1, skipna=True).dropna()

momentum_analyzer = PerformanceAnalyzer(momentum_pnl)
momentum_ann_factor = momentum_analyzer.compute_ann_factor()
momentum_daily_mu = momentum_pnl.mean()
momentum_daily_sigma = momentum_pnl.std()
momentum_mu = momentum_daily_mu * momentum_ann_factor
momentum_sigma = momentum_daily_sigma * np.sqrt(momentum_ann_factor)

momentum_full_kelly = kelly_fraction(momentum_mu, momentum_sigma)
momentum_fractional_kelly = fractional_kelly(momentum_mu, momentum_sigma, fraction=KELLY_FRACTION)
momentum_position_size = momentum_fractional_kelly * CAPITAL

print("Momentum-only pooled book (momentum_only_pooled_book.md")
print(f"n_observations (pooled, regime-gated, lockbox-excluded): {len(momentum_pnl)}")
print(f"daily_mu: {momentum_daily_mu:.6f}  daily_sigma: {momentum_daily_sigma:.6f}")
print(f"empirical ann_factor (PerformanceAnalyzer.compute_ann_factor): {momentum_ann_factor:.4f}")
print(f"annual mu: {momentum_mu:.4f}  annual sigma: {momentum_sigma:.4f}")
print(f"full kelly f*: {momentum_full_kelly:.4f}")
print(f"{KELLY_FRACTION:.2f} fractional kelly f: {momentum_fractional_kelly:.4f}")
print(f"position on ${CAPITAL:,.0f}: ${momentum_position_size:,.2f}")
