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
from src.analysis.portfolio import var_historical, var_parametric, var_monte_carlo, cvar
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

CONFIDENCE_LEVELS = [0.95, 0.99]
MC_SIMULATIONS = 100_000
MC_SEED = 28

print("Day 53 VaR/CVaR report -- momentum-only pooled book")
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

analyzer = PerformanceAnalyzer(momentum_pnl)
ann_factor = analyzer.compute_ann_factor()

print(f"n_observations (pooled, regime-gated, lockbox-excluded): {len(momentum_pnl)}")
print(f"daily_mu: {momentum_pnl.mean():.6f}  daily_sigma: {momentum_pnl.std():.6f}")
print(f"empirical ann_factor: {ann_factor:.4f}")
print()

for confidence in CONFIDENCE_LEVELS:
    hist = var_historical(momentum_pnl, confidence)
    param = var_parametric(momentum_pnl, confidence)
    mc = var_monte_carlo(momentum_pnl, confidence, n_simulations=MC_SIMULATIONS, seed=MC_SEED)
    es = cvar(momentum_pnl, confidence)

    divergence = hist - param
    divergence_pct = (divergence / param * 100) if param != 0 else float("nan")

    print(f"--- confidence = {confidence:.2f} ---")
    print(f"var_historical:   {hist:.6f}")
    print(f"var_parametric:   {param:.6f}")
    print(f"var_monte_carlo:  {mc:.6f}")
    print(f"cvar:             {es:.6f}")
    print(f"divergence (hist - param): {divergence:.6f}  ({divergence_pct:+.1f}% of parametric)")
    print(f"cvar >= var_historical: {es >= hist}")
    print()
