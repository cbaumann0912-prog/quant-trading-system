import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.framework.data_loader import DataLoader
from src.features.regime_classifier import classify_regime
from src.signals.regime_refit import compute_composite_regime_score_walkforward
from src.signals.momentum import momentum_signal
from src.stats.stochastic import simulate_gbm
from src.analysis.performance_analyzer import PerformanceAnalyzer
from research.strategies.s04_vol_regime_breakout.vol_regime_signal_report_pipeline import (
    load_rate_diff,
    fit_windows,
    regime_gated_pnl,
)

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

N_PATHS = 1000
SEED = 28
DRAWDOWN_THRESHOLD = -0.20


def main() -> None:
    momentum_pnl_chunks: list[pd.Series] = []

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
    ann_factor = momentum_analyzer.compute_ann_factor()

    n_steps = len(momentum_pnl)

    daily_mu = momentum_pnl.mean()
    daily_sigma = momentum_pnl.std()
    annual_mu = daily_mu * ann_factor
    annual_sigma = daily_sigma * np.sqrt(ann_factor)
    T = n_steps / ann_factor

    simulated_paths = simulate_gbm(
        S0=1.0,
        mu=annual_mu,
        sigma=annual_sigma,
        T=T,
        n_steps=n_steps,
        n_paths=N_PATHS,
        seed=SEED,
    )

    synthetic_index = pd.bdate_range("2000-01-01", periods=n_steps + 1)
    simulated_max_drawdowns = np.empty(N_PATHS)
    for i in range(N_PATHS):
        path_returns = pd.Series(simulated_paths[i], index=synthetic_index).pct_change().dropna()
        simulated_max_drawdowns[i] = PerformanceAnalyzer(path_returns).compute_max_drawdown()["value"]

    pct_5th_max_drawdown = np.percentile(simulated_max_drawdowns, 5)
    prob_20pct_drawdown = float((simulated_max_drawdowns <= DRAWDOWN_THRESHOLD).mean())

    historical_max_drawdown = momentum_analyzer.compute_max_drawdown()["value"]

    print(f"n_observations (pooled momentum leg): {len(momentum_pnl)}")
    print(f"daily_mu: {daily_mu:.6f}  daily_sigma: {daily_sigma:.6f}")
    print(f"empirical ann_factor (n_obs / years_spanned): {ann_factor:.4f}")
    print(f"annual_mu (x{ann_factor:.4f}): {annual_mu:.4f}  annual_sigma (x sqrt({ann_factor:.4f})): {annual_sigma:.4f}")
    print(f"n_steps (= full development-period series length, lockbox excluded): {n_steps}")
    print(f"T (years, {n_steps} steps @ empirical ann_factor): {T:.4f}")
    print(f"simulated 5th pct max drawdown: {pct_5th_max_drawdown:.4f}")
    print(f"simulated median max drawdown: {np.median(simulated_max_drawdowns):.4f}")
    print(f"P(max drawdown <= -20%) over full {n_steps}-step development horizon: {prob_20pct_drawdown:.4f}")
    print(f"historical realized max drawdown (full pooled series): {historical_max_drawdown:.4f}")

    out_path = REPO_ROOT / "research" / "notes" / "day50_monte_carlo_drawdown.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Day 50 Monte Carlo Drawdown Output",
        "",
        f"n_observations (pooled momentum leg, regime-gated): {len(momentum_pnl)}",
        f"daily_mu: {daily_mu:.6f}",
        f"daily_sigma: {daily_sigma:.6f}",
        f"empirical ann_factor (n_obs / years_spanned): {ann_factor:.4f}",
        f"annual_mu: {annual_mu:.4f}",
        f"annual_sigma: {annual_sigma:.4f}",
        f"T (years): {T:.4f}",
        f"n_paths: {N_PATHS}",
        f"n_steps: {n_steps}",
        f"seed: {SEED}",
        f"simulated 5th percentile max drawdown: {pct_5th_max_drawdown:.4f}",
        f"simulated median max drawdown: {np.median(simulated_max_drawdowns):.4f}",
        f"P(max drawdown <= -20%): {prob_20pct_drawdown:.4f}",
        f"historical realized max drawdown (pooled momentum leg, full sample): {historical_max_drawdown:.4f}",
    ]
    out_path.write_text("\n".join(lines))
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
