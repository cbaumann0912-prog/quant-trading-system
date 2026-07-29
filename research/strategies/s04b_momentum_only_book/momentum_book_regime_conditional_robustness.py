import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.framework.data_loader import DataLoader
from src.features.regime_classifier import classify_regime
from src.features.garch import fit_garch, classify_vol_regime
from src.signals.regime_refit import compute_composite_regime_score_walkforward
from src.signals.momentum import momentum_signal
from src.analysis.performance_analyzer import PerformanceAnalyzer, regime_conditional_performance
from research.strategies.s04_vol_regime_breakout.vol_regime_signal_report_pipeline import fit_windows, regime_gated_pnl

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT.parent / "data"

RATE_FILES_ALL = {
    "EURUSD": ("ea", "us"),
    "GBPUSD": ("uk", "us"),
    "USDJPY": ("us", "jp"),
    "USDCHF": ("us", "ch"),
    "AUDUSD": ("au", "us"),
    "USDCAD": ("us", "ca"),
    "NZDUSD": ("nz", "us"),
    "EURGBP": ("ea", "uk"),
    "EURJPY": ("ea", "jp"),
    "EURCHF": ("ea", "ch"),
}
VALIDATED_PAIRS = {"EURUSD", "GBPUSD", "USDJPY"}
PAIRS = list(RATE_FILES_ALL.keys())

START = "2011-01-01"
END = "2023-12-31"
MOMENTUM_LOOKBACK = 78
REGIME_WINDOW = 78
TURBULENT_THRESHOLD = 1.5
CALM_THRESHOLD = 1.0
PUBLICATION_LAG_MONTHS = 2


def load_rate_diff(pair: str) -> pd.Series:
    a, b = RATE_FILES_ALL[pair]
    a_series = pd.read_csv(DATA_DIR / f"{a}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
    b_series = pd.read_csv(DATA_DIR / f"{b}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
    diff_monthly = (a_series - b_series).dropna()
    return diff_monthly.shift(PUBLICATION_LAG_MONTHS)


print("Day 56 -- Regime-conditional performance, momentum-only pooled book")
print("Baseline (deployed) regime gate: 78d realized vol + rate-diff composite, |z|>1.5 turbulent")
print("New lens: GARCH(1,1) conditional vol, 2-means high/low split")
print(f"Universe: {len(PAIRS)} pairs -- {sorted(VALIDATED_PAIRS)} validated, rest diagnostic-only")
print()

momentum_pnl_chunks = {}
conditional_vol_by_pair = {}

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
    pnl = regime_gated_pnl(momentum, log_returns, turbulent_dummy)
    momentum_pnl_chunks[pair] = pnl

    garch_result = fit_garch(log_returns)
    conditional_vol_by_pair[pair] = garch_result["conditional_vol"]

    flag = "  <-- degenerate fit (persistence>=0.999)" if garch_result["persistence"] >= 0.999 else ""
    tag = "validated" if pair in VALIDATED_PAIRS else "diagnostic-only"
    print(f"{pair} [{tag}]: GARCH persistence={garch_result['persistence']:.4f}  "
          f"long_run_vol={garch_result['long_run_vol']:.5f}  "
          f"n_conditional_vol_obs={len(garch_result['conditional_vol'])}{flag}")

print()

print("=== Per-pair: deployed momentum PnL split by own-pair GARCH vol regime ===")
per_pair_results = {}
for pair in PAIRS:
    pnl = momentum_pnl_chunks[pair].dropna()
    vol_regime = classify_vol_regime(conditional_vol_by_pair[pair], n_regimes=2)

    result = regime_conditional_performance(pnl, vol_regime)
    per_pair_results[pair] = result

    print(f"{pair}: high_vol_sharpe={result['high_vol_sharpe']:.4f}  "
          f"low_vol_sharpe={result['low_vol_sharpe']:.4f}  "
          f"high_vol_pct={result['high_vol_pct']:.3f}  low_vol_pct={result['low_vol_pct']:.3f}")

print()

print("=== Pooled book (validated 3 pairs): equal-weight momentum PnL split by book-level GARCH vol regime ===")
validated_list = [p for p in PAIRS if p in VALIDATED_PAIRS]
pooled_pnl = pd.concat([momentum_pnl_chunks[p] for p in validated_list], axis=1).mean(axis=1, skipna=True).dropna()
book_conditional_vol = pd.concat(
    [conditional_vol_by_pair[p] for p in validated_list], axis=1
).mean(axis=1, skipna=True).dropna()
book_vol_regime = classify_vol_regime(book_conditional_vol, n_regimes=2)
pooled_result = regime_conditional_performance(pooled_pnl, book_vol_regime)
unconditional = PerformanceAnalyzer(pooled_pnl).compute_sharpe()

print(f"n_observations: {len(pooled_pnl)}")
print(f"high_vol_sharpe={pooled_result['high_vol_sharpe']:.4f}  low_vol_sharpe={pooled_result['low_vol_sharpe']:.4f}")
print(f"high_vol_pct={pooled_result['high_vol_pct']:.3f}  low_vol_pct={pooled_result['low_vol_pct']:.3f}")
print(f"unconditional_pooled_sharpe (reference, matches prior audits): {unconditional:.4f}")

print()

print("=== Pooled book (all 10 pairs, diagnostic-only): equal-weight momentum PnL split by book-level GARCH vol regime ===")
pooled_pnl_all = pd.concat(momentum_pnl_chunks.values(), axis=1).mean(axis=1, skipna=True).dropna()
book_conditional_vol_all = pd.concat(conditional_vol_by_pair.values(), axis=1).mean(axis=1, skipna=True).dropna()
book_vol_regime_all = classify_vol_regime(book_conditional_vol_all, n_regimes=2)
pooled_result_all = regime_conditional_performance(pooled_pnl_all, book_vol_regime_all)
unconditional_all = PerformanceAnalyzer(pooled_pnl_all).compute_sharpe()

print(f"n_observations: {len(pooled_pnl_all)}")
print(f"high_vol_sharpe={pooled_result_all['high_vol_sharpe']:.4f}  low_vol_sharpe={pooled_result_all['low_vol_sharpe']:.4f}")
print(f"high_vol_pct={pooled_result_all['high_vol_pct']:.3f}  low_vol_pct={pooled_result_all['low_vol_pct']:.3f}")
print(f"unconditional_pooled_sharpe_all10: {unconditional_all:.4f}")
