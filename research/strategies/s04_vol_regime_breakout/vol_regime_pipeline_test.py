import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.framework.data_loader import DataLoader
from src.signals.signal_builder import SignalBuilder
from src.signals.regime_gated import make_regime_gated_signal_fn
from src.features.regime_classifier import compute_composite_regime_score, classify_regime

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
START = "2011-01-01"
END = "2023-12-31"

MOMENTUM_LOOKBACK = 78
REVERSION_LOOKBACK = 26 
HOLDING_PERIOD = 26
TURBULENT_THRESHOLD = 1.5
CALM_THRESHOLD = 1.0
ENTRY_THRESHOLDS = (2.0, 2.5, 3.0) 
EXIT_Z = 0.5                       
TIME_STOP = 26                    
ROLLING_WINDOW = 60               

PUBLICATION_LAG_MONTHS = 2

_RATE_FILES = {"EURUSD": ("ea", "us"), "GBPUSD": ("uk", "us"), "USDJPY": ("us", "jp")}


def load_rate_diff(pair: str) -> pd.Series:
    a, b = _RATE_FILES[pair]
    a_series = pd.read_csv(DATA_DIR / f"{a}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
    b_series = pd.read_csv(DATA_DIR / f"{b}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
    diff_monthly = (a_series - b_series).dropna()
    diff_monthly_lagged = diff_monthly.shift(PUBLICATION_LAG_MONTHS)
    return diff_monthly_lagged


for pair in PAIRS:
    loader = DataLoader(pairs=[pair], start=START, end=END, data_dir=str(DATA_DIR))
    prices = loader.load()[pair]
    data = prices.to_frame(name="price")

    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.rolling(MOMENTUM_LOOKBACK).std()

    rate_diff_monthly = load_rate_diff(pair)
    rate_diff_daily = rate_diff_monthly.reindex(
        pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    ).ffill()

    composite_z = compute_composite_regime_score(vol, rate_diff_daily)
    regime = classify_regime(
        composite_z, turbulent_threshold=TURBULENT_THRESHOLD, calm_threshold=CALM_THRESHOLD
    )

    signal_fn = make_regime_gated_signal_fn(
        regime,
        reversion_lookback=REVERSION_LOOKBACK,
        entry_thresholds=ENTRY_THRESHOLDS,
        exit_z=EXIT_Z,
        time_stop=TIME_STOP,
    )

    builder = SignalBuilder(
        signal_fn=signal_fn,
        data=data,
        price_col="price",
        lookback=MOMENTUM_LOOKBACK,
        holding_period=HOLDING_PERIOD,
    )

    signal = builder.compute(data)
    forward_returns = builder.compute_forward_returns()
    ic = builder.compute_ic(forward_returns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aligned = pd.concat(
            [signal.rename("s"), forward_returns.rename("f")], axis=1, join="inner"
        ).dropna()
        n_possible_windows = len(aligned) // ROLLING_WINDOW
        rolling_ic = builder.compute_rolling_ic(forward_returns, window=ROLLING_WINDOW)

    n_degenerate = n_possible_windows - len(rolling_ic)
    cutoff = data.index[len(data) // 2]
    causal_ok = builder.validate_no_lookahead(cutoff)

    regime_aligned = regime.reindex(data.index)
    regime_counts = regime_aligned.value_counts(normalize=True) * 100

    print(f"--- {pair} ---")
    print(f"n_obs={len(data)}  range={data.index.min().date()}..{data.index.max().date()}")
    print(
        f"regime mix: turbulent={regime_counts.get('turbulent', 0.0):.1f}%  "
        f"calm={regime_counts.get('calm', 0.0):.1f}%  "
        f"deadzone={regime_counts.get('deadzone', 0.0):.1f}%  "
        f"(Day 43 baseline: turbulent 9.2-15.9%, deadzone 13.6-24.6%)"
    )
    print(f"signal: n_non_nan={int(signal.notna().sum())}  n_zero={int((signal == 0.0).sum())}")
    print(f"pooled_ic(spearman)={ic:.4f}")
    print(
        f"rolling_ic(window={ROLLING_WINDOW}): n_valid={len(rolling_ic)}/{n_possible_windows} possible "
        f"(n_degenerate_skipped={n_degenerate})  mean={rolling_ic.mean():.4f}  std={rolling_ic.std():.4f}"
    )
    print(f"validate_no_lookahead(cutoff={cutoff.date()})={causal_ok}")
    print()
