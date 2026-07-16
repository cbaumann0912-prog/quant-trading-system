import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.framework.data_loader import DataLoader
from src.signals.signal_builder import SignalBuilder
from src.signals.momentum import momentum_signal

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"
IMAGE_PATH = REPO_ROOT / "research" / "audit_images" / "day44_rolling_ic.png"
IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
START = "2011-01-01"
END = "2026-05-01"
LOOKBACK = 78
HOLDING_PERIOD = 26
ROLLING_WINDOW = 60

rolling_ic_by_pair = {}

for pair in PAIRS:
    loader = DataLoader(pairs=[pair], start=START, end=END, data_dir=str(DATA_DIR))
    prices = loader.load()[pair]
    data = prices.to_frame(name="price")

    builder = SignalBuilder(
        signal_fn=momentum_signal,
        data=data,
        price_col="price",
        lookback=LOOKBACK,
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
    n_negative = int((rolling_ic < 0).sum())
    fwd_autocorr = forward_returns.autocorr(lag=1)
    cutoff = data.index[len(data) // 2]
    causal_ok = builder.validate_no_lookahead(cutoff)

    rolling_ic_by_pair[pair] = rolling_ic

    print(f"--- {pair} ---")
    print(f"n_obs={len(data)}  range={data.index.min().date()}..{data.index.max().date()}")
    print(f"signal: n_non_nan={int(signal.notna().sum())}")
    print(f"forward_returns: n_non_nan={int(forward_returns.notna().sum())}  lag1_autocorr={fwd_autocorr:.4f}")
    print(f"pooled_ic(spearman)={ic:.4f}")
    print(
        f"rolling_ic(window={ROLLING_WINDOW}): n_valid={len(rolling_ic)}/{n_possible_windows} possible "
        f"(n_degenerate_skipped={n_degenerate})  mean={rolling_ic.mean():.4f}  std={rolling_ic.std():.4f}  "
        f"n_negative={n_negative}/{len(rolling_ic)}"
    )
    print(f"validate_no_lookahead(cutoff={cutoff.date()})={causal_ok}")
    print()

fig, ax = plt.subplots(figsize=(10, 5))
for pair, series in rolling_ic_by_pair.items():
    ax.plot(series.index, series.values, marker="o", markersize=3, label=pair)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title(
    f"Rolling Spearman IC, momentum signal (lookback={LOOKBACK}, "
    f"holding_period={HOLDING_PERIOD}, window={ROLLING_WINDOW}, "
    f"degenerate windows dropped)"
)
ax.set_xlabel("Window start")
ax.set_ylabel("IC")
ax.legend()
fig.tight_layout()
fig.savefig(IMAGE_PATH, dpi=120)
plt.close(fig)
print(f"chart saved to {IMAGE_PATH}")
