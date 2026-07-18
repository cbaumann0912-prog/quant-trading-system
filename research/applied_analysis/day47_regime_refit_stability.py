import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.framework.data_loader import DataLoader
from src.framework.walk_forward import WalkForwardValidator
from src.features.pca import pca
from src.features.regime_classifier import classify_regime, compute_composite_regime_score
from src.signals.regime_refit import compute_composite_regime_score_walkforward

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
START = "2011-01-01"
END = "2026-05-01"

REGIME_WINDOW = 78
TURBULENT_THRESHOLD = 1.5
CALM_THRESHOLD = 1.0

TRAIN_YEARS = 5
TEST_MONTHS = 12
EMBARGO_DAYS = 5

PUBLICATION_LAG_MONTHS = 2
_RATE_FILES = {"EURUSD": ("ea", "us"), "GBPUSD": ("uk", "us"), "USDJPY": ("us", "jp")}


def load_rate_diff(pair: str) -> pd.Series:
    a, b = _RATE_FILES[pair]
    a_series = pd.read_csv(DATA_DIR / f"{a}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
    b_series = pd.read_csv(DATA_DIR / f"{b}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
    diff_monthly = (a_series - b_series).dropna()
    return diff_monthly.shift(PUBLICATION_LAG_MONTHS)


def _fit_windows(prices: pd.Series) -> list[dict]:
    """Largest n_windows that fits [START, END] at TRAIN_YEARS/TEST_MONTHS,
    reusing WalkForwardValidator purely for boundary generation (its
    .run()/signal_fn machinery is not invoked here -- Day 49's job)."""
    dummy_signal_fn = lambda data, lookback: pd.Series(np.nan, index=data.index)
    for n_windows in range(12, 0, -1):
        validator = WalkForwardValidator(
            signal_fn=dummy_signal_fn,
            data=prices.to_frame(name="price"),
            n_windows=n_windows,
            train_years=TRAIN_YEARS,
            test_months=TEST_MONTHS,
            embargo_days=EMBARGO_DAYS,
        )
        try:
            return validator.generate_windows()
        except ValueError:
            continue
    raise RuntimeError("No n_windows in [1, 12] fits the available date range.")


for pair in PAIRS:
    loader = DataLoader(pairs=[pair], start=START, end=END, data_dir=str(DATA_DIR))
    prices = loader.load()[pair]

    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.rolling(REGIME_WINDOW).std()

    rate_diff_monthly = load_rate_diff(pair)
    rate_diff = rate_diff_monthly.reindex(
        pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    ).ffill()

    windows = _fit_windows(prices)

    baseline_composite = compute_composite_regime_score(vol, rate_diff)
    baseline_regime = classify_regime(
        baseline_composite, turbulent_threshold=TURBULENT_THRESHOLD, calm_threshold=CALM_THRESHOLD
    )
    combined = pd.concat([vol.rename("vol"), rate_diff.rename("rate_diff")], axis=1, sort=False).dropna()
    z_full = (combined - combined.mean()) / combined.std()
    baseline_components, baseline_explained_var, _ = pca(z_full.to_numpy(), n_components=1)
    baseline_pc1 = baseline_components[:, 0]
    if baseline_pc1[0] < 0:
        baseline_pc1 = -baseline_pc1
    baseline_rho = float(np.corrcoef(combined["vol"], combined["rate_diff"])[0, 1])

    wf_composite, diagnostics = compute_composite_regime_score_walkforward(vol, rate_diff, windows)
    wf_regime = classify_regime(
        wf_composite, turbulent_threshold=TURBULENT_THRESHOLD, calm_threshold=CALM_THRESHOLD
    )

    print(f"=== {pair} ===")
    print(f"n_windows={len(windows)}  train_years={TRAIN_YEARS}  test_months={TEST_MONTHS}")
    print(
        f"Day 43 full-sample baseline: rho={baseline_rho:+.4f}  "
        f"explained_var={float(baseline_explained_var[0]):.4f}  "
        f"mean=(vol={combined['vol'].mean():.6f}, rate_diff={combined['rate_diff'].mean():.4f})  "
        f"std=(vol={combined['vol'].std():.6f}, rate_diff={combined['rate_diff'].std():.4f})"
    )

    print()
    print("--- per-window refit vs. full-sample baseline ---")
    print(
        f"{'win':>3} {'train_start':>11} {'train_end':>11} {'rho':>8} "
        f"{'exp_var':>8} {'mean_vol':>10} {'std_vol':>10}  {'sign_flip':>9}"
    )
    prior_rho = None
    for _, row in diagnostics.iterrows():
        sign_flip = prior_rho is not None and np.sign(row["rho"]) != np.sign(prior_rho)
        prior_rho = row["rho"]
        print(
            f"{int(row['window_idx']):>3} {row['train_start'].date()!s:>11} "
            f"{row['train_end'].date()!s:>11} {row['rho']:>+8.4f} "
            f"{row['explained_variance_ratio']:>8.4f} "
            f"{row['mean_vol']:>10.6f} {row['std_vol']:>10.6f}  {str(sign_flip):>9}"
        )

    n_sign_flips = int((np.sign(diagnostics["rho"]).diff().fillna(0) != 0).sum())
    print(f"\nrho sign flips across consecutive windows: {n_sign_flips} of {len(diagnostics) - 1} transitions")

    aligned_baseline = baseline_regime.reindex(wf_regime.index)
    comparable = aligned_baseline.notna()
    agree = (wf_regime[comparable] == aligned_baseline[comparable])
    pct_agree = 100 * agree.mean()
    n_compared = int(comparable.sum())
    n_disagree = int((~agree).sum())

    print(
        f"\nregime label agreement (walk-forward refit vs. full-sample fit), "
        f"n={n_compared} out-of-sample bars: {pct_agree:.2f}% agree, "
        f"{n_disagree} bars flip label"
    )
    wf_counts = wf_regime.value_counts(normalize=True) * 100
    print(
        f"walk-forward regime mix (OOS only): turbulent={wf_counts.get('turbulent', 0.0):.1f}%  "
        f"calm={wf_counts.get('calm', 0.0):.1f}%  deadzone={wf_counts.get('deadzone', 0.0):.1f}%  "
        f"(Day 43 baseline for reference: turbulent 9.2-15.9%, deadzone 13.6-24.6%)"
    )
    print()
