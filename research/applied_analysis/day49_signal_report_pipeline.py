import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.framework.data_loader import DataLoader
from src.framework.walk_forward import WalkForwardValidator
from src.features.regime_classifier import classify_regime
from src.signals.regime_refit import compute_composite_regime_score_walkforward
from src.signals.momentum import momentum_signal
from src.signals.mean_reversion import price_zscore_signal
from src.stats.regression import interaction_regression_centered
from src.analysis.signal_report import build_signal_report

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
START = "2011-01-01"
END = "2026-05-01"

MOMENTUM_LOOKBACK = 78
PRICE_Z_LOOKBACK = 26
FORWARD_HORIZON = 26
REVERSION_ENTRY_Z = 2.0
TRADING_DAYS_PER_YEAR = 312

REGIME_WINDOW = 78
TURBULENT_THRESHOLD = 1.5
CALM_THRESHOLD = 1.0

TRAIN_YEARS = 5
TEST_MONTHS = 12
EMBARGO_DAYS = 5

PUBLICATION_LAG_MONTHS = 2
_RATE_FILES = {"EURUSD": ("ea", "us"), "GBPUSD": ("uk", "us"), "USDJPY": ("us", "jp")}

STRATEGY_NAME = "Volatility Regime Breakout/Mean-Reversion"
N_TRIALS = 4


def load_rate_diff(pair: str) -> pd.Series:
    a, b = _RATE_FILES[pair]
    a_series = pd.read_csv(DATA_DIR / f"{a}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
    b_series = pd.read_csv(DATA_DIR / f"{b}_3m_interbank.csv", parse_dates=["date"]).set_index("date")["value"]
    diff_monthly = (a_series - b_series).dropna()
    return diff_monthly.shift(PUBLICATION_LAG_MONTHS)


def fit_windows(prices: pd.Series) -> list[dict]:
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


def spearman_ic(signal: pd.Series, forward_returns: pd.Series) -> float:
    aligned = pd.concat([signal.rename("s"), forward_returns.rename("f")], axis=1, join="inner").dropna()
    if len(aligned) < 2 or aligned["s"].nunique() < 2 or aligned["f"].nunique() < 2:
        return float("nan")
    return aligned["s"].corr(aligned["f"], method="spearman")


def window_sharpe(exposure: pd.Series, daily_log_return: pd.Series) -> float:
    pnl = exposure.shift(1) * daily_log_return
    pnl = pnl.dropna()
    if len(pnl) < 2 or pnl.std() == 0:
        return float("nan")
    return (pnl.mean() / pnl.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def regime_gated_pnl(exposure: pd.Series, daily_log_return: pd.Series, active_mask: pd.Series) -> pd.Series:

    gated_exposure = exposure.where(active_mask.reindex(exposure.index).fillna(0).astype(bool))
    return gated_exposure.shift(1) * daily_log_return


def main() -> None:
    momentum_ic_by_window: list[float] = []
    reversion_ic_by_window: list[float] = []
    momentum_sharpe_by_window: list[float] = []
    reversion_sharpe_by_window: list[float] = []
    momentum_pnl_chunks: list[pd.Series] = []
    reversion_pnl_chunks: list[pd.Series] = []
    momentum_y, momentum_x1, momentum_x2 = [], [], []
    reversion_y, reversion_x1, reversion_x2 = [], [], []

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
        calm_dummy = (regime == "calm").astype(float)

        momentum = momentum_signal(data, MOMENTUM_LOOKBACK)
        price_z = price_zscore_signal(data, PRICE_Z_LOOKBACK)

        reversion_exposure = pd.Series(0.0, index=price_z.index)
        reversion_exposure[price_z > REVERSION_ENTRY_Z] = -1.0
        reversion_exposure[price_z < -REVERSION_ENTRY_Z] = 1.0
        reversion_exposure = reversion_exposure.where(price_z.notna())

        forward_return = np.log(prices.shift(-FORWARD_HORIZON) / prices)

        print(f"--- {pair} ---")
        for w in windows:
            test_mask = (data.index >= w["test_start"]) & (data.index < w["test_end"])
            test_idx = data.index[test_mask]

            turb_active = turbulent_dummy.reindex(test_idx).fillna(0).astype(bool)
            calm_active = calm_dummy.reindex(test_idx).fillna(0).astype(bool)

            m_ic = spearman_ic(momentum.loc[test_idx].where(turb_active), forward_return.loc[test_idx])
            r_ic = spearman_ic(price_z.loc[test_idx].where(calm_active), forward_return.loc[test_idx])
            m_sharpe = window_sharpe(momentum.loc[test_idx].where(turb_active), log_returns.loc[test_idx])
            r_sharpe = window_sharpe(reversion_exposure.loc[test_idx].where(calm_active), log_returns.loc[test_idx])

            momentum_ic_by_window.append(m_ic)
            reversion_ic_by_window.append(r_ic)
            momentum_sharpe_by_window.append(m_sharpe)
            reversion_sharpe_by_window.append(r_sharpe)

            print(
                f"  window test=[{w['test_start'].date()}, {w['test_end'].date()}) "
                f"n_test={test_mask.sum()}  momentum(IC={m_ic:.4f}, Sharpe={m_sharpe:.3f})  "
                f"reversion(IC={r_ic:.4f}, Sharpe={r_sharpe:.3f})"
            )

        momentum_pnl_chunks.append(regime_gated_pnl(momentum, log_returns, turbulent_dummy))
        reversion_pnl_chunks.append(regime_gated_pnl(reversion_exposure, log_returns, calm_dummy))

        aligned = pd.concat(
            [forward_return.rename("y"), momentum.rename("x1"), turbulent_dummy.rename("x2")], axis=1, join="inner"
        ).dropna()
        momentum_y.append(aligned["y"].to_numpy())
        momentum_x1.append(aligned["x1"].to_numpy())
        momentum_x2.append(aligned["x2"].to_numpy())

        aligned = pd.concat(
            [forward_return.rename("y"), price_z.rename("x1"), calm_dummy.rename("x2")], axis=1, join="inner"
        ).dropna()
        reversion_y.append(aligned["y"].to_numpy())
        reversion_x1.append(aligned["x1"].to_numpy())
        reversion_x2.append(aligned["x2"].to_numpy())

    def to_series(chunks):
        arr = np.concatenate(chunks)
        return pd.Series(arr, index=pd.RangeIndex(len(arr)))

    momentum_primary = interaction_regression_centered(
        to_series(momentum_y), to_series(momentum_x1), to_series(momentum_x2),
        x1_label="momentum_signal", x2_label="turbulent_dummy",
    )
    reversion_primary = interaction_regression_centered(
        to_series(reversion_y), to_series(reversion_x1), to_series(reversion_x2),
        x1_label="price_z", x2_label="calm_dummy",
    )

    momentum_pnl = pd.concat(momentum_pnl_chunks).dropna()
    reversion_pnl = pd.concat(reversion_pnl_chunks).dropna()

    report = build_signal_report(
        strategy_name=STRATEGY_NAME,
        leg_ic_by_window={"momentum": momentum_ic_by_window, "reversion": reversion_ic_by_window},
        leg_sharpe_by_window={"momentum": momentum_sharpe_by_window, "reversion": reversion_sharpe_by_window},
        leg_primary_p_value={
            "momentum": momentum_primary["p_values"]["interaction"],
            "reversion": reversion_primary["p_values"]["interaction"],
        },
        leg_regime_gated_returns={"momentum": momentum_pnl, "reversion": reversion_pnl},
        n_trials=N_TRIALS,
    )

    print()
    print(report.to_markdown())

    out_path = REPO_ROOT / "research" / "daily_audit" / "day49_signalreport_design.md"
    out_path.write_text(report.to_markdown())
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
