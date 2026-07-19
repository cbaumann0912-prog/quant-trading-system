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
from src.evaluation.significance import permutation_test_interaction_coefficient

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
START = "2011-01-01"
END = "2026-05-01"

MOMENTUM_LOOKBACK = 78
PRICE_Z_LOOKBACK = 26
FORWARD_HORIZON = 26 

REGIME_WINDOW = 78
ALT_REGIME_WINDOW = 156
TURBULENT_THRESHOLD = 1.5
CALM_THRESHOLD = 1.0

TRAIN_YEARS = 5
TEST_MONTHS = 12
EMBARGO_DAYS = 5

N_PERMUTATIONS = 1000
PERMUTATION_SEED = 42

CONDITION_NUMBER_THRESHOLD = 1e10
VIF_THRESHOLD = 10.0
ALPHA = 0.05

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
    .run()/signal_fn machinery is not invoked here)."""
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


def _regime_dummies(vol: pd.Series, rate_diff: pd.Series, windows: list[dict]) -> tuple[pd.Series, pd.Series]:
    composite_z, _diagnostics = compute_composite_regime_score_walkforward(vol, rate_diff, windows)
    regime = classify_regime(composite_z, turbulent_threshold=TURBULENT_THRESHOLD, calm_threshold=CALM_THRESHOLD)
    turbulent_dummy = (regime == "turbulent").astype(float)
    calm_dummy = (regime == "calm").astype(float)
    return turbulent_dummy, calm_dummy


def _aligned_arrays(y: pd.Series, x1: pd.Series, x2: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_a, x1_a = y.align(x1, join="inner")
    y_a, x2_a = y_a.align(x2, join="inner")
    x1_a = x1_a.reindex(y_a.index)
    x2_a = x2_a.reindex(y_a.index)
    valid = y_a.notna() & x1_a.notna() & x2_a.notna()
    return y_a[valid].to_numpy(), x1_a[valid].to_numpy(), x2_a[valid].to_numpy()


def _to_series(arr: np.ndarray) -> pd.Series:
    return pd.Series(arr, index=pd.RangeIndex(len(arr)))


def _verdict(primary: dict, robustness1: dict, robustness2: dict, leg_name: str) -> dict:
    b3_primary = primary["coefficients"]["interaction"]
    b3_robust1 = robustness1["coefficients"]["interaction"]

    gate_ok = (
        primary["reliability_gate_passed"]
        and robustness1["reliability_gate_passed"]
    )
    primary_significant = primary["p_values"]["interaction"] < ALPHA
    robust1_significant = robustness1["p_values"]["interaction"] < ALPHA
    robust1_same_sign = np.sign(b3_robust1) == np.sign(b3_primary)
    robust2_significant = robustness2["p_value"] < ALPHA

    leg_pass = bool(
        gate_ok
        and primary_significant
        and robust1_significant
        and robust1_same_sign
        and robust2_significant
    )

    return {
        "leg": leg_name,
        "gate_ok": bool(gate_ok),
        "primary_b3": float(b3_primary),
        "primary_p": float(primary["p_values"]["interaction"]),
        "robustness1_b3": float(b3_robust1),
        "robustness1_p": float(robustness1["p_values"]["interaction"]),
        "robustness1_same_sign": bool(robust1_same_sign),
        "robustness2_p": float(robustness2["p_value"]),
        "leg_pass": leg_pass,
    }


def main() -> None:
    momentum_y, momentum_x1, momentum_x2 = [], [], []
    momentum_y_alt, momentum_x1_alt, momentum_x2_alt = [], [], []
    reversion_y, reversion_x1, reversion_x2 = [], [], []
    reversion_y_alt, reversion_x1_alt, reversion_x2_alt = [], [], []

    for pair in PAIRS:
        loader = DataLoader(pairs=[pair], start=START, end=END, data_dir=str(DATA_DIR))
        prices = loader.load()[pair]

        log_returns = np.log(prices / prices.shift(1))
        vol = log_returns.rolling(REGIME_WINDOW).std()
        vol_alt = log_returns.rolling(ALT_REGIME_WINDOW).std()

        rate_diff_monthly = load_rate_diff(pair)
        rate_diff = rate_diff_monthly.reindex(
            pd.date_range(prices.index.min(), prices.index.max(), freq="D")
        ).ffill()

        windows = _fit_windows(prices)

        turbulent_dummy, calm_dummy = _regime_dummies(vol, rate_diff, windows)
        turbulent_dummy_alt, calm_dummy_alt = _regime_dummies(vol_alt, rate_diff, windows)

        mom_signal = momentum_signal(prices.to_frame(name="price"), lookback=MOMENTUM_LOOKBACK)
        price_z = price_zscore_signal(prices.to_frame(name="price"), lookback=PRICE_Z_LOOKBACK)
        forward_return = np.log(prices.shift(-FORWARD_HORIZON) / prices)

        y, x1, x2 = _aligned_arrays(forward_return, mom_signal, turbulent_dummy)
        momentum_y.append(y); momentum_x1.append(x1); momentum_x2.append(x2)
        y, x1, x2 = _aligned_arrays(forward_return, mom_signal, turbulent_dummy_alt)
        momentum_y_alt.append(y); momentum_x1_alt.append(x1); momentum_x2_alt.append(x2)

        y, x1, x2 = _aligned_arrays(forward_return, price_z, calm_dummy)
        reversion_y.append(y); reversion_x1.append(x1); reversion_x2.append(x2)
        y, x1, x2 = _aligned_arrays(forward_return, price_z, calm_dummy_alt)
        reversion_y_alt.append(y); reversion_x1_alt.append(x1); reversion_x2_alt.append(x2)

    momentum_y = _to_series(np.concatenate(momentum_y))
    momentum_x1 = _to_series(np.concatenate(momentum_x1))
    momentum_x2 = _to_series(np.concatenate(momentum_x2))
    momentum_y_alt = _to_series(np.concatenate(momentum_y_alt))
    momentum_x1_alt = _to_series(np.concatenate(momentum_x1_alt))
    momentum_x2_alt = _to_series(np.concatenate(momentum_x2_alt))

    reversion_y = _to_series(np.concatenate(reversion_y))
    reversion_x1 = _to_series(np.concatenate(reversion_x1))
    reversion_x2 = _to_series(np.concatenate(reversion_x2))
    reversion_y_alt = _to_series(np.concatenate(reversion_y_alt))
    reversion_x1_alt = _to_series(np.concatenate(reversion_x1_alt))
    reversion_x2_alt = _to_series(np.concatenate(reversion_x2_alt))

    print()
    print("=== Momentum leg ===")
    momentum_primary = interaction_regression_centered(
        momentum_y, momentum_x1, momentum_x2, x1_label="momentum_signal", x2_label="turbulent_dummy"
    )
    print("Primary (78-day regime window):")
    print(f"  coefficients={momentum_primary['coefficients']}")
    print(f"  p_values={momentum_primary['p_values']}")
    print(f"  condition_number={momentum_primary['condition_number']:.4e}  vif={momentum_primary['vif']}")
    print(f"  n_obs={momentum_primary['n_obs']}  reliability_gate_passed={momentum_primary['reliability_gate_passed']}")

    momentum_robust1 = interaction_regression_centered(
        momentum_y_alt, momentum_x1_alt, momentum_x2_alt, x1_label="momentum_signal", x2_label="turbulent_dummy"
    )
    print("Robustness 1 (156-day alternate regime window):")
    print(f"  interaction coef={momentum_robust1['coefficients']['interaction']:.4f}  "
          f"p={momentum_robust1['p_values']['interaction']:.4f}  "
          f"condition_number={momentum_robust1['condition_number']:.4e}  "
          f"reliability_gate_passed={momentum_robust1['reliability_gate_passed']}")

    momentum_robust2 = permutation_test_interaction_coefficient(
        momentum_y, momentum_x1, momentum_x2, n_permutations=N_PERMUTATIONS, seed=PERMUTATION_SEED
    )
    print(f"Robustness 2 ({N_PERMUTATIONS}-permutation regime-dummy shuffle): "
          f"observed_b3={momentum_robust2['observed_b3']:.4f}  p={momentum_robust2['p_value']:.4f}")

    momentum_verdict = _verdict(momentum_primary, momentum_robust1, momentum_robust2, "momentum")

    print()
    print("=== Reversion leg ===")
    reversion_primary = interaction_regression_centered(
        reversion_y, reversion_x1, reversion_x2, x1_label="price_z", x2_label="calm_dummy"
    )
    print("Primary (78-day regime window):")
    print(f"  coefficients={reversion_primary['coefficients']}")
    print(f"  p_values={reversion_primary['p_values']}")
    print(f"  condition_number={reversion_primary['condition_number']:.4e}  vif={reversion_primary['vif']}")
    print(f"  n_obs={reversion_primary['n_obs']}  reliability_gate_passed={reversion_primary['reliability_gate_passed']}")

    reversion_robust1 = interaction_regression_centered(
        reversion_y_alt, reversion_x1_alt, reversion_x2_alt, x1_label="price_z", x2_label="calm_dummy"
    )
    print("Robustness 1 (156-day alternate regime window):")
    print(f"  interaction coef={reversion_robust1['coefficients']['interaction']:.4f}  "
          f"p={reversion_robust1['p_values']['interaction']:.4f}  "
          f"condition_number={reversion_robust1['condition_number']:.4e}  "
          f"reliability_gate_passed={reversion_robust1['reliability_gate_passed']}")

    reversion_robust2 = permutation_test_interaction_coefficient(
        reversion_y, reversion_x1, reversion_x2, n_permutations=N_PERMUTATIONS, seed=PERMUTATION_SEED
    )
    print(f"Robustness 2 ({N_PERMUTATIONS}-permutation regime-dummy shuffle): "
          f"observed_b3={reversion_robust2['observed_b3']:.4f}  p={reversion_robust2['p_value']:.4f}")

    reversion_verdict = _verdict(reversion_primary, reversion_robust1, reversion_robust2, "reversion")

    print()
    print("=== Section 10 verdict ===")
    for v in (momentum_verdict, reversion_verdict):
        print(f"{v['leg']}: gate_ok={v['gate_ok']}  primary_p={v['primary_p']:.4f}  "
              f"robustness1_p={v['robustness1_p']:.4f} (same_sign={v['robustness1_same_sign']})  "
              f"robustness2_p={v['robustness2_p']:.4f}  -> leg_pass={v['leg_pass']}")

    strategy_pass = momentum_verdict["leg_pass"] and reversion_verdict["leg_pass"]
    print(f"\nStrategy-level PASS (both legs required): {strategy_pass}")

if __name__ == "__main__":
    main()
