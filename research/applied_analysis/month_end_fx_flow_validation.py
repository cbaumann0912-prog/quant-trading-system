"""H1 primary validation for research/strategies/month_end_fx_flow.md.

Stages 1-minute bars from raw data on every run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.bootstrap import block_bootstrap
from src.features.sessions import FILE_UTC_OFFSET_HOURS
from src.stats.regression import compute_vif

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
]
START = "2011-01-01"
END = "2023-12-31"
LONDON = "Europe/London"

FIX_WINDOW = (15 * 60 + 30, 16 * 60 + 15)
FIX_WINDOW_NARROW = (15 * 60 + 45, 16 * 60 + 5)
CONTROL_WINDOW = (10 * 60, 10 * 60 + 45)
POST_FIX_WINDOW = (16 * 60 + 15, 17 * 60)

MONTH_END_DAYS = 2
BREAK_DATE = pd.Timestamp("2015-02-15")
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 2000
BOOTSTRAP_BLOCK_DAYS = 21
SEED = 42
ALPHA = 0.05
CONDITION_NUMBER_THRESHOLD = 1e10
VIF_THRESHOLD = 10.0

TERMS = ["signal", "month_end", "fix",
         "signal:month_end", "signal:fix", "month_end:fix",
         "signal:month_end:fix"]
KEY = "signal:month_end:fix"

staged_pairs = {}
for pair in PAIRS:
    print(f"staging {pair} ...", flush=True)
    raw = pd.read_csv(DATA_DIR / f"{pair}.csv", usecols=["Datetime", "Close"])
    parsed = pd.to_datetime(raw["Datetime"], format="%Y%m%d %H%M%S")
    london = pd.DatetimeIndex(
        (parsed + pd.Timedelta(hours=FILE_UTC_OFFSET_HOURS)).dt.tz_localize("UTC")
    ).tz_convert(LONDON)

    bars = pd.DataFrame({
        "close": raw["Close"].to_numpy(),
        "london_date": london.normalize().tz_localize(None),
        "minute_of_day": london.hour.values * 60 + london.minute.values,
    })

    staged = {}
    for name, (entry, exit_) in [
        ("fix_return", FIX_WINDOW),
        ("fix_return_narrow", FIX_WINDOW_NARROW),
        ("control_return", CONTROL_WINDOW),
        ("post_fix_return", POST_FIX_WINDOW),
    ]:
        sel = bars.loc[(bars["minute_of_day"] >= entry) & (bars["minute_of_day"] <= exit_)]
        grouped = sel.groupby("london_date")["close"]
        staged[name] = np.log(grouped.last() / grouped.first())

    out = pd.DataFrame(staged).sort_index()
    daily_close = bars.groupby("london_date")["close"].last()
    out["daily_log_return"] = np.log(daily_close / daily_close.shift(1))
    out = out.loc[(out.index >= pd.Timestamp(START)) & (out.index <= pd.Timestamp(END))]
    out.index.name = "date"
    staged_pairs[pair] = out

print("\nH1 primary: month-end x fix-window interaction")
print(f"Universe: {len(PAIRS)} pairs   Sample: {START} to {END} (lockbox sealed)")
print("Prediction: b4 > 0. Significance AND sign both required (spec criterion 2).")

panels = {}
for fix_col in ("fix_return", "fix_return_narrow"):
    frames = []
    for pair in PAIRS:
        df = staged_pairs[pair]
        period = df.index.to_period("M")

        month_to_date = df["daily_log_return"].shift(1).groupby(period).cumsum()
        signal = -np.sign(month_to_date)
        month_end = (df.groupby(period).cumcount(ascending=False) < MONTH_END_DAYS).astype(float)

        for col, is_fix in [(fix_col, 1.0), ("control_return", 0.0)]:
            frames.append(pd.DataFrame({
                "date": df.index,
                "y": df[col].to_numpy(),
                "signal": signal.to_numpy(),
                "month_end": month_end.to_numpy(),
                "fix": is_fix,
            }))

    built = pd.concat(frames, ignore_index=True).dropna(subset=["y", "signal", "month_end"])
    built = built.loc[built["signal"] != 0.0]
    panels[fix_col] = built.sort_values("date").reset_index(drop=True)

panel = panels["fix_return"]
panel_narrow = panels["fix_return_narrow"]


def b4_core(s, m, f, y):
    s = s - s.mean()
    m = m - m.mean()
    f = f - f.mean()
    X = np.column_stack([np.ones(len(y)), s, m, f, s * m, s * f, m * f, s * m * f])
    return float((np.linalg.pinv(X.T @ X) @ X.T @ y)[-1])


def fit(p):
    s = p["signal"].to_numpy() - p["signal"].mean()
    m = p["month_end"].to_numpy() - p["month_end"].mean()
    f = p["fix"].to_numpy() - p["fix"].mean()
    y = p["y"].to_numpy()
    X = np.column_stack([np.ones(len(y)), s, m, f, s * m, s * f, m * f, s * m * f])

    n, k = X.shape
    xtx = X.T @ X
    condition_number = float(np.linalg.cond(xtx))
    beta = np.linalg.pinv(xtx) @ X.T @ y
    residuals = y - X @ beta
    se = np.sqrt(np.diag(float(residuals @ residuals / (n - k)) * np.linalg.pinv(xtx)))
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
    vif = compute_vif(X[:, 1:], labels=TERMS)
    labels = ["intercept"] + TERMS
    return {
        "coefficients": dict(zip(labels, beta)),
        "std_errors": dict(zip(labels, se)),
        "t_stats": dict(zip(labels, t_stats)),
        "p_values": dict(zip(labels, p_values)),
        "n_obs": n,
        "condition_number": condition_number,
        "vif": vif,
        "reliability_gate_passed": bool(
            condition_number < CONDITION_NUMBER_THRESHOLD
            and all(v < VIF_THRESHOLD for v in vif.values())
        ),
    }


def report(name, result):
    print(f"\n--- {name} ---")
    print(f"  b4 ({KEY}) = {result['coefficients'][KEY]:+.6e}   "
          f"p = {result['p_values'][KEY]:.5f}   t = {result['t_stats'][KEY]:+.3f}")
    print(f"  n_obs = {result['n_obs']}   condition_number = {result['condition_number']:.4e}")
    print(f"  max VIF = {max(result['vif'].values()):.3f}   gate = {result['reliability_gate_passed']}")


primary = fit(panel)
report("Primary (fix window 15:30-16:15)", primary)

signal_arr = panel["signal"].to_numpy()
month_end_arr = panel["month_end"].to_numpy()
fix_arr = panel["fix"].to_numpy()
y_arr = panel["y"].to_numpy()

dates, date_code = np.unique(panel["date"].to_numpy(), return_inverse=True)
order = np.argsort(date_code, kind="stable")
starts = np.searchsorted(date_code[order], np.arange(len(dates)))
ends = np.searchsorted(date_code[order], np.arange(len(dates)), side="right")
rows_by_date = [order[starts[i]:ends[i]] for i in range(len(dates))]

boot = block_bootstrap(
    series=np.arange(len(dates)),
    block_size=BOOTSTRAP_BLOCK_DAYS,
    n_samples=N_BOOTSTRAP,
    statistic_fn=lambda idx: (
        lambda r: b4_core(signal_arr[r], month_end_arr[r], fix_arr[r], y_arr[r])
    )(np.concatenate([rows_by_date[int(i)] for i in idx])),
    seed=SEED,
)
lo, hi = np.percentile(boot, [2.5, 97.5])
boot_p = float(2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))

print(f"\n  Block bootstrap ({N_BOOTSTRAP} draws, {BOOTSTRAP_BLOCK_DAYS}-day date blocks):")
print(f"    95% CI = [{lo:+.6e}, {hi:+.6e}]   two-sided p = {boot_p:.5f}")
print(f"    bootstrap SE = {boot.std(ddof=1):.6e}  vs  OLS SE = {primary['std_errors'][KEY]:.6e}")

narrow = fit(panel_narrow)
report("Robustness 1: narrow fix window 15:45-16:05", narrow)

pre_fit = fit(panel.loc[panel["date"] < BREAK_DATE])
post_fit = fit(panel.loc[panel["date"] >= BREAK_DATE])
report(f"Robustness 2a: pre-reform (< {BREAK_DATE.date()})", pre_fit)
report(f"Robustness 2b: post-reform (>= {BREAK_DATE.date()})", post_fit)

observed = b4_core(signal_arr, month_end_arr, fix_arr, y_arr)
date_flag = np.array([month_end_arr[rows_by_date[i]][0] for i in range(len(dates))])
rng = np.random.default_rng(SEED)
null = np.empty(N_PERMUTATIONS)
for i in range(N_PERMUTATIONS):
    null[i] = b4_core(signal_arr, rng.permutation(date_flag)[date_code], fix_arr, y_arr)
perm_p = float(np.mean(np.abs(null) >= abs(observed)))

print(f"\n--- Robustness 3: {N_PERMUTATIONS}-permutation month_end shuffle ---")
print(f"  observed b4 = {observed:+.6e}   empirical p = {perm_p:.5f}")

b4_primary = primary["coefficients"][KEY]
checks = {
    "gate (primary)": primary["reliability_gate_passed"],
    "gate (narrow)": narrow["reliability_gate_passed"],
    "b4 > 0": b4_primary > 0,
    "bootstrap p < 0.05": boot_p < ALPHA,
    "narrow p < 0.05, same sign": (
        narrow["p_values"][KEY] < ALPHA
        and np.sign(narrow["coefficients"][KEY]) == np.sign(b4_primary)
    ),
    "break: same sign both halves": (
        np.sign(pre_fit["coefficients"][KEY]) == np.sign(post_fit["coefficients"][KEY])
    ),
    "break: post-reform p < 0.05": post_fit["p_values"][KEY] < ALPHA,
    "permutation p < 0.05": perm_p < ALPHA,
}

print("\n=== H1 verdict ===")
for label, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {label}")
print(f"\nH1 {'PASS' if all(checks.values()) else 'FAIL'}")
