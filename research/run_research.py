"""
Unified walk-forward research CLI.

Merged 2026-08-02 from two scripts (run_research.py for momentum/mean-
reversion, run_research_overshoot.py for intraday overshoot) into one.
Every signal writes one JSON report per pair to results/, all with the
same envelope:

    {pair, signal, parameters, sample, window_results, ic_summary,
     ic_status_counts, sharpe_summary, caveats}

Adding a signal
---------------
A signal is an "adapter": a function `(pair, data_dir, **signal_kwargs) ->
SignalRun`, registered under a name via `@register_signal("name")`. The
adapter owns everything signal-specific: how its data loads, how its
walk-forward windows are built, how a window's IC and Sharpe get scored.
It returns a `SignalRun` (parameters, sample, window_results, caveats) and
never touches JSON, file paths, or the report envelope.

Everything else -- IC/Sharpe summarization, the ic_status_counts tally,
common caveats, JSON serialization, output paths, the lockbox guard, and
`--pairs` resolution ("all" / comma-list) -- lives once in `write_report`
and the shared CLI helpers. No signal name is branched on anywhere outside
SIGNAL_ADAPTERS and the two adapter functions below.

Each signal keeps its own thin Click subcommand, since the CLI parameters
genuinely differ (lookback/holding-period vs. k/entry-delay/scan-window).
That's a real interface difference, not hardcoding.

Usage
-----
python research/run_research.py momentum --pairs EURUSD
python research/run_research.py mean-reversion --pairs EURUSD,GBPUSD --lookback 20
python research/run_research.py intraday-overshoot --pairs all
python research/run_research.py intraday-overshoot --pairs EURUSD --k 2.5 --entry-delay 15
python research/run_research.py --help          # lists all registered signals
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import click
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.performance_analyzer import PerformanceAnalyzer, information_coefficient
from src.framework.data_loader import SUPPORTED_PAIRS, DataLoader
from src.framework.walk_forward import WalkForwardValidator
from src.signals.intraday_overshoot import build_overshoot_sessions, overshoot_trades
from src.signals.mean_reversion import price_zscore_signal
from src.signals.momentum import momentum_signal

DEFAULT_DATA_DIR = REPO_ROOT.parent / "data"
LOCKBOX_START = pd.Timestamp("2024-01-01")


@dataclass
class SignalRun:
    """What an adapter returns. write_report adds pair, signal,
    ic_summary, ic_status_counts, sharpe_summary, and the common caveats,
    then handles serialization -- adapters never touch any of that."""

    parameters: dict
    sample: dict
    window_results: list[dict]
    caveats: list[str] = field(default_factory=list)


SIGNAL_ADAPTERS: dict[str, Callable[..., SignalRun]] = {}


def register_signal(name: str) -> Callable[[Callable[..., SignalRun]], Callable[..., SignalRun]]:
    """Register `fn` as the adapter for signal `name`. The only extension
    point a new signal needs: no other file, registry, or edit to the CLI
    driver or report-writing logic."""

    def deco(fn: Callable[..., SignalRun]) -> Callable[..., SignalRun]:
        SIGNAL_ADAPTERS[name] = fn
        return fn

    return deco


COMMON_CAVEATS = [
    "Raw unsized signal-level diagnostics, gross of transaction costs. Not a claim of tradable performance.",
    "No multiple-testing correction applied. This run does not increment project n_trials and is not a research verdict.",
]


def resolve_pairs(pairs: str) -> list[str]:
    """'all' -> all 10 SUPPORTED_PAIRS; otherwise a comma-separated list."""
    pair_list = sorted(SUPPORTED_PAIRS) if pairs.lower() == "all" else [p.strip().upper() for p in pairs.split(",")]
    invalid = set(pair_list) - SUPPORTED_PAIRS
    if invalid:
        raise click.BadParameter(
            f"Unsupported pair(s) {sorted(invalid)}. Must be a subset of {sorted(SUPPORTED_PAIRS)}."
        )
    return pair_list


def guard_lockbox(end: pd.Timestamp, allow_lockbox: bool) -> None:
    if end >= LOCKBOX_START and not allow_lockbox:
        raise click.BadParameter(
            f"end date {end.date()} reaches the reserved lockbox slice "
            f"(>= {LOCKBOX_START.date()}). The lockbox is held for a single "
            f"unbiased final evaluation and must not be touched during "
            f"development. Pass --allow-lockbox only if you are deliberately "
            f"spending it."
        )


def annualization_factor(index: pd.DatetimeIndex) -> float:
    """Empirical bars-per-year from the observed index span."""
    if len(index) < 2:
        return float("nan")
    years = (index.max() - index.min()).days / 365.25
    if years <= 0:
        return float("nan")
    return len(index) / years


def summarize(values: list[float]) -> dict[str, float]:
    """Mean, std, IR and positive fraction over the non-NaN subset."""
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ir": float("nan"), "frac_positive": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else float("nan")
    ir = mean / std if std and not math.isnan(std) and std != 0 else float("nan")
    return {"n": n, "mean": mean, "std": std, "ir": ir, "frac_positive": float((arr > 0).mean())}


def json_safe(obj: Any) -> Any:
    """Recursively coerce pandas/numpy scalars into strict-JSON primitives.
    NaN/inf become null, since Python's json module writes bare NaN/Infinity
    tokens by default and no strict JSON parser accepts those."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def write_report(pair: str, signal_name: str, run: SignalRun, output: Path) -> Path:
    """Shared envelope and serialization for every signal. Computes
    ic_summary, sharpe_summary, and ic_status_counts generically off
    whatever `run.window_results` contains -- every adapter's windows
    carry at least ic/ic_status/sharpe, however differently they were
    computed."""
    ic_summary = summarize([w.get("ic") for w in run.window_results])
    sharpe_summary = summarize([w.get("sharpe") for w in run.window_results])

    ic_status_counts: dict[str, int] = {}
    for w in run.window_results:
        status = w.get("ic_status", "ok")
        ic_status_counts[status] = ic_status_counts.get(status, 0) + 1

    report = {
        "pair": pair,
        "signal": signal_name,
        "parameters": run.parameters,
        "sample": run.sample,
        "window_results": run.window_results,
        "ic_summary": ic_summary,
        "ic_status_counts": ic_status_counts,
        "sharpe_summary": sharpe_summary,
        "caveats": COMMON_CAVEATS + run.caveats,
    }

    output.mkdir(parents=True, exist_ok=True)
    out_path = output / f"{pair}_{signal_name}.json"
    out_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")

    click.echo(f"{pair} / {signal_name}")
    click.echo(
        f"  IC     mean={ic_summary['mean']:.4f} std={ic_summary['std']:.4f} "
        f"n={ic_summary['n']}/{len(run.window_results)}"
    )
    unscored = {k: v for k, v in ic_status_counts.items() if k != "ok"}
    if unscored:
        click.echo(f"  IC unscored windows: {unscored}")
    click.echo(
        f"  Sharpe mean={sharpe_summary['mean']:.4f} std={sharpe_summary['std']:.4f} n={sharpe_summary['n']}"
    )
    click.echo(f"  written: {out_path}")
    return out_path


REVERSION_ENTRY_Z = 2.0


def _momentum_exposure(signal: pd.Series) -> pd.Series:
    """Momentum signal is already a unit-exposure sign in {-1, 0, 1}."""
    return signal


def _reversion_exposure(signal: pd.Series) -> pd.Series:
    """Fade the z-score once it clears +/- REVERSION_ENTRY_Z, flat otherwise."""
    exposure = pd.Series(0.0, index=signal.index)
    exposure[signal > REVERSION_ENTRY_Z] = -1.0
    exposure[signal < -REVERSION_ENTRY_Z] = 1.0
    return exposure.where(signal.notna())


SIGNAL_REGISTRY: dict[str, dict[str, Any]] = {
    "momentum": {
        "signal_fn": momentum_signal,
        "exposure_fn": _momentum_exposure,
        "default_lookback": 78,
        "expected_ic_sign": "positive",
    },
    "mean-reversion": {
        "signal_fn": price_zscore_signal,
        "exposure_fn": _reversion_exposure,
        "default_lookback": 26,
        "expected_ic_sign": "negative",
    },
}


def _daily_spearman_ic(signal: pd.Series, forward_returns: pd.Series) -> tuple[float, str]:
    """Guarded wrapper around information_coefficient: dropping incomplete
    pairs and rejecting degenerate inputs is the caller's job, since
    information_coefficient aligns but does not drop NaN."""
    aligned = pd.concat([signal.rename("signal"), forward_returns.rename("forward")], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan"), "insufficient_obs"
    if aligned["signal"].nunique() < 2:
        return float("nan"), "constant_signal"
    if aligned["forward"].nunique() < 2:
        return float("nan"), "constant_forward_returns"
    ic = information_coefficient(aligned["signal"], aligned["forward"], method="spearman")
    return ic, "ok"


def _daily_window_sharpe(exposure: pd.Series, log_returns: pd.Series) -> float:
    """Annualized Sharpe of a lag-1 executed unit-exposure series."""
    pnl = (exposure.shift(1) * log_returns).dropna()
    if len(pnl) < 2:
        return float("nan")
    sd = pnl.std()
    if sd == 0 or math.isnan(sd):
        return float("nan")
    ann = annualization_factor(pnl.index)
    if math.isnan(ann):
        return float("nan")
    return float((pnl.mean() / sd) * math.sqrt(ann))


def _truncate_forward_returns(forward_returns: pd.Series, test_index: pd.DatetimeIndex, holding_period: int) -> pd.Series:
    """Mask the final `holding_period` bars of a test window to NaN -- their
    forward return reads price beyond test_end, inside the embargo or the
    next training region."""
    sliced = forward_returns.reindex(test_index).copy()
    if holding_period > 0 and len(sliced) > 0:
        sliced.iloc[-holding_period:] = np.nan
    return sliced


def _build_daily_signal(
    signal_name: str,
    pair: str,
    data_dir: str | Path,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    windows: int,
    train_years: int,
    test_months: int,
    embargo_days: int,
    lookback: int,
    holding_period: int,
) -> SignalRun:
    spec = SIGNAL_REGISTRY[signal_name]

    loader = DataLoader(
        pairs=[pair], start=str(train_start.date()), end=str(train_end.date()),
        embargo_days=embargo_days, data_dir=str(data_dir),
    )
    prices = loader.load()[pair]
    frame = prices.to_frame(name="price")

    log_returns = np.log(prices / prices.shift(1))
    forward_returns = np.log(prices.shift(-holding_period) / prices)

    signal = spec["signal_fn"](frame, lookback)
    exposure = spec["exposure_fn"](signal)

    validator = WalkForwardValidator(
        signal_fn=spec["signal_fn"], data=frame, n_windows=windows,
        train_years=train_years, test_months=test_months, embargo_days=embargo_days,
    )
    try:
        generated = validator.generate_windows()
    except ValueError as exc:
        raise click.BadParameter(
            f"{exc} Try fewer --windows, a shorter --train-years, a shorter "
            f"--test-months, or a wider --train-start/--train-end range."
        ) from exc

    window_results = []
    for idx, w in enumerate(generated):
        mask = (signal.index >= w["test_start"]) & (signal.index < w["test_end"])
        test_index = signal.index[mask]

        window_forward = _truncate_forward_returns(forward_returns, test_index, holding_period)
        window_signal = signal.reindex(test_index)
        ic, ic_status = _daily_spearman_ic(window_signal, window_forward)
        sharpe = _daily_window_sharpe(exposure.reindex(test_index), log_returns.reindex(test_index))

        window_results.append({
            "window_idx": idx,
            "train_start": w["train_start"], "train_end": w["train_end"],
            "embargo_end": w["embargo_end"], "test_start": w["test_start"], "test_end": w["test_end"],
            "n_test": int(len(test_index)), "n_scored": int(window_forward.notna().sum()),
            "signal_nunique": int(window_signal.nunique()),
            "ic": ic, "ic_status": ic_status, "sharpe": sharpe,
        })

    return SignalRun(
        parameters={
            "train_start": train_start, "train_end": train_end,
            "n_windows": windows, "train_years": train_years, "test_months": test_months,
            "embargo_days": embargo_days, "lookback": lookback, "holding_period": holding_period,
            "reversion_entry_z": REVERSION_ENTRY_Z, "expected_ic_sign": spec["expected_ic_sign"],
        },
        sample={
            "n_daily_bars": int(len(prices)), "first_bar": prices.index.min(), "last_bar": prices.index.max(),
            "annualization_factor": annualization_factor(prices.index),
        },
        window_results=window_results,
        caveats=[
            "Forward returns are masked over the final holding_period bars of each test window so no scored "
            "bar reads price beyond test_end.",
            "SignalReport.build_signal_report is not invoked: its contract requires a regime-interaction p(b3) "
            "that an ungated single-signal run does not produce.",
            f"Mean-reversion IC is reported on the raw z-score, so a working reversion signal produces a "
            f"{spec['expected_ic_sign']} IC.",
        ],
    )


@register_signal("momentum")
def build_momentum(pair: str, data_dir: str | Path, **kwargs: Any) -> SignalRun:
    return _build_daily_signal("momentum", pair, data_dir, **kwargs)


@register_signal("mean-reversion")
def build_mean_reversion(pair: str, data_dir: str | Path, **kwargs: Any) -> SignalRun:
    return _build_daily_signal("mean-reversion", pair, data_dir, **kwargs)


START = "2011-01-01"
END = "2023-12-31"

SCAN_OPEN = 9 * 60
SCAN_CLOSE = 12 * 60
EXIT_MIN = 13 * 60
VOL_RATIO_MIN_OBS = 250
GARCH_MIN_TRAIN = 500

K = 2.0
DELAY = 5

TRAIN_YEARS = 5
TEST_MONTHS = 3
EMBARGO_DAYS = 5
MAX_WINDOWS_SEARCHED = 200


def max_fitting_n_windows(data: pd.DataFrame, train_years: int, test_months: int, embargo_days: int) -> int:
    """Largest n_windows for which WalkForwardValidator.generate_windows()
    doesn't raise -- the most rolling windows that fit inside the data span
    without reading past its end.

    A fixed n_windows=10 works for momentum/mean-reversion's ~8-year sample
    but badly under-uses overshoot's ~13-year one. Searching for the max
    window count instead uses the full available span for every pair.
    """
    last_good = 0
    for n in range(1, MAX_WINDOWS_SEARCHED + 1):
        validator = WalkForwardValidator(
            signal_fn=None, data=data, n_windows=n,
            train_years=train_years, test_months=test_months, embargo_days=embargo_days,
        )
        try:
            validator.generate_windows()
        except ValueError:
            break
        last_good = n
    return last_good


def _overshoot_spearman_ic(disp: pd.Series, ret: pd.Series) -> tuple[float, str]:
    aligned = pd.concat([disp.rename("disp"), ret.rename("ret")], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan"), "insufficient_trades"
    if aligned["disp"].nunique() < 2:
        return float("nan"), "constant_disp"
    if aligned["ret"].nunique() < 2:
        return float("nan"), "constant_ret"
    ic, _ = spearmanr(aligned["disp"].to_numpy(), aligned["ret"].to_numpy())
    return float(ic), "ok"


def _overshoot_window_sharpe(ret: pd.Series) -> float:
    if len(ret) < 2:
        return float("nan")
    try:
        return PerformanceAnalyzer(ret).compute_sharpe()
    except ValueError:
        return float("nan")


def _build_overshoot(
    pair: str,
    data_dir: str | Path,
    *,
    start: str,
    end: str,
    k: float,
    entry_delay: int,
    scan_open: int,
    scan_close: int,
    exit_min: int,
    vol_ratio_min_obs: int,
    garch_min_train: int,
    train_years: int,
    test_months: int,
    embargo_days: int,
) -> SignalRun:
    sessions = build_overshoot_sessions(
        pair=pair, data_dir=data_dir, start=start, end=end,
        ks=[k], entry_delays=[entry_delay],
        scan_open=scan_open, scan_close=scan_close, exit_min=exit_min,
        vol_ratio_min_obs=vol_ratio_min_obs, garch_min_train=garch_min_train,
    )

    trades = overshoot_trades({pair: sessions}, k=k, delay=entry_delay).set_index("date").sort_index()
    disp_series = sessions[f"disp_{k}"]

    n_windows = max_fitting_n_windows(sessions[["sigma"]], train_years, test_months, embargo_days)
    validator = WalkForwardValidator(
        signal_fn=None, data=sessions[["sigma"]], n_windows=n_windows,
        train_years=train_years, test_months=test_months, embargo_days=embargo_days,
    )
    windows = validator.generate_windows()

    window_results = []
    for idx, w in enumerate(windows):
        session_mask = (sessions.index >= w["test_start"]) & (sessions.index < w["test_end"])
        n_sessions = int(session_mask.sum())

        trade_mask = (trades.index >= w["test_start"]) & (trades.index < w["test_end"])
        test_trades = trades.loc[trade_mask]
        n_scored = int(len(test_trades))

        disp_aligned = disp_series.reindex(test_trades.index)
        ic, ic_status = _overshoot_spearman_ic(disp_aligned, test_trades["ret"])
        sharpe = _overshoot_window_sharpe(test_trades["ret"])

        window_results.append({
            "window_idx": idx,
            "train_start": w["train_start"], "train_end": w["train_end"],
            "embargo_end": w["embargo_end"], "test_start": w["test_start"], "test_end": w["test_end"],
            "n_test": n_sessions, "n_scored": n_scored,
            "ic": ic, "ic_status": ic_status, "sharpe": sharpe,
        })

    return SignalRun(
        parameters={
            "start": start, "end": end, "k": k, "entry_delay_min": entry_delay,
            "scan_open_min": scan_open, "scan_close_min": scan_close, "exit_min": exit_min,
            "vol_ratio_min_obs": vol_ratio_min_obs, "garch_min_train": garch_min_train,
            "n_windows": n_windows, "train_years": train_years, "test_months": test_months,
            "embargo_days": embargo_days,
        },
        sample={
            "n_sessions": int(len(sessions)), "n_trades_total": int(len(trades)),
            "first_session": sessions.index.min(), "last_session": sessions.index.max(),
            "annualization_factor": annualization_factor(sessions.index),
        },
        window_results=window_results,
        caveats=[
            "k and entry_delay are pre-registered constants from section10, not fit on any training slice here -- "
            "the walk-forward windows exist for OOS-style reporting geometry, not parameter fitting.",
            "IC is Spearman(disp_k, trade_return) on trade days -- a diagnostic stand-in for a continuous signal, "
            "not one of section10's pre-registered validation criteria.",
            "GARCH conditional vol underlying the k*sigma threshold is itself fit walk-forward by calendar year "
            "(see walk_forward_conditional_vol); this script's windows are a second, coarser reporting layer on top.",
        ],
    )


@register_signal("intraday-overshoot")
def build_intraday_overshoot(pair: str, data_dir: str | Path, **kwargs: Any) -> SignalRun:
    return _build_overshoot(pair, data_dir, **kwargs)


@click.group()
def cli() -> None:
    """Unified walk-forward research CLI. `python research/run_research.py
    --help` lists every registered signal as a subcommand."""


def _daily_signal_command(signal_name: str) -> Callable[[Callable], click.Command]:
    """Builds a Click subcommand for a daily-signal adapter. Momentum and
    mean-reversion share the same option set, so it's declared once here
    instead of copy-pasted. The two subcommands differ only in
    `signal_name`, which selects the SIGNAL_REGISTRY spec and
    SIGNAL_ADAPTERS entry -- never in bespoke logic."""

    def decorator(fn: Callable) -> click.Command:
        fn = click.option(
            "--pairs", required=True,
            help="Comma-separated pair codes, or 'all' for all 10 SUPPORTED_PAIRS.",
        )(fn)
        fn = click.option(
            "--train-start", type=click.DateTime(formats=["%Y-%m-%d"]), default="2015-01-01",
            show_default=True, help="Inclusive start of the full research sample.",
        )(fn)
        fn = click.option(
            "--train-end", type=click.DateTime(formats=["%Y-%m-%d"]), default="2022-12-31",
            show_default=True, help="Inclusive end of the full research sample. Must precede the lockbox.",
        )(fn)
        fn = click.option("--windows", type=click.IntRange(min=1), default=10, show_default=True)(fn)
        fn = click.option("--train-years", type=click.IntRange(min=1), default=5, show_default=True)(fn)
        fn = click.option("--test-months", type=click.IntRange(min=1), default=3, show_default=True)(fn)
        fn = click.option("--embargo-days", type=click.IntRange(min=0), default=5, show_default=True)(fn)
        default_lookbacks = ", ".join(
            f"{name}={spec['default_lookback']}" for name, spec in SIGNAL_REGISTRY.items()
        )
        fn = click.option(
            "--lookback", type=click.IntRange(min=1), default=None,
            help=f"Signal lookback in bars. Defaults to each signal's own default ({default_lookbacks}).",
        )(fn)
        fn = click.option("--holding-period", type=click.IntRange(min=1), default=26, show_default=True)(fn)
        fn = click.option(
            "--output", type=click.Path(file_okay=False, path_type=Path), default=Path("results"), show_default=True,
        )(fn)
        fn = click.option(
            "--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
            default=DEFAULT_DATA_DIR, show_default=True,
        )(fn)
        fn = click.option("--allow-lockbox", is_flag=True, default=False)(fn)
        return cli.command(signal_name)(fn)

    return decorator


def _run_daily(signal_name, pairs, train_start, train_end, windows, train_years, test_months,
                embargo_days, lookback, holding_period, output, data_dir, allow_lockbox) -> None:
    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)
    if start >= end:
        raise click.BadParameter("--train-start must precede --train-end.")
    guard_lockbox(end, allow_lockbox)

    spec = SIGNAL_REGISTRY[signal_name]
    resolved_lookback = lookback if lookback is not None else spec["default_lookback"]

    for pair in resolve_pairs(pairs):
        run = SIGNAL_ADAPTERS[signal_name](
            pair=pair, data_dir=data_dir, train_start=start, train_end=end,
            windows=windows, train_years=train_years, test_months=test_months,
            embargo_days=embargo_days, lookback=resolved_lookback, holding_period=holding_period,
        )
        write_report(pair, signal_name, run, output)


@_daily_signal_command("momentum")
def momentum_cmd(pairs, train_start, train_end, windows, train_years, test_months,
                  embargo_days, lookback, holding_period, output, data_dir, allow_lockbox):
    """Time-series momentum: sign(P_t / P_(t-lookback) - 1)."""
    _run_daily("momentum", pairs, train_start, train_end, windows, train_years, test_months,
               embargo_days, lookback, holding_period, output, data_dir, allow_lockbox)


@_daily_signal_command("mean-reversion")
def mean_reversion_cmd(pairs, train_start, train_end, windows, train_years, test_months,
                        embargo_days, lookback, holding_period, output, data_dir, allow_lockbox):
    """Rolling price z-score, faded past +/- REVERSION_ENTRY_Z."""
    _run_daily("mean-reversion", pairs, train_start, train_end, windows, train_years, test_months,
               embargo_days, lookback, holding_period, output, data_dir, allow_lockbox)


@cli.command("intraday-overshoot")
@click.option("--pairs", default="all", show_default=True,
              help="Comma-separated pair codes, or 'all' for all 10 SUPPORTED_PAIRS.")
@click.option("--k", type=float, default=K, show_default=True, help="Crossing threshold, multiple of session sigma.")
@click.option("--entry-delay", type=int, default=DELAY, show_default=True, help="Minutes after crossing to enter.")
@click.option("--scan-open", type=int, default=SCAN_OPEN, show_default=True, help="Scan start, minutes since NY midnight.")
@click.option("--scan-close", type=int, default=SCAN_CLOSE, show_default=True, help="Scan end, minutes since NY midnight.")
@click.option("--exit-min", type=int, default=EXIT_MIN, show_default=True, help="Exit time, minutes since NY midnight.")
@click.option("--vol-ratio-min-obs", type=int, default=VOL_RATIO_MIN_OBS, show_default=True)
@click.option("--garch-min-train", type=int, default=GARCH_MIN_TRAIN, show_default=True)
@click.option("--start", default=START, show_default=True, help="Inclusive sample start (pre-lockbox).")
@click.option("--end", default=END, show_default=True, help="Inclusive sample end. Must precede the lockbox.")
@click.option("--train-years", type=int, default=TRAIN_YEARS, show_default=True)
@click.option("--test-months", type=int, default=TEST_MONTHS, show_default=True)
@click.option("--embargo-days", type=int, default=EMBARGO_DAYS, show_default=True)
@click.option("--output", type=click.Path(file_okay=False, path_type=Path), default=Path("results"), show_default=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=DEFAULT_DATA_DIR, show_default=True)
@click.option("--allow-lockbox", is_flag=True, default=False)
def overshoot_cmd(pairs, k, entry_delay, scan_open, scan_close, exit_min, vol_ratio_min_obs,
                   garch_min_train, start, end, train_years, test_months, embargo_days,
                   output, data_dir, allow_lockbox):
    """Intraday overshoot fade (strategy #6). n_windows isn't a CLI knob:
    it's auto-maximized per pair (max_fitting_n_windows) so the full
    sample is always used."""
    guard_lockbox(pd.Timestamp(end), allow_lockbox)

    for pair in resolve_pairs(pairs):
        run = SIGNAL_ADAPTERS["intraday-overshoot"](
            pair=pair, data_dir=data_dir, start=start, end=end, k=k, entry_delay=entry_delay,
            scan_open=scan_open, scan_close=scan_close, exit_min=exit_min,
            vol_ratio_min_obs=vol_ratio_min_obs, garch_min_train=garch_min_train,
            train_years=train_years, test_months=test_months, embargo_days=embargo_days,
        )
        write_report(pair, "intraday-overshoot", run, output)


if __name__ == "__main__":
    cli()
