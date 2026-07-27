from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import click
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.analysis.performance_analyzer import information_coefficient
from src.framework.data_loader import SUPPORTED_PAIRS, DataLoader
from src.framework.walk_forward import WalkForwardValidator
from src.signals.momentum import momentum_signal
from src.signals.mean_reversion import price_zscore_signal

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT.parent / "data"

STRATEGY_PAIRS = tuple(sorted(SUPPORTED_PAIRS))

LOCKBOX_START = pd.Timestamp("2024-01-01")

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


def annualization_factor(index: pd.DatetimeIndex) -> float:
    """Empirical bars-per-year from the observed index span."""
    if len(index) < 2:
        return float("nan")
    years = (index.max() - index.min()).days / 365.25
    if years <= 0:
        return float("nan")
    return len(index) / years


def spearman_ic(signal: pd.Series, forward_returns: pd.Series) -> tuple[float, str]:
    """
    Guarded wrapper around the framework's `information_coefficient`.

    `information_coefficient` aligns but does not drop NaN, and scipy's
    spearmanr propagates NaN through the whole statistic. Dropping incomplete
    pairs and rejecting degenerate inputs is the caller's job.

    Returns the IC and a status string. A sign-valued signal evaluated over a
    short test window is frequently constant, which leaves Spearman undefined;
    that case is reported as `constant_signal` rather than collapsed into a
    bare NaN, so the summary can distinguish "no variation to measure" from
    "measured and found to be zero".
    """
    aligned = pd.concat(
        [signal.rename("signal"), forward_returns.rename("forward")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        return float("nan"), "insufficient_obs"
    if aligned["signal"].nunique() < 2:
        return float("nan"), "constant_signal"
    if aligned["forward"].nunique() < 2:
        return float("nan"), "constant_forward_returns"
    ic = information_coefficient(
        aligned["signal"], aligned["forward"], method="spearman"
    )
    return ic, "ok"


def window_sharpe(exposure: pd.Series, log_returns: pd.Series) -> float:
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


def truncate_forward_returns(
    forward_returns: pd.Series,
    test_index: pd.DatetimeIndex,
    holding_period: int,
) -> pd.Series:
    """
    Restrict forward returns to bars whose full holding horizon resolves
    inside the test window.

    A forward return at bar t reads price at bar t + holding_period. For the
    final `holding_period` bars of a test window that target price sits beyond
    test_end, inside the embargo or the following training region. Those bars
    are masked to NaN rather than scored.
    """
    sliced = forward_returns.reindex(test_index).copy()
    if holding_period > 0 and len(sliced) > 0:
        sliced.iloc[-holding_period:] = np.nan
    return sliced


def score_windows(
    windows: list[dict],
    signal: pd.Series,
    exposure: pd.Series,
    forward_returns: pd.Series,
    log_returns: pd.Series,
    holding_period: int,
) -> list[dict]:
    """Per-window out-of-sample IC and Sharpe on the test slice only."""
    results: list[dict] = []
    for idx, w in enumerate(windows):
        mask = (signal.index >= w["test_start"]) & (signal.index < w["test_end"])
        test_index = signal.index[mask]

        window_forward = truncate_forward_returns(
            forward_returns, test_index, holding_period
        )
        window_signal = signal.reindex(test_index)
        ic, ic_status = spearman_ic(window_signal, window_forward)
        sharpe = window_sharpe(
            exposure.reindex(test_index), log_returns.reindex(test_index)
        )

        results.append(
            {
                "window_idx": idx,
                "train_start": w["train_start"],
                "train_end": w["train_end"],
                "embargo_end": w["embargo_end"],
                "test_start": w["test_start"],
                "test_end": w["test_end"],
                "n_test": int(len(test_index)),
                "n_scored": int(window_forward.notna().sum()),
                "signal_nunique": int(window_signal.nunique()),
                "ic": ic,
                "ic_status": ic_status,
                "sharpe": sharpe,
            }
        )
    return results


def summarize(values: list[float]) -> dict[str, float]:
    """Mean, std, IR and positive fraction over the non-NaN subset."""
    arr = np.array([v for v in values if not math.isnan(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ir": float("nan"),
            "frac_positive": float("nan"),
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else float("nan")
    ir = mean / std if std and not math.isnan(std) and std != 0 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ir": ir,
        "frac_positive": float((arr > 0).mean()),
    }


def json_safe(obj: Any) -> Any:
    """
    Recursively coerce pandas and numpy scalars into strict-JSON primitives.

    NaN and infinity are emitted as null. Python's json module writes bare
    NaN/Infinity tokens by default, which no strict JSON parser accepts.
    """
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


@click.command()
@click.option(
    "--signal",
    "signal_name",
    type=click.Choice(sorted(SIGNAL_REGISTRY), case_sensitive=False),
    required=True,
    help="Signal to evaluate. momentum = sign of lookback return; mean-reversion = rolling price z-score.",
)
@click.option(
    "--pair",
    type=click.Choice(STRATEGY_PAIRS, case_sensitive=False),
    required=True,
    help="Currency pair. Choices track DataLoader.SUPPORTED_PAIRS.",
)
@click.option(
    "--train-start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default="2015-01-01",
    show_default=True,
    help="Inclusive start of the full research sample that walk-forward carves windows from.",
)
@click.option(
    "--train-end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default="2022-12-31",
    show_default=True,
    help="Inclusive end of the full research sample. Must precede the 2024-01-01 lockbox.",
)
@click.option(
    "--windows",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Number of rolling walk-forward windows to generate.",
)
@click.option(
    "--train-years",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="Training window length in calendar years.",
)
@click.option(
    "--test-months",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
    help="Test window length in calendar months, and the step between consecutive windows.",
)
@click.option(
    "--embargo-days",
    type=click.IntRange(min=0),
    default=5,
    show_default=True,
    help="Calendar-day gap enforced between train_end and test_start.",
)
@click.option(
    "--lookback",
    type=click.IntRange(min=1),
    default=None,
    help="Signal lookback in bars. Defaults to 78 for momentum, 26 for mean-reversion.",
)
@click.option(
    "--holding-period",
    type=click.IntRange(min=1),
    default=26,
    show_default=True,
    help="Forward-return horizon in bars used for the IC target.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Directory for the JSON report. Created if absent.",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_DATA_DIR,
    show_default=True,
    help="Directory holding {PAIR}.csv minute bars.",
)
@click.option(
    "--allow-lockbox",
    is_flag=True,
    default=False,
    help="Override the 2024-01-01 lockbox guard. Do not set this during development.",
)
def main(
    signal_name: str,
    pair: str,
    train_start: Any,
    train_end: Any,
    windows: int,
    train_years: int,
    test_months: int,
    embargo_days: int,
    lookback: int | None,
    holding_period: int,
    output: Path,
    data_dir: Path,
    allow_lockbox: bool,
) -> None:
    """
    Run one signal on one pair through the walk-forward research pipeline and
    write a JSON report.

    Chain: DataLoader -> signal construction -> WalkForwardValidator window
    generation -> per-window out-of-sample scoring -> JSON report.

    All reported IC and Sharpe figures are raw, unsized, and gross of
    transaction costs. They are signal-level diagnostics, not tradable
    performance.
    """
    signal_name = signal_name.lower()
    pair = pair.upper()
    spec = SIGNAL_REGISTRY[signal_name]
    lookback = lookback if lookback is not None else spec["default_lookback"]

    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)

    if start >= end:
        raise click.BadParameter("--train-start must precede --train-end.")

    if end >= LOCKBOX_START and not allow_lockbox:
        raise click.BadParameter(
            f"--train-end {end.date()} reaches the reserved lockbox slice "
            f"(>= {LOCKBOX_START.date()}). The lockbox is held for a single "
            f"unbiased final evaluation and must not be touched during "
            f"development. Pass --allow-lockbox only if you are deliberately "
            f"spending it."
        )

    loader = DataLoader(
        pairs=[pair],
        start=str(start.date()),
        end=str(end.date()),
        embargo_days=embargo_days,
        data_dir=str(data_dir),
    )
    prices = loader.load()[pair]
    frame = prices.to_frame(name="price")

    log_returns = np.log(prices / prices.shift(1))
    forward_returns = np.log(prices.shift(-holding_period) / prices)

    signal = spec["signal_fn"](frame, lookback)
    exposure = spec["exposure_fn"](signal)

    validator = WalkForwardValidator(
        signal_fn=spec["signal_fn"],
        data=frame,
        n_windows=windows,
        train_years=train_years,
        test_months=test_months,
        embargo_days=embargo_days,
    )
    try:
        generated = validator.generate_windows()
    except ValueError as exc:
        raise click.BadParameter(
            f"{exc} Try fewer --windows, a shorter --train-years, a shorter "
            f"--test-months, or a wider --train-start/--train-end range."
        ) from exc

    window_results = score_windows(
        generated, signal, exposure, forward_returns, log_returns, holding_period
    )

    ic_summary = summarize([w["ic"] for w in window_results])
    sharpe_summary = summarize([w["sharpe"] for w in window_results])

    ic_status_counts: dict[str, int] = {}
    for w in window_results:
        ic_status_counts[w["ic_status"]] = ic_status_counts.get(w["ic_status"], 0) + 1

    report = {
        "pair": pair,
        "signal": signal_name,
        "parameters": {
            "train_start": start,
            "train_end": end,
            "n_windows": windows,
            "train_years": train_years,
            "test_months": test_months,
            "embargo_days": embargo_days,
            "lookback": lookback,
            "holding_period": holding_period,
            "reversion_entry_z": REVERSION_ENTRY_Z,
            "expected_ic_sign": spec["expected_ic_sign"],
        },
        "sample": {
            "n_daily_bars": int(len(prices)),
            "first_bar": prices.index.min(),
            "last_bar": prices.index.max(),
            "annualization_factor": annualization_factor(prices.index),
        },
        "window_results": window_results,
        "ic_summary": ic_summary,
        "ic_status_counts": ic_status_counts,
        "sharpe_summary": sharpe_summary,
        "caveats": [
            "Raw unsized signal-level diagnostics gross of transaction costs. Not a claim of tradable performance.",
            "No multiple-testing correction applied. This run does not increment project n_trials and is not a research verdict.",
            "Forward returns are masked over the final holding_period bars of each test window so no scored bar reads price beyond test_end.",
            "SignalReport.build_signal_report is not invoked: its contract requires a regime-interaction p(b3) that an ungated single-signal run does not produce.",
            f"Mean-reversion IC is reported on the raw z-score, so a working reversion signal produces a {spec['expected_ic_sign']} IC.",
        ],
    }

    output.mkdir(parents=True, exist_ok=True)
    out_path = output / f"{pair}_{signal_name}.json"
    out_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")

    click.echo(f"{pair} / {signal_name}")
    click.echo(
        f"  sample: {len(prices)} daily bars "
        f"[{prices.index.min().date()} -> {prices.index.max().date()}]"
    )
    click.echo(
        f"  IC     mean={ic_summary['mean']:.4f} std={ic_summary['std']:.4f} "
        f"IR={ic_summary['ir']:.4f} pos={ic_summary['frac_positive']:.0%} "
        f"n={ic_summary['n']}/{len(window_results)}"
    )
    unscored = {k: v for k, v in ic_status_counts.items() if k != "ok"}
    if unscored:
        click.echo(f"  IC unscored windows: {unscored}")
    click.echo(
        f"  Sharpe mean={sharpe_summary['mean']:.4f} "
        f"std={sharpe_summary['std']:.4f} n={sharpe_summary['n']}"
    )
    click.echo(f"  written: {out_path}")


if __name__ == "__main__":
    main()
