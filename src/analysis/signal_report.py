from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.evaluation.significance import benjamini_hochberg_correction

DEFAULT_PROJECT_WIDE_N_TRIALS = 4

PROJECT_WIDE_PREREGISTERED_BAR_P = 0.0125

@dataclass
class LegSignalStats:
    """Out-of-sample diagnostics for a single leg."""

    leg_name: str

    ic_n_windows: int
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_frac_positive: float

    sharpe_n_windows: int
    sharpe_mean: float
    sharpe_std: float

    primary_p_value: float

    dsr: float
    dsr_n_trials: int
    dsr_observed_sharpe: float
    dsr_n_obs: int
    dsr_skewness: float
    dsr_excess_kurtosis: float


@dataclass
class SignalReport:
    strategy_name: str
    legs: dict

    bh_alpha: float
    bh_rejected: dict

    project_wide_bar_p: float
    caveats: list

    @property
    def strategy_significant(self) -> bool:
        return all(self.bh_rejected.values())

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# SignalReport -- {self.strategy_name}")

        for leg_name, leg in self.legs.items():
            lines.append(f"## {leg_name.title()} leg")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            lines.append(f"| IC mean (n={leg.ic_n_windows} windows) | {leg.ic_mean:.4f} |")
            lines.append(f"| IC std | {leg.ic_std:.4f} |")
            lines.append(f"| IC-derived IR (mean/std) | {leg.ic_ir:.4f} |")
            lines.append(f"| IC frac. positive windows | {leg.ic_frac_positive:.2%} |")
            lines.append(
                f"| OOS Sharpe mean (n={leg.sharpe_n_windows} windows) | {leg.sharpe_mean:.4f} |"
            )
            lines.append(f"| OOS Sharpe std | {leg.sharpe_std:.4f} |")
            lines.append(f"| Primary regression p(b3) | {leg.primary_p_value:.5f} |")
            lines.append(
                f"| BH-significant (alpha={self.bh_alpha}, 2-leg family) | "
                f"{self.bh_rejected[leg_name]} |"
            )
            lines.append(
                f"| Survives project-wide pre-registered bar (p<{self.project_wide_bar_p}) | "
                f"{leg.primary_p_value < self.project_wide_bar_p} |"
            )
            lines.append(
                f"| Deflated Sharpe (n_trials={leg.dsr_n_trials}, n_obs={leg.dsr_n_obs}) | "
                f"{leg.dsr:.4f} |"
            )
            lines.append(f"| DSR input: observed Sharpe | {leg.dsr_observed_sharpe:.4f} |")
            lines.append(
                f"| DSR input: skewness / excess kurtosis | "
                f"{leg.dsr_skewness:.4f} / {leg.dsr_excess_kurtosis:.4f} |"
            )
            lines.append("")

        verdict = "PASS" if self.strategy_significant else "FAIL"
        lines.append(
            f"**Multiple-testing-corrected verdict (2-leg BH, alpha={self.bh_alpha}): "
            f"{verdict}**"
        )
        lines.append("")

        if self.caveats:
            lines.append("## Caveats")
            lines.append("")
            for c in self.caveats:
                lines.append(f"- {c}")
            lines.append("")

        return "\n".join(lines)


def _summarize_ic(ic_by_window: Sequence[float]) -> dict:
    arr = np.array([v for v in ic_by_window if not np.isnan(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"), ir=float("nan"), frac_positive=float("nan"))
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else float("nan")
    ir = mean / std if std and not np.isnan(std) and std != 0 else float("nan")
    frac_positive = float((arr > 0).mean())
    return dict(n=n, mean=mean, std=std, ir=ir, frac_positive=frac_positive)


def _summarize_sharpe(sharpe_by_window: Sequence[float]) -> dict:
    arr = np.array([v for v in sharpe_by_window if not np.isnan(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"))
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else float("nan")
    return dict(n=n, mean=mean, std=std)


def _compute_leg_dsr(regime_gated_returns: pd.Series, n_trials: int) -> dict:
    clean = regime_gated_returns.dropna()
    analyzer = PerformanceAnalyzer(returns=clean)
    observed_sharpe = analyzer.compute_sharpe()
    jb = analyzer.jarque_bera_test()

    dsr = analyzer.deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        n_obs=len(clean),
        skewness=jb["skewness"],
        kurtosis=jb["excess_kurtosis"],
    )
    return dict(
        dsr=dsr,
        observed_sharpe=observed_sharpe,
        n_obs=len(clean),
        skewness=jb["skewness"],
        excess_kurtosis=jb["excess_kurtosis"],
    )


def build_signal_report(
    strategy_name: str,
    leg_ic_by_window: dict,
    leg_sharpe_by_window: dict,
    leg_primary_p_value: dict,
    leg_regime_gated_returns: dict,
    n_trials: int = DEFAULT_PROJECT_WIDE_N_TRIALS,
    bh_alpha: float = 0.05,
    project_wide_bar_p: float = PROJECT_WIDE_PREREGISTERED_BAR_P,
    extra_caveats: Sequence[str] = (),
) -> SignalReport:
    """Aggregate pre-computed per-leg walk-forward results into a SignalReport.

    Parameters
    ----------
    strategy_name : str
    leg_ic_by_window : dict[str, list[float]]
        Per leg, one (possibly NaN) regime-gated IC value per walk-forward
        window.
    leg_sharpe_by_window : dict[str, list[float]]
        Per leg, one (possibly NaN) regime-gated window Sharpe per
        walk-forward window. Same leg-name keys as leg_ic_by_window.
    leg_primary_p_value : dict[str, float]
        Per leg, the Section 10 primary interaction-regression p(b3).
    leg_regime_gated_returns : dict[str, pd.Series]
        Per leg, the pooled (across pairs/windows) regime-gated
        exposure x forward-return series feeding that leg's DSR.
    n_trials : int, default 4
        See DEFAULT_PROJECT_WIDE_N_TRIALS docstring above.
    bh_alpha : float, default 0.05
        Alpha for the in-report two-leg Benjamini-Hochberg correction.
    project_wide_bar_p : float, default 0.0125
        Reported for comparison only -- see PROJECT_WIDE_PREREGISTERED_BAR_P.
    extra_caveats : sequence[str], optional
        Additional caveats appended after the default set below.

    Returns
    -------
    SignalReport

    Raises
    ------
    ValueError
        If the leg_* dict arguments don't all share the same set of leg
        names.
    """
    leg_names = list(leg_ic_by_window.keys())
    if (
        set(leg_sharpe_by_window) != set(leg_names)
        or set(leg_primary_p_value) != set(leg_names)
        or set(leg_regime_gated_returns) != set(leg_names)
    ):
        raise ValueError("leg_* dict arguments must all share the same set of leg names.")

    p_values = [leg_primary_p_value[name] for name in leg_names]
    bh_reject = benjamini_hochberg_correction(p_values, alpha=bh_alpha)
    bh_rejected = dict(zip(leg_names, bh_reject))

    legs: dict[str, LegSignalStats] = {}
    for name in leg_names:
        ic_summary = _summarize_ic(leg_ic_by_window[name])
        sharpe_summary = _summarize_sharpe(leg_sharpe_by_window[name])
        dsr_result = _compute_leg_dsr(leg_regime_gated_returns[name], n_trials)

        legs[name] = LegSignalStats(
            leg_name=name,
            ic_n_windows=ic_summary["n"],
            ic_mean=ic_summary["mean"],
            ic_std=ic_summary["std"],
            ic_ir=ic_summary["ir"],
            ic_frac_positive=ic_summary["frac_positive"],
            sharpe_n_windows=sharpe_summary["n"],
            sharpe_mean=sharpe_summary["mean"],
            sharpe_std=sharpe_summary["std"],
            primary_p_value=leg_primary_p_value[name],
            dsr=dsr_result["dsr"],
            dsr_n_trials=n_trials,
            dsr_observed_sharpe=dsr_result["observed_sharpe"],
            dsr_n_obs=dsr_result["n_obs"],
            dsr_skewness=dsr_result["skewness"],
            dsr_excess_kurtosis=dsr_result["excess_kurtosis"],
        )

    default_caveats = [
        "OOS Sharpe (per-window and the DSR input) is a raw, unsized, "
        "no-transaction-cost signal-level proxy: sign(signal) x forward "
        "return on regime-active days only. Section 7 vol-targeted sizing "
        "and the Day 57 transaction-cost model are not applied here -- this "
        "is not a claim of tradable performance.",
        "DSR's observed Sharpe is annualized from a DatetimeIndex pooled "
        "across all 3 pairs over the same calendar span, so "
        "compute_ann_factor()'s n_obs/years_spanned is inflated roughly 3x "
        "versus a single-pair basis -- DSR here is optimistically biased. "
        "Known, documented gap; not corrected in this build.",
        f"The BH correction above (alpha={bh_alpha}) is applied only across "
        f"this strategy's 2 legs, not the full project-wide 4 strategies the "
        f"spec pre-registers (Section 1, item 5: needs p<{project_wide_bar_p} "
        f"at rank 1 of 4). The other 3 strategies' final p-values are not "
        "logged anywhere in the repo yet, so a true 4-way BH cannot be run. "
        "The project-wide bar column above is shown for comparison, not as "
        "a substitute for that correction.",
        "This report covers only the two-leg multiple-testing check and "
        "DSR. It does not replace the full Section 10 verdict (reliability "
        "gate + both robustness checks) in "
        "research/strategies/validation_falsification/vol_regime_two_leg_section10_validation.md, and the Section "
        "10 lockbox holdout (2024-2026) has still not been opened.",
    ]

    return SignalReport(
        strategy_name=strategy_name,
        legs=legs,
        bh_alpha=bh_alpha,
        bh_rejected=bh_rejected,
        project_wide_bar_p=project_wide_bar_p,
        caveats=list(default_caveats) + list(extra_caveats),
    )
