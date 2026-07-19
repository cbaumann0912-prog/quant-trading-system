import numpy as np
import pandas as pd
import pytest

from src.analysis.signal_report import (
    LegSignalStats,
    SignalReport,
    _summarize_ic,
    _summarize_sharpe,
    build_signal_report,
)


def _make_returns(mean: float, std: float, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2016-01-01", periods=n, freq="D")
    values = rng.normal(mean, std, n)
    return pd.Series(values, index=index)


def _default_kwargs(momentum_mean=0.001, reversion_mean=0.001):
    return dict(
        strategy_name="Test Strategy",
        leg_ic_by_window={
            "momentum": [0.3, 0.4, 0.35, np.nan],
            "reversion": [0.2, 0.25, 0.1],
        },
        leg_sharpe_by_window={
            "momentum": [1.0, 1.2, 0.9],
            "reversion": [0.5, 0.6, 0.4],
        },
        leg_primary_p_value={"momentum": 0.0001, "reversion": 0.02},
        leg_regime_gated_returns={
            "momentum": _make_returns(momentum_mean, 0.01, 500, seed=1),
            "reversion": _make_returns(reversion_mean, 0.01, 500, seed=2),
        },
    )


def test_summarize_ic_basic_stats():
    result = _summarize_ic([0.1, 0.2, -0.1, np.nan, 0.3])

    assert result["n"] == 4
    assert result["mean"] == pytest.approx(np.mean([0.1, 0.2, -0.1, 0.3]))
    assert result["frac_positive"] == pytest.approx(0.75)


def test_summarize_ic_all_nan_returns_nan_stats():
    result = _summarize_ic([np.nan, np.nan])

    assert result["n"] == 0
    assert np.isnan(result["mean"])


def test_summarize_ic_single_value_std_is_nan():
    result = _summarize_ic([0.5])

    assert result["n"] == 1
    assert np.isnan(result["std"])
    assert np.isnan(result["ir"])


def test_summarize_sharpe_basic_stats():
    result = _summarize_sharpe([1.0, -0.5, 0.25, np.nan])

    assert result["n"] == 3
    assert result["mean"] == pytest.approx(np.mean([1.0, -0.5, 0.25]))


def test_summarize_sharpe_empty_input():
    result = _summarize_sharpe([])

    assert result["n"] == 0
    assert np.isnan(result["mean"])


def test_build_signal_report_returns_signal_report_with_both_legs():
    report = build_signal_report(**_default_kwargs())

    assert isinstance(report, SignalReport)
    assert set(report.legs.keys()) == {"momentum", "reversion"}
    assert isinstance(report.legs["momentum"], LegSignalStats)


def test_build_signal_report_ic_and_sharpe_fields_match_summary():
    report = build_signal_report(**_default_kwargs())
    m = report.legs["momentum"]

    assert m.ic_n_windows == 3
    assert m.ic_mean == pytest.approx(np.mean([0.3, 0.4, 0.35]))
    assert m.sharpe_n_windows == 3


def test_build_signal_report_default_n_trials_is_four():
    report = build_signal_report(**_default_kwargs())

    assert report.legs["momentum"].dsr_n_trials == 4
    assert report.legs["reversion"].dsr_n_trials == 4


def test_build_signal_report_n_trials_override_propagates():
    report = build_signal_report(**_default_kwargs(), n_trials=1)

    assert report.legs["momentum"].dsr_n_trials == 1


def test_build_signal_report_more_trials_never_increases_dsr():
    kwargs = _default_kwargs()
    report_1_trial = build_signal_report(**kwargs, n_trials=1)
    report_4_trials = build_signal_report(**kwargs, n_trials=4)

    assert report_4_trials.legs["momentum"].dsr <= report_1_trial.legs["momentum"].dsr


def test_build_signal_report_bh_correction_applied_across_both_legs():
    report = build_signal_report(**_default_kwargs())

    assert report.bh_rejected["momentum"] is True
    assert report.bh_rejected["reversion"] is True
    assert report.strategy_significant is True


def test_build_signal_report_bh_correction_can_fail_a_leg():
    kwargs = _default_kwargs()
    kwargs["leg_primary_p_value"] = {"momentum": 0.0001, "reversion": 0.9}
    report = build_signal_report(**kwargs)

    assert report.bh_rejected["reversion"] is False
    assert report.strategy_significant is False


def test_build_signal_report_mismatched_leg_keys_raises():
    kwargs = _default_kwargs()
    kwargs["leg_sharpe_by_window"] = {"momentum": [1.0]}

    with pytest.raises(ValueError):
        build_signal_report(**kwargs)


def test_build_signal_report_default_caveats_present():
    report = build_signal_report(**_default_kwargs())

    assert len(report.caveats) >= 4
    assert any("transaction-cost" in c for c in report.caveats)
    assert any("4 strategies" in c for c in report.caveats)


def test_build_signal_report_extra_caveats_appended():
    report = build_signal_report(**_default_kwargs(), extra_caveats=["custom note"])
    
    assert report.caveats[-1] == "custom note"


def test_build_signal_report_to_markdown_contains_both_legs_and_verdict():
    report = build_signal_report(**_default_kwargs())
    md = report.to_markdown()
    
    assert "Momentum leg" in md
    assert "Reversion leg" in md
    assert "Multiple-testing-corrected verdict" in md
    assert "PASS" in md


def test_build_signal_report_project_wide_bar_column_reflects_threshold():
    report = build_signal_report(**_default_kwargs(), project_wide_bar_p=0.01)
    
    assert report.legs["momentum"].primary_p_value < 0.01
    assert report.legs["reversion"].primary_p_value > 0.01
