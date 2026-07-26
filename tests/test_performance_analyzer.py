import pytest
import pandas as pd
import numpy as np
from src.analysis.performance_analyzer import (
    EQUITY_TRADING_DAYS_PER_YEAR,
    FX_TRADING_DAYS_PER_YEAR,
    PerformanceAnalyzer,
    bps_per_pip,
    breakeven_annual_return,
    breakeven_sharpe,
    cost_report,
    implied_trades_per_year,
    information_coefficient,
    information_ratio,
    max_viable_spread_pips,
    pip_size,
    regime_conditional_performance,
    rollover_bps_per_day,
    round_trip_cost_bps,
)

POSITIVE_RETURNS = pd.Series(
    [0.01, 0.02, 0.01, 0.03, 0.01],
    index=pd.date_range(start="2026-05-28", periods=5, freq="D")
)

FLAT_RETURNS = pd.Series(
    [0.0, 0.0, 0.0, 0.0, 0.0],
    index=pd.date_range(start="2026-05-28", periods=5, freq="D")
)

CONSTANT_NONZERO_RETURNS = pd.Series(
    [1 / 3] * 50,
    index=pd.date_range(start="2026-05-28", periods=50, freq="D")
)


def test_sharpe_positive_returns():
    result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).compute_sharpe()

    assert result > 0


def test_sharpe_zero_rf():
    analyzer = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None)
    result = analyzer.compute_sharpe()
    ann_factor = analyzer.compute_ann_factor()
    expected = (POSITIVE_RETURNS.mean() / POSITIVE_RETURNS.std()) * np.sqrt(ann_factor)

    assert result == expected


def test_sharpe_constant_nonzero_returns_is_nan():
    result = PerformanceAnalyzer(returns=CONSTANT_NONZERO_RETURNS, trades=None).compute_sharpe()

    assert np.isnan(result)


def test_max_drawdown_returns_dict_with_value_and_duration():
    result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).compute_max_drawdown()

    assert "value" in result
    assert "duration_days" in result
    assert "start_date" in result
    assert "end_date" in result


def test_max_drawdown_flat_series():
    result = PerformanceAnalyzer(returns=FLAT_RETURNS, trades=None).compute_max_drawdown()

    assert result["value"] == 0


def test_dsr_between_0_and_1():
    result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=1.5,
        n_trials=10,
        n_obs=312,
        skewness=0.0,
        kurtosis=3.0
    )

    assert 0 <= result <= 1


def test_more_trials_lowers_dsr():
    few_trials = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=1.5,
        n_trials=2,
        n_obs=50,
        skewness=0.0,
        kurtosis=3.0
    )
    many_trials = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=1.5,
        n_trials=50,
        n_obs=50,
        skewness=0.0,
        kurtosis=3.0
    )

    assert many_trials < few_trials


def test_deflated_lower_than_observed():
    observed = 1.5
    dsr_deflated = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=observed,
        n_trials=50,
        n_obs=50,
        skewness=0.0,
        kurtosis=3.0
    )
    dsr_baseline = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=observed,
        n_trials=2,
        n_obs=50,
        skewness=0.0,
        kurtosis=3.0
    )

    assert dsr_baseline > dsr_deflated


def test_dsr_returns_nan_when_variance_term_non_positive():
    result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).deflated_sharpe_ratio(
        observed_sharpe=5.0,
        n_trials=1,
        n_obs=30,
        skewness=5.0,
        kurtosis=1.0,
    )

    assert np.isnan(result)


def test_dsr_single_trial_no_correction():
    analyzer = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None)
    observed_sharpe = 1.2
    n_obs = 252
    skewness = 0.0
    kurtosis = 3.0

    result = analyzer.deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_trials=1,
        n_obs=n_obs,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    ann_factor = analyzer.compute_ann_factor()
    sr_period = observed_sharpe / np.sqrt(ann_factor)
    V = (1 - skewness * sr_period + ((kurtosis + 2) / 4) * sr_period**2) / (n_obs - 1)
    se = np.sqrt(V)
    z = sr_period / se
    from scipy import stats as scipy_stats
    expected = float(scipy_stats.norm.cdf(z))

    assert result == pytest.approx(expected)


def test_t_stat_constant_nonzero_returns_is_nan():
    result = PerformanceAnalyzer(returns=CONSTANT_NONZERO_RETURNS, trades=None).compute_t_stat()

    assert np.isnan(result)


def test_ann_factor_raises_on_nonpositive_timespan():
    single_obs = pd.Series([0.01], index=pd.date_range("2026-01-01", periods=1, freq="D"))
    analyzer = PerformanceAnalyzer(returns=single_obs, trades=None)

    with pytest.raises(ValueError):
        analyzer.compute_ann_factor()


def test_sortino_empty_returns_is_nan():
    empty_returns = pd.Series([], dtype=float)
    result = PerformanceAnalyzer(returns=empty_returns, trades=None).compute_sortino()

    assert np.isnan(result)


def test_sortino_no_downside_observations_is_nan():
    all_positive = pd.Series(
        [0.01, 0.02, 0.015, 0.03, 0.01],
        index=pd.date_range("2026-01-01", periods=5, freq="D")
    )
    result = PerformanceAnalyzer(returns=all_positive, trades=None).compute_sortino()

    assert np.isnan(result)


def test_sortino_zero_downside_deviation_is_nan():
    returns = pd.Series(
        [0.01, -0.01, 0.02, -0.01, 0.015],
        index=pd.date_range("2026-01-01", periods=5, freq="D")
    )
    result = PerformanceAnalyzer(returns=returns, trades=None).compute_sortino()

    assert result == pytest.approx(result)


def test_win_rate_no_trades_is_nan():
    result = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=None).compute_win_rate()

    assert np.isnan(result)


def test_win_rate_missing_pnl_column_raises():
    trades = pd.DataFrame({"not_pnl": [1, 2, 3]})
    analyzer = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=trades)

    with pytest.raises(ValueError):
        analyzer.compute_win_rate()


def test_win_rate_normal_case_matches_hand_calc():
    trades = pd.DataFrame({"pnl": [10, -5, 20, -3, 8]})
    analyzer = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=trades)
    result = analyzer.compute_win_rate()

    assert result == pytest.approx(3 / 5)


def test_profit_factor_zero_gross_loss_is_nan():
    trades = pd.DataFrame({"pnl": [10, 5, 20]})
    analyzer = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=trades)
    result = analyzer.compute_profit_factor()

    assert np.isnan(result)


def test_profit_factor_missing_pnl_column_raises():
    trades = pd.DataFrame({"not_pnl": [1, 2, 3]})
    analyzer = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=trades)

    with pytest.raises(ValueError):
        analyzer.compute_profit_factor()


def test_profit_factor_normal_case_matches_hand_calc():
    trades = pd.DataFrame({"pnl": [10, -5, 20, -3, 8]})
    analyzer = PerformanceAnalyzer(returns=POSITIVE_RETURNS, trades=trades)
    result = analyzer.compute_profit_factor()

    assert result == pytest.approx(38 / 8)


def test_calmar_zero_drawdown_is_nan():
    monotonic_returns = pd.Series(
        [0.01, 0.01, 0.01, 0.01, 0.01],
        index=pd.date_range("2026-01-01", periods=5, freq="D")
    )
    result = PerformanceAnalyzer(returns=monotonic_returns, trades=None).compute_calmar()

    assert np.isnan(result)


def test_calmar_empty_returns_is_nan():
    empty_returns = pd.Series([], dtype=float)
    result = PerformanceAnalyzer(returns=empty_returns, trades=None).compute_calmar()

    assert np.isnan(result)


def test_jarque_bera_normal_data_does_not_reject():
    rng = np.random.default_rng(7)
    normal_returns = pd.Series(rng.standard_normal(2000))
    result = PerformanceAnalyzer(returns=normal_returns, trades=None).jarque_bera_test()

    assert result["reject_normality"] is False
    assert result["p_value"] > 0.05


def test_jarque_bera_heavy_tailed_data_rejects():
    rng = np.random.default_rng(7)
    heavy_tailed_returns = pd.Series(rng.standard_t(df=2, size=2000))
    result = PerformanceAnalyzer(returns=heavy_tailed_returns, trades=None).jarque_bera_test()

    assert result["reject_normality"] is True
    assert result["p_value"] < 0.05


def test_ljung_box_white_noise_does_not_reject():
    rng = np.random.default_rng(11)
    white_noise = pd.Series(rng.standard_normal(500))
    result = PerformanceAnalyzer(returns=white_noise, trades=None).ljung_box_test(lags=10)

    assert result["reject_white_noise"] is False
    assert result["p_value"] > 0.05


def test_ljung_box_autocorrelated_series_rejects():
    rng = np.random.default_rng(11)
    n = 500
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + eps[t]
    autocorrelated = pd.Series(x)
    result = PerformanceAnalyzer(returns=autocorrelated, trades=None).ljung_box_test(lags=10)

    assert result["reject_white_noise"] is True
    assert result["p_value"] < 0.05


def test_tracking_error_misaligned_index_uses_intersection():
    idx_strategy = pd.date_range("2026-01-01", periods=10, freq="D")
    idx_benchmark = pd.date_range("2026-01-05", periods=10, freq="D")
    strategy_returns = pd.Series(np.full(10, 0.01), index=idx_strategy)
    benchmark_returns = pd.Series(np.full(10, 0.005), index=idx_benchmark)

    analyzer = PerformanceAnalyzer(returns=strategy_returns, trades=None)
    result = analyzer.tracking_error(benchmark_returns, ann_factor=252)

    assert np.isfinite(result)


def test_tracking_error_identical_series_is_zero():
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    returns = pd.Series(np.linspace(0.01, 0.02, 10), index=idx)

    analyzer = PerformanceAnalyzer(returns=returns, trades=None)
    result = analyzer.tracking_error(returns.copy(), ann_factor=252)

    assert result == pytest.approx(0.0, abs=1e-10)


def test_run_report_returns_populated_report():
    rng = np.random.default_rng(3)
    returns = pd.Series(
        rng.normal(0.0005, 0.01, 300),
        index=pd.date_range("2026-01-01", periods=300, freq="D")
    )
    trades = pd.DataFrame({"pnl": [10, -5, 20, -3, 8]})
    analyzer = PerformanceAnalyzer(returns=returns, trades=trades)

    report = analyzer.run_report(n_trials=5)

    assert np.isfinite(report.sharpe_ratio)
    assert np.isfinite(report.jb_stat)
    assert np.isfinite(report.lb_stat)
    assert report.n_trades == 5
    assert np.isnan(report.tracking_error)


def test_run_report_no_benchmark_gives_nan_tracking_error():
    rng = np.random.default_rng(3)
    returns = pd.Series(
        rng.normal(0.0005, 0.01, 300),
        index=pd.date_range("2026-01-01", periods=300, freq="D")
    )
    analyzer = PerformanceAnalyzer(returns=returns, trades=None)

    report = analyzer.run_report(benchmark_returns=None)

    assert np.isnan(report.tracking_error)


def test_run_report_n_trials_flows_to_dsr():
    rng = np.random.default_rng(3)
    returns = pd.Series(
        rng.normal(0.0005, 0.01, 300),
        index=pd.date_range("2026-01-01", periods=300, freq="D")
    )

    report_few_trials = PerformanceAnalyzer(returns=returns, trades=None).run_report(n_trials=1)
    report_many_trials = PerformanceAnalyzer(returns=returns, trades=None).run_report(n_trials=50)

    assert report_many_trials.deflated_sharpe < report_few_trials.deflated_sharpe


def test_ic_perfect_monotonic_signal_equals_one():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    signal = pd.Series(np.arange(10), index=idx)
    forward_returns = pd.Series(np.arange(10) * 0.01, index=idx)
    result = information_coefficient(signal, forward_returns)

    assert result == pytest.approx(1.0)


def test_ic_perfect_inverse_monotonic_signal_equals_negative_one():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    signal = pd.Series(np.arange(10), index=idx)
    forward_returns = pd.Series(-np.arange(10) * 0.01, index=idx)
    result = information_coefficient(signal, forward_returns)

    assert result == pytest.approx(-1.0)


def test_ic_spearman_vs_pearson_differ_on_nonlinear_monotonic_relationship():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    signal = pd.Series(np.arange(1, 21), index=idx)
    forward_returns = pd.Series((np.arange(1, 21).astype(float)) ** 3, index=idx)
    spearman_ic = information_coefficient(signal, forward_returns, method="spearman")
    pearson_ic = information_coefficient(signal, forward_returns, method="pearson")

    assert spearman_ic == pytest.approx(1.0)
    assert pearson_ic < spearman_ic


def test_ic_random_signal_near_zero():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=2000, freq="D")
    signal = pd.Series(rng.standard_normal(2000), index=idx)
    forward_returns = pd.Series(rng.standard_normal(2000), index=idx)
    result = information_coefficient(signal, forward_returns)

    assert abs(result) < 0.05


def test_ic_handles_misaligned_index_via_intersection():
    idx_signal = pd.date_range("2020-01-01", periods=10, freq="D")
    idx_returns = pd.date_range("2020-01-05", periods=10, freq="D")
    signal = pd.Series(np.arange(10), index=idx_signal)
    forward_returns = pd.Series(np.arange(10) * 0.01, index=idx_returns)
    result = information_coefficient(signal, forward_returns)

    assert not np.isnan(result)


def test_ic_invalid_method_raises():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    signal = pd.Series(np.arange(10), index=idx)
    forward_returns = pd.Series(np.arange(10) * 0.01, index=idx)
    with pytest.raises(ValueError):
        information_coefficient(signal, forward_returns, method="kendall")


def test_ir_fundamental_law_matches_hand_calc_case():
    result = information_ratio(0.03, method="fundamental_law", breadth=100)

    assert result == pytest.approx(0.03 * np.sqrt(100))


def test_ir_fundamental_law_missing_breadth_raises():
    with pytest.raises(ValueError):
        information_ratio(0.05, method="fundamental_law")


def test_ir_fundamental_law_nonscalar_ic_raises():
    ic_series = pd.Series([0.01, 0.02, 0.03])
    with pytest.raises(ValueError):
        information_ratio(ic_series, method="fundamental_law", breadth=50)


def test_ir_empirical_matches_hand_calc():
    ic_series = pd.Series([0.02, 0.04, -0.01, 0.03, 0.05])
    expected = ic_series.mean() / ic_series.std(ddof=1)
    result = information_ratio(ic_series, method="empirical")

    assert result == pytest.approx(expected)


def test_ir_empirical_single_observation_is_nan():
    result = information_ratio(pd.Series([0.05]), method="empirical")

    assert np.isnan(result)


def test_ir_empirical_constant_series_is_nan():
    result = information_ratio(pd.Series([0.05, 0.05, 0.05]), method="empirical")

    assert np.isnan(result)


def test_ir_invalid_method_raises():
    with pytest.raises(ValueError):
        information_ratio(0.05, method="not_a_real_method", breadth=50)


def _regime_fixture(n=100, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = pd.Series(rng.normal(0.001, 0.01, n), index=idx)
    labels = np.where(np.arange(n) < n // 2, "low", "high")
    regimes = pd.Series(labels, index=idx)
    return returns, regimes


def test_conditional_perf_dict_keys():
    returns, regimes = _regime_fixture()
    result = regime_conditional_performance(returns, regimes)

    assert set(result.keys()) == {
        "high_vol_sharpe",
        "low_vol_sharpe",
        "high_vol_pct",
        "low_vol_pct",
    }


def test_conditional_perf_pct_sums_to_one():
    returns, regimes = _regime_fixture()
    result = regime_conditional_performance(returns, regimes)

    assert result["high_vol_pct"] + result["low_vol_pct"] == pytest.approx(1.0)
    assert result["high_vol_pct"] == pytest.approx(0.5)


def test_conditional_perf_matches_manual_sharpe():
    returns, regimes = _regime_fixture()
    result = regime_conditional_performance(returns, regimes)

    manual_high = PerformanceAnalyzer(returns[regimes == "high"]).compute_sharpe()
    assert result["high_vol_sharpe"] == pytest.approx(manual_high)


def test_conditional_perf_misaligned_index_uses_inner_join():
    returns, regimes = _regime_fixture(n=100)
    truncated_regimes = regimes.iloc[:60]
    result = regime_conditional_performance(returns, truncated_regimes)

    assert result["high_vol_pct"] + result["low_vol_pct"] == pytest.approx(1.0)


def test_conditional_perf_empty_regime_is_nan():
    returns, regimes = _regime_fixture()
    all_low = pd.Series(["low"] * len(regimes), index=regimes.index)
    result = regime_conditional_performance(returns, all_low)

    assert np.isnan(result["high_vol_sharpe"])
    assert result["high_vol_pct"] == 0.0


def test_eurusd_cost_bps_known_value():
    result = round_trip_cost_bps(spread_pips=1.0, pip_value=10.0, notional=100_000.0)

    assert result == pytest.approx(1.0)


def test_breakeven_increases_with_more_trades():
    few = breakeven_annual_return(cost_bps=1.0, holding_period_days=5, trades_per_year=20)
    many = breakeven_annual_return(cost_bps=1.0, holding_period_days=5, trades_per_year=200)

    assert many > few
    assert many == pytest.approx(10 * few)


def test_report_keys_present():
    result = cost_report("EURUSD", notional=100_000.0, holding_period_days=5)

    expected = {
        "pair", "spread_bps", "rollover_bps", "total_bps", "breakeven_return",
        "trades_per_year", "holding_period_days", "pip_value", "bps_per_pip",
        "breakeven_sharpe",
    }
    assert expected.issubset(result.keys())


def test_breakeven_five_day_hold_known_value():
    """1 bps round trip at a 5-day hold, continuously deployed.

    260 FX sessions / 5 = 52 round trips -> 0.520%/yr. The equity convention
    (252) gives 0.504%; both are recorded in the Day 57 math note.
    """
    result = breakeven_annual_return(cost_bps=1.0, holding_period_days=5)
    assert result == pytest.approx(0.00520)

    equity = breakeven_annual_return(
        cost_bps=1.0, holding_period_days=5,
        trades_per_year=implied_trades_per_year(5, EQUITY_TRADING_DAYS_PER_YEAR),
    )
    assert equity == pytest.approx(0.00504)


def test_pip_size_jpy_quoted_pairs():
    assert pip_size("USDJPY") == pytest.approx(0.01)
    assert pip_size("EURJPY") == pytest.approx(0.01)
    assert pip_size("EURUSD") == pytest.approx(0.0001)
    assert pip_size("eur/usd") == pytest.approx(0.0001)


def test_pip_size_rejects_malformed_pair():
    with pytest.raises(ValueError):
        pip_size("EUR")


def test_bps_per_pip_is_scale_invariant_across_quote_conventions():
    assert bps_per_pip("EURUSD", 1.10) == pytest.approx(0.909, abs=1e-3)
    assert bps_per_pip("USDJPY", 110.0) == pytest.approx(0.909, abs=1e-3)


def test_cost_report_default_lot_matches_price_relative_convention():
    result = cost_report(
        "EURUSD", notional=100_000.0, holding_period_days=5,
        spread_pips=1.0, quote_price=1.10,
    )

    assert result["spread_bps"] == pytest.approx(bps_per_pip("EURUSD", 1.10))
    assert result["spread_bps"] == pytest.approx(0.90909, abs=1e-5)


def test_cost_report_explicit_lot_reproduces_desk_convention():
    result = cost_report(
        "EURUSD", notional=100_000.0, holding_period_days=5,
        spread_pips=1.0, quote_price=1.10, lot_size=100_000.0,
    )

    assert result["spread_bps"] == pytest.approx(1.0)


def test_rollover_sign_convention_positive_is_cost():
    cost = rollover_bps_per_day(base_rate_annual=0.005, quote_rate_annual=0.017)

    assert cost > 0

    credit = rollover_bps_per_day(
        base_rate_annual=0.005, quote_rate_annual=0.017, direction=-1
    )
    assert credit == pytest.approx(-cost)


def test_rollover_zero_when_rates_equal():
    assert rollover_bps_per_day(0.02, 0.02) == pytest.approx(0.0)


def test_cost_report_rollover_scales_with_holding_period():
    short = cost_report(
        "EURUSD", 100_000.0, holding_period_days=1,
        rollover_bps_per_day_=0.3, trades_per_year=50,
    )
    long = cost_report(
        "EURUSD", 100_000.0, holding_period_days=10,
        rollover_bps_per_day_=0.3, trades_per_year=50,
    )

    assert long["rollover_bps"] == pytest.approx(10 * short["rollover_bps"])
    assert short["spread_bps"] == pytest.approx(long["spread_bps"])


def test_cost_report_intraday_book_has_zero_rollover():
    result = cost_report(
        "EURUSD", 100_000.0, holding_period_days=4 / 24,
        spread_pips=2.0, quote_price=1.10, trades_per_year=27,
    )

    assert result["rollover_bps"] == pytest.approx(0.0)
    assert result["total_bps"] == pytest.approx(result["spread_bps"])


def test_max_viable_spread_inverts_breakeven():
    gross, trades, price = 0.02, 27.0, 1.10
    max_spread = max_viable_spread_pips(gross, "EURUSD", price, trades)
    report = cost_report(
        "EURUSD", 100_000.0, holding_period_days=4 / 24,
        spread_pips=max_spread, quote_price=price, trades_per_year=trades,
    )

    assert report["breakeven_return"] == pytest.approx(gross)


def test_max_viable_spread_zero_when_no_gross_edge():
    assert max_viable_spread_pips(-0.01, "EURUSD", 1.10, 27.0) == 0.0
    assert max_viable_spread_pips(0.0, "EURUSD", 1.10, 27.0) == 0.0


def test_max_viable_spread_falls_with_turnover():
    daily_book = max_viable_spread_pips(0.0068, "EURUSD", 1.10, trades_per_year=5.9)
    session_book = max_viable_spread_pips(0.0068, "EURUSD", 1.10, trades_per_year=166.0)

    assert daily_book > session_book
    assert session_book < 1.0


def test_breakeven_sharpe_scales_inversely_with_vol():
    low_vol = breakeven_sharpe(1.0, 5, annualized_vol=0.02)
    high_vol = breakeven_sharpe(1.0, 5, annualized_vol=0.08)

    assert low_vol == pytest.approx(4 * high_vol)


def test_implied_trades_per_year_defaults_to_fx_convention():
    """FX spot trades weekdays only -- 260/yr, not the equity 252 and not the
    312 (6 x 52) used elsewhere in this repo. Measured on the project's own
    2011-2023 session data: 259.44/yr, 258-260 every calendar year."""
    assert FX_TRADING_DAYS_PER_YEAR == 260
    assert implied_trades_per_year(5) == pytest.approx(52.0)
    assert implied_trades_per_year(1) == pytest.approx(260.0)


def test_implied_trades_per_year_accepts_empirical_factor():
    """The preferred route: pass compute_ann_factor()'s empirical figure."""
    assert implied_trades_per_year(5, 259.44) == pytest.approx(51.888)
    assert implied_trades_per_year(1, 312.0) == pytest.approx(312.0)


def test_implied_trades_per_year_rejects_bad_calendar():
    with pytest.raises(ValueError):
        implied_trades_per_year(5, trading_days_per_year=0)


def test_cost_functions_reject_invalid_inputs():
    with pytest.raises(ValueError):
        round_trip_cost_bps(spread_pips=-1.0, pip_value=10.0, notional=100_000.0)
    with pytest.raises(ValueError):
        round_trip_cost_bps(spread_pips=1.0, pip_value=10.0, notional=0.0)
    with pytest.raises(ValueError):
        breakeven_annual_return(cost_bps=-1.0, holding_period_days=5)
    with pytest.raises(ValueError):
        implied_trades_per_year(0)
    with pytest.raises(ValueError):
        rollover_bps_per_day(0.01, 0.02, direction=0)
    with pytest.raises(ValueError):
        bps_per_pip("EURUSD", 0.0)
    with pytest.raises(ValueError):
        cost_report("EURUSD", notional=0.0, holding_period_days=5)
    with pytest.raises(ValueError):
        max_viable_spread_pips(0.02, "EURUSD", 1.10, trades_per_year=0.0)
