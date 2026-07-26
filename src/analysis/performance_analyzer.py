from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import chi2, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass
class PerformanceReport:
    """Strategy performance metrics produced by PerformanceAnalyzer.

    Attributes
    ----------
    sharpe_ratio : float
        Annualized Sharpe ratio of the strategy returns.
    sortino_ratio : float
        Annualized Sortino ratio (downside-deviation adjusted).
    max_drawdown : float
        Maximum peak-to-trough drawdown as a negative decimal (e.g. -0.15).
    win_rate : float
        Fraction of trades that were profitable, in [0, 1].
    profit_factor : float
        Gross profit divided by gross loss across all trades.
    calmar_ratio : float
        Annualized return divided by absolute max drawdown.
    t_stat : float
        t-statistic testing whether mean return is significantly > 0.
    deflated_sharpe : float
        Probability that the Sharpe ratio is a true positive after correcting
        for multiple testing and non-normality (Bailey & Lopez de Prado, 2014).
    n_trades : int
        Total number of closed trades in the sample.
    annualized_return : float
        Geometric annualized return of the strategy.
    annualized_vol : float
        Annualized standard deviation of daily returns.
    skewness : float
        Sample skewness of the return distribution.
    excess_kurtosis : float
        Sample excess kurtosis of the return distribution (normal = 0).
    jb_stat : float
        Jarque-Bera test statistic for normality of returns.
    jb_p_value : float
        p-value of the Jarque-Bera test.
    lb_stat : float
        Ljung-Box test statistic for autocorrelation in returns.
    lb_p_value : float
        p-value of the Ljung-Box test.
    tracking_error : float
        Annualized tracking error relative to a benchmark. NaN when no
        benchmark is supplied to ``run_report``.
    metadata : dict
        Arbitrary extra fields (pair, timeframe, date range, etc.).
    """

    sharpe_ratio: float = float("nan")
    sortino_ratio: float = float("nan")
    max_drawdown: float = float("nan")
    win_rate: float = float("nan")
    profit_factor: float = float("nan")
    calmar_ratio: float = float("nan")
    t_stat: float = float("nan")
    deflated_sharpe: float = float("nan")
    n_trades: int = 0
    annualized_return: float = float("nan")
    annualized_vol: float = float("nan")
    skewness: float = float("nan")
    excess_kurtosis: float = float("nan")
    jb_stat: float = float("nan")
    jb_p_value: float = float("nan")
    lb_stat: float = float("nan")
    lb_p_value: float = float("nan")
    tracking_error: float = float("nan")
    metadata: dict = field(default_factory=dict)


class PerformanceAnalyzer:
    """Compute and report performance metrics for a trading strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily (or bar-level) strategy returns as decimals, indexed by datetime.
    trades : pd.DataFrame, optional
        Trade log with at minimum columns: ['entry_time', 'exit_time', 'pnl'].
        Additional columns (pair, direction, size) are preserved in reports.
    risk_free_rate : float, optional
        Daily risk-free rate as a decimal. Defaults to 0.0.
    """

    def __init__(
        self,
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.returns = returns
        self.trades = trades
        self.risk_free_rate = risk_free_rate

    def compute_ann_factor(self) -> float:
        """Compute the empirical annualization factor from the returns index.

        Returns
        -------
        float
            Empirical annualization factor (observations per year).

        Raises
        ------
        ValueError
            If the returns index spans zero or negative calendar time.
        """
        n_obs = len(self.returns)
        start = self.returns.index.min()
        end = self.returns.index.max()
        years_spanned = (end - start).days / 365.25

        if years_spanned <= 0:
            raise ValueError("Returns index must span a positive amount of time.")

        return n_obs / years_spanned

    def compute_sharpe(self) -> float:
        """Compute the annualized Sharpe ratio.

        Returns
        -------
        float
            Annualized Sharpe ratio. Returns NaN if standard deviation is zero.
        """
        ann_factor = self.compute_ann_factor()
        std = self.returns.std()
        if std < 1e-10:
            return float("nan")
        return (self.returns.mean() - self.risk_free_rate) / std * np.sqrt(ann_factor)

    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_trials: int,
        n_obs: int,
        skewness: float,
        kurtosis: float,
    ) -> float:
        """Compute the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

        Parameters
        ----------
        observed_sharpe : float
            In-sample annualized Sharpe ratio of the best strategy.
        n_trials : int
            Number of strategies or parameter sets evaluated.
        n_obs : int
            Number of return observations used to compute the Sharpe.
        skewness : float
            Sample skewness of the return distribution.
        kurtosis : float
            Sample excess kurtosis of the return distribution (normal = 0).

        Returns
        -------
        float
            Probability in [0, 1] that the observed Sharpe is a true positive.
            Returns NaN if the variance term is non-positive.
        """
        sr_period = observed_sharpe / np.sqrt(self.compute_ann_factor())

        V = (1 - skewness * sr_period + ((kurtosis + 2) / 4) * sr_period**2) / (n_obs - 1)
        if V <= 0:
            return np.nan

        se = np.sqrt(V)

        if n_trials <= 1:
            SR_star = 0.0
        else:
            gamma = 0.5772
            SR_star = se * (
                (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
                + gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
            )

        z = (sr_period - SR_star) / se

        return float(stats.norm.cdf(z))

    def compute_sortino(self) -> float:
        """Compute the annualized Sortino ratio using downside deviation only.

        Returns
        -------
        float
            Annualized Sortino ratio. Returns NaN if downside deviation is zero
            or if there are no negative excess return observations.
        """
        if self.returns is None or len(self.returns) == 0:
            return float("nan")

        excess_returns = self.returns - self.risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float("nan")

        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        if downside_deviation == 0:
            return float("nan")

        ann_factor = self.compute_ann_factor()
        return float(excess_returns.mean() / downside_deviation * np.sqrt(ann_factor))

    def compute_max_drawdown(self) -> dict:
        """Compute the maximum peak-to-trough drawdown.

        Returns
        -------
        dict
            Keys:

            - ``value`` (*float*): max drawdown as a negative decimal
              (e.g. -0.23 means -23%). Returns 0.0 if the equity curve
              never declines.
            - ``duration_days`` (*int*): calendar days from peak to trough.
            - ``start_date``: index value of the peak.
            - ``end_date``: index value of the trough.
        """
        equity_curve = (1 + self.returns).cumprod()
        rolling_peak = equity_curve.cummax()
        drawdown = (equity_curve - rolling_peak) / rolling_peak
        min_dd = drawdown.min()
        end_date = drawdown.idxmin()
        start_date = equity_curve[:end_date].idxmax()

        return {
            "value": min_dd,
            "duration_days": (end_date - start_date).days,
            "start_date": start_date,
            "end_date": end_date,
        }

    def compute_win_rate(self) -> float:
        """Compute the fraction of trades with positive PnL.

        Returns
        -------
        float
            Win rate in [0, 1]. Returns NaN if trades is None or empty.

        Raises
        ------
        ValueError
            If self.trades does not contain a 'pnl' column.
        """
        if self.trades is None or self.trades.empty:
            return float("nan")

        if "pnl" not in self.trades.columns:
            raise ValueError("self.trades must contain a 'pnl' column.")

        return float((self.trades["pnl"] > 0).mean())

    def compute_profit_factor(self) -> float:
        """Compute gross profit divided by gross loss across all trades.

        Returns
        -------
        float
            Profit factor. Returns NaN if trades is None, empty, or if there
            are no losing trades (gross loss = 0).

        Raises
        ------
        ValueError
            If self.trades does not contain a 'pnl' column.
        """
        if self.trades is None or self.trades.empty:
            return float("nan")

        if "pnl" not in self.trades.columns:
            raise ValueError("self.trades must contain a 'pnl' column.")

        pnl = self.trades["pnl"]
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl < 0].sum())

        if gross_loss == 0:
            return float("nan")

        return float(gross_profit / gross_loss)

    def compute_calmar(self) -> float:
        """Compute the Calmar ratio (annualized return / absolute max drawdown).

        Returns
        -------
        float
            Calmar ratio. Returns NaN if max drawdown is zero or returns
            is empty.
        """
        if self.returns is None or len(self.returns) == 0:
            return float("nan")

        ann_factor = self.compute_ann_factor()
        annualized_return = (1 + self.returns.mean()) ** ann_factor - 1
        max_drawdown = abs(self.compute_max_drawdown()["value"])

        if max_drawdown == 0:
            return float("nan")

        return float(annualized_return / max_drawdown)

    def compute_t_stat(self) -> float:
        """Compute the t-statistic testing whether mean return differs from zero.

        Returns
        -------
        float
            t-statistic. Values > ~2.0 suggest statistical significance
            at the 5% level for large samples. Returns NaN if returns is
            empty or has zero standard deviation.
        """
        if self.returns is None or len(self.returns) == 0:
            return float("nan")

        n = len(self.returns)
        std = self.returns.std()

        if std < 1e-10:
            return float("nan")

        return float(self.returns.mean() / (std / np.sqrt(n)))

    def jarque_bera_test(self, alpha: float = 0.05) -> dict:
        """Run a Jarque-Bera normality test on the return distribution.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the test. Defaults to 0.05.

        Returns
        -------
        dict
            Keys:

            - ``jb_stat`` (*float*): Jarque-Bera test statistic.
            - ``p_value`` (*float*): p-value of the test.
            - ``skewness`` (*float*): sample skewness of returns.
            - ``excess_kurtosis`` (*float*): sample excess kurtosis (normal = 0).
            - ``critical_value`` (*float*): chi-squared critical value at ``alpha``.
            - ``reject_normality`` (*bool*): True if p_value < alpha.
        """
        returns = np.asarray(self.returns)
        returns = returns[~np.isnan(returns)]

        jb_stat, p_value = jarque_bera(returns)

        mean = np.mean(returns)
        std = np.std(returns, ddof=0)
        skewness = float(np.mean(((returns - mean) / std) ** 3))
        kurtosis = float(np.mean(((returns - mean) / std) ** 4))
        excess_kurtosis = kurtosis - 3.0
        critical_value = float(chi2.ppf(1 - alpha, df=2))

        return {
            "jb_stat": float(jb_stat),
            "p_value": float(p_value),
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "critical_value": critical_value,
            "reject_normality": bool(p_value < alpha),
        }

    def ljung_box_test(self, lags: int = 10, alpha: float = 0.05) -> dict:
        """Run a Ljung-Box test for autocorrelation in returns.

        Parameters
        ----------
        lags : int, optional
            Number of lags to include in the test. Defaults to 10.
        alpha : float, optional
            Significance level for the test. Defaults to 0.05.

        Returns
        -------
        dict
            Keys:

            - ``lb_stat`` (*float*): Ljung-Box Q-statistic.
            - ``p_value`` (*float*): p-value of the test.
            - ``critical_value`` (*float*): chi-squared critical value at ``alpha``.
            - ``reject_white_noise`` (*bool*): True if p_value < alpha,
              indicating significant autocorrelation.
        """
        returns = np.asarray(self.returns)
        returns = returns[~np.isnan(returns)]

        result = acorr_ljungbox(returns, lags=[lags], return_df=True)
        lb_stat = float(result["lb_stat"].iloc[0])
        p_value = float(result["lb_pvalue"].iloc[0])
        critical_value = float(chi2.ppf(1 - alpha, df=lags))

        return {
            "lb_stat": lb_stat,
            "p_value": p_value,
            "critical_value": critical_value,
            "reject_white_noise": bool(p_value < alpha),
        }

    def tracking_error(
        self,
        benchmark_returns: pd.Series,
        ann_factor: Optional[float] = None,
    ) -> float:
        """Compute annualized tracking error relative to a benchmark.

        Parameters
        ----------
        benchmark_returns : pd.Series
            Benchmark daily returns as decimals, indexed by datetime. Only
            dates present in both series are used.
        ann_factor : float, optional
            Annualization factor. If None, inferred from self.returns via
            ``compute_ann_factor``.

        Returns
        -------
        float
            Annualized tracking error as a decimal.
        """
        if ann_factor is None:
            ann_factor = self.compute_ann_factor()

        strategy_returns, benchmark_returns = self.returns.align(
            benchmark_returns, join="inner"
        )
        active_returns = strategy_returns - benchmark_returns

        return float(np.std(active_returns, ddof=1) * np.sqrt(ann_factor))

    def run_report(
        self,
        n_trials: int = 1,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> PerformanceReport:
        """Compute all metrics and return a populated PerformanceReport.

        Parameters
        ----------
        n_trials : int, optional
            Number of strategies or parameter sets evaluated before selecting
            this one. Used by ``deflated_sharpe_ratio`` to correct for
            selection bias. Defaults to 1 (no multi-testing adjustment).
            Set this to the actual number of configurations tested when doing
            parameter searches or walk-forward optimization.
        benchmark_returns : pd.Series, optional
            Benchmark daily returns for computing tracking error. If None,
            ``tracking_error`` in the report is NaN.

        Returns
        -------
        PerformanceReport
            Fully populated report object. Fields default to NaN if the
            corresponding method fails or data is unavailable (e.g. win_rate
            and profit_factor are NaN when no trade log is provided).
        """
        ann_factor = self.compute_ann_factor()
        annualized_return = (1 + self.returns.mean()) ** ann_factor - 1
        annualized_vol = self.returns.std() * np.sqrt(ann_factor)

        jb = self.jarque_bera_test()
        lb = self.ljung_box_test()
        sharpe = self.compute_sharpe()

        dsr = self.deflated_sharpe_ratio(
            observed_sharpe=sharpe,
            n_trials=n_trials,
            n_obs=len(self.returns),
            skewness=jb["skewness"],
            kurtosis=jb["excess_kurtosis"],
        )

        te = (
            self.tracking_error(benchmark_returns)
            if benchmark_returns is not None
            else float("nan")
        )

        return PerformanceReport(
            sharpe_ratio=sharpe,
            sortino_ratio=self.compute_sortino(),
            max_drawdown=self.compute_max_drawdown()["value"],
            win_rate=self.compute_win_rate(),
            profit_factor=self.compute_profit_factor(),
            calmar_ratio=self.compute_calmar(),
            t_stat=self.compute_t_stat(),
            deflated_sharpe=dsr,
            n_trades=0 if self.trades is None else len(self.trades),
            annualized_return=annualized_return,
            annualized_vol=annualized_vol,
            skewness=jb["skewness"],
            excess_kurtosis=jb["excess_kurtosis"],
            jb_stat=jb["jb_stat"],
            jb_p_value=jb["p_value"],
            lb_stat=lb["lb_stat"],
            lb_p_value=lb["p_value"],
            tracking_error=te,
            metadata={
                "start_date": self.returns.index.min(),
                "end_date": self.returns.index.max(),
            },
        )


def information_coefficient(
    signal: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> float:
    """Compute the Information Coefficient between a signal and forward returns.

    Parameters
    ----------
    signal : pd.Series
        Signal values at time t. Must already be aligned with
        ``forward_returns`` (no lookahead) by the caller.
    forward_returns : pd.Series
        Realized returns over the period following each signal observation.
    method : str, optional
        "spearman" or "pearson". Defaults to "spearman", consistent with
        the rank-based, distribution-agnostic approach used throughout
        this framework's significance testing.

    Returns
    -------
    float
        IC value.
    """
    aligned_signal, aligned_forward_returns = signal.align(forward_returns, join="inner")

    if method == "spearman":
        ic, _ = stats.spearmanr(aligned_signal.to_numpy(), aligned_forward_returns.to_numpy())
    elif method == "pearson":
        ic, _ = stats.pearsonr(aligned_signal.to_numpy(), aligned_forward_returns.to_numpy())
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'spearman' or 'pearson'.")

    return float(ic)


def information_ratio(
    ic_values,
    method: str = "fundamental_law",
    breadth: Optional[int] = None,
) -> float:
    """Compute Information Ratio from IC, either via the Fundamental Law or
    empirically from a time series of realized ICs.

    Parameters
    ----------
    ic_values : float or array-like
        A single IC estimate (used with method="fundamental_law") or a
        time series of period-by-period IC values (used with
        method="empirical").
    method : str, optional
        "fundamental_law": computes ``IC * sqrt(breadth)``. Requires
        ``breadth``. Assumes constant IC across independent bets.
        "empirical": computes ``mean(ic_values) / std(ic_values)`` directly
        from a realized IC time series. Makes no independence or
        constant-IC assumption, but requires an actual IC series rather
        than a single pooled estimate.
    breadth : int, optional
        Number of independent bets per year. Required if
        method="fundamental_law", ignored otherwise.

    Returns
    -------
    float
        IR value.
    """
    if method == "fundamental_law":
        if breadth is None:
            raise ValueError("breadth is required when method='fundamental_law'.")
        if not np.isscalar(ic_values):
            raise ValueError(
                "ic_values must be a single scalar IC estimate when "
                "method='fundamental_law'."
            )
        return float(ic_values) * np.sqrt(breadth)

    elif method == "empirical":
        ic_array = np.asarray(ic_values, dtype=float)
        ic_array = ic_array[~np.isnan(ic_array)]

        if ic_array.size < 2:
            return float("nan")

        ic_std = ic_array.std(ddof=1)
        if ic_std < 1e-10:
            return float("nan")

        return float(ic_array.mean() / ic_std)

    else:
        raise ValueError(f"Unknown method: {method}. Must be 'fundamental_law' or 'empirical'.")


def _sharpe_or_nan(returns: pd.Series) -> float:
    """Sharpe of a return subset, NaN instead of raising when the subset is
    too degenerate to annualize (e.g. spans < 1 day, or is empty).
    """
    if len(returns) < 2:
        return float("nan")
    try:
        return PerformanceAnalyzer(returns).compute_sharpe()
    except ValueError:
        return float("nan")


def regime_conditional_performance(returns: pd.Series, regimes: pd.Series) -> dict:
    """Split strategy returns by volatility regime and compare Sharpe ratios
    conditional on regime.

    Parameters
    ----------
    returns : pd.Series
        Strategy daily returns as decimals, indexed by datetime.
    regimes : pd.Series
        Regime label per day, e.g. output of
        `src.features.garch.classify_vol_regime` (values "high"/"low").
        Aligned to `returns` via an inner join on the index -- days present
        in only one of the two series are dropped before splitting.

    Returns
    -------
    dict with keys:
        high_vol_sharpe : float -- annualized Sharpe on days labeled "high".
                                    NaN if fewer than 2 such days after
                                    alignment.
        low_vol_sharpe  : float -- annualized Sharpe on days labeled "low".
                                    Same NaN condition.
        high_vol_pct    : float -- fraction of aligned days labeled "high".
        low_vol_pct     : float -- fraction of aligned days labeled "low".
    """
    aligned_returns, aligned_regimes = returns.align(regimes, join="inner")

    high_mask = aligned_regimes == "high"
    low_mask = aligned_regimes == "low"
    n = len(aligned_regimes)

    return {
        "high_vol_sharpe": _sharpe_or_nan(aligned_returns[high_mask]),
        "low_vol_sharpe": _sharpe_or_nan(aligned_returns[low_mask]),
        "high_vol_pct": float(high_mask.sum() / n) if n > 0 else float("nan"),
        "low_vol_pct": float(low_mask.sum() / n) if n > 0 else float("nan"),
    }


# Nominal FX trading days per year, used ONLY as a fallback when a realized
# trade count is unavailable.
#
# Which number is right is not a matter of taste, and this repo currently
# carries two wrong ones:
#   252  -- US *equity* convention (260 weekdays minus ~8 exchange holidays).
#           FX spot does not close for most US equity holidays, so 252
#           understates it.
#   312  -- used in portfolio.markowitz_sharpe's default, day34/43/46 scripts,
#           and the momentum/mean_reversion docstrings. 312 = 6 x 52, which
#           assumes a six-session week. FX spot runs Sunday 17:00 ET to Friday
#           17:00 ET, and the Sunday open carries Monday's value date, so the
#           week is five sessions, not six.
#   260  -- weekdays per year, which is what FX spot actually trades.
#
# Measured on this project's own data (research/applied_analysis/
# _overshoot_cache, 2011-2023): 3,369 session days over 12.986 years = 259.44
# per year, 258-260 in every single calendar year, zero weekend bars. 260 is
# correct to within 0.2%; 312 is 20% high and inflates any Sharpe annualized
# with it by sqrt(312/259.4) = 1.10.
#
# Prefer the empirical route regardless: PerformanceAnalyzer.compute_ann_factor
# derives observations-per-year from the index and needs no constant at all.
# That is the convention the rest of this module uses.
FX_TRADING_DAYS_PER_YEAR = 260
EQUITY_TRADING_DAYS_PER_YEAR = 252

# Backwards-compatible alias. Points at the FX figure, since this is an FX repo.
TRADING_DAYS_PER_YEAR = FX_TRADING_DAYS_PER_YEAR
DEFAULT_DAY_COUNT = 360
_JPY_PIP = 0.01
_STANDARD_PIP = 0.0001

def pip_size(pair: str) -> float:
    """Price increment of one pip for an FX pair.

    Parameters
    ----------
    pair : str
        Six-character pair code, e.g. "EURUSD" or "USDJPY". Case-insensitive;
        an embedded separator ("EUR/USD") is tolerated.

    Returns
    -------
    float
        0.01 for JPY-quoted pairs, 0.0001 otherwise.

    Raises
    ------
    ValueError
        If `pair` does not resolve to a six-character currency pair.
    """
    cleaned = pair.replace("/", "").replace("_", "").replace("-", "").upper()
    if len(cleaned) != 6:
        raise ValueError(
            f"Expected a six-character currency pair, got {pair!r}."
        )
    return _JPY_PIP if cleaned[3:] == "JPY" else _STANDARD_PIP


def pip_value_from_price(
    pair: str,
    quote_price: float,
    lot_size: float = 100_000.0,
) -> float:
    """Quote-currency value of one pip on a position of `lot_size` base units.

    Parameters
    ----------
    pair : str
        Six-character pair code.
    quote_price : float
        Current quote (units of quote currency per unit of base currency).
        Used only to validate positivity; pip value in the quote currency does
        not depend on the level of the quote.
    lot_size : float, optional
        Position size in units of the base currency. Defaults to one standard
        lot (100,000).

    Returns
    -------
    float
        Value of one pip, in units of the quote currency.

    Raises
    ------
    ValueError
        If `quote_price` or `lot_size` is not strictly positive.
    """
    if quote_price <= 0:
        raise ValueError(f"quote_price must be positive, got {quote_price}.")
    if lot_size <= 0:
        raise ValueError(f"lot_size must be positive, got {lot_size}.")

    return pip_size(pair) * lot_size


def bps_per_pip(pair: str, quote_price: float) -> float:
    """Basis points of notional represented by one pip, price-relative.

    Parameters
    ----------
    pair : str
        Six-character pair code.
    quote_price : float
        Current quote.

    Returns
    -------
    float
        Basis points of notional per pip.

    Raises
    ------
    ValueError
        If `quote_price` is not strictly positive.
    """
    if quote_price <= 0:
        raise ValueError(f"quote_price must be positive, got {quote_price}.")

    return pip_size(pair) / quote_price * 1e4


def round_trip_cost_bps(
    spread_pips: float,
    pip_value: float,
    notional: float,
) -> float:
    """Round-trip transaction cost in basis points of notional.

    Parameters
    ----------
    spread_pips : float
        Round-trip spread in pips. Must be non-negative.
    pip_value : float
        Currency value of one pip on the position.
    notional : float
        Position notional, in the same currency as `pip_value`.

    Returns
    -------
    float
        Round-trip cost in basis points of notional.

    Raises
    ------
    ValueError
        If `spread_pips` is negative, or `notional` is not strictly positive.
    """
    if spread_pips < 0:
        raise ValueError(f"spread_pips must be non-negative, got {spread_pips}.")
    if notional <= 0:
        raise ValueError(f"notional must be positive, got {notional}.")

    return spread_pips * pip_value / notional * 1e4


def rollover_bps_per_day(
    base_rate_annual: float,
    quote_rate_annual: float,
    direction: int = 1,
    day_count: int = DEFAULT_DAY_COUNT,
) -> float:
    """Daily rollover as a **cost** in basis points of notional.

    Parameters
    ----------
    base_rate_annual : float
        Annualized deposit rate of the base currency, as a decimal (0.0425 for
        4.25%, not 4.25).
    quote_rate_annual : float
        Annualized deposit rate of the quote currency, as a decimal.
    direction : int, optional
        +1 for long base / short quote, -1 for the reverse. Defaults to +1.
    day_count : int, optional
        Day-count basis. Defaults to 360.

    Returns
    -------
    float
        Rollover cost in basis points of notional per calendar day held.
        Positive is a cost, negative is a credit.

    Raises
    ------
    ValueError
        If `direction` is not +1 or -1, or `day_count` is not positive.
    """
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {direction}.")
    if day_count <= 0:
        raise ValueError(f"day_count must be positive, got {day_count}.")

    return direction * (quote_rate_annual - base_rate_annual) / day_count * 1e4


def implied_trades_per_year(
    holding_period_days: float,
    trading_days_per_year: float = FX_TRADING_DAYS_PER_YEAR,
) -> float:
    """Round trips per year implied by a holding period under continuous
    deployment.

    This is an **upper bound** on trade count, not a measurement, and the gap
    is usually large. A strategy holding 5 days but in the market only 40% of
    the time trades far less than 260/5 times a year. The intraday overshoot
    book holds 0.167 days but trades 24.7 times a year, not 1,560. Where a
    realized trade count exists, pass it explicitly to
    `breakeven_annual_return` rather than relying on this.

    Parameters
    ----------
    holding_period_days : float
        Average holding period in trading days. Fractional values are valid --
        an intraday book holding 09:00 to 13:00 has a holding period of ~0.167.
    trading_days_per_year : float, optional
        Sessions per year. Defaults to the FX weekday count (260). Pass
        `PerformanceAnalyzer.compute_ann_factor()` to use the empirical figure
        derived from the actual return index, which is preferred -- see the
        note on `FX_TRADING_DAYS_PER_YEAR` for why the repo's other constants
        (252 and 312) are both wrong for this data.

    Returns
    -------
    float
        Implied round trips per year.

    Raises
    ------
    ValueError
        If `holding_period_days` or `trading_days_per_year` is not strictly
        positive.
    """
    if holding_period_days <= 0:
        raise ValueError(
            f"holding_period_days must be positive, got {holding_period_days}."
        )
    if trading_days_per_year <= 0:
        raise ValueError(
            f"trading_days_per_year must be positive, got {trading_days_per_year}."
        )

    return trading_days_per_year / holding_period_days


def breakeven_annual_return(
    cost_bps: float,
    holding_period_days: float,
    trades_per_year: Optional[float] = None,
) -> float:
    """Annualized gross return required to cover transaction costs.

    Parameters
    ----------
    cost_bps : float
        Total round-trip cost in basis points of notional.
    holding_period_days : float
        Average holding period in trading days. Used to imply trade count when
        `trades_per_year` is not supplied.
    trades_per_year : float, optional
        Realized round trips per year. Overrides the implied count when given.
        Prefer this whenever a realized trade count is available.

    Returns
    -------
    float
        Breakeven annualized return as a decimal (0.00504 for 0.504%).

    Raises
    ------
    ValueError
        If `cost_bps` is negative, `trades_per_year` is negative, or
        `holding_period_days` is not positive and no `trades_per_year` is given.
    """
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be non-negative, got {cost_bps}.")

    if trades_per_year is None:
        n_trades = implied_trades_per_year(holding_period_days)
    else:
        if trades_per_year < 0:
            raise ValueError(
                f"trades_per_year must be non-negative, got {trades_per_year}."
            )
        n_trades = float(trades_per_year)

    return n_trades * cost_bps / 1e4


def breakeven_sharpe(
    cost_bps: float,
    holding_period_days: float,
    annualized_vol: float,
    trades_per_year: Optional[float] = None,
) -> float:
    """Gross Sharpe ratio required to break even after costs.

    Parameters
    ----------
    cost_bps : float
        Total round-trip cost in basis points of notional.
    holding_period_days : float
        Average holding period in trading days.
    annualized_vol : float
        Annualized standard deviation of strategy returns, as a decimal.
    trades_per_year : float, optional
        Realized round trips per year. Overrides the implied count.

    Returns
    -------
    float
        Breakeven Sharpe ratio.

    Raises
    ------
    ValueError
        If `annualized_vol` is not strictly positive, or any input fails
        `breakeven_annual_return`'s validation.
    """
    if annualized_vol <= 0:
        raise ValueError(
            f"annualized_vol must be positive, got {annualized_vol}."
        )

    cost = breakeven_annual_return(cost_bps, holding_period_days, trades_per_year)
    return cost / annualized_vol


def max_viable_spread_pips(
    gross_annual_return: float,
    pair: str,
    quote_price: float,
    trades_per_year: float,
    rollover_bps_per_day_: float = 0.0,
    holding_period_days: float = 0.0,
) -> float:
    """Round-trip spread at which a strategy's gross edge is exactly consumed.

    Parameters
    ----------
    gross_annual_return : float
        Realized gross annualized return, as a decimal.
    pair : str
        Six-character pair code, for the pip-size lookup.
    quote_price : float
        Representative quote over the sample, for the pip-to-bps conversion.
    trades_per_year : float
        Realized round trips per year. Must be strictly positive.
    rollover_bps_per_day_ : float, optional
        Rollover cost in bps per day held, per `rollover_bps_per_day`.
        Defaults to 0.0, which is exact for a book that is flat overnight.
    holding_period_days : float, optional
        Average holding period, used only to scale rollover. Defaults to 0.0.

    Returns
    -------
    float
        Maximum viable round-trip spread in pips. Returns 0.0 when the gross
        edge is already non-positive net of rollover -- no spread is viable,
        including zero.

    Raises
    ------
    ValueError
        If `trades_per_year` is not strictly positive.
    """
    if trades_per_year <= 0:
        raise ValueError(
            f"trades_per_year must be positive, got {trades_per_year}."
        )

    rollover_drag = (
        trades_per_year * rollover_bps_per_day_ * holding_period_days / 1e4
    )
    net_of_rollover = gross_annual_return - rollover_drag

    if net_of_rollover <= 0:
        return 0.0

    bpp = bps_per_pip(pair, quote_price)
    return net_of_rollover * 1e4 / (trades_per_year * bpp)


def cost_report(
    pair: str,
    notional: float,
    holding_period_days: float,
    spread_pips: float = 1.0,
    quote_price: float = 1.0,
    lot_size: Optional[float] = None,
    trades_per_year: Optional[float] = None,
    rollover_bps_per_day_: float = 0.0,
    annualized_vol: Optional[float] = None,
) -> dict:
    """Full transaction-cost breakdown for one pair under one set of
    assumptions.

    Parameters
    ----------
    pair : str
        Six-character pair code.
    notional : float
        Position notional. Interpreted in the quote currency unless `lot_size`
        is supplied.
    holding_period_days : float
        Average holding period in trading days. Fractional values are valid.
    spread_pips : float, optional
        Assumed round-trip spread in pips. Defaults to 1.0.
    quote_price : float, optional
        Representative quote over the sample. Defaults to 1.0.
    lot_size : float, optional
        Position size in base-currency units. When omitted, derived as
        `notional / quote_price` so that pip value and notional share a
        currency.
    trades_per_year : float, optional
        Realized round trips per year. Falls back to `252 / holding_period_days`.
    rollover_bps_per_day_ : float, optional
        Rollover cost in bps/day per `rollover_bps_per_day`. Positive is a cost.
        Defaults to 0.0, exact for a book flat overnight.
    annualized_vol : float, optional
        Annualized strategy vol, as a decimal. When supplied, the report
        includes `breakeven_sharpe`; otherwise that key is NaN.

    Returns
    -------
    dict with keys:
        pair                : str   -- normalized pair code.
        spread_bps          : float -- round-trip spread cost, bps of notional.
        rollover_bps        : float -- rollover over the full holding period,
                                       bps of notional. Positive is a cost.
        total_bps           : float -- spread_bps + rollover_bps, per round trip.
        breakeven_return    : float -- annualized gross return needed to cover
                                       total_bps at the given trade count.
        trades_per_year     : float -- trade count used, realized or implied.
        holding_period_days : float -- echo of the input.
        pip_value           : float -- quote-currency value of one pip.
        bps_per_pip         : float -- price-relative pip-to-bps conversion.
        breakeven_sharpe    : float -- breakeven_return / annualized_vol, or
                                       NaN when no vol was supplied.

    Raises
    ------
    ValueError
        If `notional` or `quote_price` is not positive, or `spread_pips` is
        negative, or `holding_period_days` is not positive and no
        `trades_per_year` was given.
    """
    if notional <= 0:
        raise ValueError(f"notional must be positive, got {notional}.")
    if quote_price <= 0:
        raise ValueError(f"quote_price must be positive, got {quote_price}.")

    normalized = pair.replace("/", "").replace("_", "").replace("-", "").upper()

    effective_lot = notional / quote_price if lot_size is None else lot_size
    pv = pip_value_from_price(normalized, quote_price, effective_lot)

    spread_bps = round_trip_cost_bps(spread_pips, pv, notional)
    rollover_bps = rollover_bps_per_day_ * holding_period_days
    total_bps = spread_bps + rollover_bps

    n_trades = (
        implied_trades_per_year(holding_period_days)
        if trades_per_year is None
        else float(trades_per_year)
    )

    breakeven = breakeven_annual_return(
        max(total_bps, 0.0), holding_period_days, n_trades
    )

    be_sharpe = float("nan")
    if annualized_vol is not None and annualized_vol > 0:
        be_sharpe = breakeven / annualized_vol

    return {
        "pair": normalized,
        "spread_bps": float(spread_bps),
        "rollover_bps": float(rollover_bps),
        "total_bps": float(total_bps),
        "breakeven_return": float(breakeven),
        "trades_per_year": float(n_trades),
        "holding_period_days": float(holding_period_days),
        "pip_value": float(pv),
        "bps_per_pip": bps_per_pip(normalized, quote_price),
        "breakeven_sharpe": float(be_sharpe),
    }
