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