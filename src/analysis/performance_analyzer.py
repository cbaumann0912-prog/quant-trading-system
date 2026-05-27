"""
PerformanceAnalyzer: computes and reports standard performance metrics
for a trading strategy given a returns series and trade log.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PerformanceReport:
    """Container for all computed performance metrics.

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
    calmar_ratio : float
        Annualized return divided by absolute max drawdown.
    t_stat : float
        t-statistic testing whether mean return is significantly > 0.
    n_trades : int
        Total number of closed trades in the sample.
    annualized_return : float
        Geometric annualized return of the strategy.
    annualized_vol : float
        Annualized standard deviation of daily returns.
    metadata : dict
        Arbitrary extra fields (pair, timeframe, date range, etc.).
    """

    sharpe_ratio: float = float("nan")
    sortino_ratio: float = float("nan")
    max_drawdown: float = float("nan")
    win_rate: float = float("nan")
    calmar_ratio: float = float("nan")
    t_stat: float = float("nan")
    n_trades: int = 0
    annualized_return: float = float("nan")
    annualized_vol: float = float("nan")
    metadata: dict = field(default_factory=dict)


class PerformanceAnalyzer:
    """Compute and report performance metrics for a systematic trading strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily (or bar-level) strategy returns as decimals, indexed by datetime.
    trades : pd.DataFrame, optional
        Trade log with at minimum columns: ['entry_time', 'exit_time', 'pnl'].
        Additional columns (pair, direction, size) are preserved in reports.
    ann_factor : int, optional
        Annualization factor. Defaults to 252 (trading days). Use 52 for
        weekly returns, 12 for monthly, 365 for crypto.
    risk_free_rate : float, optional
        Daily risk-free rate as a decimal. Defaults to 0.0.

    Examples
    --------
    >>> analyzer = PerformanceAnalyzer(returns=my_returns, trades=my_trades)
    >>> report = analyzer.run_report()
    >>> print(report.sharpe_ratio)
    """

    def __init__(
        self,
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        ann_factor: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.returns = returns
        self.trades = trades
        self.ann_factor = ann_factor
        self.risk_free_rate = risk_free_rate


    def compute_sharpe(self) -> float:
        """Compute the annualized Sharpe ratio.

        Uses the returns series and risk_free_rate provided at construction.
        Excess returns are divided by their standard deviation, then scaled
        by sqrt(ann_factor).

        Returns
        -------
        float
            Annualized Sharpe ratio. Returns NaN if standard deviation is zero.

        Raises
        ------
        NotImplementedError
            Until Phase 2 implementation.
        """
        raise NotImplementedError("compute_sharpe is not yet implemented.")

    def compute_sortino(self) -> float:
        """Compute the annualized Sortino ratio.

        Like Sharpe, but the denominator uses only downside deviation
        (returns below the risk-free rate), penalizing harmful volatility only.

        Returns
        -------
        float
            Annualized Sortino ratio. Returns NaN if downside deviation is zero.

        Raises
        ------
        NotImplementedError
            Until Phase 2 implementation.
        """
        raise NotImplementedError("compute_sortino is not yet implemented.")

    def compute_max_drawdown(self) -> float:
        """Compute the maximum peak-to-trough drawdown.

        Reconstructs the cumulative equity curve from self.returns, then
        measures the largest percentage decline from any rolling peak to
        any subsequent trough.

        Returns
        -------
        float
            Max drawdown as a negative decimal (e.g. -0.23 means -23%).
            Returns 0.0 if the equity curve never declines.

        Raises
        ------
        NotImplementedError
            Until Phase 2 implementation.
        """
        raise NotImplementedError("compute_max_drawdown is not yet implemented.")

    def compute_win_rate(self) -> float:
        """Compute the fraction of trades with positive PnL.

        Requires self.trades to be set and contain a 'pnl' column.

        Returns
        -------
        float
            Win rate in [0, 1]. Returns NaN if trades is None or empty.

        Raises
        ------
        ValueError
            If self.trades does not contain a 'pnl' column.
        NotImplementedError
            Until Phase 2 implementation.
        """
        raise NotImplementedError("compute_win_rate is not yet implemented.")

    def compute_calmar(self) -> float:
        """Compute the Calmar ratio.

        Defined as annualized return divided by the absolute value of max
        drawdown. A higher Calmar indicates better return per unit of
        drawdown risk.

        Returns
        -------
        float
            Calmar ratio. Returns NaN if max drawdown is zero.

        Raises
        ------
        NotImplementedError
            Until Phase 2 implementation.
        """
        raise NotImplementedError("compute_calmar is not yet implemented.")

    def compute_t_stat(self) -> float:
        """Compute the t-statistic for mean return > 0.

        Tests the null hypothesis that the mean daily return equals zero.
        t = mean(returns) / (std(returns) / sqrt(n)).

        Returns
        -------
        float
            t-statistic. Values > ~2.0 suggest statistical significance
            at the 5% level for large samples.

        Raises
        ------
        NotImplementedError
            Until Phase 2 implementation.
        """
        raise NotImplementedError("compute_t_stat is not yet implemented.")

    def run_report(self) -> PerformanceReport:
        """Compute all metrics and return a populated PerformanceReport.

        Calls each compute_* method in sequence and assembles results into
        a PerformanceReport dataclass. Metadata includes the date range of
        self.returns and the number of trades if a trade log is present.

        Returns
        -------
        PerformanceReport
            Fully populated report object. Fields default to NaN if the
            corresponding compute_* method fails or returns NaN.

        Raises
        ------
        NotImplementedError
            Until Phase 2 implementation.
        """
        raise NotImplementedError("run_report is not yet implemented.")