import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

def adf_test(series: pd.Series, max_lag: int = None, regression: str = 'c') -> dict:
    """Run Augmented Dickey-Fuller test on a time series.

    Parameters
    ----------
    series : pd.Series
        Time series to test. For cointegration use case, pass OLS residuals.
    max_lag : int, optional
        Maximum lag order passed to adfuller autolag selection.
        If None, statsmodels uses 12*(T/100)^(1/4) by default.

    Returns
    -------
    dict with keys:
        adf_stat    : float  — test statistic (negative; more negative = stronger rejection)
        adf_p       : float  — p-value against non-standard ADF critical values
        n_lags      : int    — lag count selected by AIC
        n_obs       : int    — number of observations used
        critical_values : dict — {'1%': ..., '5%': ..., '10%': ...}
        reject_null : bool   — True if p < 0.05 (evidence of stationarity)
    """
    series = series.dropna()

    result = adfuller(
        series,
        maxlag = max_lag,
        autolag = "AIC", 
        regression = regression
    )

    return {
        "adf_stat": result[0],
        "adf_p": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "reject_null": bool(result[1] < 0.05)
    }


def kpss_test(series: pd.Series, regression: str = "c") -> dict:
    """Run KPSS stationarity test on a time series.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    regression : str
        Deterministic terms included in the regression.
        'c'  — constant only (level-stationary null)
        'ct' — constant + trend (trend-stationary null)
        For OLS residuals from a levels regression: use 'c'.

    Returns
    -------
    dict with keys:
        kpss_stat       : float — test statistic (large = evidence of unit root)
        kpss_p          : float — p-value (statsmodels returns interpolated bound)
        n_lags          : int   — lags used in long-run variance estimator
        critical_values : dict  — {'10%': ..., '5%': ..., '2.5%': ..., '1%': ...}
        reject_null     : bool  — True if stat > 5% critical value (evidence of unit root)
    """
    series = series.dropna()

    result = kpss(
        series,
        regression = regression,
        nlags = "auto",
    )

    return {
        "kpss_stat": result[0],
        "kpss_p": result[1],
        "n_lags": result[2],
        "critical_values": result[3],
        "reject_null": bool(result[0] > result[3]["5%"])
    }


def check_stationarity(series: pd.Series, regression: str = "c") -> dict:
    """Run ADF and KPSS together and return a combined stationarity verdict.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    regression : str
        Passed to kpss_test. Use 'c' for residuals, 'ct' for raw price series with drift.

    Returns
    -------
    dict with keys:
        adf_stat        : float
        adf_p           : float
        kpss_stat       : float
        kpss_p          : float
        is_stationary   : bool  — True only when both tests agree on stationarity
        recommendation  : str   — human-readable verdict string
    """
    adf_result = adf_test(series, regression = regression)
    kpss_result = kpss_test(series, regression = regression)

    adf_rejects = adf_result["reject_null"]
    kpss_rejects = kpss_result["reject_null"]

    if adf_rejects and not kpss_rejects:
        is_stationary = True
        recommendation = (
            "Both tests support stationarity. Treat the series as I(0)."
        )

    elif not adf_rejects and kpss_rejects:
        is_stationary = False
        recommendation = (
            "Both tests support the presence of a unit root. "
            "Difference the series or investigate cointegration."
        )

    elif not adf_rejects and not kpss_rejects:
        is_stationary = False
        recommendation = (
            "Tests are inconclusive. Consider a larger sample, "
            "examining plots/ACF, or using additional tests."
        )

    else:
        is_stationary = False
        recommendation = (
            "Both tests reject their null hypotheses. "
            "Possible structural break, regime shift, or other "
            "departure from simple I(0)/I(1) behavior."
        )

    return {
        "adf_stat": adf_result["adf_stat"],
        "adf_p": adf_result["adf_p"],
        "kpss_stat": kpss_result["kpss_stat"],
        "kpss_p": kpss_result["kpss_p"],
        "is_stationary": is_stationary,
        "recommendation": recommendation,
    }


def plot_acf_pacf(
    series: np.ndarray,
    lags: int = 40,
    title: str = "",
    alpha: float = 0.05,
    save_path: Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series, lags=lags, alpha=alpha, ax=axes[0])
    axes[0].set_title(f"{title} ACF")
    plot_pacf(series, lags=lags, alpha=alpha, ax=axes[1])
    axes[1].set_title(f"{title} PACF")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close("all")


def ljung_box_test(series: np.ndarray, lags: int = 20,) -> pd.DataFrame:
    """
    Ljung-Box Q test for autocorrelation up to given lag.

    Parameters
    ----------
    series : np.ndarray
        Series to test (typically residuals or returns).
    lags : int
        Maximum lag to test.

    Returns
    -------
    pd.DataFrame
        Columns: lb_stat, lb_pvalue, indexed by lag.
    """
    return acorr_ljungbox(series, lags=lags, return_df=True)