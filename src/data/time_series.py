import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def fit_arima(series: pd.Series,
order: tuple[int, int, int],
trend: str = "n",
) -> dict:
    """
    Fit an ARIMA(p, d, q) model to a univariate time series.

    Parameters
    ----------
    series : pd.Series
        Univariate time series. Pass d=0 for stationary return series (I(0),
        confirmed via ADF/KPSS). Pass d=1 for price levels with a unit root.
    order : tuple[int, int, int]
        (p, d, q) — AR order, integration order, MA order.
    trend : str, optional
        'n' — no constant (default; appropriate for mean-zero return series).
        'c' — estimate a constant term.

    Returns
    -------
    dict with keys:
        order         : tuple[int, int, int]
        aic           : float
        bic           : float
        params        : dict[str, float]   — statsmodels param names → values
        residuals     : np.ndarray
        fitted_values : np.ndarray
        result        : ARIMAResults       — raw statsmodels object

    Raises
    ------
    ValueError   — NaN in series, or series too short for requested order.
    RuntimeError — statsmodels optimization fails to converge.

    Notes
    -----
    AIC = -2 * log_likelihood + 2 * k
    BIC = -2 * log_likelihood + k * ln(n)

    For n > 7, ln(n) > 2, so BIC always penalises extra parameters more
    heavily than AIC.  At ~4 700 daily FX observations ln(4700) ≈ 8.5,
    roughly 4× harsher per extra parameter.  When AIC and BIC disagree,
    trust BIC on large samples.
    """
    if series.isnull().any():
        raise ValueError(
            "series contains NaN values; clean before fitting."
        )

    p, d, q = order
    n_params = p + q + (1 if trend == "c" else 0)
    min_obs = n_params + d + 1
    if len(series) < min_obs:
        raise ValueError(
            f"Series length {len(series)} is too short for ARIMA{order} "
            f"with trend='{trend}' ({n_params} parameters, {d} difference(s)). "
            f"Need at least {min_obs} observations."
        )

    try:
        model = ARIMA(series, order=order, trend=trend)
        result = model.fit()
    except Exception as exc:
        raise RuntimeError(f"ARIMA{order} fitting failed: {exc}") from exc

    return {
        "order": order,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "params": dict(zip(result.param_names, result.params.tolist())),
        "residuals": np.asarray(result.resid),
        "fitted_values": np.asarray(result.fittedvalues),
        "result": result,
    }




def select_arima_order(
    series: pd.Series,
    max_p: int,
    max_q: int,
    d: int = 0,
    criterion: str = "aic",
) -> tuple[int, int, int]:
    """
    Select ARIMA order by exhaustive grid search over p and q.

    Evaluates all (p, q) combinations in range(0, max_p+1) x range(0, max_q+1).
    Fits each via fit_arima and returns the order minimising the chosen
    information criterion.  Failed fits are skipped and logged to stderr.

    Parameters
    ----------
    series    : pd.Series               — stationary univariate series
    max_p     : int                     — maximum AR order to consider
    max_q     : int                     — maximum MA order to consider
    d         : int, optional           — integration order (default 0)
    criterion : str, optional           — 'aic' or 'bic' (default 'aic')

    Returns
    -------
    tuple[int, int, int]   — best (p, d, q) found

    Raises
    ------
    ValueError   — invalid criterion string.
    RuntimeError — no (p, q) combination converged successfully.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError(f"criterion must be 'aic' or 'bic', got '{criterion}'.")

    best_order: tuple[int, int, int] | None = None
    best_score: float = float("inf")
    failed: list[tuple[int, int]] = []

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try: 
                result = fit_arima(series, order=(p, d, q))
                score = result[criterion]
                if score < best_score:
                    best_order = (p, d, q)
                    best_score = score
            except Exception:
                failed.append((p,q))

    if failed:
        print(f"Warning: {len(failed)} combinations failed to converge: {failed}")
    if best_order is None:
        raise RuntimeError
    return best_order
