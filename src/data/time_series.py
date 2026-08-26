"""
ARIMA fitting and order selection.

`select_arima_order` searches over (p, d, q) by information criterion. Note
that the selected order is itself an estimate chosen on the data, so any
subsequent inference that treats it as fixed and known understates
uncertainty. Order selection performed on the full sample and then
evaluated on the same sample is a leakage path.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def fit_arima(
    series: pd.Series,
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
    """Select ARIMA order by exhaustive grid search over (p, q).

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
    total_candidates = (max_p + 1) * (max_q + 1)

    logger.info(
        "ARIMA order search over %d candidates (p<=%d, q<=%d, d=%d) by %s.",
        total_candidates, max_p, max_q, d, criterion.upper(),
    )

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                result = fit_arima(series, order=(p, d, q))
                score = result[criterion]
                if score < best_score:
                    best_order = (p, d, q)
                    best_score = score
            except Exception as exc:
                logger.debug("ARIMA(%d, %d, %d) failed: %s", p, d, q, exc)
                failed.append((p, q))

    if failed:
        logger.warning(
            "ARIMA order search: %d of %d (p, q) combinations failed to converge: %s. "
            "Selected order was chosen from the surviving subset only.",
            len(failed), total_candidates, failed,
        )
    if best_order is None:
        logger.error("ARIMA order search: no candidate converged.")
        raise RuntimeError(
            "No ARIMA order converged over the requested search grid. "
            "Widen the grid, lengthen the sample, or difference the series first."
        )
    return best_order
