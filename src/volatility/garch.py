from __future__ import annotations

import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, GARCH, Normal

_SCALE = 100.0


def fit_garch(returns: pd.Series) -> dict:
    """Fit a GARCH(1,1) model to a return series via maximum likelihood.

    Model
    -----
    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    where eps_t is the (demeaned) return shock. Volatility clustering shows
    up as alpha > 0 (yesterday's shock size feeds today's variance) and
    beta > 0 (yesterday's variance forecast persists into today).

    Parameters
    ----------
    returns : pd.Series
        Return series (e.g. daily log returns). NaNs are dropped before
        fitting. Internally rescaled by 100x before fitting -- the arch
        package's optimizer is numerically unstable on series with variance
        far from O(1), which daily log/simple returns usually are -- and
        every output below is converted back to the original scale.

    Returns
    -------
    dict with keys:
        omega           : float     -- baseline variance level, original scale
        alpha           : float     -- ARCH coefficient, reaction to shocks
        beta            : float     -- GARCH coefficient, variance persistence
        persistence     : float     -- alpha + beta; decay speed of shocks.
                                       Close to 1 => shocks decay slowly.
        long_run_vol    : float     -- sqrt(omega / (1 - persistence)),
                                       the unconditional volatility the
                                       process reverts to. NaN if
                                       persistence >= 1 (non-stationary
                                       variance, no finite long-run level).
        conditional_vol : pd.Series -- fitted sigma_t path, same index as
                                       the (NaN-dropped) input, original
                                       scale.
    """
    clean = returns.dropna()
    scaled = clean * _SCALE

    model = ConstantMean(scaled)
    model.volatility = GARCH(p=1, o=0, q=1)
    model.distribution = Normal()
    result = model.fit(disp="off")

    omega = result.params["omega"] / _SCALE**2
    alpha = result.params["alpha[1]"]
    beta = result.params["beta[1]"]
    persistence = alpha + beta

    long_run_vol = (
        float(np.sqrt(omega / (1 - persistence))) if persistence < 1 else float("nan")
    )

    conditional_vol = pd.Series(
        result.conditional_volatility / _SCALE,
        index=clean.index,
        name="conditional_vol",
    )

    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(persistence),
        "long_run_vol": long_run_vol,
        "conditional_vol": conditional_vol,
    }
