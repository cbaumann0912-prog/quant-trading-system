"""
Simulation of stochastic processes: GBM and Ornstein-Uhlenbeck.

Used to generate data with known parameters, which is the only reliable way
to validate an estimator: a half-life estimator can be checked against a
process whose true half-life is set by construction. `ou_stress_test`
sweeps the parameter space to map where estimation degrades -- typically
when the observation window is short relative to the true half-life, where
mean reversion becomes statistically indistinguishable from a random
walk.
"""
import numpy as np
import pandas as pd

from src.analysis.performance_analyzer import PerformanceAnalyzer

from src.utils.random_state import get_rng

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate geometric Brownian motion price paths using the exact
    lognormal solution S(t) = S0 * exp((mu - 0.5*sigma**2)*t + sigma*W(t)).

    Returns an array of shape (n_paths, n_steps + 1), column 0 is S0.
    """
    rng = get_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    cumulative_log_returns = np.cumsum(log_increments, axis=1)
    cumulative_log_returns = np.hstack(
        [np.zeros((n_paths, 1)), cumulative_log_returns]
    )

    return S0 * np.exp(cumulative_log_returns)


def simulate_ou(
    theta: float,
    mu: float,
    sigma: float,
    X0: float,
    n_steps: int,
    n_paths: int,
    dt: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate Ornstein-Uhlenbeck paths via Euler-Maruyama discretization of
    dX_t = theta*(mu - X_t)*dt + sigma*dW_t:
        X_{t+dt} = X_t + theta*(mu - X_t)*dt + sigma*sqrt(dt)*Z,  Z ~ N(0, 1)

    Parameters
    ----------
    theta : float
        Mean-reversion speed.
    mu : float
        Long-run mean the process reverts to.
    sigma : float
        Diffusion coefficient. Must be >= 0.
    X0 : float
        Initial value of every path at t=0.
    n_steps : int
        Number of discretization steps. Output has n_steps + 1 columns.
    n_paths : int
        Number of independent paths to simulate.
    dt : float, optional
        Time step size. Defaults to 1.0 (one bar per step).
    seed : int, optional
        Random seed. None resolves to utils.random_state.DEFAULT_SEED.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps + 1). Column 0 is X0..
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")

    rng = get_rng(seed)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = X0

    z = rng.standard_normal((n_paths, n_steps))
    sqrt_dt = np.sqrt(dt)

    for t in range(n_steps):
        paths[:, t + 1] = (
            paths[:, t] + theta * (mu - paths[:, t]) * dt + sigma * sqrt_dt * z[:, t]
        )

    return paths


def _fit_ou_params(x: np.ndarray, dt: float) -> tuple[float, float, float]:
    """
    Calibrate (theta, mu, sigma) from a single observed series via the
    *exact* discrete-time OU transition, fit as an AR(1) regression:

        X_{t+dt} = a + b * X_t + eta_t,   b = exp(-theta * dt)

    Parameters
    ----------
    x : np.ndarray
        1-D array of consecutive observations, evenly spaced by dt.
    dt : float
        Time step between observations.

    Returns
    -------
    tuple[float, float, float]
        (theta, mu, sigma).

    Raises
    ------
    ValueError
        If the fitted AR(1) coefficient b implies a non-mean-reverting
        process (b <= 0 or b >= 1).
    """
    x_t = x[:-1]
    x_next = x[1:]

    b, a = np.polyfit(x_t, x_next, 1)

    if not (0 < b < 1):
        raise ValueError(
            f"Fitted AR(1) coefficient b={b:.4f} is outside (0, 1); the "
            "series is not consistent with a mean-reverting OU process "
        )

    theta = -np.log(b) / dt
    mu = a / (1 - b)

    residuals = x_next - (a + b * x_t)
    resid_var = np.var(residuals, ddof=2)

    sigma = np.sqrt(resid_var * 2 * theta / (1 - b**2))

    return float(theta), float(mu), float(sigma)


def ou_stress_test(
    strategy_returns: pd.Series,
    n_simulations: int = 500,
    seed: int | None = None,
) -> dict:
    """
    Monte Carlo stress test of a strategy's realized Sharpe ratio against
    parameter uncertainty in an OU model fit to the return series.

    Parameters
    ----------
    strategy_returns : pd.Series
        Observed strategy returns, indexed by datetime ascending.
    n_simulations : int, optional
        Number of Monte Carlo replicates. Defaults to 500.
    seed : int, optional
        Random seed forwarded to `simulate_ou`. None resolves to
        utils.random_state.DEFAULT_SEED.
        Defaults to 28.

    Returns
    -------
    dict
        Keys: 'p5_sharpe', 'p50_sharpe', 'p95_sharpe' -- percentiles of
        the simulated Sharpe ratio distribution. NaN Sharpes are dropped before
        computing percentiles.

    Raises
    ------
    ValueError
        If strategy_returns has fewer than 3 observations or if the fitted
        process is not mean-reverting.
    """
    if len(strategy_returns) < 3:
        raise ValueError(
            "strategy_returns must have >= 3 observations to fit an OU "
            f"process, got {len(strategy_returns)}"
        )

    x = strategy_returns.to_numpy(dtype=float)
    dt = 1.0
    theta, mu, sigma = _fit_ou_params(x, dt)

    n_steps = len(x) - 1
    sim_paths = simulate_ou(
        theta=theta,
        mu=mu,
        sigma=sigma,
        X0=x[0],
        n_steps=n_steps,
        n_paths=n_simulations,
        dt=dt,
        seed=seed,
    )

    sharpes = np.empty(n_simulations)
    for i in range(n_simulations):
        sim_returns = pd.Series(sim_paths[i], index=strategy_returns.index)
        sharpes[i] = PerformanceAnalyzer(sim_returns).compute_sharpe()

    sharpes = sharpes[~np.isnan(sharpes)]

    return {
        "p5_sharpe": float(np.percentile(sharpes, 5)),
        "p50_sharpe": float(np.percentile(sharpes, 50)),
        "p95_sharpe": float(np.percentile(sharpes, 95)),
    }
