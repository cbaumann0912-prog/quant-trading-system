import numpy as np

def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 28,
) -> np.ndarray:
    """
    Simulate geometric Brownian motion price paths using the exact
    lognormal solution S(t) = S0 * exp((mu - 0.5*sigma**2)*t + sigma*W(t)).

    Returns an array of shape (n_paths, n_steps + 1), column 0 is S0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    cumulative_log_returns = np.cumsum(log_increments, axis=1)
    cumulative_log_returns = np.hstack(
        [np.zeros((n_paths, 1)), cumulative_log_returns]
    )
    
    return S0 * np.exp(cumulative_log_returns)