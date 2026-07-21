import numpy as np
from scipy import stats

from src.stats.stochastic import simulate_gbm

def test_output_shape():
    paths = simulate_gbm(S0=100.0, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=1000, seed=28)

    assert paths.shape == (1000, 253)


def test_paths_start_at_S0():
    paths = simulate_gbm(S0=100.0, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=1000, seed=28)

    assert np.allclose(paths[:, 0], 100.0)


def test_mean_path_follows_drift():
    S0, mu, sigma, T = 100.0, 0.05, 0.2, 1.0
    paths = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, n_steps=252, n_paths=50000, seed=28)
    log_returns_terminal = np.log(paths[:, -1] / S0)
    theoretical_mean = (mu - 0.5 * sigma**2) * T
    theoretical_se = sigma * np.sqrt(T) / np.sqrt(50000)

    assert abs(log_returns_terminal.mean() - theoretical_mean) < 4 * theoretical_se


def test_log_returns_approximately_normal():
    paths = simulate_gbm(S0=100.0, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=5000, seed=28)
    step_log_returns = np.diff(np.log(paths[:, :50]), axis=1).flatten()
    _, p_value = stats.normaltest(step_log_returns)

    assert p_value > 0.01