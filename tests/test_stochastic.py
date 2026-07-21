import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.stats.stochastic import simulate_gbm, simulate_ou, ou_stress_test

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


def test_ou_output_shape():
    paths = simulate_ou(theta=0.1, mu=0.0, sigma=0.5, X0=1.0, n_steps=100, n_paths=50, dt=1.0, seed=42)

    assert paths.shape == (50, 101)


def test_ou_paths_start_at_X0():
    paths = simulate_ou(theta=0.1, mu=0.0, sigma=0.5, X0=3.0, n_steps=50, n_paths=20, seed=42)

    assert np.allclose(paths[:, 0], 3.0)


def test_paths_revert_to_mu():
    theta, mu, sigma, X0 = 0.5, 2.0, 0.3, 10.0
    n_steps, dt, n_paths = 200, 1.0, 20000

    paths = simulate_ou(theta=theta, mu=mu, sigma=sigma, X0=X0, n_steps=n_steps, n_paths=n_paths, dt=dt, seed=42)
    terminal = paths[:, -1]

    t = n_steps * dt
    theoretical_mean = mu + (X0 - mu) * np.exp(-theta * t)
    theoretical_var = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t))
    se = np.sqrt(theoretical_var / n_paths)

    assert abs(terminal.mean() - theoretical_mean) < 4 * se
    assert abs(terminal.mean() - mu) < 4 * se + 1e-6


def test_faster_theta_shorter_half_life():
    mu, X0, sigma, dt, n_steps = 0.0, 10.0, 0.0, 1.0, 200
    fast = simulate_ou(theta=0.5, mu=mu, sigma=sigma, X0=X0, n_steps=n_steps, n_paths=1, dt=dt, seed=42)[0]
    slow = simulate_ou(theta=0.05, mu=mu, sigma=sigma, X0=X0, n_steps=n_steps, n_paths=1, dt=dt, seed=42)[0]
    half_dist = abs(X0 - mu) / 2
    fast_cross = np.argmax(np.abs(fast - mu) <= half_dist)
    slow_cross = np.argmax(np.abs(slow - mu) <= half_dist)

    assert fast_cross < slow_cross
    assert (np.log(2) / 0.5) < (np.log(2) / 0.05)


def test_ou_stress_test_output_keys():
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    rng = np.random.default_rng(7)
    values = np.zeros(300)
    for i in range(1, 300):
        values[i] = 0.3 * values[i - 1] + rng.normal(0, 0.01)
    returns = pd.Series(values, index=dates)
    result = ou_stress_test(returns, n_simulations=200, seed=1)

    assert set(result.keys()) == {"p5_sharpe", "p50_sharpe", "p95_sharpe"}
    assert result["p5_sharpe"] <= result["p50_sharpe"] <= result["p95_sharpe"]


def test_ou_stress_test_rejects_non_mean_reverting_series():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    values = 1.05 ** np.arange(100, dtype=float)
    returns = pd.Series(values, index=dates)

    with pytest.raises(ValueError):
        ou_stress_test(returns, n_simulations=50)
