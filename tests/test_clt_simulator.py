import numpy as np
import pytest
from src.stats.clt_simulator import simulate_sample_means

def test_output_shape():
     result = simulate_sample_means(dist="exponential", n_samples=1000, sample_size=30, seed=28)
     assert result.shape == (1000,)

def test_mean_converges_to_population_mean():
     result = simulate_sample_means(dist="exponential", n_samples=1000, sample_size=30, seed=28)
     result_mean = result.mean()
     std_err = 1/np.sqrt(1000)
     assert abs(result_mean -1) < 2*std_err

def test_invalid_dist_raises():
     with pytest.raises(ValueError):
         simulate_sample_means(dist="normal", n_samples=1000, sample_size=30, seed=28)