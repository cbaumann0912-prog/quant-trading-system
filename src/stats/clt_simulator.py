import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_sample_means(dist: str, n_samples: int, sample_size: int, seed: int) -> np.ndarray:
    """
    Simulates the sampling distribution of the mean for a given parent distrbution
    
    Parameters
    ----------

    dist:  name of the parent distrribution.(exponential, uniform, lognorm)
    n_samples: number of simulations to be ran
    sample_size: number of observations per simulation (the n in CLT)
    seed: random seed for reproducibility

    Returns
    -------
    
    1D array of length n_samples containing the mean of each simulation
    """
    np.random.seed(seed)

    if dist == "exponential":
        samples = np.random.exponential(scale=1, size=(n_samples,sample_size))
    elif dist == "uniform":
        samples = np.random.uniform(low=0.0, high=1.0, size=(n_samples,sample_size))
    elif dist == "lognormal":
        samples = np.random.lognormal(mean=0.0, sigma=1.0, size=(n_samples,sample_size))
    else:
        raise  ValueError("dist must be exponential, uniform, or lognormal")
    
    logger.info(f"dist={dist} | n_samples={n_samples} | sample_size={sample_size} | mean={samples.mean(axis=1).mean():.4f} | std={samples.mean(axis=1).std():.4f}")

    return samples.mean(axis=1)