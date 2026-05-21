import math
import numpy as np

def normal_pdf(x,mu,sigma):
    return (1 / (sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*(x-mu)/sigma)**2

def normal_cdf(x,mu, sigma):
    z = (x-mu) / (sigma*math.sqrt(2))
    return 0.5*(1+np.vectorize(math.erf)(z))

def lognormal_pdf(X, mu, sigma):
    x=np.asarray(X,dtype=float)
    if np.any(X<=0):
        raise ValueError("lognormal_pdf is only defined for x>0")
    return (1/(X*sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((np.log(x)-mu)/sigma)**2)

def simulate_log_returns(mu, sigma, n, seed=28):
    rng=np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n)

def simulate_price_path(S0, mu, sigma, n, seed=42):
    log_returns = simulate_log_returns(mu, sigma, n - 1, seed)
    prices = np.empty(n)
    prices[0] = S0
    prices[1:] = S0 * np.exp(np.cumsum(log_returns))
    return prices
