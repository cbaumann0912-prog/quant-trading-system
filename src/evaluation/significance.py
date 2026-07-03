import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Literal

def bonferroni_correction(p_values: list[float], alpha: float) -> list[bool]:
    """Apply Bonferroni correction across m hypothesis tests.

    Parameters
    ----------
    p_values
        List of raw p-values from m independent hypothesis tests.
    alpha
        Family-wise significance level.

    Returns
    -------
    list[bool]
        Boolean mask of length m. True means reject H0 (significant).

    Raises
    ------
    ValueError
        If p_values is empty.
    """
    if len(p_values) == 0: 
        raise ValueError("the list p_values must not be empty")

    result = []
    for p in p_values:
        if p <= (alpha / len(p_values)):
            result.append(True)
        else:
            result.append(False)
    return result



def benjamini_hochberg_correction(p_values: list[float], alpha: float) -> list[bool]:
    """Apply Benjamini-Hochberg FDR correction across m hypothesis tests.

    Parameters
    ----------
    p_values
        List of raw p-values from m hypothesis tests.
    alpha
        False discovery rate threshold.

    Returns
    -------
    list[bool]
        Boolean mask of length m. True means reject H0 (significant).

    Raises
    ------
    ValueError
        If p_values is empty.
    """
    if len(p_values) == 0: 
        raise ValueError("the list p_values must not be empty")

    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_ps = np.array(p_values)[sorted_indices]

    R_max = 0
    for i, p in enumerate(sorted_ps, 1):
        if p <= ((i * alpha) / m):
            R_max = i

    rej_indices = sorted_indices[:R_max] 
    rejected = np.zeros(m,dtype=bool)
    rejected[rej_indices] = True
    
    return rejected.tolist()


def permutation_test(
    signal: pd.Series,
    forward_returns: pd.Series,
    n_permutations: int = 1000,
    seed: int = 42,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict:
    """
    Test signal predictive significance via empirical permutation testing.

    Parameters
    ----------
    signal : pd.Series
        Signal values, indexed identically to forward_returns.
    forward_returns : pd.Series
        Forward-looking return series, indexed identically to signal.
    n_permutations : int, default 1000
        Number of permutations used to build the null distribution.
    seed : int, default 42
        Seed for the random number generator, for reproducibility.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        - "greater": tests whether observed_ic is significantly greater
          than the null.
        - "less": tests whether observed_ic is significantly less than the
          null. 
        - "two-sided": no pre-committed direction; tests |observed_ic|
          against |null|. 
    Returns
    -------
    dict
        observed_ic : float
            Spearman correlation between signal and forward_returns.
        p_value : float
            Empirical p-value in the specified direction, using the
            (1 + count) / (n_permutations + 1) correction.
        null_distribution : np.ndarray
            Array of length n_permutations containing the permuted ICs.

    Raises
    ------
    ValueError
        If signal and forward_returns have mismatched lengths or indices,
        or if alternative is not one of the three allowed values.
    """
    if len(signal) != len(forward_returns):
        raise ValueError(
            f"signal (len {len(signal)}) and forward_returns "
            f"(len {len(forward_returns)}) must have the same length"
        )
    if not signal.index.equals(forward_returns.index):
        raise ValueError(
            "signal and forward_returns must share identical indices "
            "(same labels, same order)"
        )
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', "
            f"got {alternative!r}"
        )

    rng = np.random.default_rng(seed)

    signal_vals = signal.to_numpy()
    fr_vals = forward_returns.to_numpy()

    observed_ic, _ = spearmanr(signal_vals, fr_vals)

    null_distribution = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted_signal = rng.permutation(signal_vals)
        null_distribution[i], _ = spearmanr(permuted_signal, fr_vals)

    if alternative == "greater":
        count_as_extreme = np.sum(null_distribution >= observed_ic)
    elif alternative == "less":
        count_as_extreme = np.sum(null_distribution <= observed_ic)
    else:
        count_as_extreme = np.sum(np.abs(null_distribution) >= np.abs(observed_ic))

    p_value = (1 + count_as_extreme) / (n_permutations + 1)

    return {
        "observed_ic": float(observed_ic),
        "p_value": float(p_value),
        "null_distribution": null_distribution,
    }