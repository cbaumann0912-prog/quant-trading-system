import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Literal

from src.stats.regression import interaction_regression_centered

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


def paired_sign_permutation_test(
    diffs: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 28,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict:
    """
    Permutation test for whether a set of paired differences has a mean
    distinguishable from zero, via sign-flipping.

    Parameters
    ----------
    diffs : np.ndarray
        One difference per matched pair.
    n_permutations : int, default 10000
        Number of random sign-flip permutations to draw.
    seed : int, default 28
        Seed for the random number generator, for reproducibility.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Same convention as permutation_test: pre-committed direction,
        chosen before observing the sign of the true mean difference.

    Returns
    -------
    dict
        observed_mean_diff : float
            Mean of the observed paired differences.
        p_value : float
            Empirical p-value using the (1 + count) / (n_permutations + 1)
            correction.
        null_distribution : np.ndarray
            Array of length n_permutations containing the permuted mean
            differences.

    Raises
    ------
    ValueError
        If diffs is empty, or alternative is not one of the three
        allowed values.

    Notes
    -----
    Null hypothesis: each pair's difference is equally likely to have
    been positive or negative.
    """
    if len(diffs) == 0:
        raise ValueError("diffs must not be empty")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', "
            f"got {alternative!r}"
        )

    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)

    observed_mean_diff = float(np.mean(diffs))

    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
    null_distribution = (signs * diffs).mean(axis=1)

    if alternative == "greater":
        count_as_extreme = np.sum(null_distribution >= observed_mean_diff)
    elif alternative == "less":
        count_as_extreme = np.sum(null_distribution <= observed_mean_diff)
    else:
        count_as_extreme = np.sum(
            np.abs(null_distribution) >= np.abs(observed_mean_diff)
        )

    p_value = (1 + count_as_extreme) / (n_permutations + 1)

    return {
        "observed_mean_diff": observed_mean_diff,
        "p_value": float(p_value),
        "null_distribution": null_distribution,
    }


def permutation_test_interaction_coefficient(
    y: pd.Series,
    x1: pd.Series,
    dummy: pd.Series,
    n_permutations: int = 1000,
    seed: int = 42,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict:
    """
    Null hypothesis: b3 (the signal x regime interaction effect) is no
    different from what you'd see if the regime label were random noise.

    Parameters
    ----------
    y : pd.Series
        Response variable (e.g. forward return over the shared horizon).
    x1 : pd.Series
        Continuous predictor (e.g. momentum_signal or price_z). Held fixed
        across all permutations -- only `dummy` is shuffled.
    dummy : pd.Series
        0/1 regime indicator (e.g. turbulent_dummy or calm_dummy).
    n_permutations : int, default 1000
        Number of permutations used to build the null distribution, per
        Section 10 (1000 permutations, pre-registered).
    seed : int, default 42
        Seed for the random number generator, for reproducibility.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Same convention as `permutation_test`. Two-sided is appropriate
        here since Section 10 does not pre-commit to a sign for b3.

    Returns
    -------
    dict
        observed_b3 : float
            Interaction coefficient from the unpermuted fit.
        p_value : float
            Empirical p-value using the (1 + count) / (n_permutations + 1)
            correction.
        null_distribution : np.ndarray
            Array of length n_permutations containing the permuted b3
            values.
        n_obs : int
            Number of observations used in the unpermuted fit (post
            alignment/NaN-drop), for reference.

    Raises
    ------
    ValueError
        Propagated from `interaction_regression_centered` if the aligned,
        NaN-dropped sample has too few observations to fit the model, or
        if `alternative` is not one of the three allowed values.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', "
            f"got {alternative!r}"
        )

    observed_result = interaction_regression_centered(y, x1, dummy)
    observed_b3 = observed_result["coefficients"]["interaction"]
    n_obs = observed_result["n_obs"]

    y_aligned, x1_aligned = y.align(x1, join="inner")
    y_aligned, dummy_aligned = y_aligned.align(dummy, join="inner")
    x1_aligned = x1_aligned.reindex(y_aligned.index)
    dummy_aligned = dummy_aligned.reindex(y_aligned.index)
    valid = y_aligned.notna() & x1_aligned.notna() & dummy_aligned.notna()

    y_vals = y_aligned[valid].reset_index(drop=True)
    x1_vals = x1_aligned[valid].reset_index(drop=True)
    dummy_vals = dummy_aligned[valid].reset_index(drop=True).to_numpy()

    rng = np.random.default_rng(seed)
    null_distribution = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted_dummy = pd.Series(rng.permutation(dummy_vals))
        result = interaction_regression_centered(y_vals, x1_vals, permuted_dummy)
        null_distribution[i] = result["coefficients"]["interaction"]

    if alternative == "greater":
        count_as_extreme = np.sum(null_distribution >= observed_b3)
    elif alternative == "less":
        count_as_extreme = np.sum(null_distribution <= observed_b3)
    else:
        count_as_extreme = np.sum(np.abs(null_distribution) >= np.abs(observed_b3))

    p_value = (1 + count_as_extreme) / (n_permutations + 1)

    return {
        "observed_b3": float(observed_b3),
        "p_value": float(p_value),
        "null_distribution": null_distribution,
        "n_obs": n_obs,
    }