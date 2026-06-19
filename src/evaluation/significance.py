import numpy as np

def bonferroni_correction(p_values: list[float], alpha: float) -> list[bool]:
    """
    Apply Bonferroni correction across m hypothesis tests.

    Controls FWER by rejecting H0_i where p_i <= alpha / m.

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
    """
    Apply Benjamini-Hochberg correction across m hypothesis tests.

    Controls FDR by finding the largest rank i where p_(i) <= i * alpha / m,
    then rejecting all hypotheses up to and including that rank.
    Assumes independent or positively correlated tests (c_m = 1).

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