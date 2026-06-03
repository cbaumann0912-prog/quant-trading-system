import numpy as np


def bonferroni_correction(p_values: list[float], alpha: float) -> list[bool]:
    """
    Apply Bonferroni correction across m hypothesis tests.
    Controls FWER by rejecting H0_i where p_i <= alpha / m.
    Returns a boolean mask — True means reject H0 (significant).
    """

    result = []
    for p in p_values:
        if p <= (alpha / len(p_values)):
            result.append(True)
        else:
            result.append(False)
    return result



def benjamini_hochberg(p_values: list[float], alpha: float) -> list[bool]:
    """
    Apply Benjamini-Hochberg correction across m hypothesis tests.
    Controls FDR by finding the largest rank i where p_(i) <= i*alpha/m,
    then rejecting all hypotheses up to and including that rank.
    Returns a boolean mask — True means reject H0 (significant).
    Assumes independent or positively correlated tests (c_m = 1).
    """
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

  
        
