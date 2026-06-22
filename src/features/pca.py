import numpy as np
from src.analysis.portfolio_stats import compute_covariance_matrix


def matrix_inverse_via_svd(M: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
    """Compute the Moore-Penrose pseudoinverse of M via SVD.

    Parameters
    ----------
    M : np.ndarray
        Square 2D matrix. Need not be symmetric.
    threshold : float
        Singular values below this value are set to zero in the
        inverted diagonal. Guards against numerical explosion on
        ill-conditioned matrices.

    Returns
    -------
    np.ndarray
        Pseudoinverse of M, same shape as M.

    Raises
    ------
    ValueError
        If M is not a 2D square matrix.
    """

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be a 2D square matrix, got shape {M.shape}")
  
    U, s, Vt  = np.linalg.svd(M)
    s_inv = np.zeros(len(s))
    for i, val in enumerate(s):
        if val < threshold:  
            s_inv[i] = 0.0
        else:  
            s_inv[i] = 1/val
    A_inv = Vt.T @ np.diag(s_inv) @ U.T
    return A_inv


def eigendecomposition(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of M, sorted by descending magnitude.

    Parameters
    ----------
    M : np.ndarray
        Square 2D matrix.

    Returns
    -------
    eigenvalues : np.ndarray
        1D array of eigenvalues, sorted descending by magnitude.
        Returned as real if M is symmetric, otherwise may be complex.
    eigenvectors : np.ndarray
        2D array where column i is the eigenvector corresponding to
        eigenvalue i. Sorted to match eigenvalue ordering.

    Raises
    ------
    ValueError
        If M is not a 2D square matrix.
    """

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be a 2D square matrix, got shape {M.shape}")

    lambdas, v = np.linalg.eigh(M)

    idx = np.argsort(-np.abs(lambdas))

    lambdas = lambdas[idx]
    v = v[:,idx]
    
    if np.allclose(M, M.T):
        lambdas = lambdas.real
        v = v.real

    return lambdas , v


def pca(X: np.ndarray, n_components: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA from first principles via eigendecomposition of the
    covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_observations, n_features). Will be
        mean-centred internally.
    n_components : int or None
        Number of principal components to retain. If None, keep all.

    Returns
    -------
    components : np.ndarray
        Principal component directions, shape (n_components, n_features).
    explained_variance : np.ndarray
        Eigenvalues corresponding to retained components.
    projected : np.ndarray
        X projected onto the principal components,
        shape (n_observations, n_components).
    """
    X_c = X - X.mean(axis=0)

    cov_matrix = compute_covariance_matrix(X)

    lambdas , v = eigendecomposition(cov_matrix)

    Z = X_c @ v

    total_variance = lambdas.sum()

    v = v[:, :n_components]
    lambdas = lambdas[:n_components]
    Z = Z[:, :n_components]

    explained_variance = lambdas / total_variance

    return v , explained_variance , Z