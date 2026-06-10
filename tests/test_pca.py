import numpy as np
import pytest
from src.features.pca import matrix_inverse_via_svd, eigendecomposition, pca

@pytest.fixture
def symmetric_matrix():
    A = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 5.0, 3.0],
        [1.0, 3.0, 6.0]
    ])
    return A


@pytest.fixture
def ill_conditioned_matrix():
    A = np.array([
        [1.0, 2.0],
        [2.0, 4.0 + 1e-12]
    ])
    return A


@pytest.fixture
def simple_data():
    """Small deterministic dataset for exact numerical checks."""
    np.random.seed(8)
    return np.random.randn(100, 3)


def test_inverse_times_original_is_identity(symmetric_matrix):
    M_inv = matrix_inverse_via_svd(symmetric_matrix)
    n = symmetric_matrix.shape[0]
    identity = np.eye(n)
    assert np.allclose(symmetric_matrix @ M_inv, identity)


def test_inverse_invalid_input():
    with pytest.raises(ValueError):
        matrix_inverse_via_svd(np.ones((2, 3)))


def test_ill_conditioned_does_not_explode(ill_conditioned_matrix):
    result = matrix_inverse_via_svd(ill_conditioned_matrix)
    assert np.isfinite(result).all()


def test_eigenvalues_sorted_descending(symmetric_matrix):
    lambdas, v = eigendecomposition(symmetric_matrix)
    for i in range(len(lambdas) - 1):
        assert lambdas[i] >= lambdas[i + 1]


def test_reconstruct_matrix_from_eigen(symmetric_matrix):
    lambdas, v = eigendecomposition(symmetric_matrix)
    reconstructed = v @ np.diag(lambdas) @ v.T
    assert np.allclose(reconstructed, symmetric_matrix)


def test_eigendecomposition_invalid_input():
    with pytest.raises(ValueError):
        eigendecomposition(np.ones((2, 3)))


def test_eigenvectors_are_unit_length(symmetric_matrix):
    lambdas, v = eigendecomposition(symmetric_matrix)
    for i in range(v.shape[1]):
        assert np.isclose(np.linalg.norm(v[:, i]), 1.0)


def test_eigenvectors_are_orthogonal(symmetric_matrix):
    lambdas, v = eigendecomposition(symmetric_matrix)
    identity = np.eye(v.shape[1])

    assert np.allclose(v.T @ v, identity, atol=1e-8)


def test_variance_of_scores_equals_eigenvalues(simple_data):
    """Var(Z_k) must equal λ_k for each retained component."""
    components, explained_variance, projected = pca(simple_data, n_components=3)
    total_variance = np.var(simple_data, axis=0, ddof=1).sum()
    eigenvalues = explained_variance * total_variance

    for k in range(3):
        assert np.isclose(
            np.var(projected[:, k], ddof=1),
            eigenvalues[k],
            atol=1e-8,
        )


def test_pc_scores_are_orthogonal(simple_data):
    """Correlation between any two PC score series must be near zero."""
    components, explained_variance, projected = pca(simple_data, n_components=3)
    corr_matrix = np.corrcoef(projected.T)
    n = corr_matrix.shape[0]

    for i in range(n):
        for j in range(n):
            if i != j:
                assert np.isclose(corr_matrix[i, j], 0.0, atol=1e-8)


def test_explained_variance_sums_to_one(simple_data):
    """When retaining all components, explained variance must sum to 1.0."""
    components, explained_variance, projected = pca(simple_data, n_components=3)
    
    assert np.isclose(explained_variance.sum(), 1.0, atol=1e-8)


def test_n_components_none_retains_all(simple_data):
    """n_components=None must return all components."""
    n, p = simple_data.shape
    components, explained_variance, projected = pca(
        simple_data,
        n_components=None,
    )

    assert components.shape == (p, p)
    assert explained_variance.shape == (p,)
    assert projected.shape == (n, p)


def test_output_shapes(simple_data):
    """Shape contract: components (k, p), explained_variance (k,), projected (n, k)."""
    n, p = simple_data.shape
    k = 2
    components, explained_variance, projected = pca(
        simple_data,
        n_components=k,
    )

    assert components.shape == (p, k)
    assert explained_variance.shape == (k,)
    assert projected.shape == (n, k)


def test_explained_variance_descending(simple_data):
    """Each component must explain less variance than the previous."""
    components, explained_variance, projected = pca(simple_data, n_components=3)
    
    assert np.all(np.diff(explained_variance) <= 0)