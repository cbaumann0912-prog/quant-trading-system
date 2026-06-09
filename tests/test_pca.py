import numpy as np
import pytest
from src.features.pca import matrix_inverse_via_svd, eigendecomposition


@pytest.fixture
def symmetric_matrix():
    # Valid symmetric positive definite 3x3 matrix
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
        assert np.isclose(
            np.linalg.norm(v[:, i]),
            1.0
        )