from typing import Tuple, List, Union
import scipy.sparse
import numpy as np


Matrix = Union[np.ndarray, scipy.sparse.csr_matrix]


def get_upper_lower_diag(matrix: Matrix) -> Tuple[Matrix, Matrix, np.ndarray]:
    if isinstance(matrix, scipy.sparse.csr_matrix):
        upper = scipy.sparse.triu(matrix, k=1, format="csr")
        lower = scipy.sparse.tril(matrix, k=-1, format="csr")
        diag = matrix.diagonal()
        # a_lower[i] takes too much time. Let's precompute it
        lower = [lower[i] for i in range(lower.shape[0])]

    else:
        upper = np.triu(matrix, k=1)
        lower = np.tril(matrix, k=-1)
        diag = np.diagonal(matrix)

    return upper, lower, diag


def sor(a: Matrix, b: np.ndarray, omega: float, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    assert max_iter > 0, f"Max number of iteration must be > 0, got {max_iter}"
    assert 0 < omega < 2, f"Omega must be between 0 and 2, got {omega}"

    upper, lower, diag = get_upper_lower_diag(a)
    omega_a_diag = omega / diag

    x = np.zeros(a.shape[1], dtype=np.float64)

    for i in range(max_iter):
        # parallelized part
        x_next = omega_a_diag * (b - upper.dot(x)) + (1 - omega) * x
        # non-parallelized part
        for i in range(x.shape[0]):
            x_next[i] -= omega_a_diag[i] * lower[i].dot(x_next)
        if np.linalg.norm(a.dot(x_next) - b, ord=np.inf) < tol:
            break
        x_prev, x = x, x_next

    return x_next
