from typing import Union
import scipy.sparse
import numpy as np

from exam.sor_11.sor_yarik import get_upper_lower_diag

Matrix = Union[np.ndarray, scipy.sparse.csr_matrix]


# Лекция 4. Метод Зейделя
# Часть со слагаемыми x_k распараллелена
def zeidel_parallelized(
    a: Matrix, b: np.ndarray, max_iter: int = 1000, tol: float = 1e-6
) -> np.ndarray:
    assert max_iter > 0, f"Max number of iteration must be > 0, got {max_iter}"

    upper, lower, diag = get_upper_lower_diag(a)
    x = np.zeros(a.shape[1], dtype=np.float64)

    for i in range(max_iter):
        # parallelized part (слагаемые с x^k)
        x = (b - upper.dot(x)) / diag
        # non-parallelized part (слагаемое с x^k+1)
        for i in range(x.shape[0]):
            x[i] -= lower[i].dot(x) / diag[i]

        if np.linalg.norm(a.dot(x) - b, ord=np.inf) < tol:
            break

    return x
