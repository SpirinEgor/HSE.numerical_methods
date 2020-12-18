from typing import Tuple
import scipy.sparse
import numpy as np
from numpy.testing import assert_allclose

from exam.sor_11.sor_yarik import sor


def u0(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(2 * np.pi * y)


def f0(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return u0(x, y) * (5 * np.pi ** 2 + np.exp(x * y))


def build_A_u_f(N: int) -> Tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray]:
    data = []
    row_ind = []
    col_ind = []
    u = np.empty(N * N, dtype=np.float64)
    f = np.empty(N * N, dtype=np.float64)

    def insert(row, col, val):
        row_ind.append(row)
        col_ind.append(col)
        data.append(val)

    h = 1 / (N + 1)
    non_diag_val = -1 / h ** 2
    for i in range(N):
        for j in range(N):
            k = i + j * N
            x = (i + 1) * h
            y = (j + 1) * h

            u[k] = u0(x, y)
            f[k] = f0(x, y)

            insert(k, k, 4 / h ** 2 + np.exp(x * y))
            if i > 0:
                k_ = i - 1 + j * N
                insert(k, k_, non_diag_val)
            if i < N - 1:
                k_ = i + 1 + j * N
                insert(k, k_, non_diag_val)
            if j > 0:
                k_ = i + (j - 1) * N
                insert(k, k_, non_diag_val)
            if j < N - 1:
                k_ = i + (j + 1) * N
                insert(k, k_, non_diag_val)

    a = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(N * N, N * N), dtype=np.float64)
    return a, u, f


def test_sparse():
    N = 10
    a, u, f = build_A_u_f(N)
    for omega in [0.5, 1.0, 1.6, 1.8, 1.8, 1.9, 1.9]:
        x = sor(a, f, omega, max_iter=1000, tol=1e-6)
        assert_allclose(x, u, rtol=5e-2)


def test_dense():
    N = 10
    a, u, f = build_A_u_f(N)
    a = a.todense()
    for omega in [0.5, 1.0, 1.6, 1.8, 1.8, 1.9, 1.9]:
        x = sor(a, f, omega, max_iter=1000, tol=1e-6)
        assert_allclose(x, u, rtol=5e-2)


if __name__ == '__main__':
    test_sparse()
    test_dense()
