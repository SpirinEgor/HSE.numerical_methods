from typing import Tuple, NoReturn
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import lu


def LU(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L, U = np.eye(A.shape[0]), np.zeros_like(A, dtype=np.float)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
            else:
                L[i, j] = 1 / U[j, j] * (A[i, j] - L[i, :j] @ U[:j, j])
    return L, U


def print_matrix(A: np.ndarray) -> NoReturn:
    for row in A.astype(int):
        for x in row:
            print(f"{x:<4d}", end=" ")
        print()


def test(LU, A: np.ndarray, print_matrix) -> NoReturn:
    assert A.shape[0] == A.shape[1], f"Matrix should be square"
    true_L, true_U = lu(A)[1:3]
    L, U = LU(A)
    assert_allclose([true_L, true_U], [L, U], atol=1e-8), f"Wrong answer"
    for key, value in {'A': A, 'L': L, 'U': U}.items():
        print(key+':')
        print_matrix(value)


def main(A: np.ndarray, LU, test, print_matrix) -> NoReturn:
    test(LU, A, print_matrix)


'A - correct test matrix'
A = np.array([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]])
main(A, LU, test, print_matrix)
