from typing import Tuple, NoReturn
import numpy as np
from scipy.linalg import lu

# A = test matrix
A = np.array([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]])

def LU(A:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix should be square")
    L, U = np.diag(np.ones(A.shape[0])), np.zeros_like(A, dtype = np.float)
    n = A.shape[0]

    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
            else:
                L[i, j] = 1 / U[j, j] * (A[i, j] - L[i, :j] @ U[:j, j])
    return (L, U)

def test(LU, A:np.ndarray) -> NoReturn:
    true_L, true_U = lu(A)[1], lu(A)[2]
    L, U = LU(A)
    ans = np.isclose([true_L, true_U], [L, U], atol = 1e-8)

    if np.all(ans == True):
        print('Correct')
    else:
        print('Incorrect')
