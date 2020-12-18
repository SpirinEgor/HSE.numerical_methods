import numpy as np


def cholesky(A: np.ndarray) -> np.ndarray:
    L = np.zeros((A.shape[0], A.shape[0]))

    if A[0, 0] < 0:
        raise ValueError("matrix is not positive definite")

    L[0, 0] = A[0, 0] ** 0.5
    # first column
    for j in range(1, A.shape[0]):
        L[j, 0] = A[j, 0] / L[0, 0]

    for i in range(1, A.shape[0]):
        # number of elements in a row = row number + 1
        for j in range(1, i + 1):
            if i == j:
                val = A[i, i] - np.sum(np.square(L[i, :i]))
                if val < 0:
                    raise ValueError("matrix is not positive definite")
                L[i, i] = val ** 0.5
            else:
                L[i, j] = 1 / L[j, j] * (A[i, j] - L[j, :j] @ L[i, :j])
    return L
