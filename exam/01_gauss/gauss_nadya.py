import numpy as np


def gauss(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix A must be square")

    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b dimensions don't match")

    # add b as a column to A
    A_ = np.hstack((A, b.reshape(-1, 1)))
    for j in range(A_.shape[0]):
        A_[j] = A_[j] / A_[j, j]
        for i in range(j + 1, A_.shape[0]):
            A_[i] = A_[i] - A_[j] * A_[i, j]

    # extract updated b from updated A
    b_ = A_[:, -1]
    A_ = A_[:, :-1]
    x = np.zeros_like(b_)

    # backtracking
    x[-1] = b_[-1] / A_[-1, -1]
    for i in reversed(range(b_.shape[0] - 1)):
        x[i] = 1 / A_[i, i] * (b_[i] - A_[i, i + 1:] @ x[i + 1:])

    return x
