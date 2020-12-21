import numpy as np


# TODO stop crit
# TODO smarter algos
def QR_algo(A: np.ndarray, eps: float = 1e-8):
    A_ = A.copy()

    p_evals = 0
    evals = p_evals + eps + 1
    while abs(np.linalg.norm(p_evals - evals)):
        q, r = np.linalg.qr(A_)
        A_ = r @ q

        p_evals = evals
        evals = np.sort(np.unique(np.diag(A_)))[::-1]

    return evals


