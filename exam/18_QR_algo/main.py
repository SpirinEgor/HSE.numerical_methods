import numpy as np


def QR_algo(A: np.ndarray, max_iters: int = 1000, eps: float = 1e-8):
    p_evals = np.zeros(A.shape[0])
    evals = p_evals + eps + 1

    for _ in range(max_iters):
        if np.allclose(p_evals, evals, rtol=eps):
            break

        q, r = np.linalg.qr(A)
        A = r @ q

        p_evals = evals
        evals = np.diag(A)

    return np.sort(np.unique(evals))[::-1]
