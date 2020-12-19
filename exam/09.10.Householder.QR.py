import numpy as np
from numpy.testing import assert_allclose
from typing import Tuple, NoReturn


class Householder:
    def __init__(self):
        self.v = None
        self.x = None

    def _get_v(self, x: np.ndarray) -> np.ndarray:
        e1 = np.zeros_like(self.x)
        e1[0] = 1
        return x + np.sign(self.x[0]) * np.linalg.norm(self.x) * e1

    def set_x(self, x: np.ndarray) -> NoReturn:
        self.x = x
        self.v = self._get_v(x)

    @property
    def h(self) -> np.ndarray:
        return np.identity(self.v.shape[0]) - 2 / (self.v @ self.v) * np.outer(self.v, self.v)

    @property
    def hx(self) -> np.ndarray:
        e1 = np.zeros_like(self.x)
        e1[0] = 1
        return -np.sign(self.x) * np.linalg.norm(x) * e1

    def dot(self, A: np.ndarray) -> np.ndarray:
        beta = -2 / (self.v @ self.v)
        w = beta * A.T @ self.v
        return A + np.outer(self.v, w)


def qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    householder = Householder()
    R = A.copy()
    for i in range(R.shape[1] - 1):
        householder.set_x(R[i:, i])

        # idk how to calculate Q in a more efficient way :(
        if i == 0:
            Q = householder.h
        else:
            H = householder.h
            Q = Q @ np.block([[np.identity(Q.shape[0] - H.shape[0]),
                                        np.zeros((Q.shape[0] - H.shape[0], H.shape[1]))],
                              [np.zeros((H.shape[0],
                                         Q.shape[1] - H.shape[1])), H]])

        R[i:, i:] = householder.dot(R[i:, i:])

    return Q, R


def test() -> NoReturn:
    A = np.array([[1, 7, 5], [4, 5, 6], [7, 8, 22]], dtype=float)
    Q, R = qr(A)
    npq, npr = np.linalg.qr(A)
    assert_allclose([npq, npr], [Q, R], atol=1e-8)

test()