import numpy as np
from scipy import linalg


def jacobi(A: np.ndarray, b: np.ndarray, x0: np.ndarray, max_iter=1000, tol=1.e-10) -> np.ndarray:
    """
    Метод Якоби для решения СЛАУ Ax = b с условием строгого диагонального преобладания:
        A_{ii} > sum_{i \neq j}|A_{ij}|

    Parametrs
    ---------
    :param A: np.ndarray
        Матрица решаемой СЛАУ. Для сходимотси достаточно (но не необходимо), чтобы матрица А
        имела строгое диагональное преобладание
    :param b: np.ndarray
        Вектор свободных членов СЛАУ
    :param x0: np.ndarray
        Начальное приближение
    :param max_iter: int
        Максимальное число итераций метода
    :param tol: float
        Точноcть, с котрой метод ищет приближение к точном решению. Используется условие остановки
        norm(b - A @ x_i) / norm(b) <= tol, где x_i -- текущее приближение решения
    :return: np.ndarray
        Приближенное решение x СЛАУ Ax = b

    """
    n = A.shape[0]
    n_iter = 0
    b_norm = linalg.norm(b)
    E = np.identity(n)
    D_inv = np.diag(1 / A.diagonal())
    B = E - D_inv @ A
    q = np.dot(D_inv, b)
    now = x0
    err = linalg.norm(b - A @ now) / b_norm
    while n_iter < max_iter and err > tol:
        now = np.dot(B, now) + q
        err = linalg.norm(b - A @ now) / b_norm
        n_iter += 1
    return now


def test(size, times):
    b = np.ones((size, 1))
    x0 = np.ones((size, 1))
    for _ in range(times):
        A = np.diag(np.random.uniform(size + 1, 2 * size, size=size))
        for i in range(size):
            for j in range(size):
                if i != j:
                    A[i, j] = np.random.uniform(-size / (size-1), size / (size - 1))

        assert np.allclose(linalg.solve(A, b), jacobi(A, b, x0))


if __name__ == '__main__':
    test(size=10, times=100)
