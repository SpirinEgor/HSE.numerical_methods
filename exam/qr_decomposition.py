import numpy as np
from typing import Tuple


def qr_householder(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Преобразование Хаусхолдера -- линейный оператор вида H = I - 2vv^t, где v -- вектор единичной длины. В частности,
    позволяет занулить определенные компоненты вектора. Не сложно заметить (проверяется обычной подстановкой), что
    1. H^t = H
    2. HH^t = I
    Т.е. такой опреатор симметричен и ортогонален.

    Идея заключается в том, чтобы с помощью преобразований Хаусхолдера, занулять элементы, лежащие
    под главной диагональю, в итоге получаем произведение ортогональных матриц, которое превращает A в
    верхнетреугольную матрицу R: H @ A = R  ==>  A = H.T @ R.

    Если A.shape = (n, m), то итераций min(n, m). На каждой итерации i мы составляем матрицу Хаусхолдера так,
    чтобы занулить все элементы A[i:, i], кроме первого.

    Подробнее можно посмотреть здесь (http://mlwiki.org/index.php/Householder_Transformation) и в 5 лекции.
    """
    m, n = A.shape
    Q = np.eye(m)  # Orthogonal transform so far
    R = A.copy()  # Transformed matrix so far

    for j in range(min(n, m)):
        # Find H = I - beta * u * u.T to put zeros below R[j,j]
        x = R[j:, j]
        normx = np.linalg.norm(x)
        rho = -np.sign(x[0])
        u1 = x[0] - rho * normx
        u = x / u1
        u[0] = 1
        beta = -rho * u1 / normx

        u = u.reshape(-1, 1)

        # update matrices
        R[j:, :] = R[j:, :] - beta * u @ (u.T @ R[j:, :])
        Q[:, j:] = Q[:, j:] - beta * (Q[:, j:] @ u) @ u.T

    return Q, R


def test(A: np.ndarray):
    Q, R = qr_householder(A)

    # checking if Q is orthogonal
    print(f'Q orthogonality: {np.linalg.norm(Q @ Q.T - np.eye(Q.shape[0]), ord=np.inf)}')

    # checking if R's lower triangle is zero
    lower_tr_ind = np.tril_indices(n=R.shape[0], m=R.shape[1], k=-1)
    lower_tr_max = np.linalg.norm(R[lower_tr_ind], ord=np.inf)
    print(f'R lower part norm: {lower_tr_max}')

    # checking if QR = A
    print(f'A norm: {np.linalg.norm(A - Q @ R, ord=np.inf)}\n')


def rand_tests():
    A = np.random.normal(0, 1, size=(33, 17))
    test(A)

    A = np.random.normal(2, 33, size=(24, 65))
    test(A)

    A = np.random.normal(4, 77, size=(500, 500))
    test(A)


if __name__ == '__main__':
    rand_tests()
