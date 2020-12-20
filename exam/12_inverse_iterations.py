import numpy as np

from typing import Tuple

np.random.seed(42)

def inverse_iteration(A: np.ndarray, mu: float=0.0, tol: float=1e-8, max_iter: int=10000) -> Tuple[float, np.ndarray]:
    """
    Метод обратных итераций со сдвигом[1][2] & метод Рэлея[3] - итеративный методы вычисления собственных значений и векторов.
    Метод Рэлея имеет кубическую сходимость, в то время как обычный метод обратных итераций - толькой линейную. 
    
    Обычный методы обратных итераций - mu фиксирована с самого начала и не меняется.
    Метод Рэлея - mu перевычисялется на каждой итерации с помощью отношения Рэлея.
    
    Позволяет искать собственные вектора и собственные значения произвольной матрицы.
    Обычно используется для поиска собственных векторов, если известны хорошие приближения для собственных чисел.
    Сходится к ближайшему к начальному mu собственному значению. 
    При mu=0.0 сходится к минимальному по модулю собственному значению (но есть нюансы).
    
    References
    ----------
    [1]: https://en.wikipedia.org/wiki/Inverse_iteration
    [2]: https://wiki.compscicenter.ru/images/1/10/NMM20_lec6.pdf, слайд 10-11
    [3]: https://en.wikipedia.org/wiki/Rayleigh_quotient_iteration


    Parameters
    ----------
    A: np.ndarray
        Произвольная обратимая квадратная матрица.
        
    mu: float=0.0
        Начальное приближение для собственного значения.
    
    tol: float=1e-8
        Точность, после достижения которого метод останавливается.
    
    max_iter: int=10000
        Максимальное количество итераций метода.

    Returns
    -------
    eigvalue: float
        Ближайшее по модулю к mu собственное число.
    
    eigvector: np.ndarray
        Собственный вектор для найденного собственного числа.

    """
    u, I = np.ones(shape=(A.shape[1], 1)), np.identity(A.shape[0])
    # mu_const = 0.0

    for _ in range(max_iter):
        w = u / np.linalg.norm(u)
        
        # u = np.linalg.inv(A - mu_const*I) @ w
        u = np.linalg.inv(A - mu*I) @ w
        mu_new = u.T @ (A @ u) / (u.T @ u)

        if np.abs(mu - mu_new) < tol:
            break
        
        mu = mu_new
        
    return mu[0][0], u

