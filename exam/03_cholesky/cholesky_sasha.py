import numpy as np

from scipy import linalg
from typing import Tuple


def cholesky(A: np.ndarray) -> np.ndarray:
    """
    Алгоритм Холецкого-Банашевича для разложения симметричной
    положительно-определенной матрицы А в виде:
            
            A = L @ L.T
    
    где L - нижняя треугольная матрица со строго положительными
    элементами на диагонали. 
    
    Разложение Холецкого всегда существует и единственно для 
    любой симметричной положительно-определённой матрицы. 
    
    Другим, часто полезным, вариантом является LDL (square-root-free Cholesky) разложение:
    
        A = L @ D @ L.T
        
    где L - нижняя унитриугольная матрица, а D - диагональная.
    В таком виде имплементация не требует извлечения корней.
    
    Оба варианта разложения связаны между собой:
    
        (=>) A = L @ D @ L.T = (L @ D^(1/2)) @ (L @ D^(1/2))
        
        (<=) A = L @ L.T = L' @ D @ L'.T, гдe L' = L @ (1 / diag(L)), D = diag(L)^2
        
    Разложение Холецкого приемущественно используется для решения систем линейных уравнений[1],
    т.к. обладает меньшей константой (порядка O(n^3/3) операций), чем алгоритм Гаусса O(n^3) или LU разложение O(2n^3/2). 
            
    
    References
    ----------
    [1]: https://math.stackexchange.com/questions/2422012/solving-a-linear-system-with-cholesky-factorization/2422703
    
    
    Parameters
    ----------
    A: np.ndarray
        Симметричная положительно-определенная матрца

    Returns
    -------
    L: np.ndarray
        Нижняя треугольная матрица такая, что L @ L.T == A

    """
    L = np.zeros(A.shape)
    
    for i in range(A.shape[0]):
        for j in range(i + 1):  
            value = A[i, j] - np.sum(L[i, :-1] * L[j, :-1])
            
            if i == j:
                L[i, i] = np.sqrt(value)
            else:
                L[i, j] = 1 / L[j, j] * value
    return L


def test(size, times):
    for _ in range(times):
        A_ = np.random.rand(size, size)
        A = A_ @ A_.T

        assert np.allclose(linalg.cholesky(A, lower=True), cholesky(A))


if __name__ == "__main__":
    test(size=10, times=100)