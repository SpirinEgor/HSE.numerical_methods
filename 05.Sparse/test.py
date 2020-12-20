from scipy.sparse import csr_matrix
from main import CSRMatrix
import numpy as np


def test_dot():
    # Не забудьте -- нумерация в лекциях -- с единицы, тут с нуля
    seed = np.random.randint(0, int(1e8))
    print('seed:', seed)
    np.random.seed(seed)
    shapes = np.random.randint(2, int(1e1), size=(int(1e2), 2))
    for c, r in shapes:
        m_dense = np.random.random((c, r))
        m_scipy = csr_matrix(m_dense)
        m_custom = CSRMatrix(m_scipy.data, m_scipy.indptr, m_scipy.indices)

        vec = np.random.random(m_dense.shape[1])
        res_scipy, res_custom = m_scipy.dot(vec), m_custom.dot(vec)
        if not np.allclose(res_scipy, res_custom):
            print(c, r)

    print("it's fine")


def test_sum():
    # Не забудьте -- нумерация в лекциях -- с единицы, тут с нуля

    m1 = CSRMatrix(np.array([1, 3]), np.array([0, 1, 2, 2]), np.array([0, 1]))
    m2 = CSRMatrix(np.array([2, 1, 3, 2, 2]), np.array([0, 3, 3, 5]), np.array([0, 1, 2, 1, 2]))
    res = CSRMatrix(np.array([3, 1, 3, 3, 2, 2]), np.array([0, 3, 4, 6]), np.array([0, 1, 2, 1, 1, 2]))

    print(m1+m2, '\n______\n'+str(res))

    # Не должно работать, сложение матриц разного размера не определено и в scipy
    # m1 = CSRMatrix(np.array([1, 3]), np.array([0, 1, 2, 2]), np.array([0, 1]))
    # m2 = CSRMatrix(np.array([1, 2, 3, 1, 1]), np.array([0, 2, 4, 4, 5]), np.array([1, 2, 1, 3, 3]))
    # res = CSRMatrix(np.array([1, 1, 2, 6, 1, 1]), np.array([0, 3, 5, 5, 6]), np.array([0, 1, 2, 1, 3, 3]))
    #
    # print(m1+m2, '\n'+str(res))

    seed = np.random.randint(0, int(1e8))
    print('____\nseed:', seed)
    np.random.seed(seed)
    shapes = np.random.randint(2, int(1e1), size=(int(1e2), 2))
    for c, r in shapes:
        m_dense1 = np.random.random((c, r))
        m_dense2 = np.random.random((c, r))

        m_scipy1 = csr_matrix(m_dense1)
        m_scipy2 = csr_matrix(m_dense2)

        m_custom1 = CSRMatrix(m_scipy1.data, m_scipy1.indptr, m_scipy1.indices)
        m_custom2 = CSRMatrix(m_scipy2.data, m_scipy2.indptr, m_scipy2.indices)

        res_scipy, res_custom = m_scipy1+m_scipy2, m_custom1+m_custom2
        if (not np.allclose(res_scipy.data, res_custom.values)) \
                and (not np.allclose(res_scipy.indptr, res_custom.ia)) \
                and (not np.allclose(res_scipy.indices, res_custom.ja)):
            print(c, r)

        res_scipy, res_custom = m_scipy1-m_scipy2, m_custom1-m_custom2
        if (not np.allclose(res_scipy.data, res_custom.values)) \
                and (not np.allclose(res_scipy.indptr, res_custom.ia)) \
                and (not np.allclose(res_scipy.indices, res_custom.ja)):
            print(c, r)
    print("it's fine")


test_dot()
test_sum()