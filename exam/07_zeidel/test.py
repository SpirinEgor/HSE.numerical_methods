from numpy.testing import assert_allclose
import scipy.sparse.linalg as splinalg
import numpy as np

from exam.sor_11.test import build_A_u_f
from exam.zeidel_10.zeidel_yarik import zeidel_parallelized


def test_sparse():
    N = 10
    a, u, f = build_A_u_f(N)

    x = zeidel_parallelized(a, f, max_iter=1000, tol=1e-7)
    assert_allclose(x, u, rtol=5e-2)

    scipy_x = splinalg.spsolve(a, f)
    assert_allclose(x, scipy_x)


def test_dense():
    N = 10
    a, u, f = build_A_u_f(N)
    a = a.todense()

    x = zeidel_parallelized(a, f, max_iter=1000, tol=1e-7)
    assert_allclose(x, u, rtol=5e-2)

    numpy_x = np.linalg.solve(a, f)
    assert_allclose(x, numpy_x)


if __name__ == "__main__":
    test_sparse()
    test_dense()
