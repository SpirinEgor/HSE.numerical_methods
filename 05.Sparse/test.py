from scipy.sparse import csr_matrix
from main import CSRMatrix
import numpy as np

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
