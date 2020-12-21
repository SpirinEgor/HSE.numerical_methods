import numpy as np
from sklearn.decomposition import PCA


def SVD(A):
    _, u = np.linalg.eig(A@A.T)
    v, h = np.linalg.eig(A.T@A)

    return u, np.sqrt(v), h.T


def SVD_nice(A):
    tmp_mat = np.block([[np.zeros(A.T.shape), A],
                        [A.T, np.zeros(A.shape)]])
    v, h = np.linalg.eig(tmp_mat)
    h *= np.sqrt(2)
    u = h[:h.shape[0]//2, ::2]
    h = h[h.shape[0]//2:, ::2].T
    return u, v[1::2], h


def PCA_alg(A, n):
    u, v, h = SVD(A)
    inds = np.argsort(v)[::-1]

    sorted_u = u[inds]
    sorted_v = v[inds]
    sorted_h = h[inds]
    return sorted_u[:, :n] @ np.diag(sorted_v)[:n] @ sorted_h[:, :n]


mat = np.array([[1, 3, 4],
                [5, 6, 7],
                [8, 9, 10]])


u_custom, v_custom, h_custom = SVD(mat)
u_custom_nice, v_custom_nice, h_custom_nice = SVD_nice(mat)
u_scipy, v_scipy, h_scipy = np.linalg.svd(mat)

print(np.allclose(abs(u_custom_nice), abs(u_custom)))
print(np.allclose(abs(v_custom_nice), abs(v_custom)))
print(np.allclose(abs(h_custom_nice), abs(h_custom)))

print(np.allclose(abs(u_scipy), abs(u_custom)))
print(np.allclose(abs(v_scipy), abs(v_custom)))
print(np.allclose(abs(h_scipy), abs(h_custom)))
print('_____')
print(PCA_alg(mat, 2))
pca = PCA(n_components=2, svd_solver='full')
print(pca.fit_transform(mat))
print('_____')
print(PCA_alg(mat, 3))
pca = PCA(n_components=3, svd_solver='full')
print(pca.fit_transform(mat))
