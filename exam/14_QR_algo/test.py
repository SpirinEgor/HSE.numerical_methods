from main import QR_algo
import numpy as np

seed = np.random.randint(0, int(1e8))
print('seed:', seed)
np.random.seed(seed)
shapes = np.random.randint(2, int(1e1), size=(int(1e2)))
for sh in shapes:
    m = np.random.random((sh, sh))
    nump = np.sort(np.linalg.eigvals(m))[::-1]
    # Алгоритм не умеет в комплексные числа
    if nump.dtype == np.complex128:
        continue
    custom = QR_algo(m)
    if not np.allclose(custom, nump):
        print(m)
print("it's fine")
# m = np.array([[0.62477314, 0.60520494], [0.6471618, 0.43945216]])
# print(QR_algo(m))
# print(np.linalg.eigvals(m))
# print()
