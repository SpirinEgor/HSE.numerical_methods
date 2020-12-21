import numpy as np
from main import QR_algo, QR_algo_shifted


def test():
    seed = np.random.randint(0, int(1e8))
    print(f"seed: {seed}")
    np.random.seed(seed)

    shapes = np.random.randint(2, int(1e1), size=(int(1e2)))
    for sh in shapes:
        m = np.random.random((sh, sh))

        nump = np.sort(np.linalg.eigvals(m))[::-1]
        # Алгоритм не умеет в комплексные числа
        if nump.dtype == np.complex128:
            continue

        custom = QR_algo(m)
        custom_s = QR_algo_shifted(m)
        if not np.allclose(custom, nump):
            print(m)
        if not np.allclose(custom_s, nump):
            print(m)

    print("it's fine")


if __name__ == "__main__":
    for _ in range(50):
        test()
