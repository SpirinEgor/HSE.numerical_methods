import numpy as np
from matplotlib import pyplot as plt
from time import time


def true_sol(x: float):
    return x + np.exp(1) / (1 - np.exp(2)) * (np.exp(x) - np.exp(-x))


def solve_thomas_v1(A: np.ndarray, d: list):
    """
    функции a_, b_, c_, d_ нужны для сдвига
    инедксов, чтобы соответствовать формуле с лекции
    """

    def a_(i):
        return A[i - 1][i - 2]

    def b_(i):
        return A[i - 1][i - 1]

    def c_(i):
        return A[i - 1][i]

    def d_(i):
        return d[i - 1]

    t_start = time()
    n = d.shape[0]
    alphas = np.zeros(n - 1)
    betas = np.zeros(n)
    alphas[0] = c_(1) / b_(1)
    betas[0] = d_(1) / b_(1)
    for i in range(2, n + 1):
        if i <= n - 1:
            alphas[i - 1] = c_(i) / (b_(i) - a_(i) * alphas[i - 2])
        betas[i - 1] = (d_(i) - a_(i) * betas[i - 2]) / (b_(i) - a_(i) * alphas[i - 2])

    sol = np.zeros(n)
    for i in range(n, 0, -1):
        if i == n:
            sol[i - 1] = betas[i - 1]
        else:
            sol[i - 1] = betas[i - 1] - alphas[i - 1] * sol[i]

    return sol, time() - t_start


def solve_thomas_v2(b: float, c: float, d: list):
    """
    Эта функции не нужна вся матрица, а только элементы
    элементы в пределах каждой из диагоналек.
    Естественно, работает, только если в исходной матрице эти элементы
    в рамках своих диагоналек одинаковые.
    Ее можно удалить, так как, скорее всего, она не нужна на экзамен
    """
    t_start = time()  # наивный способ затаймить
    n = d.shape[0]
    a = c

    alphas = np.zeros(n - 1)
    betas = np.zeros(n)
    alphas[0] = c / b
    betas[0] = d[0] / b
    for i in range(2, n + 1):
        if i <= n - 1:
            alphas[i - 1] = c / (b - a * alphas[i - 2])
        betas[i - 1] = (d[i - 1] - a * betas[i - 2]) / (b - a * alphas[i - 2])

    sol = np.zeros(n)
    for i in range(n, 0, -1):
        if i == n:
            sol[i - 1] = betas[i - 1]
        else:
            sol[i - 1] = betas[i - 1] - alphas[i - 1] * sol[i]

    return sol, time() - t_start


def test(n: int = 100):
    """
    Наивная проверка правильности решения через модуль разности векторов
    """
    h = 1 / (n + 1)
    b = np.arange(1, n + 1) * h
    D = np.eye(n) * (2 * np.power(h, -2) + 1)
    Doff = np.eye(n, k=1) * (-np.power(h, -2))
    A = D + Doff + Doff.T

    ans_v1, _ = solve_thomas_v1(A, b)
    ans_v2, _ = solve_thomas_v2(A[0][0], A[0][1], b)

    true_ans = true_sol(b)
    print(f"n = {n}")
    print("Thomas v1:", np.linalg.norm(ans_v1 - true_ans))
    print("Thomas v2:", np.linalg.norm(ans_v2 - true_ans))


def prep_to_plot_thomas(n: int = 100, step: int = 1):
    x = []
    times_thomas_v1 = []
    diffs_thomas_v1 = []
    times_thomas_v2 = []
    diffs_thomas_v2 = []
    for i in range(2, n + 1, step):
        x.append(i)
        h = 1 / (i + 1)
        b = np.arange(1, i + 1) * h
        D = np.eye(i) * (2 * np.power(i + 1, 2) + 1)
        Doff = np.eye(i, k=1) * (-np.power(i + 1, 2))
        A = D + Doff + Doff.T

        ans1, time_spent1 = solve_thomas_v1(A, b)
        ans2, time_spent2 = solve_thomas_v2(A[0][0], A[0][1], b)
        true_ans = true_sol(b)

        times_thomas_v1.append(time_spent1)
        times_thomas_v2.append(time_spent2)
        diffs_thomas_v1.append(np.linalg.norm(ans1 - true_ans))
        diffs_thomas_v2.append(np.linalg.norm(ans2 - true_ans))
    return (
        np.array(x),
        (np.array(diffs_thomas_v1), np.array(times_thomas_v1)),
        (np.array(diffs_thomas_v2), np.array(times_thomas_v2)),
    )


if __name__ == "__main__":
    for i in range(100, 1001, 100):
        test(i)

    n = 1000
    step = 50
    x, thomas_v1, thomas_v2 = prep_to_plot_thomas(n, step)
    plt.plot(x, thomas_v1[1] * 1000, label="Thomas v1")
    plt.plot(x, thomas_v2[1] * 1000, label="Thomas v2")
    plt.xlabel("N")
    plt.ylabel("Time spent, ms")
    plt.title("Time comparison")
    plt.legend()
    plt.show()
