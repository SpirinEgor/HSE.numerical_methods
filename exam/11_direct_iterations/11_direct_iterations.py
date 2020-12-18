import numpy as np
import scipy
from scipy import sparse
from scipy import linalg
from matplotlib import pyplot as plt

# служебные функции, по которым строится матричная задача
def u(x, y):
    return np.sin(np.pi * x) * np.sin(2 * np.pi * y)


def f(x, y):
    return (5 * np.pi ** 2 + np.exp(x * y)) * u(x, y)


def construct(u: callable, f: callable, n: int = 10):
    """
    Конструирует матрицу A для СЛАУ A @ u = f
    
    Parameters
    ==========
    
    u: np.ndarray 
        "неизвестный" вектор
    
    f: np.ndarray
        вектор правой части СЛАУ
        
    n: int = 10
        показатель дробления интервала (0, 1), на котором ищется решение
    
    Returns
    =======
    
    A: np.ndarray
        разреженная матрица А в формате 'csr',
        построенная по набору (value, (row, col))
        
    u: np.ndarray
        дискретизированный вектор "неизвестных"
    
    f: np.ndarray
        дискретизированный вектор правой части
    """

    h = 1 / (n + 1)
    nn = n ** 2
    a = 4 * (n + 1) ** 2
    b = -1 * (n + 1) ** 2

    row = []
    col = []
    value = []

    xs = np.zeros(shape=(nn, 1))
    ys = np.zeros(shape=(nn, 1))

    for i in range(n):
        for j in range(n):
            k = i + j * n
            x = (i + 1) * h
            y = (j + 1) * h

            row.append(k)
            col.append(k)
            value.append(a + np.exp(x * y))

            if i >= 1:
                row.append(k)
                col.append(k - 1)
                value.append(b)
            if i <= n - 2:
                row.append(k)
                col.append(k + 1)
                value.append(b)

            if j >= 1:
                row.append(k)
                col.append(k - n)
                value.append(b)
            if j <= n - 2:
                row.append(k)
                col.append(k + n)
                value.append(b)

            xs[k] = x
            ys[k] = y
    return sparse.csr_matrix((value, (row, col)), shape=(nn, nn)), u(xs, ys), f(xs, ys)


def direct_iter(A: np.ndarray, max_iter: int = 10000, tol: float = 1e-8, normalization_step: int = 1):
    """
    Находит наибольшее собственное число матрицы A
    
    Parameters
    ==========
    
    A: np.ndarray
        матрицы, собственное значение которой надо найти
        
    max_iter: int = 10000
        ограничение максимального числа итераций
        
    tol: float = 1e-8
        точность, с которой ищется собственное число
        
    normalization_step: int = 1
        частоста, с которой проводится периодическая нормализаций собственного вектора
    
    Returns
    =======
    
    ev1: float
        наибольшее собственное число матрицы A
    
    Reference
    =========
    
    [1] https://wiki.compscicenter.ru/images/1/10/NMM20_lec6.pdf 6 -- 8 стр.
    """

    u1_prev = np.ones(shape=(A.shape[1], 1))
    u1_prev = u1_prev / np.linalg.norm(u1_prev)

    u2_prev = np.ones(shape=(A.shape[1], 1))
    u2_prev = u2_prev / np.linalg.norm(u2_prev)

    ev1 = ev1_prev = 0
    ev2 = ev2_prev = 0
    for k in range(max_iter):
        u1 = A @ u1_prev
        ev1 = (u1.T @ u1_prev / (u1_prev.T @ u1_prev)).ravel()[0]

        # здесь происходит приближенный поиск второго по величине собственного числа
        # оно пригодится для критерия остановки
        u2 = A @ u2_prev - ev1 * (u1.T @ u2_prev).ravel()[0] * u1
        ev2 = (u2.T @ u2_prev / (u2_prev.T @ u2_prev)).ravel()[0]

        err = np.abs(ev1 - ev1_prev)
        # naive break condition --> algo will not exit immediately by next condition inside
        if err < tol:
            q = ev2 / ev1
            if q / (1 - q) * err < tol:
                break

        ev1_prev = ev1
        ev2_prev = ev2
        # normilizing vector every iteration step
        # better to update with some condition or
        # at certain interation steps, e.g., every 5th step
        if k % normalization_step == 0:
            u1_prev = u1 / np.linalg.norm(u1)
            u2_prev = u2 / np.linalg.norm(u2)
        else:
            u1_prev = u1.copy()
            u2_prev = u2.copy()

    return ev1


def find_minmax(A: np.ndarray, max_iter: int = 10000, tol: float = 1e-8, normalization_step: int = 1):
    """
    Использует метод сдвига, чтобы найти наименьшее собственное число
    заодно выдает и наибольшее, так как оно всё равно нужно для расчета
    
    Parameters
    ==========
    
    A: np.ndarray
        матрицы, собственное значение которой надо найти
        
    max_iter: int = 10000
        ограничение максимального числа итераций
        
    tol: float = 1e-8
        точность, с которой ищется собственное число
        
    normalization_step: int = 1
        частоста, с которой проводится периодическая нормализаций собственного вектора
    
    Returns
    =======
    
    ev_min: float
        наименьшее собственное число матрицы A
    
    ev_max: float
        наибольшее собственное число матрицы A
        
    Reference
    =========
    
    [1] https://wiki.compscicenter.ru/images/1/10/NMM20_lec6.pdf 9 стр.
    """

    ev_max = direct_iter(A, max_iter=max_iter, tol=tol, normalization_step=normalization_step)
    # небольшая сдвижечка, так как в лекции нужно значение большее,
    # чем наибольшее собственное значение, хотя работает и без нее
    alpha = ev_max + tol
    ev_min = direct_iter(A - alpha * sparse.eye(A.shape[0]),
                         max_iter=max_iter,
                         tol=tol,
                         normalization_step=normalization_step)
    return ev_min + alpha, ev_max


def get_data(start: int = 5, finish: int = 51, step: int = 5):
    """
    Подгатавливает данные для графиков
    для различных размеров исходной матрицы A:
    shape(A) = (n ** 2, n ** 2), где n in range(start, finish, step)
    """

    x = []
    y_max = []
    y_min = []
    q_jacobi = []
    q_seidel = []
    for n in range(start, finish, step):
        """
        Time spent:
            5m 14s for range(5, 101, 5)
            49s for range(5, 51)
            10s for range(5, 51, 5)
        """

        x.append(n)
        A, _, _ = construct(u, f, n)
        D = sparse.diags(A.diagonal())

        # find min and max eigenvalue of initial matrix A
        min_, max_ = find_minmax(A)
        y_max.append(np.abs(max_))
        y_min.append(np.abs(min_))

        # find q-coefficient for  Jacobi method
        B = A - D
        D_ = sparse.diags(1 / D.diagonal())
        M_jacobi = D_ @ B
        q_jacobi.append(np.abs(direct_iter(M_jacobi)))

        # find q-coefficient for Seidel method
        L = sparse.tril(A, k=-1)
        U = sparse.triu(A, k=1).A
        # note: if inverse matrix is going to be dense then
        # it is better to use scipy.linalg.inv rather than sparse.linalg.inv
        M_seidel = linalg.inv((D + L).A) @ U
        q_seidel.append(np.abs(direct_iter(M_seidel)))
    x = np.power(np.array(x), 2)
    y_min = np.array(y_min)
    y_max = np.array(y_max)
    q_jacobi = np.array(q_jacobi)
    q_seidel = np.array(q_seidel)
    return x, y_min, y_max, q_jacobi, q_seidel


# Пачка функций для графиков
def plot_max(x, y_max):
    plt.plot(x, y_max)
    plt.xlabel("$N^2$")
    plt.ylabel("$\\lambda_{\\max}$")
    plt.title("Task №1: $\\lambda_{\\max}$ vs $N^2$")
    plt.show()


def plot_min(x, y_min):
    plt.plot(x, y_min)
    plt.xlabel("$N^2$")
    plt.ylabel("$\\lambda_{\\min}$")
    plt.title("Task №1: $\\lambda_{\\min}$ vs $N^2$")
    plt.show()


def plot_kappa(x, y_max, y_min):
    plt.plot(x, y_max / y_min)
    plt.xlabel("$N^2$")
    plt.ylabel("$\\varkappa$")
    plt.title("Task №2: $\\varkappa$ vs $N^2$")
    plt.show()


def plot_Jacobi_vs_Seidel(x, q_jacobi, q_seidel):
    plt.plot(x, 1 / (1 - q_jacobi), label="Jacobi")
    plt.plot(x, 1 / (1 - q_seidel), label="Seidel")
    plt.xlabel("$N^2$")
    plt.ylabel("$\\frac{1}{1 - q}$")
    plt.title("Task №3: Jacobi vs Seidel")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    x, y_min, y_max, q_jacobi, q_seidel = get_data()

    plot_max(x, y_max)
    plot_min(x, y_min)
    plot_kappa(x, y_max, y_min)
    plot_Jacobi_vs_Seidel(x, q_jacobi, q_seidel)
