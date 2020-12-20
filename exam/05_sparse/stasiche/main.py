import numpy as np

from typing import Tuple


class CSRMatrix:
    def __init__(self, values: np.ndarray, ia: np.ndarray, ja: np.ndarray):
        self.values = values    # список ненулевых значений
        self.ia = ia            # префиксная сумма количества элементов в строках
        self.ja = ja            # номер столбца каждого ненулевого элемента

        self.cols_num = self.ia.shape[0]-1

    def __str__(self):
        return 'values: {}\nia: {}\nja: {}'.format(self.values, self.ia, self.ja)

    def __add__(self, other):
        return self.__elw_func(other, lambda x, y: x+y)

    def __sub__(self, other):
        return self.__elw_func(other, lambda x, y: x-y)

    def dot(self, vec: Tuple[np.ndarray, 'CSRMatrix']):
        y = np.empty(self.cols_num)

        if isinstance(vec, np.ndarray):
            for i in range(self.cols_num):
                y[i] = self.values[self.ia[i]: self.ia[i+1]] @ vec[self.ja[self.ia[i]: self.ia[i+1]]]
        elif isinstance(vec, CSRMatrix):
            for i in range(self.cols_num):
                y[i] = self.values[self.ia[i]: self.ia[i + 1]] @ vec.values[self.ja[self.ia[i]: self.ia[i + 1]]]
        else:
            raise NotImplementedError

        return y

    def __elw_func(self, other, func):
        if not isinstance(other, CSRMatrix):
            raise NotImplementedError

        none_obj = (None, (None, None))
        # TODO replace by ndarray
        res_val = []
        res_ia = [0]
        res_ja = []
        for i in range(self.cols_num):
            res_ia.append(res_ia[-1])

            it1 = enumerate(zip(self.values[self.ia[i]: self.ia[i + 1]], self.ja[self.ia[i]: self.ia[i + 1]]))
            it2 = enumerate(zip(other.values[other.ia[i]: other.ia[i + 1]], other.ja[other.ia[i]: other.ia[i + 1]]))

            p1, (val1, ja1) = next(it1, none_obj)
            p2, (val2, ja2) = next(it2, none_obj)
            while p1 is not None and p2 is not None:
                if p1 is not None and ja1 < ja2:
                    val, ja = val1, ja1
                    p1, (val1, ja1) = next(it1, none_obj)
                elif p2 is not None and ja2 < ja1:
                    val, ja = val2, ja2
                    p2, (val2, ja2) = next(it2, none_obj)
                else:
                    val, ja = func(val1, val2), ja1
                    p1, (val1, ja1) = next(it1, none_obj)
                    p2, (val2, ja2) = next(it2, none_obj)

                if val:
                    res_val.append(val)
                    res_ia[-1] += 1
                    res_ja.append(ja)

            it, p, val, ja = (it1, p1, val1, ja1) if p1 is not None else (it2, p2, val2, ja2)
            while p is not None:
                res_val.append(val)
                res_ia[-1] += 1
                res_ja.append(ja)
                p, (val, ja) = next(it, none_obj)

        res_matrix = CSRMatrix(np.array(res_val), np.array(res_ia), np.array(res_ja))
        return res_matrix

