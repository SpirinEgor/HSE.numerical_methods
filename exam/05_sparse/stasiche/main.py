import numpy as np


class CSRMatrix:
    def __init__(self, values: np.ndarray, ia: np.ndarray, ja: np.ndarray):
        self.values = values
        self.ia = ia
        self.ja = ja

        self.cols_num = self.ia.shape[0]-1

    def __str__(self):
        return 'values: {}\nia: {}\nja: {}'.format(self.values, self.ia, self.ja)

    def __add__(self, other):
        return self.__elw_func(other, lambda x, y: x+y)

    def __sub__(self, other):
        return self.__elw_func(other, lambda x, y: x-y)

    def dot(self, vec: np.ndarray):
        y = np.empty(self.cols_num)
        for i in range(self.cols_num):
            y[i] = self.values[self.ia[i]: self.ia[i+1]] @ vec[self.ja[self.ia[i]: self.ia[i+1]]]
        return y

    def __elw_func(self, other, func):
        if not isinstance(other, CSRMatrix):
            raise NotImplementedError

        # TODO replace by ndarray
        res_val = []
        res_ia = [0]
        res_ja = []
        for i in range(self.cols_num):
            res_ia.append(res_ia[-1])

            vals1 = self.values[self.ia[i]: self.ia[i + 1]]
            vals2 = other.values[other.ia[i]: other.ia[i + 1]]

            l1 = vals1.shape[0]
            l2 = vals2.shape[0]

            ja1 = self.ja[self.ia[i]: self.ia[i + 1]]
            ja2 = other.ja[other.ia[i]: other.ia[i + 1]]

            p1 = p2 = 0
            while p1 != l1 and p2 != l2:
                if p1 < l1 and ja1[p1] < ja2[p2]:
                    res_val.append(vals1[p1])
                    res_ia[-1] += 1
                    res_ja.append(ja1[p1])

                    p1 += 1
                elif p2 < l2 and ja2[p2] < ja1[p1]:
                    res_val.append(vals2[p2])
                    res_ia[-1] += 1
                    res_ja.append(ja2[p2])

                    p2 += 1
                elif p1 < l1 and p2 < l2:
                    res_val.append(func(vals1[p1], vals2[p2]))
                    res_ia[-1] += 1
                    res_ja.append(ja1[p1])

                    p1 += 1
                    p2 += 1

            while p1 != l1:
                res_val.append(vals1[p1])
                res_ia[-1] += 1
                res_ja.append(ja1[p1])
                p1 += 1

            while p2 != l2:
                res_val.append(vals2[p2])
                res_ia[-1] += 1
                res_ja.append(ja2[p2])
                p2 += 1

        res_matrix = CSRMatrix(np.array(res_val), np.array(res_ia), np.array(res_ja))
        return res_matrix

