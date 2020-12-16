import numpy as np


class CSRMatrix:
    def __init__(self, values: np.ndarray, ia: np.ndarray, ja: np.ndarray):
        self.nnz = len(values)
        self.values = values
        self.ia = ia
        self.ja = ja

        self.cols_num = self.ia.shape[0]-1

    def dot(self, vec: np.ndarray):
        y = np.empty(self.cols_num)
        for i in range(self.cols_num):
            y[i] = self.values[self.ia[i]: self.ia[i+1]] @ vec[self.ja[self.ia[i]: self.ia[i+1]]]
        return y
