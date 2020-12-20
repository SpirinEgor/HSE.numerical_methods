import numpy as np

class Householder:
	'''
	Матрица Хаусхолдера (лекция 5) для ненулевого v - такая матрица, что H.dot(x) 
	это отражение x относительно гиперплоскости, ортогональной v. 

	В данной реализации можно задать v самому, но по умолчанию будет выбрано
	v = x +- ||x|| * e. В таком случае умножение H.dot(x) пропорционально вектору 
	e (элемент стандартного базиса) т.е. H.dot(x) = a * e, где a = +-||x||. 
	'''

	def __init__(self):
		# v - вектор, задающий гиперплоскость для отражения
		# mat - матрица отражения Хаусхолдера

		self.v = None
		self.mat = None

	def make_householder(self, x=None, e=None, v=None):
		"""создание матрицы Хаусхолдера"""

		# Надо задать x и e или v. По умолчанию e = [1, 0, .., 0], что удобно
		# для QR разложения

		if v is not None:
			self.mat = np.eye(v.shape[0])
			self.mat -= (2 / np.dot(v, v)) * self.v[:, None] @ self.v[None, :]
		elif x is None:
			raise TypeError("Missing argument 'x' or 'v' ")
		else:
			e = np.eye(x.shape[0])[:, 0]
			self.v = x + np.sign(x @ e) * (x.T @ x) ** .5 * e
			self.mat = np.eye(x.shape[0])
			self.mat -= (2 / (self.v @ self.v)) * self.v[:, None] @ self.v[None, :]
	
	def matr_comp(self, A, type_comp='r'):
		"""умножение на матрицу А за N^2 справа (слева)"""
		
		if self.mat is None or self.v is None:
			raise RuntimeError("Init matrix before computing dot product")

		beta = -2 / (self.v.T @ self.v)

		# умножение справа
		if type_comp == 'r':
			w = beta * A.T @ self.v
			return A + self.v[:, None] @ w[None, :]

		# умножение слева
		if type_comp == 'l':
			w = beta * A @ self.v
			return A + w[:, None] @ self.v[None, :]

def qr_decomp(A):
	"""QR-разложение с использованием Хаусхолдера для квадратных матриц"""
	
	A = A.astype(np.float)
	m, n = A.shape
	Q = np.eye(m)
	H = Householder()
	R = np.copy(A)

	for i in range(n):
		H.make_householder(R[i:, i])
		R[i:, i:] = H.matr_comp(R[i:, i:])
		Q[i:, i:] = H.matr_comp(Q[i:, i:], type_comp='l')
		if i > 0:
			Q[:i, i:] = H.matr_comp(Q[:i, i:], type_comp='l')
	return Q, R


#проверка
def main():
	A = np.array([[4., 1., -2., 2.], [1., 2., 0., 1.], [-2., 0., 3., -2.],[2., 1., -2., -1.]])

	Q, R = qr_decomp(A)
	print("сравни A: \n {} \nПроизведение матриц из разложения: \n {}".format(A, (Q @ R).round(6)))
	print("Это верхнетреугольная матрица: \n", R.round(6))

if __name__ == "__main__":
	main()