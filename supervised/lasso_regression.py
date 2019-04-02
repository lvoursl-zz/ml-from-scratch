import numpy as np

from .linear_regression import LinearRegression

class LassoRegression(LinearRegression):
	def __init__(self, learning_rate=0.001, alpha=0.1, n_iter=1000, verbose=False):
		super(LassoRegression, self).__init__(
			learning_rate=learning_rate, verbose=verbose, n_iter=n_iter
		)		
		self.alpha = alpha

	def fit(self, X, y):
		self._check_dimensions(X, y)
		self._init_weights(X)

		n = X.shape[0]
		y = y.reshape(y.shape[0], 1)

		for iter_num in range(self.n_iter):
			if self.verbose:
				print(
					'Loss =', 
					np.sum(np.power(X @ self.W + self.b - y, 2)) / n + np.sum(np.abs(self.W)) + np.abs(self.b)
				)

			dw_module = self.W.copy()
			dw_module[dw_module > 0] = 1
			dw_module[dw_module < 0] = -1
			dw_module[dw_module == 0] = 0
			dw_module = self.alpha * dw_module

			if self.b > 0:
				db_module = 1
			elif self.b < 0:
				db_module = -1
			else:
				db_module = 0

			db_module = self.alpha * db_module

			dw = X.T @ (X @ self.W + self.b - y) / n + dw_module
			db = np.sum(X @ self.W + self.b - y) / n + db_module
			self.W -= self.learning_rate * dw
			self.b -= self.learning_rate * db

	def predict(self, X):
		return X @ self.W + self.b