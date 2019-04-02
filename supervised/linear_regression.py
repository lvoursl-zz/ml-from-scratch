import numpy as np

from .base import BaseModel

class LinearRegression(BaseModel):
	def __init__(self, learning_rate=0.001, n_iter=1000, verbose=False):
		super(LinearRegression, self).__init__(verbose)
		self.learning_rate = learning_rate
		self.n_iter = n_iter

	def _init_weights(self, X):
		self.W = np.random.random(size=(X.shape[1], 1))
		self.b = np.random.random()

	def fit(self, X, y):
		self._check_dimensions(X, y)
		self._init_weights(X)

		n = X.shape[0]
		y = y.reshape(y.shape[0], 1)

		for iter_num in range(self.n_iter):
			if self.verbose:
				print('Loss =', np.sum(np.power(X @ self.W + self.b - y, 2)) / n)

			dw = X.T @ (X @ self.W + self.b - y) / n
			db = np.sum(X @ self.W + self.b - y) / n
			self.W -= self.learning_rate * dw
			self.b -= self.learning_rate * db

	def predict(self, X):
		return X @ self.W + self.b