import numpy as np

class SingularValueDecomposition(object):
	def __init__(
		self, latent_space_size=10, learning_rate=0.01, 
		regularization=0.001, n_iter=1000, verbose=True
	):
		super(SingularValueDecomposition, self).__init__()
		self.latent_space_size = latent_space_size
		self.learning_rate = learning_rate
		self.regularization = regularization
		self.n_iter = n_iter
		self.verbose = verbose

		self._loss_history = []

	def decompose(self, X):
		samples_num = X.shape[0]
		features_num = X.shape[1]

		U = np.random.random(size=(samples_num, self.latent_space_size))
		V = np.random.random(size=(self.latent_space_size, features_num))

		for i in range(self.n_iter):
			loss = np.sum((X - U @ V) ** 2)
			self._loss_history.append(loss)

			if self.verbose:
				print('MSE =', loss)

			dU = (U @ V - X) @ V.T
			dV = U.T @ (U @ V - X)

			U -= self.learning_rate * dU - self.regularization * U
			V -= self.learning_rate * dV - self.regularization * V

		return U, V