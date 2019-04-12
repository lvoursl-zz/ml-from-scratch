import numpy as np

from .svd import SingularValueDecomposition


class PrincipalComponentAnalysis(object):
	def __init__(
		self, latent_space_size=10, n_iter=1000, verbose=False
	):
		super(PrincipalComponentAnalysis, self).__init__()

		self.latent_space_size = latent_space_size
		self.n_iter = n_iter
		self.verbose = verbose

		self.U = None
		self.V = None

	def fit(self, X):
		svd = SingularValueDecomposition(
			latent_space_size=self.latent_space_size,
			n_iter=self.n_iter,
			verbose=self.verbose
		)
		self.U, self.V = svd.decompose(X)

	def transform(self, X):
		if self.U is None and self.V is None:
			raise Exception('PCA is not fitted')
		else:
			return X @ self.V.T