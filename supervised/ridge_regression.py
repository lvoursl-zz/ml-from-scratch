import numpy as np

from .linear_regression import LinearRegression

class RidgeRegression(LinearRegression):
    def __init__(self, learning_rate=0.001, alpha=0.1, n_iter=1000, verbose=False):
        super(RidgeRegression, self).__init__(
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
                    np.sum(np.power(X @ self.W + self.b - y, 2)) / n + np.sum(self.W ** 2) + self.b ** 2
                )

            dw_square = self.alpha * self.W
            db_square = self.alpha * self.b

            dw = X.T @ (X @ self.W + self.b - y) / n + dw_square
            db = np.sum(X @ self.W + self.b - y) / n + db_square
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db