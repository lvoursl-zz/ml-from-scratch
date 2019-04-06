import numpy as np

from .linear_regression import LinearRegression

class LogisticRegression(LinearRegression):
    def __init__(self, learning_rate=0.001, n_iter=1000, verbose=False):
        super(LogisticRegression, self).__init__(
            learning_rate=learning_rate, verbose=verbose, n_iter=n_iter
        )

    def fit(self, X, y):
        self._check_dimensions(X, y)
        self._init_weights(X)

        n = X.shape[0]
        y = y.reshape(y.shape[0], 1)

        for iter_num in range(self.n_iter):
            if self.verbose:
                print(
                    'Loss =', 
                    np.sum(y * np.log1p(self.predict(X)) + (1 - y) * np.log1p(1 - self.predict(X))) / n
                )

            dw = X.T @ (self.predict(X) - y) / n
            db = np.sum(self.predict(X) - y) / n
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        return 1 / (1 + 1 / np.exp(X @ self.W + self.b))

    def predict_classes(self, X, threshold=0.5):
        probabilities = self.predict(X)
        probabilities[probabilities >= threshold] = 1
        probabilities[probabilities < threshold] = 0
        return probabilities