import numpy as np

from .linear_regression import LinearRegression


class QuantileRegression(LinearRegression):
    def __init__(self, quantile=0.5, learning_rate=0.001, n_iter=1000, verbose=False):
        super(QuantileRegression, self).__init__(
            learning_rate=learning_rate, verbose=verbose, n_iter=n_iter
        )
        self.quantile = quantile
        self.loss = []

    def fit(self, X, y):
        self._check_dimensions(X, y)
        self._init_weights(X)

        y = y.reshape(y.shape[0], 1)

        for iter_num in range(self.n_iter):
            y_hat = X @ self.W + self.b
            residuals = y_hat - y

            biggest_or_equal = np.argwhere(residuals >= 0)[:, 0]
            smallest = np.argwhere(residuals < 0)[:, 0]

            loss = (
                    np.mean((self.quantile * np.abs(residuals[biggest_or_equal]).ravel()))
                    + np.mean(((1 - self.quantile) * np.abs(residuals[smallest])).ravel())
            )
            self.loss.append(loss)
            if self.verbose:
                print('Loss =', loss)

            dw = (
                ((1 - self.quantile) * np.mean(X[smallest], axis=0).reshape(-1, 1) * np.mean(residuals[smallest], axis=0).reshape(-1, 1)) +
                (self.quantile * np.mean(X[biggest_or_equal], axis=0).reshape(-1, 1) * np.mean(residuals[biggest_or_equal], axis=0).reshape(-1, 1))
            )
            db = (
                ((1 - self.quantile) * np.mean(residuals[smallest], axis=0).reshape(-1, 1)) +
                (self.quantile * np.mean(residuals[biggest_or_equal], axis=0).reshape(-1, 1))
            )

            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db
