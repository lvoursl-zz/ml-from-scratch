import numpy as np

from .base import BaseModel

class NearestNeighborsRegression(BaseModel):
    def __init__(self, k=1, distance='euclidean', verbose=False):
        super(NearestNeighborsRegression, self).__init__(verbose=verbose)
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self._check_dimensions(X, y)
        self.X = X
        self.y = y

    def predict(self, X):
        if self.distance == 'euclidean':
            if len(X.shape) == 1:
                X = [X]
            y_hat = []
            for x in X:
                distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
                neighbors_indexes = np.argsort(distances)[:self.k]
                y_hat.append(np.average(self.y[neighbors_indexes]))
            return y_hat
        else:
            raise Exception('support only euclidean distance')