import numpy as np
from scipy.stats import norm

from .base import BaseModel

class GaussianNaiveBayes(BaseModel):
    def __init__(self, verbose=False):
        super(GaussianNaiveBayes, self).__init__(verbose)
        self.features_num = 0
        self.model = dict()
        self.classes = []
        self.classes_probabilities = dict()
        self.distibution = norm

    def fit(self, X, y):
        self._check_dimensions(X, y)
        self.features_num = X.shape[1]

        n = X.shape[0]
        y = y.reshape(y.shape[0], 1)

        for y_ in np.unique(y):
            self.classes.append(y_)
            self.classes_probabilities[y_] = y[y == y_].shape[0] / n
            self.model[y_] = {}

            class_indexes = np.where(y == y_)[0]
            for feature_index in np.arange(self.features_num):
                self.model[y_][feature_index] = {}
                self.model[y_][feature_index]['mean'] = X[class_indexes, feature_index].mean()
                self.model[y_][feature_index]['variance']= X[class_indexes, feature_index].var()


    def predict(self, X):
        if len(X.shape) == 1:
            X = [X]

        predictions = []
        for x in X:
            probabilities = []
            for class_name in self.classes:
                class_probability = self.classes_probabilities[class_name]
                for feature_index in np.arange(self.features_num):
                    loc = self.model[class_name][feature_index]['mean']
                    scale = self.model[class_name][feature_index]['variance']
                    
                    feature_probability = self.distibution.pdf(x[feature_index], loc=loc, scale=scale)
                    # hack to solve numerical stability
                    if feature_probability > 1e-3:
                        class_probability *= feature_probability
                    else:
                        class_probability *= 1e-3
                probabilities.append(class_probability)
            predictions.append(probabilities)

        return predictions

    def predict_classes(self, X):
        return np.argmax(self.predict(X), axis=1)


    def predict_loglikelihood(self, X):
        if len(X.shape) == 1:
            X = [X]

        predictions = []
        for x in X:
            probabilities = []
            for class_name in self.classes:
                class_probability = np.log(self.classes_probabilities[class_name])
                for feature_index in np.arange(self.features_num):
                    loc = self.model[class_name][feature_index]['mean']
                    scale = self.model[class_name][feature_index]['variance']

                    feature_probability = - 0.5 * np.sum(np.log(2. * np.pi * scale))
                    feature_probability -= 0.5 * ((x[feature_index] - loc) ** 2) / (scale)
                    class_probability += feature_probability

                probabilities.append(class_probability)
            predictions.append(probabilities)

        return predictions

    def predict_classes_loglikelihood(self, X):
        return np.argmax(self.predict_loglikelihood(X), axis=1)
