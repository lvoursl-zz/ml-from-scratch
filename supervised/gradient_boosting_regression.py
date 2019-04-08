import numpy as np

from .base import BaseModel
from .decision_tree_regression import DecisionTreeRegressor

class GradientBoostingRegressor(BaseModel):
    def __init__(
        self, 
        learning_rate=1.0,
        n_estimators=10,
        subsample=0.7,
        col_subsample=0.7,
        criterion='mse', 
        max_depth=5, 
        leafs_num=-1, 
        max_objects_in_leaf_num=-1,
        min_impurity_decrease=0, 
        use_binning=True,
        verbose=False
    ):
        super(GradientBoostingRegressor, self).__init__(verbose)
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.col_subsample = col_subsample
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_objects_in_leaf_num = max_objects_in_leaf_num
        self.min_impurity_decrease = min_impurity_decrease        
        self.use_binning = use_binning
        
        if leafs_num != -1:
            self.leafs_num = leafs_num
        else:
            self.leafs_num = (max_depth + 1) ** 2        

        self._features_num = 0
        self._samples_num = 0
        self._ensemble = []
        self._models_weights = []

    def fit(self, X, y):
        self._check_dimensions(X, y)

        base_model = DecisionTreeRegressor(
            criterion=self.criterion, 
            max_depth=self.max_depth, 
            leafs_num=self.leafs_num, 
            max_objects_in_leaf_num=self.max_objects_in_leaf_num,
            min_impurity_decrease=self.min_impurity_decrease, 
            use_binning=self.use_binning,
            use_rsm=True,
            verbose=self.verbose
        )
        base_model.fit(X, y)

        self._ensemble.append(base_model)
        self._models_weights.append(1)

        for i in range(self.n_estimators):
            if self.criterion == 'mse':
                if self.verbose:
                    print('Train MSE:', np.mean((y - self.predict(X))**2))

                # bagging
                samples_indexes = sorted(np.random.choice(
                    np.arange(X.shape[0]), int(X.shape[0] * self.subsample), replace=True
                ))
                # RSM
                columns_indexes = sorted(np.random.choice(
                    np.arange(X.shape[1]), int(X.shape[1] * self.col_subsample), replace=False
                ))

                residuals = y - self.predict(X)

                model = DecisionTreeRegressor(
                    criterion=self.criterion, 
                    max_depth=self.max_depth, 
                    leafs_num=self.leafs_num, 
                    max_objects_in_leaf_num=self.max_objects_in_leaf_num,
                    min_impurity_decrease=self.min_impurity_decrease, 
                    use_binning=self.use_binning,
                    use_rsm=True,
                    verbose=self.verbose
                )
                model.fit(X[samples_indexes][:,columns_indexes], residuals[samples_indexes])

                weight = 1
                best_loss = 1e10
                grid = np.arange(-10, 10, 0.1)

                for candidate in grid:
                    loss = np.mean(
                        (y - self.predict(X) - self.learning_rate * candidate * model.predict(X)) ** 2
                    )
                    if loss < best_loss:
                        weight = candidate
                        best_loss = loss

                self._ensemble.append(model)
                self._models_weights.append(self.learning_rate * weight)

    def predict(self, X):
        if len(X.shape) == 1:
            X = [X]

        predictions = []
        for x in X:
            ensemble_preds = 0

            for weight, model in zip(self._models_weights, self._ensemble):
                ensemble_preds += weight * model.predict(x)[0]

            predictions.append(ensemble_preds)

        return np.array(predictions)