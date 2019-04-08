import numpy as np
from scipy.stats import mode
from .base import BaseModel
from .decision_tree_classification import DecisionTreeClassifier

EPSILON = 1e-10
BINS_NUMBER = 100

NUMERICAL_KEY = 'numerical'
CATEGORICAL_KEY = 'categorical'

class RandomForestClassifier(BaseModel):
    def __init__(
        self, 
        n_estimators=10,
        subsample=0.7,
        col_subsample=0.7,
        criterion='gini', 
        max_depth=5, 
        leafs_num=-1, 
        max_objects_in_leaf_num=-1,
        min_impurity_decrease=0, 
        use_binning=True,
        verbose=False
    ):
        super(RandomForestClassifier, self).__init__(verbose)
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

    def fit(self, X, y):
        self._check_dimensions(X, y)

        # make it parallel!
        for i in range(self.n_estimators):
            samples_indexes = sorted(np.random.choice(
                np.arange(X.shape[0]), int(X.shape[0] * self.subsample), replace=True
            ))
            columns_indexes = sorted(np.random.choice(
                np.arange(X.shape[1]), int(X.shape[1] * self.col_subsample), replace=False
            ))

            base_model = DecisionTreeClassifier(
                criterion=self.criterion, 
                max_depth=self.max_depth, 
                leafs_num=self.leafs_num, 
                max_objects_in_leaf_num=self.max_objects_in_leaf_num,
                min_impurity_decrease=self.min_impurity_decrease, 
                use_binning=self.use_binning,
                use_rsm=True,
                verbose=self.verbose
            )

            base_model.fit(X[samples_indexes][:,columns_indexes], y[samples_indexes])
            self._ensemble.append(
                base_model
            )

    def predict(self, X):
        if len(X.shape) == 1:
            X = [X]

        predictions = []
        for x in X:
            ensemble_preds = []

            for estimator in self._ensemble:
                ensemble_preds.append(estimator.predict(x))

            predictions.append(mode(ensemble_preds)[0][0])

        return predictions