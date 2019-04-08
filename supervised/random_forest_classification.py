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

            base_model = DecisionTreeClassifierWithRSM(
                criterion=self.criterion, 
                max_depth=self.max_depth, 
                leafs_num=self.leafs_num, 
                max_objects_in_leaf_num=self.max_objects_in_leaf_num,
                min_impurity_decrease=self.min_impurity_decrease, 
                use_binning=self.use_binning,
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


class DecisionTreeClassifierWithRSM(DecisionTreeClassifier):
    def __init__(
        self, 
        criterion='gini', 
        max_depth=5, 
        leafs_num=-1, 
        max_objects_in_leaf_num=-1,
        min_impurity_decrease=0, 
        use_binning=True,
        verbose=False
    ):
        super(DecisionTreeClassifierWithRSM, self).__init__(
            criterion=criterion,
            max_depth=max_depth,
            leafs_num= leafs_num,
            max_objects_in_leaf_num=max_objects_in_leaf_num,
            min_impurity_decrease=min_impurity_decrease,
            use_binning=use_binning,
            verbose=verbose
        )

    def _get_split(self, X, y):
        split_feature_index, split_feature_value = -1, -1
        final_left_indexes, final_right_indexes = [], []
        is_leaf, decision = False, None

        best_criterion_value = -1

        if X.shape[0] < self.max_objects_in_leaf_num:
            is_leaf = True
            decision = self._decision_rule(y)
            return -1, -1, -1, -1, is_leaf, decision

        for j in range(self._features_num):
            # little fix for Random Space Method
            if np.random.random() > 1 / np.sqrt(self._features_num):
                continue

            if self._features_types[j] == NUMERICAL_KEY:
                # hack to solve problems with binning when we
                # have a small number of objects
                if  self.use_binning and len(np.unique(X[:,j])) > BINS_NUMBER * 10:
                    bins = np.linspace(min(X[:,j]), max(X[:,j]), BINS_NUMBER)
                    digitized = np.digitize(X[:,j], bins)
                    bin_means = [X[:,j][digitized == i].mean() for i in range(1, len(bins))]
                else:
                    bin_means = np.unique(X[:,j])

                for bin_value in bin_means:
                    left_indexes = np.where(X[:,j] < bin_value)[0]
                    right_indexes = np.where(X[:,j] >= bin_value)[0]

                    if (len(left_indexes) == 0) or (len(right_indexes) == 0):
                        continue

                    parent_criterion = self._impurity_criterion(y)
                    left_criterion = self._impurity_criterion(y[left_indexes])
                    right_criterion = self._impurity_criterion(y[right_indexes])

                    if parent_criterion < EPSILON:
                        # if variance of parent is equal to zero
                        is_leaf = True
                        decision = self._decision_rule(y)
                        return -1, -1, -1, -1, is_leaf, decision

                    criterion_value = (
                        parent_criterion 
                        - (len(left_indexes) / X.shape[0]) * left_criterion 
                        - (len(right_indexes) / X.shape[0]) * right_criterion
                    )

                    if criterion_value > best_criterion_value:
                        best_criterion_value = criterion_value
                        split_feature_index = j
                        split_feature_value = bin_value

                        final_left_indexes = left_indexes
                        final_right_indexes = right_indexes

            else:
                raise NotImplementedError()

        if best_criterion_value == -1:
            is_leaf = True
            decision = self._decision_rule(y)
            return -1, -1, -1, -1, is_leaf, decision

        if best_criterion_value <= (self.min_impurity_decrease + EPSILON):
            is_leaf = True
            decision = self._decision_rule(y)
            return -1, -1, -1, -1, is_leaf, decision            
        
        return split_feature_index, split_feature_value, final_left_indexes, final_right_indexes, \
            is_leaf, decision        