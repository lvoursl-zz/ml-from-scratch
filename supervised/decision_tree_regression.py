import numpy as np

from .base import BaseModel

CATEGORICAL_CONST = 100
BINS_NUMBER = 3

NUMERICAL_KEY = 'numerical'
CATEGORICAL_KEY = 'categorical'

class DecisionTreeRegressor(BaseModel):
    def __init__(
        self, 
        criterion='mse', 
        max_depth=5, 
        leaves_num=10, 
        min_impurity_decrease=0, 
        max_objects_in_leaf_num=-1,
        verbose=False
    ):
        super(DecisionTreeRegressor, self).__init__(verbose)
        self.criterion = criterion
        self.max_depth = max_depth
        self.leaves_num = leaves_num
        self.min_impurity_decrease = min_impurity_decrease
        self.max_objects_in_leaf_num = max_objects_in_leaf_num

        self._features_types = []
        self._features_num = 0
        self._samples_num = 0
        self._tree = {}

    def _infer_features_types(self, X):
        self._samples_num = X.shape[0]
        self._features_num = X.shape[1]

        for i in range(self._features_num):
            if len(np.unique(X[:,i])) < self._samples_num / CATEGORICAL_CONST:
                # TODO: realize categorical features
                # self._features_types.append(CATEGORICAL_KEY)
                self._features_types.append(NUMERICAL_KEY)
            else:
                # TODO: realize categorical features
                self._features_types.append(NUMERICAL_KEY)

    def _get_left_child_index(self, parent_index):
        return parent_index * 2 + 1

    def _get_right_child_index(self, parent_index):
        return parent_index * 2 + 2

    def _impurity_criterion(self, y):
        if self.criterion == 'mse':
            return np.var(y)
        elif self.criterion == 'mae':
            raise NotImplementedError()
        else:
            raise Exception('avalible criterions: MSE and MAE')

    def _get_split(self, X, y):
        split_feature_index, split_feature_value = -1, -1
        final_left_indexes, final_right_indexes = [], []
        is_leaf, decision = False, None

        best_criterion_value = -1

        if X.shape[0] < self.max_objects_in_leaf_num:
            is_leaf = True
            decision = np.mean(y)
            return -1, -1, -1, -1, is_leaf, decision

        for j in range(self._features_num):
            if self._features_types[j] == NUMERICAL_KEY:
                bins = np.linspace(min(X[:,j]), max(X[:,j]), BINS_NUMBER)
                digitized = np.digitize(X[:,j], bins)
                bin_means = [X[:,j][digitized == i].mean() for i in range(1, len(bins))]

                for bin_value in bin_means:
                    left_indexes = np.where(X[:,j] < bin_value)
                    right_indexes = np.where(X[:,j] >= bin_value)

                    parent_criterion = self._impurity_criterion(y)
                    left_criterion = self._impurity_criterion(y[left_indexes])
                    right_criterion = self._impurity_criterion(y[right_indexes])

                    if parent_criterion == 0:
                        is_leaf = True
                        decision = np.mean(y)
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
            decision = np.mean(y)
            return -1, -1, -1, -1, is_leaf, decision
        
        return split_feature_index, split_feature_value, final_left_indexes, final_right_indexes, \
            is_leaf, decision

    def _get_available_nodes(self):
        available_nodes = []

        for node_index, node_data in self._tree.items():
            if (
                node_data['is_leaf'] == False and 
                self._get_left_child_index(node_index) not in self._tree and 
                self._get_right_child_index(node_index) not in self._tree
            ):
                available_nodes.append(node_index)

        return available_nodes

    def fit(self, X, y):
        self._check_dimensions(X, y)
        self._infer_features_types(X)

        split_feature_index, split_feature_value, left_indexes, right_indexes, \
            is_leaf, decision = self._get_split(X, y)

        self._tree[0] = {
            'split_feature_index': split_feature_index,
            'split_feature_value': split_feature_value, 
            'left_indexes': left_indexes, 
            'right_indexes': right_indexes, 
            'is_leaf': is_leaf,
            'decision': decision
        }

        available_nodes = self._get_available_nodes()

        while len(available_nodes) != 0:
            for node_index in available_nodes:
                parent_left_indexes = self._tree[node_index]['left_indexes']
                parent_right_indexes = self._tree[node_index]['right_indexes']
                
                split_feature_index, split_feature_value, left_indexes, right_indexes, \
                    is_leaf, decision = self._get_split(X[parent_left_indexes], y[parent_left_indexes])

                self._tree[self._get_left_child_index(node_index)] = {
                    'split_feature_index': split_feature_index,
                    'split_feature_value': split_feature_value, 
                    'left_indexes': left_indexes, 
                    'right_indexes': right_indexes, 
                    'is_leaf': is_leaf,
                    'decision': decision
                }

                split_feature_index, split_feature_value, left_indexes, right_indexes, \
                    is_leaf, decision = self._get_split(X[parent_right_indexes], y[parent_right_indexes])

                self._tree[self._get_right_child_index(node_index)] = {
                    'split_feature_index': split_feature_index,
                    'split_feature_value': split_feature_value, 
                    'left_indexes': left_indexes, 
                    'right_indexes': right_indexes, 
                    'is_leaf': is_leaf,
                    'decision': decision
                }

            available_nodes = self._get_available_nodes()

    def predict(self, X):
        if len(X.shape) == 1:
            X = [X]

        predictions = []
        for x in X:
            current_node_index = 0
            current_node = self._tree[current_node_index]

            while current_node['is_leaf'] != True:
                split_feature_index = current_node['split_feature_index']
                split_feature_value = current_node['split_feature_value']

                if x[split_feature_index] >= split_feature_value:
                    current_node_index = self._get_right_child_index(current_node_index)
                else:
                    current_node_index = self._get_left_child_index(current_node_index)
                
                current_node = self._tree[current_node_index]

            predictions.append(current_node['decision'])
        
        return predictions